from __future__ import annotations

import math
import struct

import pytest

from simsteer.runtime.telemetry import Telemetry
from simsteer.runtime.telemetry._shm import SharedPage
from simsteer.runtime.telemetry.ac_shm import AcTelemetry, parse_ac_physics, synth_wheel_angle
from simsteer.runtime.telemetry.scs_shm import ScsTelemetry


def physics(packet_id=42, speed=20.0, yaw=-0.075, steer=0.25):
    buf = bytearray(4096)
    struct.pack_into("<i", buf, 0, packet_id)
    struct.pack_into("<2f", buf, 24, steer, speed * 3.6)
    struct.pack_into("<3f", buf, 296, 0.3, yaw, -0.2)
    return buf


def test_parse_physics_offsets_and_units():
    p = parse_ac_physics(physics())
    assert p.packet_id == 42
    assert p.speed_mps == pytest.approx(20.0)
    assert p.steer == pytest.approx(0.25)
    assert p.yaw_rate_rad_s == pytest.approx(-0.075)


@pytest.mark.parametrize("size", [0, 28, 296, 307])
def test_truncated_physics_is_rejected(size):
    with pytest.raises(ValueError, match="physics page"):
        parse_ac_physics(bytes(size))


@pytest.mark.parametrize("speed", [0.0, 0.5, -0.99, 1.01, 20.0, -20.0])
def test_ac_synthesized_angle_and_sign(speed):
    tel = AcTelemetry(opener=lambda *_: physics(speed=speed))
    try:
        assert isinstance(tel, Telemetry)
        expected = 0.0 if abs(speed) < 1.0 else math.atan(-0.075 * 4.0 / speed)
        assert tel.wheel_angle_rad(4.0) == pytest.approx(expected)
        assert tel.user_steer() == tel.game_steer() == pytest.approx(0.25)
    finally:
        tel.close()


def test_synthesized_angle_at_the_exact_speed_threshold():
    assert synth_wheel_angle(0.1, 1.0, 4.0) == pytest.approx(math.atan(0.4))
    assert synth_wheel_angle(0.1, 0.999, 4.0) == 0.0


def test_ac_recovers_on_the_same_reader_after_start_pause_and_restart():
    now = [10.0]
    page = physics(packet_id=0)
    opens = []

    def opener(name, size):
        opens.append((name, size))
        return page

    tel = AcTelemetry(opener=opener, clock=lambda: now[0])
    try:
        assert not tel.available
        assert tel.speed_mps() is None
        assert len(opens) == 1
        page[:] = physics()
        now[0] += 1.1
        assert tel.available
        assert tel.speed_mps() == pytest.approx(20.0)
        now[0] += 1.1
        assert not tel.available
        assert tel.wheel_angle_rad() is None
        struct.pack_into("<i", page, 0, 43)
        assert tel.available
        struct.pack_into("<i", page, 0, 0)
        assert not tel.available
        page[:] = physics(packet_id=1, speed=15)
        now[0] += 1.1
        assert tel.available
        assert tel.speed_mps() == pytest.approx(15.0)
    finally:
        tel.close()


def test_shared_page_releases_empty_handles_and_throttles_retries():
    class Page(bytearray):
        closed = False

        def close(self):
            self.closed = True

    now = [5.0]
    empty, live = Page(1), Page(b"\x01")
    pending = [empty, live]
    page = SharedPage("test-page", 1, lambda b: bool(b[0]),
                      opener=lambda *_: pending.pop(0), clock=lambda: now[0])
    assert empty.closed
    assert page.get() is None
    assert len(pending) == 1
    now[0] += 1.0
    assert page.get() is live
    assert page.get() is live
    page.close()
    assert live.closed


def test_scs_reconnects_without_a_new_reader():
    now = [10.0]
    buf = bytearray(32768)
    tel = ScsTelemetry(opener=lambda *_: buf, clock=lambda: now[0])
    try:
        assert not tel.available
        buf[0] = 1
        struct.pack_into("<Q", buf, 8, 1000)
        struct.pack_into("<f", buf, 948, 22.5)
        struct.pack_into("<f", buf, 1920, 0.02)
        now[0] += 1.1
        assert tel.available
        assert tel.speed_mps() == pytest.approx(22.5)
        assert tel.yaw_rate_rad_s() == pytest.approx(0.02)
        buf[0] = 0
        assert not tel.available
        assert tel.speed_mps() is None
        buf[0] = 1
        now[0] += 1.1
        assert tel.available
    finally:
        tel.close()
