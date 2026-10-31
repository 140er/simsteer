from __future__ import annotations

import math
import socket
import struct
import time

import pytest

from simsteer.runtime.telemetry.forza_udp import ForzaTelemetry, parse_forza_dash


def packet(length=324, *, race_on=True, speed=23.5, yaw=-0.12, steer=-64):
    buf = bytearray(length)
    struct.pack_into("<i", buf, 0, int(race_on))
    struct.pack_into("<3f", buf, 32, 3.0, 4.0, 12.0)
    struct.pack_into("<3f", buf, 44, 0.9, yaw, 0.7)
    if length >= 311:
        shift = 12 if length >= 324 else 0
        struct.pack_into("<f", buf, 244 + shift, speed)
        struct.pack_into("<b", buf, 308 + shift, steer)
    return bytes(buf)


@pytest.mark.parametrize("length", [311, 324])
def test_parse_dash_offsets_units_and_sign(length):
    p = parse_forza_dash(packet(length))
    assert p.race_on
    assert p.speed_mps == pytest.approx(23.5)
    assert p.yaw_rate_rad_s == pytest.approx(-0.12)
    assert p.steer == pytest.approx(-64 / 127.0)
    assert not parse_forza_dash(packet(length, race_on=False)).race_on


def test_sled_uses_velocity_magnitude_and_has_no_steer():
    p = parse_forza_dash(packet(232))
    assert p.speed_mps == pytest.approx(13.0)
    assert p.yaw_rate_rad_s == pytest.approx(-0.12)
    assert p.steer is None


@pytest.mark.parametrize("size", [0, 4, 48, 231])
def test_short_packet_is_ignored(size):
    assert parse_forza_dash(bytes(size)) is None


@pytest.mark.parametrize("field", ["speed", "yaw"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_packet_is_ignored(field, value):
    assert parse_forza_dash(packet(**{field: value})) is None


def test_signed_byte_minimum_stays_in_axis_range():
    assert parse_forza_dash(packet(steer=-128)).steer == -1.0


def wait_for(predicate):
    until = time.monotonic() + 2.0
    while not predicate():
        assert time.monotonic() < until, "UDP receiver did not update"
        time.sleep(0.005)


def test_udp_reader_keeps_one_socket_through_menus_and_packet_gaps(capsys):
    now = [10.0]
    tel = ForzaTelemetry(port=0, host="localhost", clock=lambda: now[0])
    receiver = tel._sock
    thread = tel._thread
    try:
        assert not tel.available
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            dest = ("localhost", tel.port)
            sender.sendto(packet(), dest)
            wait_for(lambda: tel.available)
            assert tel.speed_mps() == pytest.approx(23.5)
            assert tel.game_steer() == pytest.approx(-64 / 127)
            assert tel.user_steer() is None
            assert tel.wheel_angle_rad(4.0) == pytest.approx(math.atan(-0.12 * 4 / 23.5))
            sender.sendto(packet(race_on=False), dest)
            wait_for(lambda: not tel.available)
            assert tel.game_steer() is None
            for _ in range(100):
                assert not tel.available
            assert tel._sock is receiver
            sender.sendto(packet(311, speed=19.0), dest)
            wait_for(lambda: tel.speed_mps() == 19.0)
            now[0] += 10.0
            assert not tel.available
            assert tel.speed_mps() is None
            sender.sendto(packet(speed=21.0), dest)
            wait_for(lambda: tel.speed_mps() == 21.0)
            assert tel._sock is receiver
            assert tel._thread is thread
        assert capsys.readouterr().out.count("listening on UDP port") == 1
    finally:
        tel.close()
    assert not thread.is_alive()


def test_udp_bind_failure_recovers_on_the_same_reader():
    now = [10.0]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as busy:
        busy.bind(("localhost", 0))
        tel = ForzaTelemetry(port=busy.getsockname()[1], host="localhost",
                             clock=lambda: now[0])
        assert not tel.available
        assert tel._sock is None
    try:
        now[0] += 5.1
        assert not tel.available
        assert tel._sock is not None
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(packet(), ("localhost", tel.port))
            wait_for(lambda: tel.available)
    finally:
        tel.close()
