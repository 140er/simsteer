from __future__ import annotations

from types import SimpleNamespace

import pytest

from simsteer.runtime.telemetry.ac_shm import parse_ac_physics
from simsteer.runtime.telemetry.forza_udp import parse_forza_dash
from simsteer.runtime.telemetry.scs_shm import ScsTelemetry
from tools import fake_telemetry as fake


def test_fake_packets_match_the_adapter_contract():
    for length in (311, 324):
        p = parse_forza_dash(fake.forza_packet(length, True, 21.5, 0.03, -0.5))
        assert p.speed_mps == pytest.approx(21.5)
        assert p.yaw_rate_rad_s == pytest.approx(0.03)
        assert p.steer == pytest.approx(-64 / 127)
    ac = parse_ac_physics(fake.ac_physics(12, 21.5, 0.03, -0.5))
    assert ac.packet_id == 12
    assert ac.speed_mps == pytest.approx(21.5)
    assert ac.yaw_rate_rad_s == pytest.approx(0.03)
    page = bytearray(32768)
    for offset, data in fake.scs_page(12, 21.5, 0.03, -0.5):
        page[offset:offset + len(data)] = data
    scs = ScsTelemetry(opener=lambda *_: page)
    try:
        assert scs.speed_mps() == pytest.approx(21.5)
        assert scs.yaw_rate_rad_s() == pytest.approx(0.03)
    finally:
        scs.close()


@pytest.mark.parametrize("extra", [
    ["--hz", "0"], ["--period", "0"], ["--seconds", "-1"],
    ["--gap", "bad"], ["--gap", "10:5"], ["--port", "65536"],
])
def test_fake_sender_rejects_invalid_arguments(extra):
    with pytest.raises(SystemExit) as e:
        fake.main(["forza", *extra])
    assert e.value.code == 2


def test_fake_sender_resumes_at_its_requested_rate_after_a_gap(monkeypatch):
    now = [0.0]
    sent = []

    class Sender:
        def sendto(self, packet, destination):
            sent.append(now[0])

        def close(self):
            pass

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(fake, "time", SimpleNamespace(
        monotonic=lambda: now[0], time=lambda: now[0], sleep=sleep))
    monkeypatch.setattr(fake.socket, "socket", lambda *_: Sender())
    assert fake.main(["forza", "--seconds", "1.2", "--hz", "10",
                      "--gap", "0.2:0.8"]) == 0
    assert 4 <= len(sent) <= 7
    assert all(b - a >= 0.099 for a, b in zip(sent, sent[1:]))
