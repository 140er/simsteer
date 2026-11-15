"""In-process force feedback: the wheel gets force only while armed."""

from __future__ import annotations

import pytest

from simsteer.runtime.ffb import bridge as br
from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.ffb.passthrough import FfbPassthrough
from test_ffb_bridge import Clock, Source, constant_game, run_until


def disarmed_bridge(cap=2.0):
    clock = Clock()
    src = Source(clock)
    sink = di.LogSink()
    b = br.FfbBridge(src, sink, cap, attended=True, armed=False, clock=clock)
    return b, src, sink, clock


def test_disarmed_bridge_tracks_the_game_but_never_opens_the_wheel():
    b, src, sink, clock = disarmed_bridge()
    constant_game(src)
    run_until(b, clock, 3.0)
    assert sink.opened == 0 and sink.calls == [] and not b.device_open
    assert b.model.counters.forwarded == 5
    assert b.status == "FFB off until engaged (cap 2%)"


def test_arming_opens_with_the_capped_gain_first_then_replays_the_game():
    b, src, sink, clock = disarmed_bridge(cap=2.0)
    constant_game(src)
    run_until(b, clock, 0.5)
    b.arm(True)
    run_until(b, clock, 0.6)
    assert b.device_open and sink.opened == 1
    ops = [c.op for c in sink.calls]
    assert ops[0] == "SetGain" and sink.calls[0].value == br.capped_gain(10_000, 2.0)
    assert ops[:6] == ["SetGain", "SetGain", "Command", "CreateEffect", "SetParameters", "Start"]
    assert all(c.value <= br.capped_gain(10_000, 2.0) for c in sink.calls if c.op == "SetGain")


def test_disarming_releases_the_wheel_within_one_step():
    b, src, sink, clock = disarmed_bridge()
    constant_game(src)
    b.arm(True)
    run_until(b, clock, 0.5)
    assert b.device_open
    b.arm(False)
    t0 = clock.t
    b._step()
    assert not b.device_open and sink.device is None
    assert clock.t - t0 <= br.IDLE_WAIT_S + 1e-9
    assert b.stats.device_events[-1].endswith("wheel released")
    calls = len(sink.calls)
    run_until(b, clock, 2.0)  # stays released while disarmed
    assert len(sink.calls) == calls and sink.opened == 1


def test_rearming_reopens_and_replays():
    b, src, sink, clock = disarmed_bridge()
    constant_game(src)
    b.arm(True)
    run_until(b, clock, 0.5)
    b.arm(False)
    run_until(b, clock, 1.0)
    sink.calls.clear()
    b.arm(True)
    run_until(b, clock, 1.1)
    assert b.device_open and sink.opened == 2
    assert [c.op for c in sink.calls][:6] == ["SetGain", "SetGain", "Command", "CreateEffect", "SetParameters", "Start"]


def test_standalone_cli_bridge_stays_armed_by_default():
    clock = Clock()
    b = br.FfbBridge(Source(clock), di.LogSink(), 5.0, clock=clock)
    assert b.armed
    assert not br.FfbBridge(Source(clock), di.LogSink(), 5.0, attended=False, clock=clock).armed


class FakeBridge:
    def __init__(self):
        self.armed_log = []
        self.running = False
        self.stopped = False
        self.status = "FFB off until engaged (cap 2%)"

    def start(self):
        self.running = True

    def arm(self, on):
        self.armed_log.append(on)

    def stop(self):
        self.stopped = True
        self.running = False


class FakeReceiver:
    closed = False

    def close(self):
        self.closed = True


def test_passthrough_arms_only_while_active_and_releases_on_close():
    fake, rx, lines = FakeBridge(), FakeReceiver(), []
    p = FfbPassthrough(2.0, say=lines.append, make_bridge=lambda: (fake, rx))
    assert p.start()
    p.update(True)
    p.update(False)
    p.close()
    assert fake.armed_log == [True, False, False] and fake.stopped and rx.closed
    assert "cap 2%" in lines[0]


def test_passthrough_never_arms_a_stopped_bridge():
    fake = FakeBridge()
    p = FfbPassthrough(2.0, say=lambda line: None, make_bridge=lambda: (fake, FakeReceiver()))
    p.start()
    fake.running = False  # bridge thread died (e.g. the wheel refused the cap)
    p.update(True)
    assert fake.armed_log == [False]


def test_passthrough_start_failure_disables_ffb_without_raising():
    lines = []

    def boom():
        raise OSError("vJoy FFB: owned by another process")

    p = FfbPassthrough(2.0, say=lines.append, make_bridge=boom)
    assert not p.start()
    p.update(True)
    p.close()
    assert lines == ["FFB disabled: vJoy FFB: owned by another process"]


@pytest.mark.parametrize("cap", [0, 16, float("nan")])
def test_passthrough_refuses_caps_outside_1_to_15(cap):
    with pytest.raises(ValueError):
        FfbPassthrough(cap)
