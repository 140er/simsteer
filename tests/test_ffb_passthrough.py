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


# --------------------------------------------------------------------------
# the live loop arms force feedback only while engaged and releases it on
# every stop path


class Recorder:
    instances: list["Recorder"] = []

    def __init__(self, cap, vjoy_device=1):
        self.cap = cap
        self.updates: list[bool] = []
        self.closed = False
        Recorder.instances.append(self)

    def start(self):
        return True

    def update(self, active):
        self.updates.append(bool(active))

    def close(self):
        self.closed = True


def drive_with_ffb(monkeypatch, **kw):
    import functools

    import simsteer.runtime.loop as loop
    import simsteer.runtime.ffb.passthrough as passthrough
    import simsteer.runtime.output.base as devices
    from test_output_safety import drive

    Recorder.instances.clear()
    monkeypatch.setattr(passthrough, "FfbPassthrough", Recorder)
    monkeypatch.setattr(devices.DeviceManager, "kind", property(lambda self: "wheel"))
    monkeypatch.setattr(loop, "run", functools.partial(loop.run, ffb_cap=2.0))
    st, pad, manager = drive(monkeypatch, **kw)
    (ffb,) = Recorder.instances
    return st, ffb


def test_loop_arms_ffb_while_engaged_and_releases_it_on_a_fault(monkeypatch):
    st, ffb = drive_with_ffb(monkeypatch, fault="vision")
    assert ffb.cap == 2.0
    assert True in ffb.updates  # armed during the genuinely engaged frames
    first_release = ffb.updates.index(False, ffb.updates.index(True))
    assert not any(ffb.updates[first_release:])  # never re-armed after the fault
    assert ffb.closed


def test_loop_releases_ffb_when_insert_disengages(monkeypatch):
    from simsteer.app.global_keys import KEY_INSERT

    st, ffb = drive_with_ffb(monkeypatch, no_frames=True, key=KEY_INSERT)
    assert True in ffb.updates
    assert ffb.updates[-1] is False and ffb.closed


def test_loop_without_ffb_cap_never_creates_a_passthrough(monkeypatch):
    import simsteer.runtime.ffb.passthrough as passthrough
    from test_output_safety import drive

    Recorder.instances.clear()
    monkeypatch.setattr(passthrough, "FfbPassthrough", Recorder)
    drive(monkeypatch, fault="vision")
    assert Recorder.instances == []


# --------------------------------------------------------------------------
# gate follow-ups


def test_disarm_mid_step_sends_nothing_more_to_the_wheel():
    b, src, sink, clock = disarmed_bridge()
    constant_game(src)
    b.arm(True)
    run_until(b, clock, 0.5)
    calls = len(sink.calls)
    b.arm(False)
    b._execute(b.model.rearm(clock.t + 1.0))  # a renewal already computed
    assert len(sink.calls) == calls


def test_passthrough_reports_a_bridge_that_dies_at_start():
    fake, rx, lines = FakeBridge(), FakeReceiver(), []
    fake.start = lambda: None  # thread never runs
    fake.status = "FFB stopped: wheel refused the capped gain"
    p = FfbPassthrough(2.0, say=lines.append, make_bridge=lambda: (fake, rx))
    assert not p.start()
    assert lines == ["FFB disabled: FFB stopped: wheel refused the capped gain"] and rx.closed


def test_close_keeps_the_receiver_while_a_stuck_bridge_thread_uses_it():
    fake, rx, lines = FakeBridge(), FakeReceiver(), []
    p = FfbPassthrough(2.0, say=lines.append, make_bridge=lambda: (fake, rx))
    p.start()
    fake.stop = lambda: None  # join timed out, still running
    p.close()
    assert fake.armed_log[-1] is False and not rx.closed
    assert lines[-1] == "FFB bridge did not stop in time; vJoy receiver left open"


def test_ffb_cap_is_refused_for_a_replay(tmp_path, capsys):
    from simsteer.app import main as app
    from simsteer.games.base import registry

    args = app._build_parser(registry().ids).parse_args(
        ["--device", "wheel", "--ffb-cap", "2"])
    assert app._main(args, registry(), tmp_path) == 2
    assert "does not apply to a replay" in capsys.readouterr().out
