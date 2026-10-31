"""Force feedback on the simulated base: the real DirectInputSink, the real
vJoy receiver and bridge, and what the base received."""

from __future__ import annotations

import ctypes
import json
import math
import time

import pytest

from simsteer.core import ffb_model as fm
from simsteer.runtime.ffb import bridge as br
from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.ffb import vjoy_ffb as vj
from simsteer.testing.fanatec import FanatecSim, FfbLog, level_of, load_packets
import ffb_fixtures as fx

FULL = fm.DIEP_AXES | fm.DIEP_DIRECTION | fm.DIEP_TYPESPECIFICPARAMS | fm.DIEP_DURATION | fm.DIEP_GAIN


class Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def constant(magnitude=5000, duration=fm.INFINITE, gain=fm.DI_FFNOMINALMAX, **kw):
    return fm.DiParams(duration=duration, gain=gain, type_specific=fm.DiConstant(magnitude), **kw)


def constant_game(t0=0.0, magnitude=4000):
    return [(t0, *fx.gain(255)), (t0 + 0.001, *fx.new_effect(1, 1)), (t0 + 0.002, *fx.constant(1, magnitude)),
            (t0 + 0.003, *fx.effect_report(1, 1)), (t0 + 0.004, *fx.effect_op(1, fm.OP_START))]


def wait_for(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


# --------------------------------------------------------------------------
# the sink on the base


def test_sink_opens_with_the_dd2_order_and_closes_clean():
    with FanatecSim() as sim:
        sink = di.DirectInputSink()
        assert sink.open(500)
        assert sink.device.is_fanatec and sink.device.instance == sim.wheel.instances[0].instance
        assert sink.axis_id == 0x01000002
        ops = [(e["op"], e.get("detail"), e["hr"]) for e in sim.ffb.calls()]
        assert ops == [("SetProperty", "AUTOCENTER=0", "DI_OK"), ("Acquire", "exclusive", "DI_OK"),
                       ("SetProperty", "AUTOCENTER=0", "0x800700AA"),  # the DD2 refuses it once acquired
                       ("SetProperty", "FFGAIN=500", "DI_OK")]
        assert sim.wheel.autocenter == 0
        sink.execute(fm.DiCall("CreateEffect", 1, "ConstantForce"))
        assert sink.execute(fm.DiCall("SetParameters", 1, "ConstantForce", FULL | fm.DIEP_START, constant())) == 0
        assert sim.wheel.force() == pytest.approx(0.05 * 0.5)
        sink.close()
        tail = [(e["op"], e.get("detail")) for e in sim.ffb.calls()][-3:]
        assert tail == [("SendForceFeedbackCommand", "STOPALL"), ("Unload", None), ("Release", None)]
        assert sim.wheel.force() == 0 and sim.wheel.exclusive is None
        sim.ffb.assert_gain_set_before_first_start(cap=0.05)
        sim.assert_no_leaks()
        sim.assert_single_thread()


def test_refused_gain_closes_the_wheel_before_any_effect():
    with FanatecSim() as sim:
        sim.wheel.refuse_gain = lambda value: True
        with pytest.raises(di.GainRefused):
            di.DirectInputSink().open(500)
        assert not sim.ffb.calls("CreateEffect") and sim.ffb.plays() == []
        sim.assert_no_leaks()


def test_effect_lifecycle_on_the_simulator_clock():
    clock = Clock()
    with FanatecSim(clock=clock) as sim:
        sink = di.DirectInputSink()
        sink.open(10000)
        sink.execute(fm.DiCall("CreateEffect", 1, "ConstantForce"))
        sink.execute(fm.DiCall("SetParameters", 1, "ConstantForce", FULL | fm.DIEP_START,
                               constant(4000, duration=200_000)))
        clock.t += 0.1
        assert sim.wheel.force() == pytest.approx(0.4)
        clock.t += 0.15
        assert sim.wheel.force() == 0  # its 200 ms ended without a renewal
        sink.execute(fm.DiCall("CreateEffect", 2, "Sine"))
        sink.execute(fm.DiCall("SetParameters", 2, "Sine", FULL, fm.DiParams(
            duration=fm.INFINITE, type_specific=fm.DiPeriodic(3000, -1000, 0, 100_000))))
        assert sink.execute(fm.DiCall("Start", 1, iterations=fm.INFINITE)) == 0
        assert sink.execute(fm.DiCall("Start", 2, iterations=fm.INFINITE)) == 0
        assert sim.wheel.force() == pytest.approx(0.8)
        clock.t += 0.01
        sink.execute(fm.DiCall("SetParameters", 1, "ConstantForce", fm.DIEP_TYPESPECIFICPARAMS,
                               constant(9000)))  # update while playing
        assert sim.wheel.force() == pytest.approx(1.0)  # summed, clipped
        sink.execute(fm.DiCall("SetGain", value=2500))
        assert sim.wheel.force() == pytest.approx(0.25)
        sink.execute(fm.DiCall("Start", 2, iterations=fm.INFINITE, flags=fm.DIES_SOLO))
        assert sim.wheel.force() == pytest.approx(0.1)
        sink.execute(fm.DiCall("Command", flags=fm.DISFFC_PAUSE))
        assert sim.wheel.force() == 0
        sink.execute(fm.DiCall("Command", flags=fm.DISFFC_CONTINUE))
        clock.t += 0.01
        sink.execute(fm.DiCall("Stop", 2))
        assert sim.wheel.force() == 0
        assert sim.ffb.force_at(100.255) == pytest.approx(0.8)  # history stays queryable
        sink.close()
        sim.assert_no_leaks()


def test_effects_obey_directinput_and_dd2_rules():
    with FanatecSim() as sim:
        sink = di.DirectInputSink()
        sink.open(500)
        sink.execute(fm.DiCall("CreateEffect", 1, "ConstantForce"))
        ptr = sink._effects[1]
        eff, keep = di.dieffect_for(constant(), sink.axis_id)

        def set_params(flags):
            return di._com_call(ptr, "IDirectInputEffect", "SetParameters", [], ctypes.byref(eff), flags)

        eff.dwFlags = fm.DIEFF_OBJECTOFFSETS | fm.DIEFF_CARTESIAN
        assert set_params(FULL) == di.DIERR_INVALIDPARAM  # what the DD2 answered on hardware
        eff.dwFlags = fm.DIEFF_OBJECTIDS | fm.DIEFF_CARTESIAN
        eff.rgdwAxes[0] = 0x00000102  # not the actuator
        assert set_params(FULL) == di.DIERR_INVALIDPARAM
        eff.rgdwAxes[0] = sink.axis_id
        eff.cbTypeSpecificParams = 8  # a ramp block on a constant effect
        assert set_params(FULL) == di.DIERR_INVALIDPARAM
        eff.cbTypeSpecificParams = 4
        assert set_params(fm.DIEP_DURATION) == 3  # DI_DOWNLOADSKIPPED: not complete yet
        assert sink.execute(fm.DiCall("Start", 1)) == di.DIERR_INCOMPLETEEFFECT  # no axes or force yet
        assert set_params(FULL | fm.DIEP_START) == di.DI_OK
        assert sim.wheel.force() > 0
        assert di._dev(sink._ptr, "Unacquire", []) == di.DI_OK
        assert sim.wheel.force() == 0  # unacquiring stops the device's effects
        assert sink.execute(fm.DiCall("Start", 1)) == di.DIERR_NOTEXCLUSIVEACQUIRED
        assert di._com_call(ptr, "IDirectInputEffect", "Unload", []) == di.DIERR_NOTEXCLUSIVEACQUIRED
        assert set_params(FULL) == 3  # DI_DOWNLOADSKIPPED
        sink.close()
        sim.assert_no_leaks()


def test_safety_switch_cuts_the_force():
    clock = Clock()
    with FanatecSim(clock=clock) as sim:
        sink = di.DirectInputSink()
        sink.open(1500)
        sink.execute(fm.DiCall("CreateEffect", 1, "Spring"))
        sink.execute(fm.DiCall("SetParameters", 1, "Spring", FULL | fm.DIEP_START, fm.DiParams(
            duration=fm.INFINITE, type_specific=fm.DiCondition(0, 5000, 5000, 8000, 6000, 0))))
        assert sim.wheel.force() == pytest.approx(0.8 * 0.15)
        clock.t += 1
        sim.wheel.emergency_stop()
        assert sim.wheel.force() == 0
        state = ctypes.c_uint32()
        di._dev(sink._ptr, "GetForceFeedbackState", [], ctypes.byref(state))
        assert state.value & 0x200  # DIGFFS_SAFETYSWITCHOFF
        sim.wheel.safety_switch(True)
        assert sim.wheel.force() > 0
        sink.close()


# --------------------------------------------------------------------------
# the real bridge on the base


def run_bridge(sim, packets, cap, *, speed=1.0, settle=0.3, reopen_s=br.REOPEN_S, during=None):
    rx = vj.VJoyFfbReceiver(1)
    assert rx.open().ok
    bridge = br.FfbBridge(rx, di.DirectInputSink(), cap, reopen_s=reopen_s)
    bridge.start()
    sender = sim.game.send(packets, realtime=True, speed=speed)
    try:
        sender.join(10)
        if during:
            during(bridge)
        time.sleep(settle)
    finally:
        bridge.stop()
        stopped = sim.clock()
        rx.close()
    return bridge, rx, sender, stopped


@pytest.mark.parametrize("fixture,cap,speed", [
    ("recorded_sender_default", 5, 10.0),
    ("synthetic_all_types", 15, 2.0),
    ("synthetic_directions", 1, 1.0),
])
def test_game_forces_reach_the_base_capped(fixture, cap, speed):
    with FanatecSim() as sim:
        bridge, rx, sender, stopped = run_bridge(sim, fixture, cap, speed=speed)
        packets = load_packets(fixture)
        assert sender.delivered == len(packets) and sender.dropped == 0
        assert rx.stats.received == len(packets)
        s = bridge.summary()
        model, _ = fm.run_packets(json.loads((fx.FIXTURES / f"{fixture}.json").read_text())["packets"])
        # The same counts as the model alone: the synthetic set has deliberate unknowns.
        assert (s["unknown"], s["orphans"], s["adapted"]) == (
            model.counters.unknown, model.counters.orphans, model.counters.adapted)
        assert (s["errors"], s["failure"]) == (0, None)
        if fixture.startswith("recorded"):
            assert rx.stats.mismatches == 0  # cross-checked against the recorded DLL readings
        assert sim.ffb.plays(), "nothing played"
        sim.ffb.assert_force_never_above(cap / 100)
        sim.ffb.assert_gain_set_before_first_start(cap / 100)
        sim.ffb.assert_all_stopped(by=stopped)
        sim.ffb.assert_quiet_after(stopped)
        assert sim.ffb.commands()[-1] == "STOPALL"
        sim.assert_no_leaks()
        sim.assert_single_thread()


def game_with_reset(enact: bool):
    """A constant, a PID Device Reset, optionally Enable Actuators, then a new constant."""
    after = [(0.25, *fx.control(fm.CTRL_ENACT))] if enact else []
    return (constant_game() + [(0.2, *fx.control(fm.CTRL_DEVRST))] + after
            + [(t + 0.3, cmd, data) for t, cmd, data in constant_game(magnitude=-6000)])


@pytest.mark.parametrize("enact", [True, False], ids=["reset-then-enable", "reset-only"])
def test_game_reset_disables_the_actuators_until_the_game_enables_them(enact):
    with FanatecSim() as sim:
        marks = {}

        def note_second_effect(bridge):
            wait_for(lambda: any(e["t"] > sim.ffb.plays()[0]["t"] + 0.25 for e in sim.ffb.plays()))
            marks["second"] = sim.clock()

        run_bridge(sim, game_with_reset(enact), 5, during=note_second_effect)
        gains = [g for _, g in sim.ffb.gain_timeline()]
        assert 10000 in gains and gains[-1] == 500  # reset to the wheel's own gain, then capped again
        sim.ffb.assert_force_never_above(0.05)
        assert "RESET" in sim.ffb.commands()
        # The bridge forwards the game's commands only: after a reset with no
        # Enable Actuators the second effect plays on a disabled motor.
        expected = 0.6 * 0.05 if enact else 0.0
        assert sim.ffb.force_at(marks["second"]) == pytest.approx(expected)


def test_reset_disables_the_actuators_until_actuators_on():
    clock = Clock()
    with FanatecSim(clock=clock) as sim:
        sink = di.DirectInputSink()
        sink.open(10000)
        sink.execute(fm.DiCall("CreateEffect", 1, "ConstantForce"))
        assert sink.execute(fm.DiCall("Command", flags=fm.DISFFC_RESET)) == 0
        assert not sim.wheel.actuators
        sink.execute(fm.DiCall("CreateEffect", 2, "ConstantForce"))
        assert sink.execute(fm.DiCall("SetParameters", 2, "ConstantForce", FULL | fm.DIEP_START, constant())) == 0
        assert sim.wheel.force() == 0  # playing, but the actuators are off
        clock.t += 0.1
        sink.execute(fm.DiCall("Command", flags=fm.DISFFC_SETACTUATORSON))
        assert sim.wheel.force() == 0.5
        sink.close()
    with FanatecSim(clock=clock, reset_disables_actuators=False) as sim:
        sink = di.DirectInputSink()
        sink.open(10000)
        sink.execute(fm.DiCall("Command", flags=fm.DISFFC_RESET))
        sink.execute(fm.DiCall("CreateEffect", 1, "ConstantForce"))
        sink.execute(fm.DiCall("SetParameters", 1, "ConstantForce", FULL | fm.DIEP_START, constant()))
        assert sim.wheel.actuators and sim.wheel.force() == 0.5
        sink.close()


def test_pause_rules():
    with FanatecSim() as sim:
        sink = di.DirectInputSink()
        sink.open(10000)
        assert sink.execute(fm.DiCall("Command", flags=fm.DISFFC_CONTINUE)) == di.DIERR_INVALIDPARAM  # not paused
        sink.execute(fm.DiCall("CreateEffect", 1, "ConstantForce"))
        sink.execute(fm.DiCall("SetParameters", 1, "ConstantForce", FULL | fm.DIEP_START, constant(3000)))
        assert sink.execute(fm.DiCall("Command", flags=fm.DISFFC_PAUSE)) == 0
        sink.execute(fm.DiCall("CreateEffect", 2, "ConstantForce"))
        new = fm.DiCall("SetParameters", 2, "ConstantForce", FULL | fm.DIEP_START, constant())
        assert sink.execute(new) == di.DIERR_INVALIDPARAM  # nothing starts while paused
        assert sink.execute(fm.DiCall("SetParameters", 1, "ConstantForce", fm.DIEP_TYPESPECIFICPARAMS,
                                      constant(9000))) == di.DIERR_INVALIDPARAM  # nor changes
        assert sink.execute(fm.DiCall("SetParameters", 2, "ConstantForce", FULL | fm.DIEP_NODOWNLOAD,
                                      constant())) == 0  # storing without download is fine
        assert sim.wheel.force() == 0
        assert sink.execute(fm.DiCall("Command", flags=fm.DISFFC_CONTINUE)) == 0
        assert sim.wheel.force() == 0.3 and sink.execute(new) == 0 and sim.wheel.force() == 0.8
        sink.close()


def test_unplugged_wheel_is_reopened_with_the_cap_before_any_force():
    with FanatecSim() as sim:
        marks = {}

        def unplug_and_back(bridge):
            wait_for(lambda: sim.ffb.plays())
            marks["unplug"] = sim.clock()
            sim.wheel.unplug()
            wait_for(lambda: not bridge.device_open)
            time.sleep(0.2)
            assert sim.wheel.force() == 0
            marks["replug"] = sim.clock()
            sim.wheel.replug()
            wait_for(lambda: bridge.device_open)

        bridge, *_ = run_bridge(sim, constant_game(), 5, reopen_s=0.05, during=unplug_and_back)
        assert any("wheel lost: DIERR_INPUTLOST" in e for e in bridge.stats.device_events)
        log = sim.ffb
        assert log.force_at(marks["unplug"] + 0.1) == 0
        after = [e for e in log.plays() if e["t"] > marks["replug"]]
        assert after, "the game's effect was not replayed"
        gains = [(t, g) for t, g in log.gain_timeline() if t > marks["replug"]]
        assert gains[0][1] == 10000 and gains[1][1] == 500 and gains[1][0] <= after[0]["t"]
        log.assert_force_never_above(0.05)
        sim.assert_no_leaks()


def test_waiting_without_a_wheel_holds_no_directinput_objects():
    with FanatecSim("none") as sim:
        run_bridge(sim, constant_game(), 5, reopen_s=0.05, settle=0.2)
        sim.assert_no_leaks()


# --------------------------------------------------------------------------
# the force model and the log


def test_effect_levels():
    env = fm.DiEnvelope(attack_level=9000, attack_time=1000, fade_level=0, fade_time=0)
    assert level_of("ConstantForce", constant(-4000)) == 0.4
    assert level_of("ConstantForce", constant(4000, gain=5000)) == 0.2
    assert level_of("ConstantForce", constant(4000, envelope=env)) == 0.9
    assert level_of("RampForce", fm.DiParams(type_specific=fm.DiRamp(-2000, 7000))) == 0.7
    assert level_of("Sine", fm.DiParams(type_specific=fm.DiPeriodic(3000, -2000, 0, 1))) == 0.5
    assert level_of("Spring", fm.DiParams(type_specific=fm.DiCondition(0, 1, 1, 3000, 6000, 0))) == 0.6
    assert level_of("Damper", fm.DiParams(type_specific=fm.DiCondition(0, 100, 0, 0, 0, 0))) == 1.0
    assert level_of("ConstantForce", constant(20000)) == 1.0


def test_log_from_events_and_a_torn_file(tmp_path):
    events = [
        {"t": 0.0, "ev": "gain", "value": 500},
        {"t": 1.0, "ev": "play", "effect": 1, "kind": "ConstantForce", "level": 1.0, "start": 1.0, "until": 1.2},
        {"t": 1.1, "ev": "play", "effect": 1, "kind": "ConstantForce", "level": 0.5, "start": 1.1, "until": 1.3},
        {"t": 1.15, "ev": "play", "effect": 2, "kind": "Sine", "level": 0.4, "start": 1.15, "until": None},
        {"t": 1.25, "ev": "end", "effect": 2},
    ]
    path = tmp_path / "ev.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + '\n{"t": 1.3, "ev": "pl')
    log = FfbLog.load(path)
    assert log.force_at(0.5) == 0 and log.force_at(1.05) == 0.05 and log.force_at(1.2) == pytest.approx(0.045)
    assert log.force_at(1.27) == pytest.approx(0.025) and log.force_at(1.3) == 0
    assert log.peak_force() == 0.05 and log.last_force_time() == 1.3
    log.assert_force_never_above(0.05)
    log.assert_all_stopped(by=1.3)
    log.assert_all_stopped(after=1.1, within_ms=200)
    with pytest.raises(AssertionError, match="still on"):
        log.assert_all_stopped(after=1.0, within_ms=250)
    with pytest.raises(AssertionError, match="force reached"):
        log.assert_force_never_above(0.04)
    with pytest.raises(AssertionError, match="effect start"):
        log.assert_quiet_after(1.12)
    with pytest.raises(AssertionError, match="read-only"):
        FfbLog.from_events([{"t": 0, "ev": "call", "op": "Acquire", "hr": "DI_OK", "detail": "exclusive"}]
                           ).assert_no_force_writes()
    unbounded = FfbLog.from_events([{"t": 0, "ev": "play", "effect": 1, "kind": "x", "level": 1, "until": None}])
    assert unbounded.last_force_time() == math.inf


def test_gain_checks_catch_an_uncapped_start():
    log = FfbLog.from_events([
        {"t": 1.0, "ev": "play", "effect": 1, "kind": "ConstantForce", "level": 0.01, "start": 1.0, "until": None}])
    with pytest.raises(AssertionError, match="before any device gain"):
        log.assert_gain_set_before_first_start()
    with pytest.raises(AssertionError, match="device gain 10000"):
        log.assert_force_never_above(0.15)  # tiny force, but the wheel was never capped
    rising = FfbLog.from_events([{"t": 0, "ev": "gain", "value": 500}, {"t": 2, "ev": "gain", "value": 1500}])
    with pytest.raises(AssertionError, match="gain rose"):
        rising.assert_quiet_after(1)
