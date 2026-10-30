"""The FFB bridge with a fake vJoy source and the logging sink, plus the
ctypes layouts the DirectInput and vJoy layers depend on."""

from __future__ import annotations

import ctypes

import pytest

from simsteer.core import ffb_model as fm
from simsteer.runtime.ffb import bridge as br
from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.ffb import vjoy_ffb as vj
from simsteer.runtime.ffb.vjoy_ffb import Packet
import ffb_fixtures as fx


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class Source:
    """Hands out packets at their time; waiting advances the clock."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.pending: list[Packet] = []
        self.mismatches: list[list[str]] = []

    def push(self, t: float, pkt: tuple[int, bytes], dll=None) -> None:
        self.pending.append(Packet(t, pkt[0], pkt[1], dll))

    def get(self, timeout):
        if self.pending and self.pending[0].t <= self.clock.t + (timeout or 0):
            p = self.pending.pop(0)
            self.clock.t = max(self.clock.t, p.t)
            return p
        self.clock.t += timeout or 0
        return None

    def note_mismatch(self, problems, packet):
        self.mismatches.append(problems)


def make(cap=5.0, present=True, attended=True, devices=()):
    clock = Clock()
    src = Source(clock)
    sink = di.LogSink(present=present)
    b = br.FfbBridge(src, sink, cap, attended=attended, clock=clock, list_devices=lambda: list(devices))
    b._try_open(0.0)
    return b, src, sink, clock


def run_until(b, clock, t):
    while clock.t < t:
        b._step()


def constant_game(src, t0=0.0):
    src.push(t0, fx.gain(255))
    src.push(t0 + 0.001, fx.new_effect(1, 1))
    src.push(t0 + 0.002, fx.constant(1, 4000))
    src.push(t0 + 0.003, fx.effect_report(1, 1))
    src.push(t0 + 0.004, fx.effect_op(1, fm.OP_START))


# --------------------------------------------------------------------------
# cap


@pytest.mark.parametrize("cap", [0, 0.5, 15.1, 100, float("nan"), -5])
def test_cap_outside_1_to_15_is_refused(cap):
    with pytest.raises(ValueError):
        br.validate_cap(cap)


def test_cli_refuses_without_a_cap(capsys):
    assert br.main([]) == 2
    assert "--ffb-test-cap PERCENT is required" in capsys.readouterr().err
    assert br.main(["--ffb-test-cap", "20"]) == 2


def test_cap_scales_every_device_gain():
    b, src, sink, clock = make(cap=5)
    assert sink.lines()[0] == "SetGain 500"  # before any effect
    constant_game(src)
    src.push(0.1, fx.gain(128))
    src.push(0.2, fx.control(fm.CTRL_DEVRST))
    run_until(b, clock, 0.3)
    gains = [c.value for c in sink.calls if c.op == "SetGain"]
    assert gains == [500, 500, 251, 251]
    assert max(gains) <= 10000 * 5 / 100
    assert b.model.device_gain == 5020  # the model keeps the game's own value


def test_capped_gain():
    assert br.capped_gain(10000, 15) == 1500
    assert br.capped_gain(10000, 1) == 100
    assert br.capped_gain(5020, 5) == 251


# --------------------------------------------------------------------------
# forwarding


def test_forwards_and_renews_while_the_game_plays():
    b, src, sink, clock = make()
    constant_game(src)
    run_until(b, clock, 0.35)
    ops = [c.op for c in sink.calls]
    assert ops[:5] == ["SetGain", "SetGain", "CreateEffect", "SetParameters", "SetParameters"]
    renewals = [c for c in sink.calls if c.op == "SetParameters" and c.flags & fm.DIEP_START]
    assert len(renewals) == 3
    assert b.status == "FFB pass 0 unknown"
    assert len(b.stats.latencies_ms) == 5


def test_unknown_packets_reach_the_hud_and_not_the_wheel():
    b, src, sink, clock = make()
    src.push(0.01, fx.raw(fm.IOCTL_HID_WRITE_REPORT, [0x17, 1, 0, 0]))
    run_until(b, clock, 0.05)
    assert [c.op for c in sink.calls] == ["SetGain"]
    assert b.status == "FFB pass 1 unknown"


def test_dll_mismatch_is_reported():
    b, src, sink, clock = make()
    src.push(0.01, fx.new_effect(1, 1))
    src.push(0.02, fx.constant(1, -3000), dll={"rc": 0, "type": fm.PT_CONSTREP, "EffectBlockIndex": 1, "Magnitude": 1234})
    run_until(b, clock, 0.05)
    assert src.mismatches == [["magnitude: ours -3000, DLL 1234"]]


# --------------------------------------------------------------------------
# the wheel coming and going


def test_waits_without_a_wheel_then_rebuilds_it():
    b, src, sink, clock = make(present=False)
    assert b.status == "FFB waiting for wheel"
    constant_game(src)
    run_until(b, clock, 0.5)
    assert sink.calls == [] and b.model.counters.forwarded == 5
    sink.present = True
    run_until(b, clock, 1.2)
    assert b.device_open
    assert [c.op for c in sink.calls][:5] == ["SetGain", "CreateEffect", "SetParameters", "Start", "SetParameters"]
    assert sink.calls[0].value == 500


def test_lost_wheel_is_reopened_and_replayed():
    b, src, sink, clock = make()
    constant_game(src)
    run_until(b, clock, 0.05)
    sink.results["SetParameters"] = di.DIERR_INPUTLOST
    run_until(b, clock, 0.2)
    assert not b.device_open and "wheel lost: DIERR_INPUTLOST" in b.stats.device_events[-1]
    sink.results.clear()
    sink.calls.clear()
    run_until(b, clock, 1.3)
    assert b.device_open and sink.opened == 2
    assert [c.op for c in sink.calls][:4] == ["SetGain", "CreateEffect", "SetParameters", "Start"]


def test_unattended_never_opens_a_wheel():
    wheel = di.DeviceInfo("FANATEC Podium Wheel Base DD2", 0x0EB7, 0x0007, "x")
    b, src, sink, clock = make(attended=False, devices=[wheel])
    constant_game(src)
    run_until(b, clock, 2.5)
    assert sink.opened == 0 and sink.calls == []
    assert b.status == "FFB waiting for wheel"
    events = [e.split(" ", 1)[1] for e in b.stats.device_events]
    assert events == ["wheel present but run is unattended, not opening it: FANATEC Podium Wheel Base DD2 (0EB7:0007)"]


def test_unattended_without_a_wheel_reports_waiting():
    b, *_ = make(attended=False)
    assert [e.split(" ", 1)[1] for e in b.stats.device_events] == ["no force feedback wheel attached; waiting"]


def test_failure_stops_forwarding_and_closes_the_wheel():
    class Boom(di.LogSink):
        def execute(self, call):
            if call.op == "CreateEffect":
                raise OSError("boom")
            return super().execute(call)

    clock = Clock()
    src = Source(clock)
    sink = Boom()
    b = br.FfbBridge(src, sink, 5, clock=clock)
    constant_game(src)
    b.start()
    b._thread.join(2)
    assert not b.running and b.status.startswith("FFB stopped: boom")
    assert sink.device is None


# --------------------------------------------------------------------------
# ctypes layouts, 64-bit Windows sizes (lane 3)


@pytest.mark.parametrize("struct,size", [
    (di.DIEFFECT, 80), (di.DIENVELOPE, 20), (di.DICONSTANTFORCE, 4), (di.DIRAMPFORCE, 8),
    (di.DIPERIODIC, 16), (di.DICONDITION, 24), (di.DIPROPHEADER, 16), (di.DIPROPDWORD, 20),
    (di.DIDEVICEINSTANCEW, 1100), (di.DIDEVICEOBJECTINSTANCEW, 576), (di.DIDEVCAPS, 44),
    (di.DIEFFECTINFOW, 552), (di.DIOBJECTDATAFORMAT, 24), (di.DIDATAFORMAT, 32), (di.GUID, 16),
    (vj.FFB_DATA, 16), (vj.FFB_EFF_REPORT, 28), (vj.FFB_EFF_CONSTANT, 8), (vj.FFB_EFF_RAMP, 12),
    (vj.FFB_EFF_OP, 12), (vj.FFB_EFF_PERIOD, 20), (vj.FFB_EFF_COND, 32), (vj.FFB_EFF_ENVLP, 20),
])
def test_struct_sizes(struct, size):
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        pytest.skip("layouts are for 64-bit Python")
    assert ctypes.sizeof(struct) == size


def test_dieffect_offsets():
    assert [getattr(di.DIEFFECT, f).offset for f in ("cAxes", "rgdwAxes", "lpEnvelope", "cbTypeSpecificParams",
                                                     "lpvTypeSpecificParams", "dwStartDelay")] == [28, 32, 48, 56, 64, 72]
    assert vj.FFB_EFF_REPORT.Polar.offset == 20 and vj.FFB_EFF_REPORT.DirX.offset == 24


@pytest.mark.parametrize("interface,method,slot", [
    ("IDirectInputEffect", "Release", 2), ("IDirectInputEffect", "SetParameters", 6),
    ("IDirectInputEffect", "Start", 7), ("IDirectInputEffect", "Stop", 8),
    ("IDirectInputEffect", "Download", 10), ("IDirectInputEffect", "Unload", 11),
    ("IDirectInputDevice8W", "SetProperty", 6), ("IDirectInputDevice8W", "Acquire", 7),
    ("IDirectInputDevice8W", "Unacquire", 8), ("IDirectInputDevice8W", "SetDataFormat", 11),
    ("IDirectInputDevice8W", "SetCooperativeLevel", 13), ("IDirectInputDevice8W", "CreateEffect", 18),
    ("IDirectInputDevice8W", "EnumEffects", 19), ("IDirectInputDevice8W", "SendForceFeedbackCommand", 22),
    ("IDirectInput8W", "CreateDevice", 3), ("IDirectInput8W", "EnumDevices", 4),
])
def test_vtable_slots(interface, method, slot):
    assert di.vtable_slot(interface, method) == slot


def test_dinput_constants():
    assert (fm.DIEFF_OBJECTIDS, fm.DIEFF_CARTESIAN, fm.DIEFF_POLAR, fm.DIEFF_SPHERICAL) == (0x01, 0x10, 0x20, 0x40)
    assert (fm.DIEP_TYPESPECIFICPARAMS, fm.DIEP_START, fm.DIEP_NODOWNLOAD) == (0x100, 0x20000000, 0x80000000)
    assert (di.DIERR_NOTACQUIRED, di.DIERR_INPUTLOST) == (0x8007000C, 0x8007001E)


def test_dieffect_names_the_axis_by_object_id():
    params = fm.DiParams(duration=200_000, gain=7500, type_specific=fm.DiCondition(1, 2, 3, 4, 5, 6),
                         envelope=fm.DiEnvelope(1, 2, 3, 4))
    eff, keep = di.dieffect_for(params, 0x01000002)
    assert eff.dwFlags == fm.DIEFF_OBJECTIDS | fm.DIEFF_CARTESIAN
    assert (eff.cAxes, eff.rgdwAxes[0], eff.rglDirection[0]) == (1, 0x01000002, 0)
    assert (eff.dwDuration, eff.dwGain, eff.dwTriggerButton) == (200_000, 7500, fm.DIEB_NOTRIGGER)
    assert eff.cbTypeSpecificParams == ctypes.sizeof(di.DICONDITION)
    cond = ctypes.cast(eff.lpvTypeSpecificParams, ctypes.POINTER(di.DICONDITION)).contents
    assert (cond.lOffset, cond.lDeadBand) == (1, 6)
    env = ctypes.cast(eff.lpEnvelope, ctypes.POINTER(di.DIENVELOPE)).contents
    assert (env.dwSize, env.dwAttackLevel, env.dwFadeTime) == (20, 1, 4)
    eff, _ = di.dieffect_for(fm.DiParams(), 0x01000002)
    assert eff.lpEnvelope is None and eff.cbTypeSpecificParams == 0


def test_effect_guids_cover_every_kind():
    assert set(di.EFFECT_GUIDS) == set(fm.EFFECT_KINDS.values())


def test_vid_pid_from_product_guid():
    assert di.vid_pid(0x00070EB7) == (0x0EB7, 0x0007)
    assert di.DeviceInfo("vJoy Device", 0x1234, 0xBEAD, "").is_vjoy


# --------------------------------------------------------------------------
# vJoy check (3A calls this) and the DLL cross-check


def test_vjoy_check_rows():
    full = vj.VJoyFfbProbe(True, True, 0x222, True, True, True, tuple(vj.EFFECT_USAGES), "FREE")
    assert vj.vjoy_ffb_check(full) == vj.CheckResult(True, "vJoy device 1 force feedback on, 11 effects")
    assert vj.vjoy_ffb_check(vj.VJoyFfbProbe()).row == "vJoy is not installed"
    off = vj.vjoy_ffb_check(vj.VJoyFfbProbe(True, True, 0x222, True, True, False, (), "FREE"))
    assert off.row == "vJoy device 1 has force feedback off" and "Enable Effects" in off.fix
    some = vj.vjoy_ffb_check(vj.VJoyFfbProbe(True, True, 0x222, True, True, True, ("ConstantForce",), "FREE"))
    assert some.row.startswith("vJoy device 1 lacks effects: RampForce")
    busy = vj.vjoy_ffb_check(vj.VJoyFfbProbe(True, True, 0x222, True, True, True, tuple(vj.EFFECT_USAGES), "BUSY", 4242))
    assert not busy.ok and "process 4242" in busy.row


def test_crosscheck_sign_extends_the_dll_fields():
    rec = fm.ConstantForce(1, -3000)
    assert vj.crosscheck(rec, {"rc": 0, "type": fm.PT_CONSTREP, "EffectBlockIndex": 1, "Magnitude": 62536}) == []
    assert vj.crosscheck(rec, {"rc": 0, "type": fm.PT_CONSTREP, "EffectBlockIndex": 1, "Magnitude": 3000}) == \
        ["magnitude: ours -3000, DLL 3000"]
    cond = fm.Condition(2, 1, -1, -2, 3, 4, 5, 6)
    dll = {"rc": 0, "type": fm.PT_CONDREP, "EffectBlockIndex": 2, "isY": 1, "CenterPointOffset": 0xFFFF,
           "PosCoeff": 0xFFFE, "NegCoeff": 3, "PosSatur": 4, "NegSatur": 5, "DeadBand": 6}
    assert vj.crosscheck(cond, dll) == []
