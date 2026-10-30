"""Receive the game's force feedback packets from vJoy through ctypes.

`vJoyInterface.dll` (vJoy 2.2.2.0, BrunnerInnovation fork) delivers every
PID report the game's DirectInput sends to vJoy device N through the
callback registered with `FfbRegisterGenCB` (SDK/inc/vjoyinterface.h).
The DLL only reads a device's FFB queue after this process has called
`AcquireVJD` on it (apps/common/vJoyInterface/vJoyInterface.cpp,
AcquireVJD -> FfbStartThread), so the receiver owns the vJoy device while
it runs.

The callback runs on the DLL's window thread and blocks the driver's FFB
read loop until it returns, so it only copies the packet into a queue.
Decoding happens in `simsteer.core.ffb_model`. Each packet is also decoded
with the DLL's own `Ffb_h_*` helpers, and any difference is counted, so a
game that sends something the fixtures never covered shows up as a
mismatch instead of a silent misread. The helpers leave 16-bit signed
fields unextended (Ffb_h_Eff_Constant reads (data[3] << 8) + data[2] into a
LONG), so the comparison sign-extends those fields first.

Run `python -m simsteer.runtime.ffb.vjoy_ffb --check` for the setup row,
or `--record FILE` to save packets as test fixtures.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

from simsteer.core import ffb_model as fm

DEFAULT_DLL = r"C:\Program Files\vJoy\x64\vJoyInterface.dll"

# HID usages of the 11 effects vJoyConf can enable (public.h HID_USAGE_*).
EFFECT_USAGES = {
    "ConstantForce": 0x26, "RampForce": 0x27, "Square": 0x30, "Sine": 0x31,
    "Triangle": 0x32, "SawtoothUp": 0x33, "SawtoothDown": 0x34, "Spring": 0x40,
    "Damper": 0x41, "Inertia": 0x42, "Friction": 0x43,
}
VJD_STATUS = {0: "OWN", 1: "FREE", 2: "BUSY", 3: "MISSING", 4: "UNKNOWN"}

UINT8 = ctypes.c_uint8
UINT16 = ctypes.c_uint16
UINT32 = ctypes.c_uint32
INT32 = ctypes.c_int32


class FFB_DATA(ctypes.Structure):
    _fields_ = [("size", UINT32), ("cmd", UINT32), ("data", ctypes.POINTER(UINT8))]


# The Ffb_h_* output structs, default packing as declared in vjoyinterface.h.
class FFB_EFF_REPORT(ctypes.Structure):
    _fields_ = [
        ("EffectBlockIndex", UINT8), ("EffectType", UINT32), ("Duration", UINT16),
        ("TrigerRpt", UINT16), ("SamplePrd", UINT16), ("StartDelay", UINT16), ("Gain", UINT8),
        ("TrigerBtn", UINT8), ("AxesEnabledDirection", UINT8), ("Polar", INT32),
        ("DirX", UINT16), ("DirY", UINT16),
    ]


class FFB_EFF_CONSTANT(ctypes.Structure):
    _fields_ = [("EffectBlockIndex", UINT8), ("Magnitude", INT32)]


class FFB_EFF_RAMP(ctypes.Structure):
    _fields_ = [("EffectBlockIndex", UINT8), ("Start", INT32), ("End", INT32)]


class FFB_EFF_OP(ctypes.Structure):
    _fields_ = [("EffectBlockIndex", UINT8), ("EffectOp", UINT32), ("LoopCount", UINT8)]


class FFB_EFF_PERIOD(ctypes.Structure):
    _fields_ = [("EffectBlockIndex", UINT8), ("Magnitude", UINT32), ("Offset", INT32),
                ("Phase", UINT32), ("Period", UINT32)]


class FFB_EFF_COND(ctypes.Structure):
    _fields_ = [
        ("EffectBlockIndex", UINT8), ("isY", INT32), ("CenterPointOffset", INT32), ("PosCoeff", INT32),
        ("NegCoeff", INT32), ("PosSatur", UINT32), ("NegSatur", UINT32), ("DeadBand", INT32),
    ]


class FFB_EFF_ENVLP(ctypes.Structure):
    _fields_ = [("EffectBlockIndex", UINT8), ("AttackLevel", UINT32), ("FadeLevel", UINT32),
                ("AttackTime", UINT32), ("FadeTime", UINT32)]


# packet type -> (helper, struct, field -> record attribute, signed 16-bit fields)
DLL_DECODERS: dict[int, tuple[str, type, dict[str, str], frozenset[str]]] = {
    fm.PT_EFFREP: ("Ffb_h_Eff_Report", FFB_EFF_REPORT, {
        "EffectBlockIndex": "ebi", "EffectType": "effect_type", "Duration": "duration_ms",
        "TrigerRpt": "trigger_repeat_ms", "SamplePrd": "sample_period_ms", "StartDelay": "start_delay_ms",
        "Gain": "gain", "TrigerBtn": "trigger_button", "AxesEnabledDirection": "axes",
    }, frozenset()),
    fm.PT_CONSTREP: ("Ffb_h_Eff_Constant", FFB_EFF_CONSTANT,
                     {"EffectBlockIndex": "ebi", "Magnitude": "magnitude"}, frozenset({"Magnitude"})),
    fm.PT_RAMPREP: ("Ffb_h_Eff_Ramp", FFB_EFF_RAMP,
                    {"EffectBlockIndex": "ebi", "Start": "start", "End": "end"}, frozenset({"Start", "End"})),
    fm.PT_EFOPREP: ("Ffb_h_EffOp", FFB_EFF_OP,
                    {"EffectBlockIndex": "ebi", "EffectOp": "op", "LoopCount": "loop_count"}, frozenset()),
    fm.PT_PRIDREP: ("Ffb_h_Eff_Period", FFB_EFF_PERIOD, {
        "EffectBlockIndex": "ebi", "Magnitude": "magnitude", "Offset": "offset",
        "Phase": "phase", "Period": "period_ms",
    }, frozenset({"Offset"})),
    fm.PT_CONDREP: ("Ffb_h_Eff_Cond", FFB_EFF_COND, {
        "EffectBlockIndex": "ebi", "isY": "block", "CenterPointOffset": "center_offset",
        "PosCoeff": "positive_coefficient", "NegCoeff": "negative_coefficient",
        "PosSatur": "positive_saturation", "NegSatur": "negative_saturation", "DeadBand": "dead_band",
    }, frozenset({"CenterPointOffset", "PosCoeff", "NegCoeff"})),
    fm.PT_ENVREP: ("Ffb_h_Eff_Envlp", FFB_EFF_ENVLP, {
        "EffectBlockIndex": "ebi", "AttackLevel": "attack_level", "FadeLevel": "fade_level",
        "AttackTime": "attack_time_ms", "FadeTime": "fade_time_ms",
    }, frozenset()),
}


def _s16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def crosscheck(record: fm.Record, dll: dict[str, int] | None) -> list[str]:
    """Fields where our decode and the DLL's Ffb_h_* decode disagree. `dll`
    holds the helper's raw struct fields, as recorded in the fixtures."""
    if dll is None:
        return []
    tp = dll.get("type")
    problems = []
    if isinstance(record, fm.Unknown) or dll.get("rc"):
        # Unknown packets are already counted; a helper error has no fields.
        return []
    if tp in DLL_DECODERS:
        _, _, fields, signed = DLL_DECODERS[tp]
        for name, attr in fields.items():
            if name not in dll:
                continue
            want = _s16(dll[name]) if name in signed else dll[name]
            have = getattr(record, attr)
            if name == "isY":
                have, want = bool(have & 0x0F), bool(want)
            if have != want:
                problems.append(f"{attr}: ours {have}, DLL {want}")
        if tp == fm.PT_EFFREP and "DirX" in dll:
            d1, d2 = record.direction
            if d1 != dll["DirX"] or (not dll.get("Polar") and d2 != dll.get("DirY", d2)):
                problems.append(f"direction: ours {record.direction}, DLL ({dll['DirX']}, {dll.get('DirY')})")
    elif tp == fm.PT_NEWEFREP and isinstance(record, fm.CreateNewEffect):
        if (record.effect_type, record.ebi) != (dll.get("EffectType"), dll.get("EffectId")):
            problems.append(f"new effect: ours {(record.effect_type, record.ebi)}, "
                            f"DLL {(dll.get('EffectType'), dll.get('EffectId'))}")
    elif tp == fm.PT_CTRLREP and isinstance(record, fm.DeviceControl):
        if record.control != dll.get("Control"):
            problems.append(f"control: ours {record.control}, DLL {dll.get('Control')}")
    elif tp == fm.PT_GAINREP and isinstance(record, fm.DeviceGain):
        if record.gain != dll.get("Gain"):
            problems.append(f"gain: ours {record.gain}, DLL {dll.get('Gain')}")
    elif tp == fm.PT_BLKFRREP and isinstance(record, fm.BlockFree):
        if record.ebi != dll.get("EffectId"):
            problems.append(f"block free: ours {record.ebi}, DLL {dll.get('EffectId')}")
    return problems


# --------------------------------------------------------------------------
# The setup check, pure so the setup window (3A) can call it with any probe


@dataclass(frozen=True)
class VJoyFfbProbe:
    dll_found: bool = False
    driver_enabled: bool = False
    version: int | None = None
    driver_ffb: bool = False
    device_exists: bool = False
    device_ffb: bool = False
    effects: tuple[str, ...] = ()
    status: str = "UNKNOWN"
    owner_pid: int | None = None


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    row: str
    fix: str | None = None


VJOY_CONF_FIX = (
    "Open Configure vJoy (Start menu), select device {id}, tick Enable Effects and every effect, "
    "click Apply. SimSteer never changes this setting itself."
)


def vjoy_ffb_check(probe: VJoyFfbProbe, device_id: int = 1) -> CheckResult:
    """Whether vJoy device `device_id` can carry the game's force feedback,
    and how to fix it if not."""
    if not probe.dll_found:
        return CheckResult(False, "vJoy is not installed", "Install vJoy 2.2.2.0 (BrunnerInnovation).")
    if not probe.driver_enabled:
        return CheckResult(False, "the vJoy driver is not running", "Reinstall vJoy 2.2.2.0, then restart Windows.")
    if not probe.driver_ffb:
        return CheckResult(False, "this vJoy driver has no force feedback", "Install vJoy 2.2.2.0 (BrunnerInnovation).")
    if not probe.device_exists:
        return CheckResult(False, f"vJoy device {device_id} does not exist",
                           f"Open Configure vJoy and add device {device_id}.")
    if not probe.device_ffb:
        return CheckResult(False, f"vJoy device {device_id} has force feedback off", VJOY_CONF_FIX.format(id=device_id))
    missing = [e for e in EFFECT_USAGES if e not in probe.effects]
    if missing:
        return CheckResult(False, f"vJoy device {device_id} lacks effects: {', '.join(missing)}",
                           VJOY_CONF_FIX.format(id=device_id))
    row = f"vJoy device {device_id} force feedback on, {len(probe.effects)} effects"
    if probe.status == "BUSY":
        return CheckResult(False, f"{row}, but owned by process {probe.owner_pid}",
                           "Close the other program that feeds vJoy, or run the bridge inside SimSteer.")
    return CheckResult(True, row)


# --------------------------------------------------------------------------
# ctypes binding (Windows only)


class VJoyError(RuntimeError):
    pass


def load_dll(path: str | None = None):
    path = path or os.environ.get("SIMSTEER_VJOY_DLL") or DEFAULT_DLL
    try:
        dll = ctypes.CDLL(path)
    except OSError as e:
        raise VJoyError(f"cannot load {path}: {e}") from e
    for name in ("vJoyEnabled", "vJoyFfbCap", "isVJDExists", "IsDeviceFfb", "IsDeviceFfbEffect",
                 "AcquireVJD", "GetvJoyMaxDevices"):
        getattr(dll, name).restype = ctypes.c_int
    dll.GetvJoyVersion.restype = ctypes.c_short
    dll.GetVJDStatus.restype = ctypes.c_int
    dll.GetOwnerPid.restype = ctypes.c_int
    dll.RelinquishVJD.restype = None
    dll.FfbRegisterGenCB.restype = None
    dll.FfbRegisterGenCB.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    for name in ("Ffb_h_Type", "Ffb_h_DeviceID", "Ffb_h_EffectBlockIndex", "Ffb_h_CreateNewEffect",
                 "Ffb_h_DevCtrl", "Ffb_h_DevGain", *(d[0] for d in DLL_DECODERS.values())):
        getattr(dll, name).restype = UINT32
    return dll


def probe_vjoy(dll=None, device_id: int = 1) -> VJoyFfbProbe:
    """Read-only state of vJoy and its FFB setting."""
    if dll is None:
        try:
            dll = load_dll()
        except VJoyError:
            return VJoyFfbProbe()
    if not dll.vJoyEnabled():
        return VJoyFfbProbe(dll_found=True)
    cap = ctypes.c_int(0)
    dll.vJoyFfbCap(ctypes.byref(cap))
    exists = bool(dll.isVJDExists(device_id))
    device_ffb = exists and bool(dll.IsDeviceFfb(device_id))
    effects = tuple(name for name, usage in EFFECT_USAGES.items()
                    if device_ffb and dll.IsDeviceFfbEffect(device_id, usage))
    status = VJD_STATUS.get(dll.GetVJDStatus(device_id), "UNKNOWN")
    owner = dll.GetOwnerPid(device_id) if status == "BUSY" else None
    return VJoyFfbProbe(True, True, dll.GetvJoyVersion() & 0xFFFF, bool(cap.value), exists,
                        device_ffb, effects, status, owner)


@dataclass
class Packet:
    t: float
    cmd: int
    data: bytes
    dll: dict[str, int] | None = None


# CALLBACK is __stdcall; WINFUNCTYPE exists only on Windows.
_FUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
FfbGenCB = _FUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


def dll_decode(dll, address: int) -> dict[str, int]:
    """The DLL's own reading of the FFB_DATA at `address`, as raw struct fields."""
    ptr = ctypes.c_void_p(address)
    tp = UINT32(0)
    out: dict[str, int] = {"rc": dll.Ffb_h_Type(ptr, ctypes.byref(tp))}
    out["type"] = tp.value
    if out["rc"]:
        return out
    t = tp.value
    if t in DLL_DECODERS:
        helper, struct_type, _fields, _signed = DLL_DECODERS[t]
        st = struct_type()
        out["rc"] = getattr(dll, helper)(ptr, ctypes.byref(st))
        for name, _ in struct_type._fields_:
            out[name] = getattr(st, name)
    elif t == fm.PT_NEWEFREP:
        et, eid = UINT32(0), UINT32(0)
        out["rc"] = dll.Ffb_h_CreateNewEffect(ptr, ctypes.byref(et), ctypes.byref(eid))
        out["EffectType"], out["EffectId"] = et.value, eid.value
    elif t == fm.PT_CTRLREP:
        ctl = UINT32(0)
        out["rc"] = dll.Ffb_h_DevCtrl(ptr, ctypes.byref(ctl))
        out["Control"] = ctl.value
    elif t == fm.PT_GAINREP:
        gain = UINT8(0)
        out["rc"] = dll.Ffb_h_DevGain(ptr, ctypes.byref(gain))
        out["Gain"] = gain.value
    elif t == fm.PT_BLKFRREP:
        eid = UINT32(0)
        out["rc"] = dll.Ffb_h_EffectBlockIndex(ptr, ctypes.byref(eid))
        out["EffectId"] = eid.value
    return out


@dataclass
class ReceiverStats:
    received: int = 0
    other_device: int = 0
    mismatches: int = 0
    mismatch_log: list[str] = field(default_factory=list)


class VJoyFfbReceiver:
    """Owns vJoy device `device_id` and queues the game's FFB packets."""

    def __init__(self, device_id: int = 1, dll_path: str | None = None, dll_check: bool = True) -> None:
        self.device_id = device_id
        self.dll_path = dll_path
        self.dll_check = dll_check
        self.queue: queue.SimpleQueue[Packet] = queue.SimpleQueue()
        self.stats = ReceiverStats()
        self._dll = None
        self._cb = None
        self._lock = threading.Lock()

    def open(self) -> CheckResult:
        self._dll = dll = load_dll(self.dll_path)
        check = vjoy_ffb_check(probe_vjoy(dll, self.device_id), self.device_id)
        if not check.ok:
            return check
        if not dll.AcquireVJD(self.device_id):
            status = VJD_STATUS.get(dll.GetVJDStatus(self.device_id), "UNKNOWN")
            return CheckResult(False, f"cannot acquire vJoy device {self.device_id} ({status})",
                               "Close other programs that feed vJoy.")
        self._cb = FfbGenCB(self._on_packet)
        dll.FfbRegisterGenCB(ctypes.cast(self._cb, ctypes.c_void_p), None)
        return check

    def _on_packet(self, data, _user) -> None:
        t = time.perf_counter()
        try:
            pkt = ctypes.cast(data, ctypes.POINTER(FFB_DATA)).contents
            if pkt.size < 10:
                return
            raw = ctypes.string_at(pkt.data, pkt.size - 8)
            if fm.device_id(raw) != self.device_id:
                self.stats.other_device += 1
                return
            dll = dll_decode(self._dll, data) if self.dll_check else None
            self.stats.received += 1
            self.queue.put(Packet(t, pkt.cmd, raw, dll))
        except Exception as e:  # never raise into the DLL's thread
            with self._lock:
                self.stats.mismatch_log.append(f"callback error: {e!r}")

    def get(self, timeout: float | None) -> Packet | None:
        try:
            return self.queue.get(timeout=timeout) if timeout is None or timeout > 0 else self.queue.get_nowait()
        except queue.Empty:
            return None

    def note_mismatch(self, problems: list[str], packet: Packet) -> None:
        self.stats.mismatches += 1
        if len(self.stats.mismatch_log) < 50:
            self.stats.mismatch_log.append(f"{packet.data.hex()}: {'; '.join(problems)}")

    def close(self) -> None:
        if self._dll is not None:
            self._dll.FfbRegisterGenCB(None, None)
            self._dll.RelinquishVJD(self.device_id)
            self._dll = None


# --------------------------------------------------------------------------
# CLI: --check, --record


def record(device_id: int, seconds: float, idle_s: float, path: str, note: str) -> int:
    rx = VJoyFfbReceiver(device_id)
    check = rx.open()
    if not check.ok:
        print(f"vJoy FFB: {check.row}. {check.fix or ''}".strip(), file=sys.stderr)
        rx.close()
        return 2
    print(f"recording vJoy device {device_id} for up to {seconds:g} s", flush=True)
    packets: list[Packet] = []
    t0 = time.perf_counter()
    last = t0
    try:
        while time.perf_counter() - t0 < seconds:
            p = rx.get(0.05)
            if p is not None:
                packets.append(p)
                last = p.t
            elif packets and time.perf_counter() - last > idle_s:
                break
    finally:
        rx.close()
    first = packets[0].t if packets else t0
    doc = {
        "source": {
            "kind": "recorded",
            "note": note,
            "host": platform.node(),
            "windows": platform.platform(),
            "vjoy": f"0x{probe_vjoy(load_dll(), device_id).version or 0:X}",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "packets": [{"t": round(p.t - first, 6), "cmd": p.cmd, "data": p.data.hex(), "dll": p.dll} for p in packets],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"saved {len(packets)} packets to {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m simsteer.runtime.ffb.vjoy_ffb",
                                description="Check vJoy force feedback, or record its packets.")
    p.add_argument("--device", type=int, default=1, help="vJoy device (default 1)")
    p.add_argument("--check", action="store_true", help="print whether vJoy FFB is set up (read only)")
    p.add_argument("--record", metavar="FILE", help="record packets to a fixture file")
    p.add_argument("--seconds", type=float, default=30.0, help="longest recording (default 30)")
    p.add_argument("--idle", type=float, default=2.0, help="stop after this long without packets (default 2)")
    p.add_argument("--note", default="", help="what played the effects, for the fixture header")
    args = p.parse_args(argv)
    if sys.platform != "win32":
        print("vJoy needs Windows", file=sys.stderr)
        return 2
    if args.record:
        return record(args.device, args.seconds, args.idle, args.record, args.note)
    probe = probe_vjoy(device_id=args.device)
    check = vjoy_ffb_check(probe, args.device)
    print(json.dumps({"ok": check.ok, "row": check.row, "fix": check.fix, "probe": asdict(probe)}, indent=1))
    return 0 if check.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
