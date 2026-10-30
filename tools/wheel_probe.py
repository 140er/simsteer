"""Probe a Fanatec wheel and pedals on the simrig and write a JSON report.

The report decides two things for SimSteer 1.0 (plan step H1):

* which backend reads the wheel and pedals while a game has focus:
  SDL through pygame-ce with background events on, or DirectInput8
  (non-exclusive, background) through ctypes;
* whether DirectInput force feedback works on the base.

    python tools\\wheel_probe.py --report probe.json
    python tools\\wheel_probe.py --ffb-test --ffb-level 5 --report probe-ffb.json

The vJoy section only reads. The tool never changes vJoy or ViGEm.

The module imports on any OS. Windows APIs and pygame load lazily inside
the functions that need them, so the pure parts are unit tested on Linux.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import math
import os
import platform
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

FANATEC_VID = 0x0EB7
VJOY_VID, VJOY_PID = 0x1234, 0xBEAD

DEFAULT_FFB_LEVEL = 5.0
MAX_FFB_LEVEL = 15.0
FFB_RUN_S = 2.0
FFB_UPDATE_S = 0.05
EFFECT_DURATION_US = 200_000

DEFAULT_SECONDS = 20.0
DEFAULT_COUNTDOWN = 5
POLL_SLEEP_S = 0.0005
FOCUS_CHECK_S = 0.05
# Consecutive value changes closer than this count as one movement, so
# pauses between movements do not drag the read rate down.
ACTIVE_GAP_S = 0.05
REACQUIRE_S = 0.5
# Windows.Gaming.Input reports controllers asynchronously after init.
SDL_SETTLE_S = 2.0
SDL_STABLE_S = 0.5

EXIT_OK = 0
EXIT_CANCELLED = 1
EXIT_USAGE = 2
EXIT_FFB_FAILED = 3
EXIT_INTERRUPTED = 130

PYGAME_HINT = "pygame-ce is not installed. Install it with: python -m pip install pygame-ce"
NO_FANATEC_NOTE = (
    "No Fanatec wheel found (vendor 0x0EB7). The base may be unplugged; "
    "that is normal when nobody is at the rig."
)


# ---------------------------------------------------------------------------
# Pure helpers


def validate_ffb_level(level: float) -> float:
    level = float(level)
    if not math.isfinite(level) or level <= 0:
        raise ValueError(f"--ffb-level must be a positive percent, got {level:g}")
    if level > MAX_FFB_LEVEL:
        raise ValueError(
            f"refusing --ffb-level {level:g}: the limit is {MAX_FFB_LEVEL:g} percent. "
            "Raise the level in separate runs, at most 2.5 percent per step."
        )
    return level


def parse_vid(text: str) -> int:
    value = int(text, 16)
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"vendor ID out of range: {text}")
    return value


def hex16(value: int | None) -> str | None:
    return None if value is None else f"0x{value:04X}"


def is_fanatec(vid: int | None, name: str = "") -> bool:
    if vid is not None:
        return vid == FANATEC_VID
    return "fanatec" in name.lower()


def is_vjoy(vid: int | None, pid: int | None) -> bool:
    return vid == VJOY_VID and pid == VJOY_PID


def sdl_vid_pid(guid_hex: str) -> tuple[int | None, int | None]:
    """VID/PID from an SDL2 joystick GUID string, or (None, None).

    SDL packs bus, crc, vendor, 0, product, 0, version as little-endian
    16-bit words. GUIDs built from a device name do not carry the IDs.
    """
    try:
        raw = bytes.fromhex(guid_hex)
    except (TypeError, ValueError):
        return None, None
    if len(raw) != 16:
        return None, None
    word = lambda i: int.from_bytes(raw[i:i + 2], "little")
    vid, pid = word(4), word(8)
    if word(6) != 0 or word(10) != 0 or vid == 0:
        return None, None
    return vid, pid


def dinput_vid_pid(product_data1: int) -> tuple[int, int]:
    """DirectInput guidProduct.Data1 is MAKELONG(vid, pid)."""
    return product_data1 & 0xFFFF, (product_data1 >> 16) & 0xFFFF


DI8_DEVTYPES = {
    0x14: "joystick",
    0x15: "gamepad",
    0x16: "driving",
    0x17: "flight",
    0x18: "1stperson",
    0x19: "devicectrl",
    0x1A: "screenpointer",
    0x1B: "remote",
    0x1C: "supplemental",
}


def device_type_name(dev_type: int) -> str:
    return DI8_DEVTYPES.get(dev_type & 0xFF, f"0x{dev_type & 0xFF:02X}")


def format_vjoy_version(raw: int) -> str:
    """GetvJoyVersion returns 0x0219 for 2.1.9."""
    return f"{(raw >> 8) & 0xF}.{(raw >> 4) & 0xF}.{raw & 0xF}"


HRESULT_NAMES = {
    0x00000000: "DI_OK",
    0x00000001: "S_FALSE",
    0x00000002: "DI_POLLEDDEVICE",
    0x00000003: "DI_DOWNLOADSKIPPED",
    0x00000004: "DI_EFFECTRESTARTED",
    0x00000008: "DI_TRUNCATED",
    0x0000000B: "DI_SETTINGSNOTSAVED",
    0x0000000C: "DI_TRUNCATEDANDRESTARTED",
    0x8007047E: "DIERR_OLDDIRECTINPUTVERSION",
    0x80070481: "DIERR_BETADIRECTINPUTVERSION",
    0x80070077: "DIERR_BADDRIVERVER",
    0x80040154: "DIERR_DEVICENOTREG",
    0x80070002: "DIERR_NOTFOUND",
    0x80070057: "DIERR_INVALIDPARAM",
    0x80004002: "DIERR_NOINTERFACE",
    0x80004005: "DIERR_GENERIC",
    0x8007000E: "DIERR_OUTOFMEMORY",
    0x80004001: "DIERR_UNSUPPORTED",
    0x80070015: "DIERR_NOTINITIALIZED",
    0x800704DF: "DIERR_ALREADYINITIALIZED",
    0x80040110: "DIERR_NOAGGREGATION",
    # E_ACCESSDENIED is also DIERR_READONLY and DIERR_HANDLEEXISTS; for
    # Acquire it means another app holds the device exclusively.
    0x80070005: "DIERR_OTHERAPPHASPRIO",
    0x80070006: "E_HANDLE",
    0x8007001E: "DIERR_INPUTLOST",
    0x800700AA: "DIERR_ACQUIRED",
    0x8007000C: "DIERR_NOTACQUIRED",
    0x8000000A: "E_PENDING",
    0x80040200: "DIERR_INSUFFICIENTPRIVS",
    0x80040201: "DIERR_DEVICEFULL",
    0x80040202: "DIERR_MOREDATA",
    0x80040203: "DIERR_NOTDOWNLOADED",
    0x80040204: "DIERR_HASEFFECTS",
    0x80040205: "DIERR_NOTEXCLUSIVEACQUIRED",
    0x80040206: "DIERR_INCOMPLETEEFFECT",
    0x80040207: "DIERR_NOTBUFFERED",
    0x80040208: "DIERR_EFFECTPLAYING",
    0x80040209: "DIERR_UNPLUGGED",
    0x8004020A: "DIERR_REPORTFULL",
    0x8004020B: "DIERR_MAPFILEFAIL",
}


def hresult_name(hr: int) -> str:
    hr &= 0xFFFFFFFF
    return HRESULT_NAMES.get(hr, f"0x{hr:08X}")


def hresult_failed(hr: int) -> bool:
    return bool(hr & 0x80000000)


DIGFFS_FLAGS = {
    0x00000001: "EMPTY",
    0x00000002: "STOPPED",
    0x00000004: "PAUSED",
    0x00000010: "ACTUATORSON",
    0x00000020: "ACTUATORSOFF",
    0x00000040: "POWERON",
    0x00000080: "POWEROFF",
    0x00000100: "SAFETYSWITCHON",
    0x00000200: "SAFETYSWITCHOFF",
    0x00000400: "USERFFSWITCHON",
    0x00000800: "USERFFSWITCHOFF",
    0x80000000: "DEVICELOST",
}


def ff_state_flags(state: int) -> list[str]:
    return [name for bit, name in DIGFFS_FLAGS.items() if state & bit]


def spring_parameters(level: float) -> dict[str, int]:
    """Spring centred on zero whose force never exceeds `level` percent.

    The coefficient carries the level, so full lock reaches the level and
    no further. The saturation repeats it as a cap in case a driver
    treats the coefficient loosely. Effect gain stays nominal so the two
    never multiply into a weaker or stronger force than asked.
    """
    level = validate_ffb_level(level)
    magnitude = round(level * DI_FFNOMINALMAX / 100)
    return {
        "offset": 0,
        "coefficient": magnitude,
        "saturation": magnitude,
        "deadband": 0,
        "gain": DI_FFNOMINALMAX,
        "duration_us": EFFECT_DURATION_US,
    }


def filter_devices(devices: Iterable[dict], only_vid: int | None) -> list[dict]:
    if only_vid is None:
        return list(devices)
    return [d for d in devices if d.get("vid") == only_vid]


def update_rate(times: Sequence[float], active_gap: float = ACTIVE_GAP_S) -> float | None:
    """Value updates per second while something was moving."""
    intervals = [b - a for a, b in pairwise(times) if 0 < b - a <= active_gap]
    if not intervals:
        return None
    return round(len(intervals) / sum(intervals), 1)


def summarize_samples(
    samples: Sequence[tuple[float, Sequence[float], Sequence[int], bool | None]],
    axis_names: Sequence[str] | None = None,
    active_gap: float = ACTIVE_GAP_S,
) -> dict[str, Any]:
    """Summarise one device's polls.

    Each sample is (t seconds, axis values, button states, foreign_focus)
    where foreign_focus is True when another window (the game) had focus.
    An axis is frozen when it never changed. background_ok means at least
    one axis changed while another window had focus.
    """
    n_axes = len(samples[0][1]) if samples else len(axis_names or ())
    names = list(axis_names or [])[:n_axes]
    names += [f"axis{i}" for i in range(len(names), n_axes)]
    out: dict[str, Any] = {
        "polls": len(samples),
        "duration_s": 0.0,
        "poll_hz": None,
        "read_hz": None,
        "background_read_hz": None,
        "max_gap_ms": None,
        "foreign_focus_fraction": None,
        "background_ok": False,
        "axes": [],
        "buttons_changed": [],
        "buttons_changed_in_background": [],
    }
    if not samples:
        out["axes"] = [_axis_entry(name, None, None, 0, 0) for name in names]
        return out

    changes = [0] * n_axes
    bg_changes = [0] * n_axes
    lows = list(samples[0][1])
    highs = list(samples[0][1])
    change_times: list[float] = []
    bg_change_times: list[float] = []
    buttons: set[int] = set()
    bg_buttons: set[int] = set()
    max_gap = 0.0
    focus_known = any(s[3] is not None for s in samples)
    foreign_count = 1 if samples[0][3] else 0

    for prev, cur in pairwise(samples):
        t, values, states, foreign = cur
        max_gap = max(max_gap, t - prev[0])
        if foreign:
            foreign_count += 1
        moved = False
        for i in range(n_axes):
            v = values[i]
            lows[i] = min(lows[i], v)
            highs[i] = max(highs[i], v)
            if v != prev[1][i]:
                changes[i] += 1
                moved = True
                if foreign:
                    bg_changes[i] += 1
        if moved:
            change_times.append(t)
            if foreign:
                bg_change_times.append(t)
        for i, (a, b) in enumerate(zip(prev[2], states)):
            if a != b:
                buttons.add(i)
                if foreign:
                    bg_buttons.add(i)

    duration = samples[-1][0] - samples[0][0]
    out.update(
        duration_s=round(duration, 3),
        poll_hz=round((len(samples) - 1) / duration, 1) if duration > 0 else None,
        read_hz=update_rate(change_times, active_gap),
        background_read_hz=update_rate(bg_change_times, active_gap),
        max_gap_ms=round(max_gap * 1000, 2),
        foreign_focus_fraction=round(foreign_count / len(samples), 3) if focus_known else None,
        background_ok=any(bg_changes),
        axes=[
            _axis_entry(names[i], lows[i], highs[i], changes[i], bg_changes[i])
            for i in range(n_axes)
        ],
        buttons_changed=sorted(buttons),
        buttons_changed_in_background=sorted(bg_buttons),
    )
    return out


def _axis_entry(name: str, low, high, changes: int, bg_changes: int) -> dict[str, Any]:
    return {
        "name": name,
        "min": low,
        "max": high,
        "changes": changes,
        "background_changes": bg_changes,
        "frozen": changes == 0,
        "changed_in_background": bg_changes > 0,
    }


def summarize_backend(devices: Sequence[dict]) -> dict[str, Any]:
    """Backend verdict from its Fanatec devices, else its physical ones, so
    a vJoy device fed by another app cannot pass a backend. Each device
    dict carries its `summary`."""
    basis, chosen = "fanatec", [d for d in devices if d.get("fanatec")]
    if not chosen:
        basis, chosen = "physical", [d for d in devices if not d.get("virtual")]
    if not chosen:
        basis, chosen = "all", list(devices)
    summaries = [d["summary"] for d in chosen]

    def best(key: str):
        values = [s[key] for s in summaries if s.get(key) is not None]
        return max(values) if values else None

    return {
        "summary_basis": basis if summaries else None,
        "background_ok": any(s["background_ok"] for s in summaries) if summaries else None,
        "read_hz": best("read_hz"),
        "background_read_hz": best("background_read_hz"),
        "poll_hz": best("poll_hz"),
    }


def build_report(
    *,
    mode: str,
    args: dict[str, Any],
    backends: dict[str, dict],
    vjoy: dict,
    ffb: dict | None = None,
    focus: dict | None = None,
    events: Sequence[dict] = (),
    interrupted: bool = False,
    host: dict | None = None,
) -> dict[str, Any]:
    devices = [d for b in backends.values() for d in b.get("devices", [])]
    fanatec_found = any(d.get("fanatec") for d in devices)
    notes = [] if fanatec_found else [NO_FANATEC_NOTE]
    return {
        "tool": "wheel_probe",
        "schema": 1,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": host or host_info(),
        "mode": mode,
        "args": args,
        "devices": devices,
        "fanatec_found": fanatec_found,
        "notes": notes,
        "backends": {name: {k: v for k, v in b.items() if k != "devices"} for name, b in backends.items()},
        "focus": focus,
        "events": list(events),
        "vjoy": vjoy,
        "ffb": ffb or {"attempted": False},
        "interrupted": interrupted,
    }


def host_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "bits": ctypes.sizeof(ctypes.c_void_p) * 8,
    }


def device_entry(
    *,
    backend: str,
    key: str,
    name: str,
    vid: int | None,
    pid: int | None,
    axis_names: Sequence[str],
    buttons: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "key": key,
        "name": name,
        "vid": vid,
        "pid": pid,
        "vid_hex": hex16(vid),
        "pid_hex": hex16(pid),
        "axes": len(axis_names),
        "axis_names": list(axis_names),
        "buttons": buttons,
        "fanatec": is_fanatec(vid, name),
        "virtual": is_vjoy(vid, pid),
        **extra,
    }


# ---------------------------------------------------------------------------
# DirectInput8 definitions. Plain ctypes types keep struct sizes identical on
# every OS, so the layouts are checked by the Linux unit tests.

DWORD = ctypes.c_uint32
LONG = ctypes.c_int32
WORD = ctypes.c_uint16
WCHAR260 = ctypes.c_uint16 * 260

DIRECTINPUT_VERSION = 0x0800
DI8DEVCLASS_GAMECTRL = 4
DIEDFL_ATTACHEDONLY = 0x00000001
DIEDFL_FORCEFEEDBACK = 0x00000100
DISCL_EXCLUSIVE = 0x00000001
DISCL_NONEXCLUSIVE = 0x00000002
DISCL_BACKGROUND = 0x00000008
DIDFT_ALL = 0x00000000
DIDFT_AXIS = 0x00000003
DIDFT_BUTTON = 0x0000000C
DIDFT_POV = 0x00000010
DIDFT_ANYINSTANCE = 0x00FFFF00
DIDFT_FFACTUATOR = 0x01000000
DIDFT_OPTIONAL = 0x80000000
DIDF_ABSAXIS = 0x00000001
DIEFT_ALL = 0x00000000
DIDOI_ASPECTPOSITION = 0x00000100
DIDC_FORCEFEEDBACK = 0x00000100
DIPH_DEVICE = 0
DIPH_BYOFFSET = 1
DIPROP_FFGAIN = 7
DIPROP_AUTOCENTER = 9
DIPROPAUTOCENTER_OFF = 0
DISFFC_STOPALL = 0x00000002
DISFFC_SETACTUATORSON = 0x00000010
DIEFF_OBJECTIDS = 0x00000001
DIEFF_CARTESIAN = 0x00000010
DIEP_DURATION = 0x00000001
DIEP_TYPESPECIFICPARAMS = 0x00000100
DIEP_START = 0x20000000
DIEB_NOTRIGGER = 0xFFFFFFFF
DI_FFNOMINALMAX = 10000
DIERR_UNSUPPORTED = 0x80004001
DIERR_OTHERAPPHASPRIO = 0x80070005
DIERR_INPUTLOST = 0x8007001E
DIERR_NOTACQUIRED = 0x8007000C
DIERR_UNPLUGGED = 0x80040209
DEVICE_LOST = (DIERR_INPUTLOST, DIERR_NOTACQUIRED, DIERR_UNPLUGGED)

IID_IDirectInput8W = "BF798031-483A-4DA2-AA99-5D64ED369700"
GUID_XAxis = "A36D02E0-C9F3-11CF-BFC7-444553540000"
GUID_YAxis = "A36D02E1-C9F3-11CF-BFC7-444553540000"
GUID_ZAxis = "A36D02E2-C9F3-11CF-BFC7-444553540000"
GUID_RxAxis = "A36D02F4-C9F3-11CF-BFC7-444553540000"
GUID_RyAxis = "A36D02F5-C9F3-11CF-BFC7-444553540000"
GUID_RzAxis = "A36D02E3-C9F3-11CF-BFC7-444553540000"
GUID_Slider = "A36D02E4-C9F3-11CF-BFC7-444553540000"
GUID_POV = "A36D02F2-C9F3-11CF-BFC7-444553540000"
GUID_Spring = "13541C27-8E33-11D0-9AD0-00A0C9A06E35"

# Method order as declared in dinput.h. Calls go through these by name so
# every slot index used is the position in this table.
VTABLES: dict[str, tuple[str, ...]] = {
    "IDirectInput8W": (
        "QueryInterface", "AddRef", "Release", "CreateDevice", "EnumDevices",
        "GetDeviceStatus", "RunControlPanel", "Initialize", "FindDevice",
        "EnumDevicesBySemantics", "ConfigureDevices",
    ),
    "IDirectInputDevice8W": (
        "QueryInterface", "AddRef", "Release", "GetCapabilities", "EnumObjects",
        "GetProperty", "SetProperty", "Acquire", "Unacquire", "GetDeviceState",
        "GetDeviceData", "SetDataFormat", "SetEventNotification",
        "SetCooperativeLevel", "GetObjectInfo", "GetDeviceInfo", "RunControlPanel",
        "Initialize", "CreateEffect", "EnumEffects", "GetEffectInfo",
        "GetForceFeedbackState", "SendForceFeedbackCommand",
        "EnumCreatedEffectObjects", "Escape", "Poll", "SendDeviceData",
        "EnumEffectsInFile", "WriteEffectToFile", "BuildActionMap", "SetActionMap",
        "GetImageInfo",
    ),
    "IDirectInputEffect": (
        "QueryInterface", "AddRef", "Release", "Initialize", "GetEffectGuid",
        "GetParameters", "SetParameters", "Start", "Stop", "GetEffectStatus",
        "Download", "Unload", "Escape",
    ),
}

# Every (interface, method) this tool calls, for the vtable review.
USED_METHODS: tuple[tuple[str, str], ...] = (
    ("IDirectInput8W", "Release"),
    ("IDirectInput8W", "CreateDevice"),
    ("IDirectInput8W", "EnumDevices"),
    ("IDirectInputDevice8W", "Release"),
    ("IDirectInputDevice8W", "GetCapabilities"),
    ("IDirectInputDevice8W", "EnumObjects"),
    ("IDirectInputDevice8W", "GetProperty"),
    ("IDirectInputDevice8W", "SetProperty"),
    ("IDirectInputDevice8W", "Acquire"),
    ("IDirectInputDevice8W", "Unacquire"),
    ("IDirectInputDevice8W", "GetDeviceState"),
    ("IDirectInputDevice8W", "SetDataFormat"),
    ("IDirectInputDevice8W", "SetCooperativeLevel"),
    ("IDirectInputDevice8W", "GetObjectInfo"),
    ("IDirectInputDevice8W", "CreateEffect"),
    ("IDirectInputDevice8W", "EnumEffects"),
    ("IDirectInputDevice8W", "GetForceFeedbackState"),
    ("IDirectInputDevice8W", "SendForceFeedbackCommand"),
    ("IDirectInputDevice8W", "Poll"),
    ("IDirectInputEffect", "Release"),
    ("IDirectInputEffect", "SetParameters"),
    ("IDirectInputEffect", "Start"),
    ("IDirectInputEffect", "Stop"),
    ("IDirectInputEffect", "Unload"),
)


def vtable_slot(interface: str, method: str) -> int:
    return VTABLES[interface].index(method)


class GUID(ctypes.Structure):
    _fields_ = [("Data1", DWORD), ("Data2", WORD), ("Data3", WORD), ("Data4", ctypes.c_uint8 * 8)]

    @classmethod
    def parse(cls, text: str) -> GUID:
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)

    def __str__(self) -> str:
        return str(uuid.UUID(bytes_le=bytes(self))).upper()


class DIDEVICEINSTANCEW(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("guidInstance", GUID),
        ("guidProduct", GUID),
        ("dwDevType", DWORD),
        ("tszInstanceName", WCHAR260),
        ("tszProductName", WCHAR260),
        ("guidFFDriver", GUID),
        ("wUsagePage", WORD),
        ("wUsage", WORD),
    ]


class DIDEVICEOBJECTINSTANCEW(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("guidType", GUID),
        ("dwOfs", DWORD),
        ("dwType", DWORD),
        ("dwFlags", DWORD),
        ("tszName", WCHAR260),
        ("dwFFMaxForce", DWORD),
        ("dwFFForceResolution", DWORD),
        ("wCollectionNumber", WORD),
        ("wDesignatorIndex", WORD),
        ("wUsagePage", WORD),
        ("wUsage", WORD),
        ("dwDimension", DWORD),
        ("wExponent", WORD),
        ("wReportId", WORD),
    ]


class DIDEVCAPS(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("dwFlags", DWORD),
        ("dwDevType", DWORD),
        ("dwAxes", DWORD),
        ("dwButtons", DWORD),
        ("dwPOVs", DWORD),
        ("dwFFSamplePeriod", DWORD),
        ("dwFFMinTimeResolution", DWORD),
        ("dwFirmwareRevision", DWORD),
        ("dwHardwareRevision", DWORD),
        ("dwFFDriverVersion", DWORD),
    ]


class DIEFFECTINFOW(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("guid", GUID),
        ("dwEffType", DWORD),
        ("dwStaticParams", DWORD),
        ("dwDynamicParams", DWORD),
        ("tszName", WCHAR260),
    ]


class DIOBJECTDATAFORMAT(ctypes.Structure):
    _fields_ = [("pguid", ctypes.POINTER(GUID)), ("dwOfs", DWORD), ("dwType", DWORD), ("dwFlags", DWORD)]


class DIDATAFORMAT(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("dwObjSize", DWORD),
        ("dwFlags", DWORD),
        ("dwDataSize", DWORD),
        ("dwNumObjs", DWORD),
        ("rgodf", ctypes.POINTER(DIOBJECTDATAFORMAT)),
    ]


class DIPROPHEADER(ctypes.Structure):
    _fields_ = [("dwSize", DWORD), ("dwHeaderSize", DWORD), ("dwObj", DWORD), ("dwHow", DWORD)]


class DIPROPDWORD(ctypes.Structure):
    _fields_ = [("diph", DIPROPHEADER), ("dwData", DWORD)]


class DICONDITION(ctypes.Structure):
    _fields_ = [
        ("lOffset", LONG),
        ("lPositiveCoefficient", LONG),
        ("lNegativeCoefficient", LONG),
        ("dwPositiveSaturation", DWORD),
        ("dwNegativeSaturation", DWORD),
        ("lDeadBand", LONG),
    ]


class DIEFFECT(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("dwFlags", DWORD),
        ("dwDuration", DWORD),
        ("dwSamplePeriod", DWORD),
        ("dwGain", DWORD),
        ("dwTriggerButton", DWORD),
        ("dwTriggerRepeatInterval", DWORD),
        ("cAxes", DWORD),
        ("rgdwAxes", ctypes.POINTER(DWORD)),
        ("rglDirection", ctypes.POINTER(LONG)),
        ("lpEnvelope", ctypes.c_void_p),
        ("cbTypeSpecificParams", DWORD),
        ("lpvTypeSpecificParams", ctypes.c_void_p),
        ("dwStartDelay", DWORD),
    ]


JOY_AXES = 8
JOY_POVS = 4
JOY_BUTTONS = 128
JOY_POV_OFS = JOY_AXES * 4
JOY_BUTTON_OFS = JOY_POV_OFS + JOY_POVS * 4
JOY_AXIS_LABELS = ("X", "Y", "Z", "Rx", "Ry", "Rz", "Slider0", "Slider1")


class JOYSTATE(ctypes.Structure):
    """DIJOYSTATE with 128 buttons, read through our own data format."""

    _fields_ = [
        ("axes", LONG * JOY_AXES),
        ("povs", DWORD * JOY_POVS),
        ("buttons", ctypes.c_uint8 * JOY_BUTTONS),
    ]


def joystick_data_format() -> tuple[DIDATAFORMAT, list]:
    """c_dfDIJoystick laid out for JOYSTATE. Returns the format and the
    objects it points at, which must stay alive while DirectInput uses it."""
    axis_guids = [GUID_XAxis, GUID_YAxis, GUID_ZAxis, GUID_RxAxis, GUID_RyAxis, GUID_RzAxis, GUID_Slider, GUID_Slider]
    guids = [GUID.parse(g) for g in axis_guids] + [GUID.parse(GUID_POV)]
    rows = []
    for i in range(JOY_AXES):
        rows.append((ctypes.pointer(guids[i]), i * 4, DIDFT_OPTIONAL | DIDFT_AXIS | DIDFT_ANYINSTANCE, DIDOI_ASPECTPOSITION))
    for i in range(JOY_POVS):
        rows.append((ctypes.pointer(guids[-1]), JOY_POV_OFS + i * 4, DIDFT_OPTIONAL | DIDFT_POV | DIDFT_ANYINSTANCE, 0))
    for i in range(JOY_BUTTONS):
        rows.append((None, JOY_BUTTON_OFS + i, DIDFT_OPTIONAL | DIDFT_BUTTON | DIDFT_ANYINSTANCE, 0))
    objects = (DIOBJECTDATAFORMAT * len(rows))(*rows)
    fmt = DIDATAFORMAT(
        ctypes.sizeof(DIDATAFORMAT),
        ctypes.sizeof(DIOBJECTDATAFORMAT),
        DIDF_ABSAXIS,
        ctypes.sizeof(JOYSTATE),
        len(rows),
        ctypes.cast(objects, ctypes.POINTER(DIOBJECTDATAFORMAT)),
    )
    return fmt, [guids, objects]


def wide(buf) -> str:
    return bytes(buf).decode("utf-16-le", "replace").split("\0", 1)[0]


# ---------------------------------------------------------------------------
# DirectInput8 through ctypes (Windows only, loaded lazily)


class DIError(Exception):
    def __init__(self, stage: str, hr: int):
        self.stage = stage
        self.hr = hr & 0xFFFFFFFF
        super().__init__(f"{stage} failed: {hresult_name(self.hr)} (0x{self.hr:08X})")

    @property
    def name(self) -> str:
        return hresult_name(self.hr)


def _com_call(ptr: int, interface: str, method: str, argtypes: Sequence, *args, restype=ctypes.c_long) -> int:
    vtbl = ctypes.cast(ctypes.c_void_p(ptr), ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    result = proto(vtbl[vtable_slot(interface, method)])(ptr, *args)
    return result & 0xFFFFFFFF if restype is ctypes.c_long else result


def _check(stage: str, hr: int) -> int:
    if hresult_failed(hr):
        raise DIError(stage, hr)
    return hr


class DirectInput:
    def __init__(self) -> None:
        from ctypes import POINTER, byref, c_void_p, c_wchar_p

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.restype = c_void_p
        kernel32.GetModuleHandleW.argtypes = [c_wchar_p]
        self._user32 = user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.CreateWindowExW.restype = c_void_p
        user32.CreateWindowExW.argtypes = [
            DWORD, c_wchar_p, c_wchar_p, DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, c_void_p, c_void_p, c_void_p, c_void_p,
        ]
        user32.DestroyWindow.argtypes = [c_void_p]
        user32.DestroyWindow.restype = ctypes.c_int
        create = ctypes.WinDLL("dinput8").DirectInput8Create
        create.argtypes = [c_void_p, DWORD, POINTER(GUID), POINTER(c_void_p), c_void_p]
        create.restype = ctypes.c_long

        self.hinstance = kernel32.GetModuleHandleW(None)
        self.ptr: int | None = None
        self.hwnd: int | None = None
        out = c_void_p()
        _check("DirectInput8Create", create(self.hinstance, DIRECTINPUT_VERSION, byref(GUID.parse(IID_IDirectInput8W)), byref(out), None) & 0xFFFFFFFF)
        self.ptr = out.value
        # SetCooperativeLevel needs a top-level window; message-only windows
        # are rejected. It stays hidden and needs no message loop.
        self.hwnd = user32.CreateWindowExW(0, "STATIC", "SimSteer wheel probe", 0, 0, 0, 0, 0, None, None, self.hinstance, None)
        if not self.hwnd:
            error = ctypes.get_last_error()
            self.close()
            raise OSError(f"CreateWindowExW failed: error {error}")

    def enum_devices(self, ff_only: bool = False) -> list[DIDEVICEINSTANCEW]:
        found: list[DIDEVICEINSTANCEW] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(DIDEVICEINSTANCEW), ctypes.c_void_p)

        def on_device(inst, _ref):
            copy = DIDEVICEINSTANCEW()
            ctypes.memmove(ctypes.byref(copy), inst, ctypes.sizeof(copy))
            found.append(copy)
            return 1

        callback = callback_type(on_device)
        flags = DIEDFL_ATTACHEDONLY | (DIEDFL_FORCEFEEDBACK if ff_only else 0)
        _check("EnumDevices", _com_call(
            self.ptr, "IDirectInput8W", "EnumDevices",
            [DWORD, callback_type, ctypes.c_void_p, DWORD],
            DI8DEVCLASS_GAMECTRL, callback, None, flags,
        ))
        return found

    def create_device(self, inst: DIDEVICEINSTANCEW) -> DInputDevice:
        out = ctypes.c_void_p()
        _check("CreateDevice", _com_call(
            self.ptr, "IDirectInput8W", "CreateDevice",
            [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            ctypes.byref(inst.guidInstance), ctypes.byref(out), None,
        ))
        return DInputDevice(out.value, inst)

    def close(self) -> None:
        if self.ptr:
            _com_call(self.ptr, "IDirectInput8W", "Release", [], restype=DWORD)
            self.ptr = None
        if self.hwnd:
            self._user32.DestroyWindow(self.hwnd)
            self.hwnd = None


class DInputDevice:
    def __init__(self, ptr: int, inst: DIDEVICEINSTANCEW) -> None:
        self.ptr: int | None = ptr
        self.inst = inst
        self.name = wide(inst.tszProductName)
        self.vid, self.pid = dinput_vid_pid(inst.guidProduct.Data1)
        self.acquired = False
        self._format = None

    def call(self, method: str, argtypes: Sequence, *args, restype=ctypes.c_long) -> int:
        return _com_call(self.ptr, "IDirectInputDevice8W", method, argtypes, *args, restype=restype)

    def set_data_format(self) -> None:
        self._format = joystick_data_format()
        _check("SetDataFormat", self.call("SetDataFormat", [ctypes.POINTER(DIDATAFORMAT)], ctypes.byref(self._format[0])))

    def capabilities(self) -> DIDEVCAPS:
        caps = DIDEVCAPS()
        caps.dwSize = ctypes.sizeof(DIDEVCAPS)
        _check("GetCapabilities", self.call("GetCapabilities", [ctypes.POINTER(DIDEVCAPS)], ctypes.byref(caps)))
        return caps

    def objects(self) -> list[DIDEVICEOBJECTINSTANCEW]:
        found: list[DIDEVICEOBJECTINSTANCEW] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(DIDEVICEOBJECTINSTANCEW), ctypes.c_void_p)

        def on_object(obj, _ref):
            copy = DIDEVICEOBJECTINSTANCEW()
            ctypes.memmove(ctypes.byref(copy), obj, ctypes.sizeof(copy))
            found.append(copy)
            return 1

        callback = callback_type(on_object)
        _check("EnumObjects", self.call("EnumObjects", [callback_type, ctypes.c_void_p, DWORD], callback, None, DIDFT_ALL))
        return found

    def effects(self) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(DIEFFECTINFOW), ctypes.c_void_p)

        def on_effect(info, _ref):
            found.append({"name": wide(info.contents.tszName), "guid": str(info.contents.guid)})
            return 1

        callback = callback_type(on_effect)
        _check("EnumEffects", self.call("EnumEffects", [callback_type, ctypes.c_void_p, DWORD], callback, None, DIEFT_ALL))
        return found

    def object_at(self, offset: int) -> DIDEVICEOBJECTINSTANCEW | None:
        """The object our data format put at `offset`, or None. EnumObjects
        reports native offsets, which differ from the data format's."""
        obj = DIDEVICEOBJECTINSTANCEW()
        obj.dwSize = ctypes.sizeof(obj)
        hr = self.call("GetObjectInfo", [ctypes.POINTER(DIDEVICEOBJECTINSTANCEW), DWORD, DWORD], ctypes.byref(obj), offset, DIPH_BYOFFSET)
        return None if hresult_failed(hr) else obj

    def set_cooperative_level(self, hwnd: int, flags: int) -> None:
        _check("SetCooperativeLevel", self.call("SetCooperativeLevel", [ctypes.c_void_p, DWORD], hwnd, flags))

    def acquire(self) -> int:
        hr = _check("Acquire", self.call("Acquire", []))
        self.acquired = True
        return hr

    def unacquire(self) -> None:
        if self.ptr and self.acquired:
            self.call("Unacquire", [])
            self.acquired = False

    def read(self, state: JOYSTATE) -> int:
        self.call("Poll", [])
        return self.call("GetDeviceState", [DWORD, ctypes.c_void_p], ctypes.sizeof(JOYSTATE), ctypes.byref(state))

    def property_dword(self, prop: int) -> tuple[int, int]:
        value = DIPROPDWORD(DIPROPHEADER(ctypes.sizeof(DIPROPDWORD), ctypes.sizeof(DIPROPHEADER), 0, DIPH_DEVICE), 0)
        # Predefined DIPROP_* "GUIDs" are small integers passed as the pointer.
        hr = self.call("GetProperty", [ctypes.c_void_p, ctypes.POINTER(DIPROPDWORD)], prop, ctypes.byref(value))
        return hr, value.dwData

    def set_property_dword(self, prop: int, data: int) -> int:
        value = DIPROPDWORD(DIPROPHEADER(ctypes.sizeof(DIPROPDWORD), ctypes.sizeof(DIPROPHEADER), 0, DIPH_DEVICE), data)
        return self.call("SetProperty", [ctypes.c_void_p, ctypes.POINTER(DIPROPDWORD)], prop, ctypes.byref(value))

    def ff_state(self) -> dict[str, Any]:
        state = DWORD()
        hr = self.call("GetForceFeedbackState", [ctypes.POINTER(DWORD)], ctypes.byref(state))
        if hresult_failed(hr):
            return {"result": hresult_name(hr)}
        return {"result": hresult_name(hr), "raw": f"0x{state.value:08X}", "flags": ff_state_flags(state.value)}

    def ff_command(self, flags: int) -> int:
        return self.call("SendForceFeedbackCommand", [DWORD], flags)

    def create_effect(self, guid: str, effect: DIEFFECT) -> int:
        out = ctypes.c_void_p()
        _check("CreateEffect", self.call(
            "CreateEffect",
            [ctypes.POINTER(GUID), ctypes.POINTER(DIEFFECT), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            ctypes.byref(GUID.parse(guid)), ctypes.byref(effect), ctypes.byref(out), None,
        ))
        return out.value

    def close(self) -> None:
        if self.ptr:
            try:
                self.unacquire()
            finally:
                self.call("Release", [], restype=DWORD)
                self.ptr = None


def describe_dinput_device(dev: DInputDevice, caps: DIDEVCAPS) -> tuple[dict[str, Any], list[int], list[int]]:
    objects = dev.objects()
    axes = []
    for i in range(JOY_AXES):
        obj = dev.object_at(i * 4)
        if obj is not None:
            axes.append((i, wide(obj.tszName) or JOY_AXIS_LABELS[i]))
    buttons = [i for i in range(JOY_BUTTONS) if dev.object_at(JOY_BUTTON_OFS + i) is not None]
    actuators = [wide(o.tszName) for o in objects if o.dwType & DIDFT_FFACTUATOR]
    axes_total = sum(1 for o in objects if o.dwType & DIDFT_AXIS)
    return device_entry(
        backend="dinput",
        key=f"dinput:{dev.inst.guidInstance}",
        name=dev.name,
        vid=dev.vid,
        pid=dev.pid,
        axis_names=[n for _, n in axes],
        buttons=len(buttons),
        type=device_type_name(dev.inst.dwDevType),
        ffb=bool(caps.dwFlags & DIDC_FORCEFEEDBACK),
        ff_actuators=actuators,
        axes_total=axes_total,
        axis_slots=[JOY_AXIS_LABELS[i] for i, _ in axes],
        instance=str(dev.inst.guidInstance),
    ), [i for i, _ in axes], buttons


class BackendUnavailable(Exception):
    pass


class DInputBackend:
    name = "dinput"

    def __init__(self, only_vid: int | None) -> None:
        self.only_vid = only_vid
        self.di: DirectInput | None = None
        self.devices: list[dict] = []
        self.errors: list[dict] = []
        self._open: list[dict] = []
        self._state = None

    def open(self) -> list[dict]:
        if sys.platform != "win32":
            raise BackendUnavailable("DirectInput needs Windows")
        self.di = DirectInput()
        self._state = JOYSTATE()
        for inst in self.di.enum_devices():
            vid, _ = dinput_vid_pid(inst.guidProduct.Data1)
            if self.only_vid is not None and vid != self.only_vid:
                continue
            dev = None
            try:
                dev = self.di.create_device(inst)
                dev.set_data_format()
                caps = dev.capabilities()
                entry, axis_idx, button_idx = describe_dinput_device(dev, caps)
                if entry["ffb"]:
                    try:
                        entry["ff_effects"] = dev.effects()
                    except DIError as e:
                        entry["ff_effects"] = e.name
                dev.set_cooperative_level(self.di.hwnd, DISCL_NONEXCLUSIVE | DISCL_BACKGROUND)
                entry["acquire"] = hresult_name(dev.acquire())
            except (DIError, OSError) as e:
                stage, name = (e.stage, e.name) if isinstance(e, DIError) else ("open", str(e))
                self.errors.append({"device": wide(inst.tszProductName), "stage": stage, "error": name})
                if dev is not None:
                    _quietly(dev.close)
                continue
            self.devices.append(entry)
            self._open.append({"dev": dev, "entry": entry, "axes": axis_idx, "buttons": button_idx, "lost_at": None, "retry_at": 0.0})
        return self.devices

    def poll(self, t: float, events: list) -> list[tuple[str, tuple, tuple]]:
        out = []
        state = self._state
        for rec in self._open:
            dev = rec["dev"]
            if rec["lost_at"] is not None:
                if t < rec["retry_at"]:
                    continue
                rec["retry_at"] = t + REACQUIRE_S
                try:
                    dev.acquire()
                except DIError:
                    continue
                rec["lost_at"] = None
                events.append({"t": round(t, 4), "backend": self.name, "device": dev.name, "event": "reacquired"})
            hr = dev.read(state)
            if hresult_failed(hr):
                dev.acquired = hr not in DEVICE_LOST
                rec["lost_at"] = t
                rec["retry_at"] = t + REACQUIRE_S
                events.append({"t": round(t, 4), "backend": self.name, "device": dev.name, "event": "lost", "error": hresult_name(hr)})
                continue
            out.append((rec["entry"]["key"], tuple(state.axes[i] for i in rec["axes"]), tuple(state.buttons[i] >> 7 for i in rec["buttons"])))
        return out

    def close(self) -> None:
        for rec in self._open:
            try:
                rec["dev"].close()
            except OSError:
                pass
        self._open.clear()
        if self.di:
            self.di.close()
            self.di = None


# ---------------------------------------------------------------------------
# SDL through pygame-ce (optional)


class SdlBackend:
    name = "sdl"

    def __init__(self, only_vid: int | None) -> None:
        self.only_vid = only_vid
        self.pg = None
        self.devices: list[dict] = []
        self.errors: list[dict] = []
        self._joysticks: dict[int, tuple[Any, dict]] = {}
        self.info: dict[str, Any] = {}

    def open(self) -> list[dict]:
        os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
        # SDL2's DirectInput driver opens force feedback devices exclusively,
        # sends DISFFC_RESET and turns autocenter on (SDL_dinputjoystick.c),
        # which can move the rim and lock a game out. With it off SDL reads
        # through Windows.Gaming.Input, which sends no force commands.
        os.environ["SDL_DIRECTINPUT_ENABLED"] = "0"
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            import pygame
        except ImportError as e:
            raise BackendUnavailable(PYGAME_HINT) from e
        self.pg = pygame
        if sys.platform != "win32" and not os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        try:
            pygame.display.init()
            pygame.joystick.init()
        except pygame.error as e:
            raise BackendUnavailable(f"SDL init failed: {e}") from e
        self.info = {
            "pygame": pygame.version.ver,
            "pygame_ce": bool(getattr(pygame, "IS_CE", False)),
            "sdl": ".".join(map(str, pygame.get_sdl_version())),
            "allow_background_events": os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"],
            "directinput_driver": os.environ["SDL_DIRECTINPUT_ENABLED"],
            "sdl_window": False,
        }
        self._settle()
        for i in range(pygame.joystick.get_count()):
            self._open_index(i)
        pygame.event.clear()
        return self.devices

    def _settle(self, clock: Callable[[], float] = time.monotonic) -> None:
        pg = self.pg
        t0 = last_change = clock()
        count = pg.joystick.get_count()
        while clock() - t0 < SDL_SETTLE_S:
            pg.event.pump()
            now = pg.joystick.get_count()
            if now != count:
                count, last_change = now, clock()
            elif count and clock() - last_change >= SDL_STABLE_S:
                break
            time.sleep(0.02)
        self.info["settle_s"] = round(clock() - t0, 2)

    def _open_index(self, index: int) -> dict | None:
        pg = self.pg
        try:
            js = pg.joystick.Joystick(index)
        except pg.error as e:
            self.errors.append({"device_index": index, "stage": "open", "error": str(e)})
            return None
        if js.get_instance_id() in self._joysticks:
            return None
        vid, pid = sdl_vid_pid(js.get_guid())
        if self.only_vid is not None and vid != self.only_vid:
            js.quit()
            return None
        entry = device_entry(
            backend="sdl",
            key=f"sdl:{js.get_instance_id()}",
            name=js.get_name(),
            vid=vid,
            pid=pid,
            axis_names=[f"axis{a}" for a in range(js.get_numaxes())],
            buttons=js.get_numbuttons(),
            hats=js.get_numhats(),
            guid=js.get_guid(),
        )
        self.devices.append(entry)
        self._joysticks[js.get_instance_id()] = (js, entry)
        return entry

    def poll(self, t: float, events: list) -> list[tuple[str, tuple, tuple]]:
        pg = self.pg
        for ev in pg.event.get():
            if ev.type == pg.JOYDEVICEREMOVED and ev.instance_id in self._joysticks:
                _, entry = self._joysticks.pop(ev.instance_id)
                events.append({"t": round(t, 4), "backend": self.name, "device": entry["name"], "event": "removed"})
            elif ev.type == pg.JOYDEVICEADDED:
                entry = self._open_index(ev.device_index)
                if entry is not None:
                    events.append({"t": round(t, 4), "backend": self.name, "device": entry["name"], "event": "added"})
        out = []
        for js, entry in list(self._joysticks.values()):
            try:
                axes = tuple(js.get_axis(i) for i in range(entry["axes"]))
                buttons = tuple(js.get_button(i) for i in range(entry["buttons"]))
            except pg.error:
                continue
            out.append((entry["key"], axes, buttons))
        return out

    def close(self) -> None:
        if self.pg:
            self.pg.quit()
            self.pg = None


BACKENDS: dict[str, type] = {"sdl": SdlBackend, "dinput": DInputBackend}


# ---------------------------------------------------------------------------
# Focus tracking (Windows only)


TERMINALS = {"windowsterminal.exe", "conhost.exe", "openconsole.exe", "cmd.exe", "powershell.exe", "pwsh.exe"}


class FocusTracker:
    """Tells whether a window other than the one that started the probe
    has focus. The terminal is focused when the probe starts, so any other
    foreground window during logging is the game."""

    def __init__(self) -> None:
        self.available = sys.platform == "win32"
        self.home = None
        self.windows: list[dict] = []
        self._last_check = -1.0
        self._hwnd = None
        if self.available:
            from ctypes import POINTER, c_void_p, c_wchar_p

            self._u = ctypes.WinDLL("user32")
            self._k = ctypes.WinDLL("kernel32")
            self._u.GetForegroundWindow.restype = c_void_p
            self._u.GetWindowTextW.argtypes = [c_void_p, c_wchar_p, ctypes.c_int]
            self._u.GetWindowThreadProcessId.argtypes = [c_void_p, POINTER(DWORD)]
            self._k.OpenProcess.restype = c_void_p
            self._k.OpenProcess.argtypes = [DWORD, ctypes.c_int, DWORD]
            self._k.QueryFullProcessImageNameW.argtypes = [c_void_p, DWORD, c_wchar_p, POINTER(DWORD)]
            self._k.CloseHandle.argtypes = [c_void_p]
            self.home = self._u.GetForegroundWindow()
            self.home_window = self._describe(self.home) if self.home else {"title": None, "exe": None}

    def _describe(self, hwnd) -> dict[str, Any]:
        title = ctypes.create_unicode_buffer(256)
        self._u.GetWindowTextW(hwnd, title, 256)
        pid = DWORD()
        self._u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        exe = None
        handle = self._k.OpenProcess(0x1000, 0, pid.value)
        if handle:
            path = ctypes.create_unicode_buffer(520)
            size = DWORD(520)
            if self._k.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                exe = Path(path.value).name
            self._k.CloseHandle(handle)
        return {"title": title.value, "exe": exe}

    def foreign(self, t: float) -> bool | None:
        if not self.available:
            return None
        if t - self._last_check >= FOCUS_CHECK_S:
            self._last_check = t
            hwnd = self._u.GetForegroundWindow()
            if hwnd != self._hwnd:
                self._hwnd = hwnd
                self.windows.append({"t": round(t, 3), **self._describe(hwnd)})
        return self._hwnd is not None and self._hwnd != self.home

    def report(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False}
        out = {"available": True, "start_window": self.home_window, "windows": self.windows}
        if (self.home_window.get("exe") or "").lower() not in TERMINALS:
            out["warning"] = "the probe did not start from a focused terminal, so background_ok may be wrong"
        return out


# ---------------------------------------------------------------------------
# vJoy (read only)

VJOY_STATUS = ("own", "free", "busy", "missing", "unknown")
VJOY_AXES = {"X": 0x30, "Y": 0x31, "Z": 0x32, "Rx": 0x33, "Ry": 0x34, "Rz": 0x35, "Slider0": 0x36, "Slider1": 0x37, "Wheel": 0x38}


def _vjoy_dll_candidates() -> list[str]:
    arch = "x64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "x86"
    roots = [os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramW6432", "")]
    return ["vJoyInterface.dll"] + [str(Path(r) / "vJoy" / arch / "vJoyInterface.dll") for r in roots if r]


def probe_vjoy(device_id: int = 1) -> dict[str, Any]:
    if sys.platform != "win32":
        return {"available": False, "error": "vJoy needs Windows"}
    dll = path = None
    for candidate in _vjoy_dll_candidates():
        try:
            dll = ctypes.CDLL(candidate)
            path = candidate
            break
        except OSError:
            continue
    if dll is None:
        return {"available": False, "error": "vJoyInterface.dll not found; vJoy is probably not installed"}

    missing: list[str] = []

    def fn(name: str, restype, argtypes):
        f = getattr(dll, name, None)
        if f is None:
            missing.append(name)
            return None
        f.restype, f.argtypes = restype, argtypes
        return f

    out: dict[str, Any] = {"available": True, "dll": path, "device": device_id}
    enabled = fn("vJoyEnabled", ctypes.c_int, [])
    version = fn("GetvJoyVersion", ctypes.c_short, [])
    match = fn("DriverMatch", ctypes.c_int, [ctypes.POINTER(WORD), ctypes.POINTER(WORD)])
    status = fn("GetVJDStatus", ctypes.c_int, [ctypes.c_uint])
    axis_exist = fn("GetVJDAxisExist", ctypes.c_int, [ctypes.c_uint, ctypes.c_uint])
    button_count = fn("GetVJDButtonNumber", ctypes.c_int, [ctypes.c_uint])
    ffb_cap = fn("vJoyFfbCap", ctypes.c_int, [ctypes.POINTER(ctypes.c_int)])
    device_ffb = fn("IsDeviceFfb", ctypes.c_int, [ctypes.c_uint])

    out["enabled"] = bool(enabled()) if enabled else None
    out["version"] = format_vjoy_version(version() & 0xFFFF) if version else None
    if match:
        dll_ver, drv_ver = WORD(), WORD()
        out["driver_match"] = bool(match(ctypes.byref(dll_ver), ctypes.byref(drv_ver)))
        out["dll_version"] = format_vjoy_version(dll_ver.value)
        out["driver_version"] = format_vjoy_version(drv_ver.value)
    if status:
        code = status(device_id)
        out["device_status"] = VJOY_STATUS[code] if 0 <= code < len(VJOY_STATUS) else code
    if axis_exist:
        out["device_axes"] = [name for name, usage in VJOY_AXES.items() if axis_exist(device_id, usage)]
    if button_count:
        out["device_buttons"] = button_count(device_id)
    if ffb_cap:
        supported = ctypes.c_int()
        out["driver_ffb_capable"] = bool(ffb_cap(ctypes.byref(supported))) and bool(supported.value)
    if device_ffb:
        out["device_ffb_enabled"] = bool(device_ffb(device_id))
    out["missing_exports"] = missing
    return out


# ---------------------------------------------------------------------------
# Logging run


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def beep(freq: int, ms: int) -> None:
    if sys.platform == "win32":
        import winsound

        try:
            winsound.Beep(freq, ms)
        except RuntimeError:
            pass


def countdown(seconds: int, what: str, sleep: Callable[[float], None] = time.sleep) -> None:
    for left in range(seconds, 0, -1):
        say(f"{what} in {left}...")
        sleep(1)


def open_backends(names: Sequence[str], only_vid: int | None) -> tuple[list, dict[str, dict]]:
    opened = []
    sections: dict[str, dict] = {}
    for name in names:
        backend = BACKENDS[name](only_vid)
        try:
            devices = backend.open()
        except BackendUnavailable as e:
            sections[name] = {"available": False, "error": str(e), "devices": []}
            say(f"{name}: {e}")
            continue
        except (DIError, OSError) as e:
            sections[name] = {"available": False, "error": str(e), "devices": []}
            say(f"{name}: {e}")
            backend.close()
            continue
        sections[name] = {"available": True, "error": None, "devices": devices, "open_errors": backend.errors}
        if getattr(backend, "info", None):
            sections[name]["info"] = backend.info
        opened.append(backend)
    return opened, sections


def log_inputs(
    backends: Sequence,
    seconds: float,
    focus: FocusTracker,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, list], list[dict], bool]:
    samples: dict[str, list] = {}
    events: list[dict] = []
    t0 = clock()
    try:
        while True:
            t = clock() - t0
            if t >= seconds:
                break
            foreign = focus.foreign(t)
            for backend in backends:
                for key, axes, buttons in backend.poll(t, events):
                    samples.setdefault(key, []).append((t, axes, buttons, foreign))
            sleep(POLL_SLEEP_S)
    except KeyboardInterrupt:
        return samples, events, True
    return samples, events, False


def write_samples(path: Path, samples: dict[str, list]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key, rows in samples.items():
            prev = None
            for t, axes, buttons, foreign in rows:
                if prev is None or (axes, buttons) != prev:
                    f.write(json.dumps({"t": round(t, 5), "device": key, "axes": axes, "buttons": buttons, "foreign_focus": foreign}) + "\n")
                    prev = (axes, buttons)


def run_logging(args, only_vid: int | None) -> tuple[dict, int]:
    names = ["sdl", "dinput"] if args.backend == "both" else [args.backend]
    focus = FocusTracker()
    backends, sections = open_backends(names, only_vid)
    devices = [d for s in sections.values() for d in s["devices"]]
    for d in devices:
        say(f"{d['backend']:6} {d['vid_hex'] or '?':6} {d['pid_hex'] or '?':6} {'FANATEC ' if d['fanatec'] else ''}{d['name']} ({d['axes']} axes, {d['buttons']} buttons)")
    if not any(d["fanatec"] for d in devices):
        say(NO_FANATEC_NOTE)

    samples: dict[str, list] = {}
    events: list[dict] = []
    interrupted = False
    try:
        if devices and args.seconds > 0:
            countdown(args.countdown, "Focus the game now. Logging starts")
            beep(880, 150)
            say(f"Logging {args.seconds:g} s. Move the wheel and every pedal.")
            samples, events, interrupted = log_inputs(backends, args.seconds, focus)
            beep(440, 300)
        elif not devices:
            say("No devices matched; nothing to log.")
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for backend in backends:
            backend.close()

    for section in sections.values():
        for d in section["devices"]:
            d["summary"] = summarize_samples(samples.get(d["key"], []), d["axis_names"])
        if section["available"]:
            section.update(summarize_backend(section["devices"]))

    report = build_report(
        mode="log",
        args=vars(args),
        backends=sections,
        vjoy=probe_vjoy(),
        focus=focus.report(),
        events=events,
        interrupted=interrupted,
    )
    if args.report and samples:
        write_samples(Path(args.report).with_suffix(".samples.jsonl"), samples)
    if interrupted:
        say("Interrupted; wrote a partial report.")
    return report, EXIT_INTERRUPTED if interrupted else EXIT_OK


# ---------------------------------------------------------------------------
# FFB test


class SpringGuard:
    """Stops the spring from atexit, the console control handler and the
    main thread.

    The handler runs on its own thread, so effect calls are serialised by
    `lock`. It returns FALSE so Python still raises KeyboardInterrupt on
    Ctrl+C. Stop and STOPALL are safe to repeat, so every path sends them
    until the effect is released.
    """

    def __init__(self, effect: int, device: DInputDevice) -> None:
        self.effect = effect
        self.device = device
        self.lock = threading.Lock()
        self.stopped = False
        self.released = False
        self.reason: str | None = None
        self._handler = None
        self._set_handler = None

    def install(self) -> None:
        atexit.register(self.stop, "atexit")
        handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, DWORD)
        self._handler = handler_type(self._on_console_event)
        set_handler = ctypes.WinDLL("kernel32").SetConsoleCtrlHandler
        set_handler.argtypes = [handler_type, ctypes.c_int]
        set_handler.restype = ctypes.c_int
        set_handler(self._handler, 1)
        self._set_handler = set_handler

    def _on_console_event(self, event: int) -> int:
        self.stop(f"console event {event}")
        return 0

    def _send_stop(self) -> None:
        _quietly(lambda: _com_call(self.effect, "IDirectInputEffect", "Stop", []))
        _quietly(lambda: self.device.ff_command(DISFFC_STOPALL))

    def stop(self, reason: str) -> None:
        got = self.lock.acquire(timeout=1.0)
        try:
            if self.released:
                return
            if not self.stopped:
                self.stopped = True
                self.reason = reason
            self._send_stop()
        finally:
            if got:
                self.lock.release()

    def release(self) -> None:
        """Main thread only. Waits for the lock so the handler never touches
        a released effect."""
        with self.lock:
            if self.released:
                return
            self.stopped = True
            self._send_stop()
            _quietly(lambda: _com_call(self.effect, "IDirectInputEffect", "Unload", []))
            _quietly(lambda: _com_call(self.effect, "IDirectInputEffect", "Release", [], restype=DWORD))
            self.released = True

    def uninstall(self) -> None:
        atexit.unregister(self.stop)
        if self._set_handler is not None:
            self._set_handler(self._handler, 0)


def _quietly(fn: Callable[[], Any]) -> None:
    """Teardown keeps going when one call fails; ctypes turns a COM access
    violation into OSError."""
    try:
        fn()
    except (OSError, KeyboardInterrupt):
        pass


def autocenter_off_ok(hr: int) -> bool:
    return not hresult_failed(hr) or hr == DIERR_UNSUPPORTED


def _spring_effect(params: dict[str, int], axis_id: int) -> tuple[DIEFFECT, list]:
    """The axis is named by object ID (DIDEVICEOBJECTINSTANCE.dwType). The DD2
    rejects its EnumObjects dwOfs with DIEFF_OBJECTOFFSETS (DIERR_INVALIDPARAM),
    because that is a native offset, not one in our data format."""
    condition = (DICONDITION * 1)(DICONDITION(
        params["offset"], params["coefficient"], params["coefficient"],
        params["saturation"], params["saturation"], params["deadband"],
    ))
    axes = (DWORD * 1)(axis_id)
    direction = (LONG * 1)(0)
    eff = DIEFFECT()
    eff.dwSize = ctypes.sizeof(DIEFFECT)
    eff.dwFlags = DIEFF_CARTESIAN | DIEFF_OBJECTIDS
    eff.dwDuration = params["duration_us"]
    eff.dwGain = params["gain"]
    eff.dwTriggerButton = DIEB_NOTRIGGER
    eff.cAxes = 1
    eff.rgdwAxes = ctypes.cast(axes, ctypes.POINTER(DWORD))
    eff.rglDirection = ctypes.cast(direction, ctypes.POINTER(LONG))
    eff.cbTypeSpecificParams = ctypes.sizeof(condition)
    eff.lpvTypeSpecificParams = ctypes.cast(condition, ctypes.c_void_p)
    return eff, [condition, axes, direction]


def pick_ffb_device(devices: Sequence[dict], only_vid: int | None) -> dict | None:
    """Fanatec only unless --only-vid names another vendor. Never vJoy."""
    wanted = only_vid if only_vid is not None else FANATEC_VID
    for d in sorted(devices, key=lambda d: d.get("type") != "driving"):
        if d["vid"] == wanted and not is_vjoy(d["vid"], d["pid"]):
            return d
    return None


def run_ffb_test(
    level: float,
    only_vid: int | None,
    countdown_s: int,
    with_sdl: bool = False,
    confirm: Callable[[str], str] = input,
) -> tuple[dict, list[dict]]:
    params = spring_parameters(level)
    ffb: dict[str, Any] = {"attempted": False, "level_percent": level, "spring": params, "steps": []}
    if sys.platform != "win32":
        ffb.update(result="unavailable", error="DirectInput FFB needs Windows")
        return ffb, []

    def step(stage: str, hr: int) -> int:
        ffb["steps"].append({"stage": stage, "result": hresult_name(hr)})
        return hr

    di = dev = guard = sdl = None
    listed: list[dict] = []
    try:
        di = DirectInput()
        candidates = []
        for inst in di.enum_devices(ff_only=True):
            vid, pid = dinput_vid_pid(inst.guidProduct.Data1)
            entry = device_entry(
                backend="dinput", key=f"dinput:{inst.guidInstance}", name=wide(inst.tszProductName),
                vid=vid, pid=pid, axis_names=[], buttons=0, type=device_type_name(inst.dwDevType), ffb=True,
            )
            candidates.append((entry, inst))
        listed = filter_devices([c[0] for c in candidates], only_vid)
        target = pick_ffb_device(listed, only_vid)
        if target is None:
            wanted = hex16(only_vid if only_vid is not None else FANATEC_VID)
            ffb.update(result="no_device", error=f"no force feedback device with vendor {wanted}; the base may be unplugged")
            return ffb, listed
        inst = next(i for e, i in candidates if e is target)
        ffb["device"] = {k: target[k] for k in ("name", "vid_hex", "pid_hex")}

        say(f"FFB test on {target['name']}: spring centred on zero, max {level:g} percent, {FFB_RUN_S:g} s.")
        say("Turn the rim off centre and keep hands near it.")
        try:
            confirm("Press Enter to start, Ctrl+C to cancel: ")
        except EOFError:
            ffb.update(result="cancelled", error="no confirmation (stdin closed or interrupted); the FFB test needs a person at the rig")
            return ffb, listed
        countdown(countdown_s, "Spring starts")

        if with_sdl:
            # SDL's DirectInput backend opens devices exclusively, which
            # would keep a game's FFB off the wheel. Holding SDL open here
            # shows whether it blocks our exclusive acquire.
            sdl = SdlBackend(only_vid if only_vid is not None else FANATEC_VID)
            try:
                ffb["sdl_held"] = [d["name"] for d in sdl.open()]
            except BackendUnavailable as e:
                ffb.update(result="error", error=str(e), stage="SDL open")
                return ffb, listed

        ffb["attempted"] = True
        dev = di.create_device(inst)
        dev.set_data_format()
        dev.set_cooperative_level(di.hwnd, DISCL_EXCLUSIVE | DISCL_BACKGROUND)
        before = step("SetProperty(AUTOCENTER_OFF) before Acquire", dev.set_property_dword(DIPROP_AUTOCENTER, DIPROPAUTOCENTER_OFF))
        step("Acquire", dev.acquire())
        ffb["ff_state_after_acquire"] = dev.ff_state()
        after = step("SetProperty(AUTOCENTER_OFF)", dev.set_property_dword(DIPROP_AUTOCENTER, DIPROPAUTOCENTER_OFF))
        if not (autocenter_off_ok(before) or autocenter_off_ok(after)):
            ffb.update(result="error", error=f"could not turn autocenter off: {hresult_name(after)}", stage="SetProperty(AUTOCENTER)")
            return ffb, listed
        objects = dev.objects()
        actuators = [o for o in objects if o.dwType & DIDFT_FFACTUATOR and o.dwType & DIDFT_AXIS]
        ffb["actuators"] = [{"name": wide(o.tszName), "id": f"0x{o.dwType:08X}"} for o in actuators]
        if not actuators:
            ffb.update(result="error", error="device reports no force feedback actuator axis", stage="EnumObjects")
            return ffb, listed
        hr, gain = dev.property_dword(DIPROP_FFGAIN)
        ffb["device_gain"] = gain if not hresult_failed(hr) else hresult_name(hr)
        step("SendForceFeedbackCommand(SETACTUATORSON)", dev.ff_command(DISFFC_SETACTUATORSON))
        ffb["ff_state_actuators_on"] = dev.ff_state()

        eff, _buffers = _spring_effect(params, actuators[0].dwType)
        guard = SpringGuard(dev.create_effect(GUID_Spring, eff), dev)
        guard.install()

        updates = 0
        t0 = time.perf_counter()
        with guard.lock:
            if not guard.stopped:
                _check("Start", step("Start", _com_call(guard.effect, "IDirectInputEffect", "Start", [DWORD, DWORD], 1, 0)))
        while not guard.stopped and time.perf_counter() - t0 < FFB_RUN_S:
            time.sleep(FFB_UPDATE_S)
            with guard.lock:
                if guard.stopped:
                    break
                hr = _com_call(guard.effect, "IDirectInputEffect", "SetParameters", [ctypes.POINTER(DIEFFECT), DWORD],
                               ctypes.byref(eff), DIEP_DURATION | DIEP_TYPESPECIFICPARAMS | DIEP_START)
            if hresult_failed(hr):
                raise DIError("SetParameters", hr)
            updates += 1
        played = round(time.perf_counter() - t0, 3)
        if guard.reason:
            ffb.update(result="cancelled", error=f"stopped by {guard.reason}", updates=updates, played_s=played)
        else:
            guard.stop("done")
            ffb.update(result="ok", updates=updates, played_s=played)
        ffb["ff_state_after_stop"] = dev.ff_state()
    except DIError as e:
        ffb.update(result="error", error=e.name, hresult=f"0x{e.hr:08X}", stage=e.stage)
        if e.hr == DIERR_OTHERAPPHASPRIO and e.stage == "Acquire":
            ffb["hint"] = "another app holds the wheel exclusively: the game" + (", or SDL held by --with-sdl" if sdl else "")
    except OSError as e:
        ffb.update(result="error", error=str(e))
    except KeyboardInterrupt:
        ffb.update(result="cancelled", error="interrupted")
    finally:
        # A second Ctrl+C must not cut teardown short; the console handler
        # still stops the effect meanwhile.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if guard is not None:
                guard.release()
                _quietly(guard.uninstall)
            for obj in (dev, di, sdl):
                if obj is not None:
                    _quietly(obj.close)
        finally:
            signal.signal(signal.SIGINT, previous)
    return ffb, listed


# ---------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wheel_probe",
        description="Probe wheel and pedals through SDL and DirectInput, read vJoy, and optionally test FFB.",
    )
    p.add_argument("--report", metavar="PATH", help="write the JSON report here (default: print it)")
    p.add_argument("--backend", choices=("both", "sdl", "dinput"), default="both", help="input backends to log (default: both)")
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="logging window; 0 only enumerates (default: 20)")
    p.add_argument("--countdown", type=int, default=DEFAULT_COUNTDOWN, help="seconds to focus the game first (default: 5)")
    p.add_argument("--only-vid", metavar="HEX", help="only devices with this USB vendor ID, e.g. 0x0EB7")
    p.add_argument("--ffb-test", action="store_true", help="play a centring spring on the Fanatec base for 2 s")
    p.add_argument("--with-sdl", action="store_true", help="with --ffb-test: hold the SDL joysticks open during the test")
    p.add_argument("--ffb-level", type=float, metavar="PERCENT", help=f"spring strength, default {DEFAULT_FFB_LEVEL:g}, max {MAX_FFB_LEVEL:g}")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        say("Interrupted.")
        return EXIT_INTERRUPTED


def _main(argv: Sequence[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ffb_level is not None and not args.ffb_test:
        parser.error("--ffb-level needs --ffb-test")
    if args.with_sdl and not args.ffb_test:
        parser.error("--with-sdl needs --ffb-test")
    try:
        only_vid = parse_vid(args.only_vid) if args.only_vid else None
    except ValueError:
        parser.error(f"--only-vid expects a hex vendor ID, got {args.only_vid!r}")

    if args.ffb_test:
        try:
            level = validate_ffb_level(DEFAULT_FFB_LEVEL if args.ffb_level is None else args.ffb_level)
        except ValueError as e:
            say(f"wheel_probe: {e}")
            return EXIT_USAGE
        ffb, listed = run_ffb_test(level, only_vid, args.countdown, with_sdl=args.with_sdl)
        report = build_report(mode="ffb", args=vars(args), backends={"dinput": {"available": sys.platform == "win32", "devices": listed}}, vjoy=probe_vjoy(), ffb=ffb)
        if ffb["result"] == "ok":
            say(f"FFB ok: {ffb['updates']} updates over {ffb['played_s']} s; the spring is stopped.")
            code = EXIT_OK
        elif ffb["result"] == "cancelled":
            say(f"FFB cancelled: {ffb['error']}")
            code = EXIT_CANCELLED
        else:
            say(f"FFB {ffb['result']}: {ffb['error']}" + (f" at {ffb['stage']}" if ffb.get("stage") else ""))
            code = EXIT_FFB_FAILED
    else:
        report, code = run_logging(args, only_vid)

    text = json.dumps(report, indent=2, default=str)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
        say(f"Report: {args.report}")
    else:
        print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
