"""DirectInput8 force feedback output to the Fanatec base, through ctypes.

Reworks PR #2's `dinput_ffb.py` (origin/cursor/reliability-usability-fanatec-6177)
against dinput.h:

- IDirectInputEffect slots are QueryInterface 0, AddRef 1, Release 2,
  Initialize 3, GetEffectGuid 4, GetParameters 5, SetParameters 6, Start 7,
  Stop 8, GetEffectStatus 9, Download 10, Unload 11, Escape 12. PR #2 called
  3 and 4. Every call here goes through `VTABLES` by method name.
- SetParameters takes DIEP_* flags, not DIEFF_* ones.
- DIEFF_SPHERICAL is 0x40 (PR #2 had 0x20, which is DIEFF_POLAR).
- GetModuleHandleW and CreateWindowExW return pointers, so their restype
  is c_void_p; the default int truncates them on 64-bit Windows.
- SetCooperativeLevel needs a top-level window; a message-only window
  (HWND_MESSAGE) is refused. A hidden STATIC window needs no message loop.
- The data format reports absolute axes (DIDF_ABSAXIS).
- The force axis is named by object ID (DIEFF_OBJECTIDS) with the
  actuator's dwType from EnumObjects, 0x01000002 on the DD2. The DD2 answers
  DIEFF_OBJECTOFFSETS with DIERR_INVALIDPARAM (PR #3, simrig runs).
- DIERR_NOTACQUIRED is 0x8007000C (PR #2 had 0x8007001C).
- Autocenter goes off before Acquire; the DD2 only accepts it then (PR #3).

The output owns the wheel exclusively. That is safe because the game now
opens vJoy instead. `FfbSink` is the interface the bridge uses, so tests
and dry runs swap in `LogSink`.

Plain ctypes integer types keep every struct the size it has on 64-bit
Windows, so the Linux tests check the layouts.
"""

from __future__ import annotations

import ctypes
import sys
import uuid
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from simsteer.core import ffb_model as fm

DWORD = ctypes.c_uint32
LONG = ctypes.c_int32
WORD = ctypes.c_uint16
WCHAR260 = ctypes.c_uint16 * 260

FANATEC_VID = 0x0EB7
VJOY_VID = 0x1234
VJOY_PID = 0xBEAD

DIRECTINPUT_VERSION = 0x0800
DI8DEVCLASS_GAMECTRL = 4
DIEDFL_ATTACHEDONLY = 0x00000001
DIEDFL_FORCEFEEDBACK = 0x00000100
DISCL_EXCLUSIVE = 0x00000001
DISCL_BACKGROUND = 0x00000008
DIDFT_ALL = 0x00000000
DIDFT_AXIS = 0x00000003
DIDFT_ANYINSTANCE = 0x00FFFF00
DIDFT_FFACTUATOR = 0x01000000
DIDFT_OPTIONAL = 0x80000000
DIDF_ABSAXIS = 0x00000001
DIDOI_ASPECTPOSITION = 0x00000100
DIEFT_ALL = 0x00000000
DIDC_FORCEFEEDBACK = 0x00000100
DIPH_DEVICE = 0
DIPROP_FFGAIN = 7
DIPROP_AUTOCENTER = 9
DIPROPAUTOCENTER_OFF = 0

DI_OK = 0x00000000
S_FALSE = 0x00000001
DIERR_INVALIDPARAM = 0x80070057
DIERR_OTHERAPPHASPRIO = 0x80070005
DIERR_NOTACQUIRED = 0x8007000C
DIERR_INPUTLOST = 0x8007001E
DIERR_UNPLUGGED = 0x80040209
DIERR_NOTEXCLUSIVEACQUIRED = 0x80040205
DIERR_INCOMPLETEEFFECT = 0x80040206
DIERR_EFFECTPLAYING = 0x80040208
DIERR_NOTDOWNLOADED = 0x80040203
E_POINTER = 0x80004003
DEVICE_LOST = frozenset({DIERR_INPUTLOST, DIERR_NOTACQUIRED, DIERR_UNPLUGGED, DIERR_NOTEXCLUSIVEACQUIRED})

HRESULT_NAMES = {
    DI_OK: "DI_OK", S_FALSE: "S_FALSE", 0x00000003: "DI_DOWNLOADSKIPPED",
    0x00000004: "DI_EFFECTRESTARTED", 0x00000008: "DI_TRUNCATED",
    0x0000000C: "DI_TRUNCATEDANDRESTARTED", DIERR_INVALIDPARAM: "DIERR_INVALIDPARAM",
    DIERR_OTHERAPPHASPRIO: "DIERR_OTHERAPPHASPRIO", DIERR_NOTACQUIRED: "DIERR_NOTACQUIRED",
    DIERR_INPUTLOST: "DIERR_INPUTLOST", DIERR_UNPLUGGED: "DIERR_UNPLUGGED",
    DIERR_NOTEXCLUSIVEACQUIRED: "DIERR_NOTEXCLUSIVEACQUIRED",
    DIERR_INCOMPLETEEFFECT: "DIERR_INCOMPLETEEFFECT", DIERR_EFFECTPLAYING: "DIERR_EFFECTPLAYING",
    DIERR_NOTDOWNLOADED: "DIERR_NOTDOWNLOADED", 0x80040201: "DIERR_DEVICEFULL",
    0x80004001: "DIERR_UNSUPPORTED", 0x80070015: "DIERR_NOTINITIALIZED",
    0x80040154: "DIERR_DEVICENOTREG", E_POINTER: "E_POINTER",
}


def hresult_name(hr: int) -> str:
    hr &= 0xFFFFFFFF
    return HRESULT_NAMES.get(hr, f"0x{hr:08X}")


def failed(hr: int) -> bool:
    return bool(hr & 0x80000000)


IID_IDirectInput8W = "BF798031-483A-4DA2-AA99-5D64ED369700"
_AXIS_GUIDS = (
    "A36D02E0-C9F3-11CF-BFC7-444553540000",  # X
    "A36D02E1-C9F3-11CF-BFC7-444553540000",  # Y
    "A36D02E2-C9F3-11CF-BFC7-444553540000",  # Z
    "A36D02F4-C9F3-11CF-BFC7-444553540000",  # Rx
    "A36D02F5-C9F3-11CF-BFC7-444553540000",  # Ry
    "A36D02E3-C9F3-11CF-BFC7-444553540000",  # Rz
    "A36D02E4-C9F3-11CF-BFC7-444553540000",  # Slider
    "A36D02E4-C9F3-11CF-BFC7-444553540000",  # Slider
)
EFFECT_GUIDS = {
    "ConstantForce": "13541C20-8E33-11D0-9AD0-00A0C9A06E35",
    "RampForce": "13541C21-8E33-11D0-9AD0-00A0C9A06E35",
    "Square": "13541C22-8E33-11D0-9AD0-00A0C9A06E35",
    "Sine": "13541C23-8E33-11D0-9AD0-00A0C9A06E35",
    "Triangle": "13541C24-8E33-11D0-9AD0-00A0C9A06E35",
    "SawtoothUp": "13541C25-8E33-11D0-9AD0-00A0C9A06E35",
    "SawtoothDown": "13541C26-8E33-11D0-9AD0-00A0C9A06E35",
    "Spring": "13541C27-8E33-11D0-9AD0-00A0C9A06E35",
    "Damper": "13541C28-8E33-11D0-9AD0-00A0C9A06E35",
    "Inertia": "13541C29-8E33-11D0-9AD0-00A0C9A06E35",
    "Friction": "13541C2A-8E33-11D0-9AD0-00A0C9A06E35",
}

# Method order as declared in dinput.h. Calls go through these by name, so
# every slot index is the position in this table.
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
        ("dwSize", DWORD), ("guidInstance", GUID), ("guidProduct", GUID), ("dwDevType", DWORD),
        ("tszInstanceName", WCHAR260), ("tszProductName", WCHAR260), ("guidFFDriver", GUID),
        ("wUsagePage", WORD), ("wUsage", WORD),
    ]


class DIDEVICEOBJECTINSTANCEW(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD), ("guidType", GUID), ("dwOfs", DWORD), ("dwType", DWORD), ("dwFlags", DWORD),
        ("tszName", WCHAR260), ("dwFFMaxForce", DWORD), ("dwFFForceResolution", DWORD),
        ("wCollectionNumber", WORD), ("wDesignatorIndex", WORD), ("wUsagePage", WORD), ("wUsage", WORD),
        ("dwDimension", DWORD), ("wExponent", WORD), ("wReportId", WORD),
    ]


class DIDEVCAPS(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD), ("dwFlags", DWORD), ("dwDevType", DWORD), ("dwAxes", DWORD),
        ("dwButtons", DWORD), ("dwPOVs", DWORD), ("dwFFSamplePeriod", DWORD),
        ("dwFFMinTimeResolution", DWORD), ("dwFirmwareRevision", DWORD),
        ("dwHardwareRevision", DWORD), ("dwFFDriverVersion", DWORD),
    ]


class DIEFFECTINFOW(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD), ("guid", GUID), ("dwEffType", DWORD), ("dwStaticParams", DWORD),
        ("dwDynamicParams", DWORD), ("tszName", WCHAR260),
    ]


class DIOBJECTDATAFORMAT(ctypes.Structure):
    _fields_ = [("pguid", ctypes.POINTER(GUID)), ("dwOfs", DWORD), ("dwType", DWORD), ("dwFlags", DWORD)]


class DIDATAFORMAT(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD), ("dwObjSize", DWORD), ("dwFlags", DWORD), ("dwDataSize", DWORD),
        ("dwNumObjs", DWORD), ("rgodf", ctypes.POINTER(DIOBJECTDATAFORMAT)),
    ]


class DIPROPHEADER(ctypes.Structure):
    _fields_ = [("dwSize", DWORD), ("dwHeaderSize", DWORD), ("dwObj", DWORD), ("dwHow", DWORD)]


class DIPROPDWORD(ctypes.Structure):
    _fields_ = [("diph", DIPROPHEADER), ("dwData", DWORD)]


class DIENVELOPE(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD), ("dwAttackLevel", DWORD), ("dwAttackTime", DWORD),
        ("dwFadeLevel", DWORD), ("dwFadeTime", DWORD),
    ]


class DICONSTANTFORCE(ctypes.Structure):
    _fields_ = [("lMagnitude", LONG)]


class DIRAMPFORCE(ctypes.Structure):
    _fields_ = [("lStart", LONG), ("lEnd", LONG)]


class DIPERIODIC(ctypes.Structure):
    _fields_ = [("dwMagnitude", DWORD), ("lOffset", LONG), ("dwPhase", DWORD), ("dwPeriod", DWORD)]


class DICONDITION(ctypes.Structure):
    _fields_ = [
        ("lOffset", LONG), ("lPositiveCoefficient", LONG), ("lNegativeCoefficient", LONG),
        ("dwPositiveSaturation", DWORD), ("dwNegativeSaturation", DWORD), ("lDeadBand", LONG),
    ]


class DIEFFECT(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD), ("dwFlags", DWORD), ("dwDuration", DWORD), ("dwSamplePeriod", DWORD),
        ("dwGain", DWORD), ("dwTriggerButton", DWORD), ("dwTriggerRepeatInterval", DWORD),
        ("cAxes", DWORD), ("rgdwAxes", ctypes.POINTER(DWORD)), ("rglDirection", ctypes.POINTER(LONG)),
        ("lpEnvelope", ctypes.c_void_p), ("cbTypeSpecificParams", DWORD),
        ("lpvTypeSpecificParams", ctypes.c_void_p), ("dwStartDelay", DWORD),
    ]


def dieffect_for(params: fm.DiParams, axis_id: int) -> tuple[DIEFFECT, list]:
    """The DIEFFECT for a call's parameter snapshot on one axis named by
    object ID. Returns the struct and the buffers it points into, which
    must stay alive until the call returns."""
    axes = (DWORD * 1)(axis_id)
    direction = (LONG * 1)(0)
    keep: list = [axes, direction]
    eff = DIEFFECT()
    eff.dwSize = ctypes.sizeof(DIEFFECT)
    eff.dwFlags = fm.DIEFF_OBJECTIDS | fm.DIEFF_CARTESIAN
    eff.dwDuration = params.duration
    eff.dwSamplePeriod = params.sample_period
    eff.dwGain = params.gain
    eff.dwTriggerButton = params.trigger_button
    eff.dwTriggerRepeatInterval = params.trigger_repeat
    eff.cAxes = 1
    eff.rgdwAxes = ctypes.cast(axes, ctypes.POINTER(DWORD))
    eff.rglDirection = ctypes.cast(direction, ctypes.POINTER(LONG))
    eff.dwStartDelay = params.start_delay
    if params.envelope is not None:
        e = params.envelope
        env = DIENVELOPE(ctypes.sizeof(DIENVELOPE), e.attack_level, e.attack_time, e.fade_level, e.fade_time)
        keep.append(env)
        eff.lpEnvelope = ctypes.cast(ctypes.pointer(env), ctypes.c_void_p)
    ts = params.type_specific
    block = None
    if isinstance(ts, fm.DiConstant):
        block = DICONSTANTFORCE(ts.magnitude)
    elif isinstance(ts, fm.DiRamp):
        block = DIRAMPFORCE(ts.start, ts.end)
    elif isinstance(ts, fm.DiPeriodic):
        block = DIPERIODIC(ts.magnitude, ts.offset, ts.phase, ts.period)
    elif isinstance(ts, fm.DiCondition):
        block = DICONDITION(ts.offset, ts.positive_coefficient, ts.negative_coefficient,
                            ts.positive_saturation, ts.negative_saturation, ts.dead_band)
    if block is not None:
        keep.append(block)
        eff.cbTypeSpecificParams = ctypes.sizeof(block)
        eff.lpvTypeSpecificParams = ctypes.cast(ctypes.pointer(block), ctypes.c_void_p)
    return eff, keep


def vid_pid(product_data1: int) -> tuple[int, int]:
    """guidProduct.Data1 is MAKELONG(VID, PID) for HID devices."""
    return product_data1 & 0xFFFF, (product_data1 >> 16) & 0xFFFF


def wide(buf) -> str:
    return bytes(buf).decode("utf-16-le", "replace").split("\0", 1)[0]


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    vid: int
    pid: int
    instance: str

    @property
    def is_vjoy(self) -> bool:
        return self.vid == VJOY_VID and self.pid == VJOY_PID

    @property
    def is_fanatec(self) -> bool:
        return self.vid == FANATEC_VID


def is_fanatec(dev: DeviceInfo) -> bool:
    return dev.is_fanatec


# --------------------------------------------------------------------------
# The interface the bridge uses


class FfbSink(Protocol):
    device: DeviceInfo | None

    def open(self) -> bool:
        """Find and acquire the wheel. False when none is attached."""

    def execute(self, call: fm.DiCall) -> int:
        """Run one call, return its HRESULT."""

    def close(self) -> None:
        ...


@dataclass
class LogSink:
    """Records calls instead of running them. For tests and dry runs."""

    present: bool = True
    calls: list[fm.DiCall] = field(default_factory=list)
    results: dict[str, int] = field(default_factory=dict)
    device: DeviceInfo | None = None
    opened: int = 0

    def open(self) -> bool:
        if not self.present:
            return False
        self.opened += 1
        self.device = DeviceInfo("log sink", 0, 0, "")
        return True

    def execute(self, call: fm.DiCall) -> int:
        self.calls.append(call)
        return self.results.get(call.op, DI_OK)

    def close(self) -> None:
        self.device = None

    def lines(self) -> list[str]:
        return [c.text() for c in self.calls]


# --------------------------------------------------------------------------
# DirectInput8 through ctypes (Windows only, loaded on first use)


class DIError(Exception):
    def __init__(self, stage: str, hr: int):
        self.stage = stage
        self.hr = hr & 0xFFFFFFFF
        super().__init__(f"{stage} failed: {hresult_name(self.hr)} (0x{self.hr:08X})")


def _com_call(ptr: int, interface: str, method: str, argtypes: Sequence, *args, restype=ctypes.c_long) -> int:
    vtbl = ctypes.cast(ctypes.c_void_p(ptr), ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    result = proto(vtbl[vtable_slot(interface, method)])(ptr, *args)
    return result & 0xFFFFFFFF if restype is ctypes.c_long else result


def _check(stage: str, hr: int) -> int:
    if failed(hr):
        raise DIError(stage, hr)
    return hr


def joystick_data_format() -> tuple[DIDATAFORMAT, list]:
    """Eight optional absolute axes, enough to acquire a wheel. Returns the
    format and the objects it points at, which must outlive its use."""
    guids = [GUID.parse(g) for g in _AXIS_GUIDS]
    rows = [(ctypes.pointer(guids[i]), i * 4, DIDFT_OPTIONAL | DIDFT_AXIS | DIDFT_ANYINSTANCE, DIDOI_ASPECTPOSITION)
            for i in range(len(guids))]
    objects = (DIOBJECTDATAFORMAT * len(rows))(*rows)
    fmt = DIDATAFORMAT(
        ctypes.sizeof(DIDATAFORMAT), ctypes.sizeof(DIOBJECTDATAFORMAT), DIDF_ABSAXIS,
        len(rows) * 4, len(rows), ctypes.cast(objects, ctypes.POINTER(DIOBJECTDATAFORMAT)),
    )
    return fmt, [guids, objects]


class DirectInput:
    """IDirectInput8W plus the hidden window SetCooperativeLevel needs.
    Create, use and close it on one thread."""

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
        _check("DirectInput8Create", create(self.hinstance, DIRECTINPUT_VERSION,
                                            byref(GUID.parse(IID_IDirectInput8W)), byref(out), None) & 0xFFFFFFFF)
        self.ptr = out.value
        self.hwnd = user32.CreateWindowExW(0, "STATIC", "SimSteer FFB", 0, 0, 0, 0, 0, None, None, self.hinstance, None)
        if not self.hwnd:
            error = ctypes.get_last_error()
            self.close()
            raise OSError(f"CreateWindowExW failed: error {error}")

    def ff_devices(self) -> list[DIDEVICEINSTANCEW]:
        found: list[DIDEVICEINSTANCEW] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(DIDEVICEINSTANCEW), ctypes.c_void_p)

        def on_device(inst, _ref):
            copy = DIDEVICEINSTANCEW()
            ctypes.memmove(ctypes.byref(copy), inst, ctypes.sizeof(copy))
            found.append(copy)
            return 1

        callback = callback_type(on_device)
        _check("EnumDevices", _com_call(
            self.ptr, "IDirectInput8W", "EnumDevices", [DWORD, callback_type, ctypes.c_void_p, DWORD],
            DI8DEVCLASS_GAMECTRL, callback, None, DIEDFL_ATTACHEDONLY | DIEDFL_FORCEFEEDBACK,
        ))
        return found

    def create_device(self, inst: DIDEVICEINSTANCEW) -> int:
        out = ctypes.c_void_p()
        _check("CreateDevice", _com_call(
            self.ptr, "IDirectInput8W", "CreateDevice",
            [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            ctypes.byref(inst.guidInstance), ctypes.byref(out), None,
        ))
        return out.value

    def close(self) -> None:
        if self.ptr:
            _com_call(self.ptr, "IDirectInput8W", "Release", [], restype=DWORD)
            self.ptr = None
        if self.hwnd:
            self._user32.DestroyWindow(self.hwnd)
            self.hwnd = None


def info_of(inst: DIDEVICEINSTANCEW) -> DeviceInfo:
    vid, pid = vid_pid(inst.guidProduct.Data1)
    return DeviceInfo(wide(inst.tszProductName), vid, pid, str(inst.guidInstance))


def _dev(ptr: int, method: str, argtypes: Sequence, *args, restype=ctypes.c_long) -> int:
    return _com_call(ptr, "IDirectInputDevice8W", method, argtypes, *args, restype=restype)


def _objects(ptr: int) -> list[DIDEVICEOBJECTINSTANCEW]:
    found: list[DIDEVICEOBJECTINSTANCEW] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(DIDEVICEOBJECTINSTANCEW), ctypes.c_void_p)

    def on_object(obj, _ref):
        copy = DIDEVICEOBJECTINSTANCEW()
        ctypes.memmove(ctypes.byref(copy), obj, ctypes.sizeof(copy))
        found.append(copy)
        return 1

    callback = callback_type(on_object)
    _check("EnumObjects", _dev(ptr, "EnumObjects", [callback_type, ctypes.c_void_p, DWORD], callback, None, DIDFT_ALL))
    return found


def _effects(ptr: int) -> list[str]:
    found: list[str] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(DIEFFECTINFOW), ctypes.c_void_p)

    def on_effect(info, _ref):
        found.append(wide(info.contents.tszName))
        return 1

    callback = callback_type(on_effect)
    _check("EnumEffects", _dev(ptr, "EnumEffects", [callback_type, ctypes.c_void_p, DWORD], callback, None, DIEFT_ALL))
    return found


def _actuators(ptr: int) -> list[DIDEVICEOBJECTINSTANCEW]:
    return [o for o in _objects(ptr) if o.dwType & DIDFT_FFACTUATOR and o.dwType & DIDFT_AXIS]


def attached_ff_devices() -> list[DeviceInfo]:
    """Names and IDs of attached force feedback controllers. Enumeration
    only; no device object is created."""
    if sys.platform != "win32":
        return []
    di = DirectInput()
    try:
        return [info_of(i) for i in di.ff_devices()]
    finally:
        di.close()


def list_ff_devices() -> list[dict]:
    """Every attached force feedback controller and what it can do. Read
    only: no cooperative level, no Acquire, no effect, so nothing is sent
    to the device's motor."""
    if sys.platform != "win32":
        return []
    di = DirectInput()
    out = []
    try:
        for inst in di.ff_devices():
            info = info_of(inst)
            entry: dict = {"name": info.name, "vid": f"0x{info.vid:04X}", "pid": f"0x{info.pid:04X}",
                           "instance": info.instance, "vjoy": info.is_vjoy, "fanatec": info.is_fanatec}
            ptr = None
            try:
                ptr = di.create_device(inst)
                caps = DIDEVCAPS()
                caps.dwSize = ctypes.sizeof(caps)
                _check("GetCapabilities", _dev(ptr, "GetCapabilities", [ctypes.POINTER(DIDEVCAPS)], ctypes.byref(caps)))
                entry.update(axes=caps.dwAxes, buttons=caps.dwButtons,
                             ff_sample_period=caps.dwFFSamplePeriod,
                             ff_min_time_resolution=caps.dwFFMinTimeResolution,
                             ff_driver_version=caps.dwFFDriverVersion)
                entry["actuators"] = [{"name": wide(o.tszName), "id": f"0x{o.dwType:08X}",
                                       "max_force": o.dwFFMaxForce} for o in _actuators(ptr)]
                entry["effects"] = _effects(ptr)
            except DIError as e:
                entry["error"] = str(e)
            finally:
                if ptr:
                    _dev(ptr, "Release", [], restype=DWORD)
            out.append(entry)
    finally:
        di.close()
    return out


class DirectInputSink:
    """Exclusive force feedback on the first device `match` accepts.

    vJoy is never a target unless `allow_vjoy`: the bridge reads the game's
    effects from vJoy, so writing back to it would loop. Open, execute and
    close on one thread."""

    def __init__(self, match: Callable[[DeviceInfo], bool] = is_fanatec, allow_vjoy: bool = False) -> None:
        self.match = match
        self.allow_vjoy = allow_vjoy
        self.device: DeviceInfo | None = None
        self.axis_id = 0
        self._di: DirectInput | None = None
        self._ptr: int | None = None
        self._format = None
        self._effects: dict[int, int] = {}
        self.last_error: str | None = None

    def open(self) -> bool:
        self.close()
        try:
            self._di = DirectInput()
            inst = next((i for i in self._di.ff_devices()
                         if self._wanted(info_of(i))), None)
            if inst is None:
                self.last_error = "no matching force feedback device"
                return False
            self._ptr = ptr = self._di.create_device(inst)
            self._format = joystick_data_format()
            _check("SetDataFormat", _dev(ptr, "SetDataFormat", [ctypes.POINTER(DIDATAFORMAT)], ctypes.byref(self._format[0])))
            _check("SetCooperativeLevel", _dev(ptr, "SetCooperativeLevel", [ctypes.c_void_p, DWORD],
                                               self._di.hwnd, DISCL_EXCLUSIVE | DISCL_BACKGROUND))
            before = self._set_dword(DIPROP_AUTOCENTER, DIPROPAUTOCENTER_OFF)
            _check("Acquire", _dev(ptr, "Acquire", []))
            after = self._set_dword(DIPROP_AUTOCENTER, DIPROPAUTOCENTER_OFF)
            if failed(before) and failed(after):
                raise DIError("SetProperty(AUTOCENTER)", after)
            actuators = _actuators(ptr)
            if not actuators:
                raise DIError("EnumObjects(FFACTUATOR)", DIERR_INVALIDPARAM)
            self.axis_id = actuators[0].dwType
            self.device = info_of(inst)
            self.last_error = None
            return True
        except (DIError, OSError) as e:
            self.last_error = str(e)
            self.close()
            return False

    def _wanted(self, dev: DeviceInfo) -> bool:
        if dev.is_vjoy and not self.allow_vjoy:
            return False
        return self.match(dev)

    def _set_dword(self, prop: int, value: int) -> int:
        data = DIPROPDWORD(DIPROPHEADER(ctypes.sizeof(DIPROPDWORD), ctypes.sizeof(DIPROPHEADER), 0, DIPH_DEVICE), value)
        # Predefined DIPROP_* "GUIDs" are small integers passed as the pointer.
        return _dev(self._ptr, "SetProperty", [ctypes.c_void_p, ctypes.POINTER(DIPROPDWORD)], prop, ctypes.byref(data))

    def get_gain(self) -> int | None:
        data = DIPROPDWORD(DIPROPHEADER(ctypes.sizeof(DIPROPDWORD), ctypes.sizeof(DIPROPHEADER), 0, DIPH_DEVICE), 0)
        hr = _dev(self._ptr, "GetProperty", [ctypes.c_void_p, ctypes.POINTER(DIPROPDWORD)], DIPROP_FFGAIN, ctypes.byref(data))
        return None if failed(hr) else data.dwData

    def _effect(self, ebi: int, method: str, argtypes: Sequence, *args, restype=ctypes.c_long) -> int:
        ptr = self._effects.get(ebi)
        if ptr is None:
            return DIERR_NOTDOWNLOADED
        return _com_call(ptr, "IDirectInputEffect", method, argtypes, *args, restype=restype)

    def execute(self, call: fm.DiCall) -> int:
        if self._ptr is None:
            return DIERR_NOTACQUIRED
        op = call.op
        if op == "SetParameters":
            eff, _keep = dieffect_for(call.params or fm.DiParams(), self.axis_id)
            return self._effect(call.ebi, "SetParameters", [ctypes.POINTER(DIEFFECT), DWORD], ctypes.byref(eff), call.flags)
        if op == "Start":
            return self._effect(call.ebi, "Start", [DWORD, DWORD], call.iterations, call.flags)
        if op == "Stop":
            return self._effect(call.ebi, "Stop", [])
        if op == "CreateEffect":
            self._release(call.ebi)
            out = ctypes.c_void_p()
            # No DIEFFECT: the effect starts empty and SetParameters fills it,
            # as the game's packets arrive.
            hr = _dev(self._ptr, "CreateEffect",
                      [ctypes.POINTER(GUID), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
                      ctypes.byref(GUID.parse(EFFECT_GUIDS[call.kind])), None, ctypes.byref(out), None)
            if not failed(hr):
                self._effects[call.ebi] = out.value
            return hr
        if op == "Release":
            return self._release(call.ebi)
        if op == "Command":
            hr = _dev(self._ptr, "SendForceFeedbackCommand", [DWORD], call.flags)
            if call.flags == fm.DISFFC_RESET and not failed(hr):
                for ebi in list(self._effects):
                    self._release(ebi)
            return hr
        if op == "SetGain":
            return self._set_dword(DIPROP_FFGAIN, call.value)
        return DIERR_INVALIDPARAM

    def _release(self, ebi: int) -> int:
        ptr = self._effects.pop(ebi, None)
        if ptr is None:
            return DI_OK
        hr = _com_call(ptr, "IDirectInputEffect", "Unload", [])
        _com_call(ptr, "IDirectInputEffect", "Release", [], restype=DWORD)
        return DI_OK if hr in DEVICE_LOST else hr

    def close(self) -> None:
        if self._ptr is not None:
            try:
                _dev(self._ptr, "SendForceFeedbackCommand", [DWORD], fm.DISFFC_STOPALL)
                for ebi in list(self._effects):
                    self._release(ebi)
                _dev(self._ptr, "Unacquire", [])
            finally:
                _dev(self._ptr, "Release", [], restype=DWORD)
                self._ptr = None
        self._effects.clear()
        self.device = None
        if self._di is not None:
            self._di.close()
            self._di = None
