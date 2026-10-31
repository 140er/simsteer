"""DirectInput8 stand-in: answers SimSteer's COM calls on simulated devices.

`dinput_out._com_call(ptr, interface, method, argtypes, *args)` sends every
call here once the backend is installed. Arguments arrive exactly as
SimSteer passes them to the real vtable (`ctypes.byref` buffers, callback
objects, plain integers), so the structs are read and filled in place and
SimSteer's own enumeration callbacks run.

The rules a real device enforces are enforced here too: data format before
Acquire, a window for the cooperative level, exclusive access for force
feedback, one exclusive owner per physical device, DIERR_INPUTLOST once
and then DIERR_NOTACQUIRED after an unplug, dead device objects after a
replug. DD2 quirks seen on hardware: autocentre is accepted only before
Acquire, effects must name the axis by object ID (DIEFF_OBJECTOFFSETS is
DIERR_INVALIDPARAM).
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass, replace

from simsteer.core import ffb_model as fm
from simsteer.runtime.ffb import dinput_out as di
from simsteer.testing.fanatec import device as dv
from simsteer.testing.fanatec.profiles import (
    DI8DEVTYPE_DRIVING_HID, Obj, effect_guid,
)

HWND = 0x5151
DIERR_NOTFOUND = 0x80070002
DIERR_DEVICENOTREG = 0x80040154
DIERR_NOINTERFACE = 0x80004002
DI_NOTATTACHED = 0x00000001
DIPROP_RANGE = 4
DIPH_BYOFFSET, DIPH_BYID = 1, 2
DISCL_FOREGROUND = 0x00000004
DIDFT_BUTTON = 0x0000000C
DIDFT_POV = 0x00000010
DIDC_ATTACHED, DIDC_POLLEDDEVICE = 0x1, 0x2
PREFIX = {"IDirectInput8W": "root", "IDirectInputDevice8W": "device", "IDirectInputEffect": "effect"}
TS_SIZES = {fm.DiConstant: ctypes.sizeof(di.DICONSTANTFORCE), fm.DiRamp: ctypes.sizeof(di.DIRAMPFORCE),
            fm.DiPeriodic: ctypes.sizeof(di.DIPERIODIC), fm.DiCondition: ctypes.sizeof(di.DICONDITION)}


def deref(arg):
    """The ctypes object behind `byref(x)`, `pointer(x)` or `x`."""
    if hasattr(arg, "_obj"):
        return arg._obj
    if isinstance(arg, ctypes._Pointer):
        return arg.contents
    return arg


def view(arg, struct):
    """`arg`'s buffer read as `struct`."""
    return struct.from_address(ctypes.addressof(deref(arg)))


def fill_wide(buf, text: str) -> None:
    encoded = text.encode("utf-16-le")[: (len(buf) - 1) * 2]
    ctypes.memset(buf, 0, ctypes.sizeof(buf))
    ctypes.memmove(buf, encoded, len(encoded))


def ts_kind(kind: str):
    if kind == "ConstantForce":
        return fm.DiConstant
    if kind == "RampForce":
        return fm.DiRamp
    if kind in fm.PERIODIC_KINDS:
        return fm.DiPeriodic
    if kind in fm.CONDITION_KINDS:
        return fm.DiCondition
    return None


@dataclass
class CallRecord:
    t: float
    thread: int
    interface: str
    method: str
    handle: int
    hr: int | None


@dataclass
class Row:
    offset: int
    kind: str  # axis / button / pov
    obj: Obj | None  # None: optional and unmatched


class Root:
    def __init__(self, handle: int) -> None:
        self.handle, self.thread, self.released = handle, threading.get_ident(), False


class DeviceHandle:
    def __init__(self, handle: int, root: Root, dev: dv.SimDevice, index: int) -> None:
        self.handle, self.root, self.dev, self.index = handle, root, dev, index
        self.generation = dev.generation
        self.rows: list[Row] | None = None
        self.data_size = 0
        self.coop = 0
        self.acquired = self.released = False
        self.ranges: dict[int, tuple[int, int]] = {}

    @property
    def profile(self):
        return self.dev.instances[self.index]

    @property
    def objects(self) -> list[Obj]:
        return self.dev.objects[self.index]

    @property
    def dead(self) -> bool:
        return self.released or not self.dev.plugged or self.generation != self.dev.generation

    @property
    def exclusive_acquired(self) -> bool:
        return self.acquired and not self.dead and self.dev.exclusive is self

    def lost(self, t: float) -> int:
        """The HRESULT for a read on a dead device: INPUTLOST once, then NOTACQUIRED."""
        if self.acquired:
            self.acquired = False
            if self.dev.exclusive is self:
                self.dev.exclusive = None
            return di.DIERR_INPUTLOST
        return di.DIERR_NOTACQUIRED


class EffectHandle:
    def __init__(self, handle: int, owner: DeviceHandle, effect: dv.SimEffect) -> None:
        self.handle, self.owner, self.effect = handle, owner, effect


class User32:
    def __init__(self, backend: DirectInputBackend) -> None:
        self.backend = backend

    def DestroyWindow(self, hwnd) -> int:  # noqa: N802 - Win32 name
        self.backend.windows.discard(hwnd)
        return 1


class DirectInputBackend:
    def __init__(self, sim) -> None:
        self.sim = sim
        self.lock = sim.lock
        self.handles: dict[int, object] = {}
        self.calls: list[CallRecord] = []
        self.thread_violations: list[str] = []
        self.windows: set[int] = set()
        self._next = 0x1000
        backend = self

        class SimDirectInput(di.DirectInput):
            """`dinput_out.DirectInput` minus dinput8.dll and the window."""

            def __init__(self) -> None:  # noqa: D401 - replaces the Win32 setup only
                self.hinstance = None
                self._user32 = User32(backend)
                self.ptr = backend._new(Root)
                self.hwnd = HWND
                backend.windows.add(HWND)

        self.DirectInput = SimDirectInput

    # ---- handles

    def _new(self, cls, *args):
        with self.lock:
            handle = self._next
            self._next += 0x10
            obj = cls(handle, *args)
            self.handles[handle] = obj
            return handle

    def live_handles(self) -> list[str]:
        out = []
        for h, obj in self.handles.items():
            if isinstance(obj, Root) and not obj.released:
                out.append(f"IDirectInput8W 0x{h:X}")
            elif isinstance(obj, DeviceHandle) and not obj.released:
                out.append(f"IDirectInputDevice8W 0x{h:X} ({obj.profile.name})")
            elif isinstance(obj, EffectHandle) and not obj.effect.released:
                out.append(f"IDirectInputEffect 0x{h:X} ({obj.effect.kind})")
        return out

    def _root_of(self, obj) -> Root | None:
        if isinstance(obj, Root):
            return obj
        if isinstance(obj, DeviceHandle):
            return obj.root
        if isinstance(obj, EffectHandle):
            return obj.owner.root
        return None

    # ---- dispatch

    def com_call(self, ptr, interface, method, argtypes, *args, restype=ctypes.c_long):
        with self.lock:
            now = self.sim.clock()
            for dev in self.sim.devices:
                dev._advance(now)  # plug events from a playing track
            obj = self.handles.get(ptr)
            root = self._root_of(obj)
            if root is not None and root.thread != threading.get_ident():
                self.thread_violations.append(f"{interface}.{method} on thread {threading.get_ident()}, "
                                              f"created on {root.thread}")
            fn = getattr(self, f"_{PREFIX[interface]}_{method}", None)
            hr = None
            if obj is None:
                hr = di.E_POINTER
            else:
                dev = getattr(obj, "dev", None) or getattr(getattr(obj, "owner", None), "dev", None)
                if dev is not None:
                    hr = dev.take_fault(method)
                if hr is None:
                    hr = dv.DIERR_UNSUPPORTED if fn is None else fn(obj, now, *args)
            self.calls.append(CallRecord(now, threading.get_ident(), interface, method, ptr or 0, hr))
            return hr & 0xFFFFFFFF

    def _ffb(self, dev: dv.SimDevice, t: float, op: str, hr: int, **detail) -> int:
        event = {"t": t, "ev": "call", "op": op, "hr": di.hresult_name(hr)}
        event.update(detail)
        dev.ffb.add(event)
        return hr

    # ---- IDirectInput8W

    def _root_Release(self, root: Root, t):
        root.released = True
        return 0

    def _attached(self):
        for dev in self.sim.devices:
            if dev.plugged:
                for index, profile in enumerate(dev.instances):
                    yield dev, index, profile

    def _root_EnumDevices(self, root, t, dev_class, callback, ref, flags):
        if dev_class not in (0, di.DI8DEVCLASS_GAMECTRL):
            return di.DI_OK
        for dev, index, profile in list(self._attached()):
            if flags & di.DIEDFL_FORCEFEEDBACK and not profile.ff:
                continue
            inst = di.DIDEVICEINSTANCEW()
            inst.dwSize = ctypes.sizeof(inst)
            inst.guidInstance = di.GUID.parse(profile.instance)
            inst.guidProduct = di.GUID.parse(f"{profile.pid:04X}{profile.vid:04X}-0000-0000-0000-504944564944")
            inst.dwDevType = DI8DEVTYPE_DRIVING_HID
            fill_wide(inst.tszInstanceName, profile.name)
            fill_wide(inst.tszProductName, profile.name)
            if profile.ff:
                inst.guidFFDriver = di.GUID.parse(profile.instance)
            if not callback(ctypes.pointer(inst), ref):
                break  # DIENUM_STOP
        return di.DI_OK

    def _root_GetDeviceStatus(self, root, t, guid):
        want = str(deref(guid))
        return di.DI_OK if any(p.instance == want for _, _, p in self._attached()) else DI_NOTATTACHED

    def _root_CreateDevice(self, root, t, guid, out, outer):
        want = str(deref(guid))
        for dev, index, profile in self._attached():
            if profile.instance == want:
                deref(out).value = self._new(DeviceHandle, root, dev, index)
                return di.DI_OK
        return DIERR_DEVICENOTREG

    # ---- IDirectInputDevice8W

    def _device_Release(self, h: DeviceHandle, t):
        if not h.released:
            if h.acquired:
                self._device_Unacquire(h, t)
            h.released = True
        return 0

    def _device_GetCapabilities(self, h, t, caps):
        c = deref(caps)
        if c.dwSize != ctypes.sizeof(di.DIDEVCAPS):
            return di.DIERR_INVALIDPARAM
        p = h.profile
        c.dwFlags = (DIDC_ATTACHED if not h.dead else 0) | DIDC_POLLEDDEVICE | (di.DIDC_FORCEFEEDBACK if p.ff else 0)
        c.dwDevType = DI8DEVTYPE_DRIVING_HID
        c.dwAxes, c.dwButtons, c.dwPOVs = len(p.axes), p.buttons, p.povs
        c.dwFFSamplePeriod = c.dwFFMinTimeResolution = p.ff_period_us if p.ff else 0
        c.dwFFDriverVersion = p.ff_driver_version
        return di.DI_OK

    def _object_info(self, h: DeviceHandle, obj: Obj, out) -> None:
        o = out
        o.guidType = di.GUID.parse(obj.guid)
        o.dwOfs = self._native_offset(h, obj)
        o.dwType, o.dwFlags = obj.dw_type, obj.flags
        fill_wide(o.tszName, obj.name)
        o.dwFFMaxForce = h.profile.max_force if obj.actuator else 0
        o.dwFFForceResolution = 256 if obj.actuator else 0
        o.wUsagePage = 0x01 if obj.kind != "button" else 0x09

    @staticmethod
    def _native_offset(h: DeviceHandle, obj: Obj) -> int:
        if obj.kind == "axis":
            return obj.number * 4
        if obj.kind == "pov":
            return 32 + obj.number * 4
        return 48 + obj.number

    def _device_EnumObjects(self, h, t, callback, ref, flags):
        types = flags & 0x1F
        for obj in h.objects:
            if types and not obj.dw_type & types:
                continue
            if flags & di.DIDFT_FFACTUATOR and not obj.actuator:
                continue
            info = di.DIDEVICEOBJECTINSTANCEW()
            info.dwSize = ctypes.sizeof(info)
            self._object_info(h, obj, info)
            if not callback(ctypes.pointer(info), ref):
                break
        return di.DI_OK

    def _row_at(self, h: DeviceHandle, offset: int) -> Row | None:
        return next((r for r in h.rows or () if r.offset == offset and r.obj is not None), None)

    def _device_GetObjectInfo(self, h, t, out, obj_id, how):
        info = deref(out)
        if info.dwSize != ctypes.sizeof(di.DIDEVICEOBJECTINSTANCEW):
            return di.DIERR_INVALIDPARAM
        if how == DIPH_BYOFFSET:
            row = self._row_at(h, obj_id)
            obj = row.obj if row else None
        elif how == DIPH_BYID:
            obj = next((o for o in h.objects if o.dw_type & 0x00FFFFFF == obj_id & 0x00FFFFFF), None)
        else:
            return di.DIERR_INVALIDPARAM
        if obj is None:
            return DIERR_NOTFOUND
        self._object_info(h, obj, info)
        return di.DI_OK

    def _device_GetProperty(self, h, t, prop, data):
        header = view(data, di.DIPROPHEADER)
        if prop == DIPROP_RANGE:
            row = self._row_at(h, header.dwObj) if header.dwHow == DIPH_BYOFFSET else None
            if row is None or row.kind != "axis":
                return DIERR_NOTFOUND
            lo, hi = h.ranges.get(row.offset, (0, 65535))
            values = (di.LONG * 2).from_address(ctypes.addressof(header) + ctypes.sizeof(di.DIPROPHEADER))
            values[0], values[1] = lo, hi
            return di.DI_OK
        if prop in (di.DIPROP_FFGAIN, di.DIPROP_AUTOCENTER):
            view(data, di.DIPROPDWORD).dwData = h.dev.gain if prop == di.DIPROP_FFGAIN else h.dev.autocenter
            return di.DI_OK
        return dv.DIERR_UNSUPPORTED

    def _device_SetProperty(self, h, t, prop, data):
        if prop == DIPROP_RANGE:
            header = view(data, di.DIPROPHEADER)
            if h.acquired:
                return dv.DIERR_ACQUIRED
            values = (di.LONG * 2).from_address(ctypes.addressof(header) + ctypes.sizeof(di.DIPROPHEADER))
            axes = [r for r in h.rows or () if r.kind == "axis" and r.obj is not None]
            if header.dwHow == di.DIPH_DEVICE:
                offsets = [r.offset for r in axes]
            elif header.dwHow == DIPH_BYOFFSET:
                offsets = [r.offset for r in axes if r.offset == header.dwObj]
            elif header.dwHow == DIPH_BYID:
                offsets = [r.offset for r in axes if r.obj.dw_type & 0x00FFFFFF == header.dwObj & 0x00FFFFFF]
            else:
                return di.DIERR_INVALIDPARAM
            if not offsets:
                return DIERR_NOTFOUND
            for offset in offsets:
                h.ranges[offset] = (values[0], values[1])
            return di.DI_OK
        value = view(data, di.DIPROPDWORD).dwData
        name = {di.DIPROP_FFGAIN: "FFGAIN", di.DIPROP_AUTOCENTER: "AUTOCENTER"}.get(prop, str(prop))
        if h.dead:
            hr = di.DIERR_INPUTLOST
        elif prop == di.DIPROP_FFGAIN:
            hr = h.dev.set_gain(t, value)
        elif prop == di.DIPROP_AUTOCENTER:
            # The DD2 takes autocentre only while not acquired.
            hr = dv.DIERR_ACQUIRED if h.acquired else di.DI_OK
            if hr == di.DI_OK:
                h.dev.autocenter = value
        else:
            hr = dv.DIERR_UNSUPPORTED
        return self._ffb(h.dev, t, "SetProperty", hr, detail=f"{name}={value}")

    def _device_Acquire(self, h, t):
        if h.dead:
            return di.DIERR_UNPLUGGED
        if h.rows is None:
            return di.DIERR_INVALIDPARAM
        if h.acquired:
            return di.S_FALSE
        exclusive = bool(h.coop & di.DISCL_EXCLUSIVE)
        if exclusive:
            if h.dev.exclusive not in (None, h):
                return self._ffb(h.dev, t, "Acquire", di.DIERR_OTHERAPPHASPRIO, detail="exclusive")
            h.dev.exclusive = h
        h.acquired = True
        if exclusive:
            self._ffb(h.dev, t, "Acquire", di.DI_OK, detail="exclusive")
        return di.DI_OK

    def _device_Unacquire(self, h, t):
        if not h.acquired:
            return dv.DI_NOEFFECT
        h.acquired = False
        if h.dev.exclusive is h:
            h.dev.unacquired(t, h)
        return di.DI_OK

    def _device_Poll(self, h, t):
        if h.dead:
            return h.lost(t)
        return di.DI_OK if h.acquired else di.DIERR_NOTACQUIRED

    def _device_GetDeviceState(self, h, t, size, buf):
        if h.dead:
            return h.lost(t)
        if not h.acquired:
            return di.DIERR_NOTACQUIRED
        if size != h.data_size:
            return di.DIERR_INVALIDPARAM
        base = ctypes.addressof(deref(buf))
        ctypes.memset(base, 0, size)
        s = h.dev.state(t)
        for row in h.rows:
            if row.kind == "axis":
                if row.obj is None:
                    continue
                lo, hi = h.ranges.get(row.offset, (0, 65535))
                v = h.dev.axis_value(h.index, row.obj, s)
                di.LONG.from_address(base + row.offset).value = lo + round(v * (hi - lo))
            elif row.kind == "pov":
                pov = 0xFFFFFFFF if row.obj is None else h.dev.pov(h.index, s)
                di.DWORD.from_address(base + row.offset).value = pov
            elif row.obj is not None and h.dev.button_down(h.index, row.obj.number, s):
                ctypes.c_uint8.from_address(base + row.offset).value = 0x80
        return di.DI_OK

    def _device_SetDataFormat(self, h, t, fmt):
        if h.acquired:
            return dv.DIERR_ACQUIRED
        f = deref(fmt)
        if (f.dwSize != ctypes.sizeof(di.DIDATAFORMAT) or f.dwObjSize != ctypes.sizeof(di.DIOBJECTDATAFORMAT)
                or f.dwDataSize % 4):
            return di.DIERR_INVALIDPARAM
        used, rows = set(), []
        for i in range(f.dwNumObjs):
            r = f.rgodf[i]
            kind = "axis" if r.dwType & di.DIDFT_AXIS else "pov" if r.dwType & DIDFT_POV else \
                "button" if r.dwType & DIDFT_BUTTON else None
            size = 1 if kind == "button" else 4
            if kind is None or r.dwOfs + size > f.dwDataSize:
                return di.DIERR_INVALIDPARAM
            guid = str(r.pguid.contents) if r.pguid else None
            any_instance = r.dwType & di.DIDFT_ANYINSTANCE == di.DIDFT_ANYINSTANCE
            number = (r.dwType >> 8) & 0xFFFF
            match = next((o for o in h.objects if o.kind == kind and id(o) not in used
                          and (guid is None or o.guid == guid) and (any_instance or o.number == number)), None)
            if match is None and not r.dwType & di.DIDFT_OPTIONAL:
                return di.DIERR_INVALIDPARAM
            if match is not None:
                used.add(id(match))
            rows.append(Row(r.dwOfs, kind, match))
        h.rows, h.data_size = rows, f.dwDataSize
        return di.DI_OK

    def _device_SetCooperativeLevel(self, h, t, hwnd, flags):
        if h.acquired:
            return dv.DIERR_ACQUIRED
        excl = bool(flags & di.DISCL_EXCLUSIVE) + bool(flags & 0x2)  # DISCL_NONEXCLUSIVE
        where = bool(flags & DISCL_FOREGROUND) + bool(flags & di.DISCL_BACKGROUND)
        if excl != 1 or where != 1 or hwnd not in self.windows:
            return di.DIERR_INVALIDPARAM
        h.coop = flags
        return di.DI_OK

    def _device_CreateEffect(self, h, t, guid, params, out, outer):
        want = str(deref(guid))
        kind = next((k for k, _ in h.profile.effects if effect_guid(k) == want), None)
        if h.dead:
            hr = di.DIERR_INPUTLOST
        elif kind is None:
            hr = DIERR_DEVICENOTREG
        else:
            effect = dv.SimEffect(h.dev, 0, h, kind)
            handle = self._new(EffectHandle, h, effect)
            effect.handle = handle
            h.dev.effects.append(effect)
            deref(out).value = handle
            hr = di.DI_OK
            if params:
                hr = self._set_parameters(self.handles[handle], t, params, 0x3FF)
        return self._ffb(h.dev, t, "CreateEffect", hr, detail=kind or want)

    def _device_EnumEffects(self, h, t, callback, ref, eff_type):
        for kind, name in h.profile.effects:
            info = di.DIEFFECTINFOW()
            info.dwSize = ctypes.sizeof(info)
            info.guid = di.GUID.parse(effect_guid(kind))
            fill_wide(info.tszName, name)
            if not callback(ctypes.pointer(info), ref):
                break
        return di.DI_OK

    def _device_GetForceFeedbackState(self, h, t, out):
        if h.dead:
            return di.DIERR_INPUTLOST
        if not h.exclusive_acquired:
            return di.DIERR_NOTEXCLUSIVEACQUIRED
        deref(out).value = h.dev.ff_state(t)
        return di.DI_OK

    def _device_SendForceFeedbackCommand(self, h, t, flags):
        if h.dead:
            hr = di.DIERR_INPUTLOST
        elif not h.exclusive_acquired:
            hr = di.DIERR_NOTEXCLUSIVEACQUIRED
        else:
            hr = h.dev.command(t, flags)
        return self._ffb(h.dev, t, "SendForceFeedbackCommand", hr, detail=fm.DISFFC_NAMES.get(flags, hex(flags)))

    # ---- IDirectInputEffect

    def _effect_gate(self, e: EffectHandle) -> int | None:
        if e.effect.released:
            return dv.DIERR_NOTINITIALIZED
        if e.owner.dead:
            return di.DIERR_INPUTLOST
        return None

    def _set_parameters(self, e: EffectHandle, t, peff, flags) -> int:
        eff = deref(peff)
        if eff.dwSize != ctypes.sizeof(di.DIEFFECT):
            return di.DIERR_INVALIDPARAM
        p = e.effect.params
        fields = {}
        if flags & fm.DIEP_AXES or flags & fm.DIEP_DIRECTION:
            if eff.dwFlags & fm.DIEFF_OBJECTOFFSETS or not eff.dwFlags & fm.DIEFF_OBJECTIDS:
                return di.DIERR_INVALIDPARAM  # the DD2 refuses object offsets
            actuators = {o.dw_type for o in e.owner.objects if o.actuator}
            if eff.cAxes != 1 or not eff.rgdwAxes or eff.rgdwAxes[0] not in actuators:
                return di.DIERR_INVALIDPARAM
        if flags & fm.DIEP_DIRECTION and not eff.dwFlags & (fm.DIEFF_CARTESIAN | fm.DIEFF_POLAR | fm.DIEFF_SPHERICAL):
            return di.DIERR_INVALIDPARAM
        if flags & fm.DIEP_DURATION:
            fields["duration"] = eff.dwDuration
        if flags & fm.DIEP_SAMPLEPERIOD:
            fields["sample_period"] = eff.dwSamplePeriod
        if flags & fm.DIEP_GAIN:
            if eff.dwGain > fm.DI_FFNOMINALMAX:
                return di.DIERR_INVALIDPARAM
            fields["gain"] = eff.dwGain
        if flags & fm.DIEP_TRIGGERBUTTON:
            fields["trigger_button"] = eff.dwTriggerButton
        if flags & fm.DIEP_TRIGGERREPEATINTERVAL:
            fields["trigger_repeat"] = eff.dwTriggerRepeatInterval
        if flags & fm.DIEP_STARTDELAY:
            fields["start_delay"] = eff.dwStartDelay
        if flags & fm.DIEP_ENVELOPE:
            if eff.lpEnvelope:
                env = di.DIENVELOPE.from_address(eff.lpEnvelope)
                if env.dwSize != ctypes.sizeof(di.DIENVELOPE):
                    return di.DIERR_INVALIDPARAM
                fields["envelope"] = fm.DiEnvelope(env.dwAttackLevel, env.dwAttackTime, env.dwFadeLevel,
                                                   env.dwFadeTime)
            else:
                fields["envelope"] = None
        if flags & fm.DIEP_TYPESPECIFICPARAMS:
            cls = ts_kind(e.effect.kind)
            if cls is None or eff.cbTypeSpecificParams != TS_SIZES[cls] or not eff.lpvTypeSpecificParams:
                return di.DIERR_INVALIDPARAM
            a = eff.lpvTypeSpecificParams
            if cls is fm.DiConstant:
                fields["type_specific"] = fm.DiConstant(di.DICONSTANTFORCE.from_address(a).lMagnitude)
            elif cls is fm.DiRamp:
                r = di.DIRAMPFORCE.from_address(a)
                fields["type_specific"] = fm.DiRamp(r.lStart, r.lEnd)
            elif cls is fm.DiPeriodic:
                r = di.DIPERIODIC.from_address(a)
                fields["type_specific"] = fm.DiPeriodic(r.dwMagnitude, r.lOffset, r.dwPhase, r.dwPeriod)
            else:
                r = di.DICONDITION.from_address(a)
                fields["type_specific"] = fm.DiCondition(r.lOffset, r.lPositiveCoefficient, r.lNegativeCoefficient,
                                                         r.dwPositiveSaturation, r.dwNegativeSaturation,
                                                         r.lDeadBand)
        return e.effect.update(t, replace(p, **fields), flags)

    def _effect_Release(self, e: EffectHandle, t):
        if not e.effect.released:
            e.effect.unload(t)
            e.effect.released = True
            if e.effect in e.owner.dev.effects:
                e.owner.dev.effects.remove(e.effect)
            self._ffb(e.owner.dev, t, "Release", di.DI_OK, effect=e.handle)
        return 0

    def _effect_GetEffectGuid(self, e, t, out):
        guid = di.GUID.parse(effect_guid(e.effect.kind))
        ctypes.memmove(ctypes.addressof(deref(out)), ctypes.addressof(guid), ctypes.sizeof(guid))
        return di.DI_OK

    def _effect_SetParameters(self, e, t, peff, flags):
        hr = self._effect_gate(e)
        if hr is None:
            hr = self._set_parameters(e, t, peff, flags)
        return self._ffb(e.owner.dev, t, "SetParameters", hr, effect=e.handle, detail=f"0x{flags:08X}")

    def _effect_Start(self, e, t, iterations, flags):
        hr = self._effect_gate(e)
        if hr is None:
            hr = e.effect.start(t, iterations, flags)
        return self._ffb(e.owner.dev, t, "Start", hr, effect=e.handle, detail=str(iterations))

    def _effect_Stop(self, e, t):
        hr = self._effect_gate(e)
        if hr is None:
            hr = e.effect.stop(t) if e.owner.exclusive_acquired else di.DIERR_NOTEXCLUSIVEACQUIRED
        return self._ffb(e.owner.dev, t, "Stop", hr, effect=e.handle)

    def _effect_Download(self, e, t):
        hr = self._effect_gate(e)
        if hr is None:
            if not e.owner.exclusive_acquired:
                hr = di.DIERR_NOTEXCLUSIVEACQUIRED
            elif not e.effect.complete:
                hr = di.DIERR_INCOMPLETEEFFECT
            else:
                e.effect.downloaded, hr = True, di.DI_OK
        return self._ffb(e.owner.dev, t, "Download", hr, effect=e.handle)

    def _effect_Unload(self, e, t):
        hr = self._effect_gate(e)
        if hr is None:
            hr = e.effect.unload(t) if e.owner.exclusive_acquired else di.DIERR_NOTEXCLUSIVEACQUIRED
        return self._ffb(e.owner.dev, t, "Unload", hr, effect=e.handle)

    def _effect_GetEffectStatus(self, e, t, out):
        hr = self._effect_gate(e)
        if hr is None:
            deref(out).value = e.effect.status(t)
            hr = di.DI_OK
        return hr
