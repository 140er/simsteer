"""Offline DirectInput/vJoy transports. IDs are synthetic, never captured."""
import ctypes as c
from collections import deque
from dataclasses import replace
import threading
import time
from types import SimpleNamespace

from simsteer.core.override import Physical
from simsteer.runtime.ffb import dinput_out as di


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


def descriptor(number, vid=di.FANATEC_VID, pid=1):
    inst = di.DIDEVICEINSTANCEW()
    inst.guidInstance.Data1, inst.guidProduct.Data1 = number, vid | pid << 16
    encoded = "Fanatec DD2".encode("utf-16-le")
    c.memmove(inst.tszProductName, encoded, len(encoded))
    return inst


def profile(number=1, names=None):
    return SimpleNamespace(inst=descriptor(number), names=names or {
        0: "Wheel Axis", 1: "Accelerator", 2: "Brake", 3: "Clutch", 4: "Combined Pedals"},
        ranges={i: (0, 65535) for i in range(8)}, axes=[32768, 0, 0, 32767, 0, 0, 0, 0],
        reads=deque(), acquire_errors=deque())


class FakeDi:
    def __init__(self, *profiles):
        self.profiles = list(profiles or [profile()])
        self.handles, self.calls, self.enumerations = {}, [], 0
        self.hwnd = 1

    def __call__(self):
        self.calls.append(("open", threading.get_ident()))
        return self

    def game_controllers(self):
        self.calls.append(("EnumDevices", di.DIEDFL_ATTACHEDONLY))
        self.enumerations += 1
        return [p.inst for p in self.profiles]

    def create_device(self, inst):
        ptr = len(self.handles) + 1
        self.handles[ptr] = next(p for p in self.profiles if bytes(p.inst.guidInstance) == bytes(inst.guidInstance))
        self.calls.append(("CreateDevice", inst.guidInstance.Data1))
        return ptr

    def call(self, ptr, method, types, *args, **kwargs):
        self.calls.append((method, *args))
        p = self.handles[ptr]
        if method == "GetObjectInfo":
            slot = args[1] // 4
            if slot not in p.names:
                return di.DIERR_INVALIDPARAM
            obj = c.cast(args[0], c.POINTER(di.DIDEVICEOBJECTINSTANCEW)).contents
            encoded = p.names[slot].encode("utf-16-le")
            c.memmove(obj.tszName, encoded, len(encoded))
        if method == "GetProperty":
            from simsteer.runtime.input.physical import _Range
            prop = c.cast(args[1], c.POINTER(_Range)).contents
            prop.lo, prop.hi = p.ranges[prop.header.dwObj // 4]
        if method == "Acquire" and p.acquire_errors:
            return p.acquire_errors.popleft()
        if method == "GetDeviceState":
            value = p.reads.popleft() if p.reads else p.axes
            if isinstance(value, int):
                return value
            if isinstance(value, Exception):
                raise value
            state = c.cast(args[1], c.POINTER(di.JOYSTATE)).contents
            state.axes[:] = value
        return 0

    def close(self):
        self.calls.append(("close", threading.get_ident()))


class FakeVJoyDll:
    def __init__(self, *, clutch=True, acquire=True):
        self.calls, self.values = [], {}
        self.clutch, self.acquire = clutch, acquire
        self.fail = None

    def _record(self, name, *args):
        self.calls.append((name, args, time.monotonic(), threading.get_ident()))

    def AcquireVJD(self, device):
        self._record("AcquireVJD", device)
        return self.acquire

    def GetVJDAxisExist(self, device, usage):
        return self.clutch or usage != 0x33

    def SetAxis(self, value, device, usage):
        self._record("SetAxis", value, device, usage)
        if self.fail == usage:
            self.fail = None
            return False
        self.values[usage] = value
        return True

    def SetBtn(self, *args):
        self._record("SetBtn", *args)
        return True

    def ResetVJD(self, *args):
        self._record("ResetVJD", *args)
        self.values = {usage: 0x4000 for usage in self.values}
        return True

    def RelinquishVJD(self, *args):
        self._record("RelinquishVJD", *args)


class ScriptedPhysical:
    def __init__(self, physical=None):
        self.physical = physical or Physical(ok=True)
        self.reads = deque()
        self.name, self.status, self.instance_proven, self.mapper = "Fanatec DD2", "", "", None
        self.axes, self.closed_thread = {}, None

    def open(self):
        self.opened_thread = threading.get_ident()
        return True

    def read(self, now):
        value = self.reads.popleft() if self.reads else self.physical
        if isinstance(value, Exception):
            raise value
        return replace(value, sampled_at=now)

    def close(self):
        self.closed_thread = threading.get_ident()

    def map_pedals(self, now):
        self.status = "Press throttle fully"
