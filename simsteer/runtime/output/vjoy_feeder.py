"""ctypes vJoy writer using the installed interface DLL, without FFB."""
from __future__ import annotations

import ctypes as c
import threading

from simsteer.core.override import Output, clamp
from simsteer.runtime.ffb.vjoy_ffb import load_dll
from simsteer.runtime.output.wheel import WheelError, VJOY_AXIS_CTR, VJOY_AXIS_MIN, VJOY_AXIS_MAX

AXES = {"steer": 0x30, "throttle": 0x31, "brake": 0x32, "clutch": 0x33}


def axis_value(name, value):
    if name == "steer":
        return max(VJOY_AXIS_MIN, min(VJOY_AXIS_MAX, round(VJOY_AXIS_CTR + clamp(value) * (VJOY_AXIS_CTR - VJOY_AXIS_MIN))))
    return VJOY_AXIS_MIN + round(clamp(value, 0, 1) * (VJOY_AXIS_MAX - VJOY_AXIS_MIN))


class VJoyFeeder:
    def __init__(self, device_id=1, *, dll=None):
        self.dll, self.device_id = dll if dll is not None else load_dll(), device_id
        self.owner, self.closed = threading.get_ident(), True
        if dll is None:
            for name, args, result in (
                ("AcquireVJD", [c.c_uint32], c.c_int), ("GetVJDAxisExist", [c.c_uint32, c.c_uint32], c.c_int),
                ("SetAxis", [c.c_int32, c.c_uint32, c.c_uint32], c.c_int),
                ("SetBtn", [c.c_int, c.c_uint32, c.c_uint8], c.c_int),
                ("ResetVJD", [c.c_uint32], c.c_int), ("RelinquishVJD", [c.c_uint32], None),
            ):
                getattr(self.dll, name).argtypes, getattr(self.dll, name).restype = args, result
        if not self.dll.AcquireVJD(device_id):
            raise WheelError(f"Could not acquire vJoy device #{device_id}; check it is enabled and available")
        self.closed = False
        try:
            self.axes = {k: v for k, v in AXES.items() if self.dll.GetVJDAxisExist(device_id, v)}
            if not {"steer", "throttle", "brake"} <= self.axes.keys():
                raise WheelError("vJoy needs X, Y and Z axes")
            self.write(Output())
        except Exception:
            self.close()
            raise

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("vJoy writes must stay on their owner thread")

    def write(self, output):
        self._owner()
        error = None
        for name, usage in self.axes.items():
            try:
                if not self.dll.SetAxis(axis_value(name, getattr(output, name)), self.device_id, usage):
                    raise WheelError(f"vJoy {name} write failed")
            except Exception as e:
                error = error or e
        if error:
            raise error

    def set_axis(self, usage, value):
        self._owner()
        if not self.dll.SetAxis(value, self.device_id, usage):
            raise WheelError("vJoy axis write failed")

    def set_button(self, index, down):
        self._owner()
        if not self.dll.SetBtn(bool(down), self.device_id, index):
            raise WheelError("vJoy button write failed")

    def close(self):
        self._owner()
        if not self.closed:
            try:
                self.dll.ResetVJD(self.device_id)
            finally:
                try:
                    self.write(Output())
                finally:
                    self.dll.RelinquishVJD(self.device_id)
                    self.closed = True
