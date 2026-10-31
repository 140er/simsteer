"""vJoyInterface.dll stand-in and the game behind vJoy.

`SimVJoyDll` has the DLL functions SimSteer calls (setup check, feeding
axes and buttons, the FFB callback and the `Ffb_h_*` helpers). Each one is
a plain function object, so callers can set `argtypes`/`restype` on it as
on a real `CDLL` export.

`SimGame` is what a game sees: the axes and buttons written to vJoy, and
a sender that delivers force feedback packets to the callback SimSteer
registered, in the vJoy `FFB_DATA` layout, from the DLL's own thread when
sent in real time. A recorded packet answers the `Ffb_h_*` helpers with
its recorded DLL reading, so SimSteer's cross-check runs; synthetic packets
answer with an error code, which the cross-check skips.
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from pathlib import Path

from simsteer.runtime.ffb import vjoy_ffb as vj

VJD_OWN, VJD_FREE, VJD_BUSY, VJD_MISSING = 0, 1, 2, 3
AXIS_NAMES = {0x30: "x", 0x31: "y", 0x32: "z", 0x33: "rx", 0x34: "ry", 0x35: "rz", 0x36: "sl0", 0x37: "sl1"}
OTHER_PID = 4242
AXIS_MIN, AXIS_MAX, AXIS_CTR = 0x0001, 0x8000, 0x4000
FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "vjoy_ffb_packets"


def load_packets(source) -> list[tuple[float, int, bytes, dict | None]]:
    """Packets as (t, cmd, data, dll reading) from a fixture name, a
    fixture file, a fixture dict or (t, cmd, data[, dll]) tuples."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists() and not path.suffix:
            path = FIXTURES / f"{source}.json"
        source = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(source, dict):
        return [(float(p["t"]), int(p["cmd"]), bytes.fromhex(p["data"]), p.get("dll")) for p in source["packets"]]
    out = []
    for p in source:
        t, cmd, data, *rest = p
        out.append((float(t), int(cmd), bytes(data), rest[0] if rest else None))
    return out


class VJoyDevice:
    def __init__(self, ffb: bool, effects, axes) -> None:
        self.ffb, self.effects, self.axes = ffb, set(effects), set(axes)
        self.owner: str | None = None  # "us" or "other"
        self.values = {u: AXIS_CTR for u in self.axes}
        self.buttons: set[int] = set()


class SimVJoyDll:
    def __init__(self, sim, devices=(1,), *, enabled=True, ffb=True, effects=None, version=0x222,
                 axes=(0x30, 0x31, 0x32, 0x33), buttons=32) -> None:
        self.sim = sim
        self.enabled, self.version, self.button_count = enabled, version, buttons
        effects = vj.EFFECT_USAGES.values() if effects is None else effects
        self.devices = {d: VJoyDevice(ffb, effects, axes) for d in devices}
        self.calls: list[tuple] = []
        self.callback = None
        self.callback_data = None
        self.callback_thread: int | None = None
        self._readings: dict[int, dict] = {}
        for name in ("vJoyEnabled", "vJoyFfbCap", "isVJDExists", "IsDeviceFfb", "IsDeviceFfbEffect", "AcquireVJD",
                     "GetvJoyMaxDevices", "GetvJoyVersion", "GetVJDStatus", "GetOwnerPid", "RelinquishVJD",
                     "FfbRegisterGenCB", "GetVJDAxisExist", "SetAxis", "SetBtn", "ResetVJD", "ResetAll",
                     "GetVJDButtonNumber", "Ffb_h_Type", "Ffb_h_DeviceID", "Ffb_h_EffectBlockIndex",
                     "Ffb_h_CreateNewEffect", "Ffb_h_DevCtrl", "Ffb_h_DevGain",
                     *(d[0] for d in vj.DLL_DECODERS.values())):
            setattr(self, name, self._export(name))

    def _export(self, name):
        impl = getattr(self, "_" + name)
        record = self.calls

        def export(*args):
            with self.sim.lock:
                record.append((name, args, threading.get_ident()))
                return impl(*args)
        export.__name__ = name
        return export

    # ---- setup and status

    def _vJoyEnabled(self):
        return int(self.enabled)

    def _vJoyFfbCap(self, out):
        out._obj.value = 1
        return 1

    def _GetvJoyMaxDevices(self, out):
        out._obj.value = 16
        return 1

    def _GetvJoyVersion(self):
        return self.version

    def _isVJDExists(self, device):
        return int(device in self.devices)

    def _IsDeviceFfb(self, device):
        return int(device in self.devices and self.devices[device].ffb)

    def _IsDeviceFfbEffect(self, device, usage):
        d = self.devices.get(device)
        return int(d is not None and d.ffb and usage in d.effects)

    def _GetVJDStatus(self, device):
        d = self.devices.get(device)
        if d is None:
            return VJD_MISSING
        return {None: VJD_FREE, "us": VJD_OWN, "other": VJD_BUSY}[d.owner]

    def _GetOwnerPid(self, device):
        owner = getattr(self.devices.get(device), "owner", None)
        return {"other": OTHER_PID, "us": os.getpid()}.get(owner, 0)

    def _AcquireVJD(self, device):
        d = self.devices.get(device)
        if d is None or d.owner == "other":
            return 0
        d.owner = "us"
        return 1

    def _RelinquishVJD(self, device):
        d = self.devices.get(device)
        if d is not None and d.owner == "us":
            d.owner = None

    def _FfbRegisterGenCB(self, callback, data):
        address = callback.value if isinstance(callback, ctypes.c_void_p) else callback
        self.callback = vj.FfbGenCB(address) if address else None
        self.callback_data = data
        self.callback_thread = threading.get_ident()

    # ---- feeding

    def _GetVJDAxisExist(self, device, usage):
        d = self.devices.get(device)
        return int(d is not None and usage in d.axes)

    def _owned(self, device):
        d = self.devices.get(device)
        return d if d is not None and d.owner == "us" else None

    def _SetAxis(self, value, device, usage):
        d = self._owned(device)
        if d is None or usage not in d.axes:
            return 0
        d.values[usage] = int(value)
        return 1

    def _SetBtn(self, down, device, button):
        d = self._owned(device)
        if d is None or not 1 <= button <= self.button_count:
            return 0
        (d.buttons.add if down else d.buttons.discard)(button)
        return 1

    def _ResetVJD(self, device):
        d = self._owned(device)
        if d is None:
            return 0
        d.values = {u: AXIS_CTR for u in d.axes}
        d.buttons.clear()
        return 1

    def _ResetAll(self):
        for device in self.devices:
            self._ResetVJD(device)

    def _GetVJDButtonNumber(self, device):
        return self.button_count if device in self.devices else 0

    # ---- Ffb_h_* helpers: answered from the recorded DLL reading

    def _reading(self, ptr):
        address = ptr.value if isinstance(ptr, ctypes.c_void_p) else ptr
        return self._readings.get(address)

    @staticmethod
    def _put(out, value):
        out._obj.value = value

    def _helper(self, ptr, *outs, keys):
        r = self._reading(ptr)
        if r is None or r.get("rc"):
            return 1 if r is None else r["rc"]
        for out, key in zip(outs, keys):
            if key in r:
                self._put(out, r[key])
        return 0

    def _Ffb_h_Type(self, ptr, out):
        return self._helper(ptr, out, keys=("type",))

    def _Ffb_h_DeviceID(self, ptr, out):
        return self._helper(ptr, out, keys=("DeviceID",))

    def _Ffb_h_EffectBlockIndex(self, ptr, out):
        return self._helper(ptr, out, keys=("EffectId",))

    def _Ffb_h_CreateNewEffect(self, ptr, effect_type, effect_id):
        return self._helper(ptr, effect_type, effect_id, keys=("EffectType", "EffectId"))

    def _Ffb_h_DevCtrl(self, ptr, out):
        return self._helper(ptr, out, keys=("Control",))

    def _Ffb_h_DevGain(self, ptr, out):
        return self._helper(ptr, out, keys=("Gain",))

    def _struct_helper(self, ptr, out):
        r = self._reading(ptr)
        if r is None or r.get("rc"):
            return 1 if r is None else r["rc"]
        st = out._obj
        for name, _ in st._fields_:
            if name in r:
                setattr(st, name, r[name])
        return 0

    def __getattr__(self, name):
        # The struct helpers (Ffb_h_Eff_Report, ...) share one body.
        if name.startswith("_Ffb_h_"):
            return self._struct_helper
        raise AttributeError(name)

    # ---- delivery (called by SimGame)

    def deliver(self, cmd: int, data: bytes, reading: dict | None, device: int = 1) -> bool:
        """Hand one packet to the registered callback, as the DLL's thread does.
        False when nobody listens: no callback or the device not acquired."""
        with self.sim.lock:
            d = self.devices.get(device)
            callback, user = self.callback, self.callback_data
            if callback is None or d is None or d.owner != "us" or not d.ffb:
                return False
            buf = (ctypes.c_uint8 * max(1, len(data))).from_buffer_copy(data or b"\0")
            packet = vj.FFB_DATA(len(data) + 8, cmd, ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)))
            address = ctypes.addressof(packet)
            if reading is not None:
                self._readings[address] = reading
        try:
            callback(address, user)  # outside the lock: the callback may block
        finally:
            self._readings.pop(address, None)
        return True


class Sender(threading.Thread):
    """Delivers packets at their times, like the vJoy DLL's window thread."""

    def __init__(self, dll: SimVJoyDll, packets, speed: float = 1.0, device: int = 1) -> None:
        super().__init__(name="vjoy-ffb-sim", daemon=True)
        self.dll, self.packets, self.speed, self.device = dll, packets, speed, device
        self.delivered = self.dropped = 0
        self._halt = threading.Event()

    def run(self) -> None:
        t0 = time.monotonic()
        for t, cmd, data, reading in self.packets:
            wait = t0 + t / self.speed - time.monotonic()
            if wait > 0 and self._halt.wait(wait):
                return
            if self._halt.is_set():
                return
            if self.dll.deliver(cmd, data, reading, self.device):
                self.delivered += 1
            else:
                self.dropped += 1

    def stop(self) -> None:
        self._halt.set()
        self.join(2)


class SimGame:
    """The game on the far side of vJoy."""

    def __init__(self, dll: SimVJoyDll | None, device: int = 1) -> None:
        self.dll, self.device = dll, device

    def _dev(self) -> VJoyDevice:
        if self.dll is None:
            raise RuntimeError("the game side needs the simulated vJoy (FanatecSim(vjoy=True))")
        return self.dll.devices[self.device]

    @property
    def axes(self) -> dict[str, int]:
        with self.dll.sim.lock:
            return {AXIS_NAMES.get(u, hex(u)): v for u, v in self._dev().values.items()}

    @property
    def buttons(self) -> set[int]:
        with self.dll.sim.lock:
            return set(self._dev().buttons)

    def steer(self) -> float:
        """vJoy X as SimSteer's -1..1 steering."""
        return (self.axes["x"] - AXIS_CTR) / (AXIS_CTR - AXIS_MIN)

    def pedal(self, axis: str) -> float:
        """A vJoy axis as 0..1 (y throttle, z brake, rx clutch)."""
        return (self.axes[axis] - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)

    def send(self, packets, *, realtime: bool = False, speed: float = 1.0) -> Sender | int:
        """Send force feedback packets to SimSteer. At once (returns how many
        were delivered) or at their recorded times from a thread (returns the
        running `Sender`)."""
        loaded = load_packets(packets)
        if realtime:
            self._dev()
            sender = Sender(self.dll, loaded, speed, self.device)
            sender.start()
            return sender
        return sum(self.dll.deliver(cmd, data, reading, self.device) for _, cmd, data, reading in loaded)

    def take_vjoy(self) -> None:
        """Another program owns the vJoy device (SimSteer's acquire fails)."""
        with self.dll.sim.lock:
            self._dev().owner = "other"
