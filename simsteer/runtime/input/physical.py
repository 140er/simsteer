"""Background, non-exclusive DirectInput reader. No property or force writes."""
from __future__ import annotations

import ctypes as c
import logging
import sys
import time

from simsteer.core.override import Physical
from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.input.mapping import AxisMap, PedalMapper, resolve_slots, validate_axes

RELAY_HZ = 500
RELAY_MIN_HZ = 250
REACQUIRE_S = 0.5
INSTANCE_DETECT_DEG = 2.0
INSTANCE_DETECT_HOLD_S = 0.05
PEDAL_SANE_REST = 0.05
LOG = logging.getLogger(__name__)


class _Range(c.Structure):
    _fields_ = [("header", di.DIPROPHEADER), ("lo", di.LONG), ("hi", di.LONG)]


class DInputDevice:
    """The probe's read/acquire/object-at pattern over the shared COM layer."""
    def __init__(self, api, inst):
        self.ptr = api.create_device(inst)
        self.info = di.info_of(inst)
        try:
            self._format = di.joystick_state_format()
            self.call("SetDataFormat", [c.POINTER(di.DIDATAFORMAT)], c.byref(self._format[0]))
            self.call("SetCooperativeLevel", [c.c_void_p, di.DWORD],
                      api.hwnd, di.DISCL_NONEXCLUSIVE | di.DISCL_BACKGROUND)
            self.names, self.ranges = {}, {}
            for slot in range(8):
                obj = di.DIDEVICEOBJECTINSTANCEW()
                obj.dwSize = c.sizeof(obj)
                hr = di._dev(self.ptr, "GetObjectInfo", [c.POINTER(type(obj)), di.DWORD, di.DWORD],
                             c.byref(obj), slot * 4, 1)  # DIPH_BYOFFSET
                if not di.failed(hr):
                    self.names[slot] = di.wide(obj.tszName)
                prop = _Range(di.DIPROPHEADER(c.sizeof(_Range), c.sizeof(di.DIPROPHEADER), slot * 4, 1), 0, 65535)
                hr = di._dev(self.ptr, "GetProperty", [c.c_void_p, c.POINTER(_Range)], 4, c.byref(prop))  # DIPROP_RANGE
                self.ranges[slot] = (prop.lo, prop.hi) if not di.failed(hr) and prop.hi > prop.lo else (0, 65535)
            self.acquire()
        except Exception:
            self.close()
            raise

    def call(self, method, types, *args):
        return di._check(method, di._dev(self.ptr, method, types, *args))

    def acquire(self):
        self.call("Acquire", [])

    def read(self):
        state = di.JOYSTATE()
        hr = di._dev(self.ptr, "Poll", [])
        if di.failed(hr) and hr in di.DEVICE_LOST:
            di._check("Poll", hr)
        self.call("GetDeviceState", [di.DWORD, c.c_void_p], c.sizeof(state), c.byref(state))
        return tuple(state.axes)

    def close(self):
        if self.ptr:
            try:
                di._dev(self.ptr, "Unacquire", [])
            finally:
                di._dev(self.ptr, "Release", [], restype=di.DWORD)
                self.ptr = None


class PhysicalWheel:
    def __init__(self, settings=None, *, di_factory=None, only_vid=di.FANATEC_VID):
        self.factory, self.only_vid = di_factory, only_vid
        self.saved = getattr(settings, "wheel_instance", "")
        self.axes = validate_axes(getattr(settings, "wheel_axes", {}))
        self.rim_range_deg = getattr(settings, "rim_range_deg", 900)
        self.api, self.records, self.active = None, [], None
        self.instance_proven, self.mapper, self.status = "", None, ""
        self.raw_axes = (0,) * 8
        self.retry_at, self.reopen = 0.0, False
        self.pedals_held = None

    @property
    def name(self):
        return self.active["dev"].info.name if self.active else "No wheel"

    def open(self):
        factory = self.factory if self.factory is not None else di.DirectInput
        if factory is di.DirectInput and sys.platform != "win32":
            return False
        self.api = factory()
        candidates = [i for i in self.api.game_controllers()
                      if di.info_of(i).vid == self.only_vid and not di.info_of(i).is_vjoy]
        saved = [i for i in candidates if di.info_of(i).instance == self.saved]
        if self.saved and not saved:
            LOG.warning("Saved wheel instance missing; detecting an active wheel")
            self.axes, self.saved = {}, ""
        for inst in saved or candidates:
            try:
                dev = DInputDevice(self.api, inst)
                self.records.append(dict(dev=dev, anchor=None, detect_at=None, detect_sign=0,
                                         retry=None, failures=0))
            except (di.DIError, OSError) as e:
                LOG.warning("Wheel open: %s", e)
        self.active = self.records[0] if saved and self.records else None
        self.pedals_held, self.reopen = None, False
        return bool(self.records)

    def read(self, now=None):
        now = time.monotonic() if now is None else now
        if self.reopen or not self.records:
            if now < self.retry_at:
                return Physical()
            self.close()
            self.retry_at = now + REACQUIRE_S
            self.open()
        samples = {}
        for rec in self.records:
            try:
                if rec["retry"] is not None:
                    if now < rec["retry"]:
                        continue
                    rec["retry"] = now + REACQUIRE_S
                    rec["dev"].acquire()
                axes = rec["dev"].read()
                rec["retry"], rec["failures"] = None, 0
                slots = resolve_slots(rec["dev"].names)
                slot = self.axes.get("steer", slots["steer"])
                steer = AxisMap(slot, *rec["dev"].ranges[slot]).steering(axes)
                samples[id(rec)] = (axes, steer)
                if rec["anchor"] is None:
                    rec["anchor"] = steer
                delta = (steer - rec["anchor"]) * self.rim_range_deg / 2
                moving = abs(delta) > INSTANCE_DETECT_DEG
                sign = 1 if delta > 0 else -1
                if not moving:
                    rec["detect_at"] = None
                elif rec["detect_at"] is None or rec["detect_sign"] != sign:
                    rec["detect_at"], rec["detect_sign"] = now, sign
                if (not self.saved and moving
                        and now - rec["detect_at"] + 1e-9 >= INSTANCE_DETECT_HOLD_S):
                    if self.mapper:
                        self.mapper.done, self.mapper.result = True, {}
                        self.mapper.status = "Wheel selected; map pedals again"
                    self.active = rec
                    self.saved = self.instance_proven = rec["dev"].info.instance
                    self.axes = {"steer": slots["steer"]}
                    self.pedals_held = None
            except (di.DIError, OSError) as e:
                if rec["retry"] is None:
                    LOG.warning("Wheel lost: %s", e)
                rec["retry"] = now + REACQUIRE_S
                rec["failures"] += 1
                rec["detect_at"] = None
                if rec is self.active:
                    self.pedals_held = None
                if rec["failures"] >= 3:
                    self.reopen, self.retry_at = True, now + REACQUIRE_S
        if self.saved and len(self.records) > 1:
            for rec in self.records:
                if rec is not self.active:
                    rec["dev"].close()
            self.records = [self.active]
        sample = samples.get(id(self.active))
        if sample is None:
            if self.mapper and not self.mapper.done:
                self.mapper.done, self.mapper.result, self.mapper.status = True, {}, "Wheel lost; retry"
            return Physical()
        self.raw_axes, steer = sample
        if self.mapper and not self.mapper.done:
            self.mapper.step(now, self.raw_axes)
            if self.mapper.done and self.mapper.result:
                self.axes, self.pedals_held = self.mapper.result, None
        pedals = [AxisMap.mapped_pedal(self.raw_axes, self.axes.get(k)) for k in ("throttle", "brake", "clutch")]
        away_from_rest = {name for name in ("throttle", "brake", "clutch")
                          if (p := self.axes.get(name)) and
                          abs(self.raw_axes[p["slot"]] - p["rest"]) / abs(p["full"] - p["rest"]) > PEDAL_SANE_REST}
        if self.pedals_held is None:
            self.pedals_held = away_from_rest
        self.pedals_held.intersection_update(away_from_rest)
        self.status = "Check pedals" if self.pedals_held else (self.mapper.status if self.mapper else "")
        if self.pedals_held or (self.mapper and not self.mapper.done):
            pedals = [None] * 3
        return Physical(steer, *pedals, ok=True, sampled_at=now)

    def map_pedals(self, now):
        if self.active and self.active["retry"] is None:
            dev = self.active["dev"]
            self.axes = {"steer": resolve_slots(dev.names)["steer"]}
            self.pedals_held = None
            self.mapper = PedalMapper(now, self.raw_axes, dev.names, dev.ranges, self.axes["steer"])

    def close(self):
        for rec in self.records:
            rec["dev"].close()
        self.records, self.active = [], None
        if self.api:
            self.api.close()
            self.api = None


def main(argv=None):
    import argparse
    from simsteer.app.settings import Settings
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--monitor", action="store_true", required=True)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--only-vid", type=lambda s: int(s, 16), default=di.FANATEC_VID)
    ap.add_argument("--echo-vjoy", action="store_true")
    args = ap.parse_args(argv)
    wheel = PhysicalWheel(Settings.load(), only_vid=args.only_vid)
    echo, delays, pending = None, [], None
    start = last_print = time.monotonic()
    count, previous = 0, None
    try:
        if not wheel.open():
            print("No wheel")
            return 1
        if args.echo_vjoy:
            inst = next((i for i in wheel.api.game_controllers() if di.info_of(i).is_vjoy), None)
            if inst is None:
                raise RuntimeError("No vJoy device to read")
            echo = DInputDevice(wheel.api, inst)
        while time.monotonic() - start < args.seconds:
            now = time.monotonic()
            p = wheel.read(now)
            count += p.ok
            if echo and p.ok:
                if previous is not None and abs(p.steer - previous) > 0.001:
                    pending = (p.steer, now)
                previous = p.steer
                observed = AxisMap(0, *echo.ranges[0]).steering(echo.read())
                if pending and abs(observed - pending[0]) < 0.001:
                    delays.append((time.monotonic() - pending[1]) * 1000)
                    pending = None
            if now - last_print >= 0.1:
                print(f"{wheel.name} rim {p.steer * wheel.rim_range_deg / 2:+.1f}° "
                      f"pedals {p.throttle}/{p.brake}/{p.clutch} {count / (now - start):.0f} Hz {wheel.status}")
                last_print = now
            time.sleep(1 / RELAY_HZ)
    finally:
        if echo:
            echo.close()
        wheel.close()
    if args.echo_vjoy:
        print(f"Echo p95 {sorted(delays)[int((len(delays) - 1) * .95)]:.2f} ms" if delays else "Echo: no matched movement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
