"""One wheel I/O owner; callers only replace immutable command snapshots."""
from __future__ import annotations

import atexit
import ctypes
import logging
import math
import sys
import threading
import time
from collections import deque
from dataclasses import replace

from simsteer.core.override import Commands, Output, Override, Physical, OVERRIDE_HOLD_S, clamp
from simsteer.core.finite import NonFiniteError, require_finite
from simsteer.runtime.input.physical import PhysicalWheel, RELAY_HZ, RELAY_MIN_HZ
from simsteer.runtime.output.vjoy_feeder import VJoyFeeder
from simsteer.runtime.output.wheel import WheelError
from simsteer.runtime.output.safety import OutputGate
from simsteer.runtime.wheel_session import engage_block

RELAY_REFRESH_S = 0.10
SWITCH_INTERVAL_S = 0.001
LATENCY_TARGET_P95_MS = 5.0
LATENCY_FAIL_P95_MS = 10.0
LOG = logging.getLogger(__name__)


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * p) - 1)] if ordered else 0.0


def player_output(physical):
    return Override().step(0, physical, Commands())


class ConsoleExit:
    """Console events ask the I/O owner to neutralize and release its device."""
    def __init__(self, close, *, api=None):
        callback_type = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(ctypes.c_int, ctypes.c_uint32)
        def on_event(event):
            if event not in (2, 5, 6):  # CLOSE, LOGOFF, SHUTDOWN
                return 0
            try:
                close()
            except Exception:
                LOG.exception("Wheel console exit failed")
            return 0  # Continue to the manager's permanent shutdown handler.
        self.callback = callback_type(on_event)
        if api is None:
            api = ctypes.WinDLL("kernel32", use_last_error=True)
            api.SetConsoleCtrlHandler.argtypes = [callback_type, ctypes.c_int]
            api.SetConsoleCtrlHandler.restype = ctypes.c_int
        self.set_handler = api.SetConsoleCtrlHandler
        self.registered = bool(self.set_handler(self.callback, 1))
        if not self.registered:
            raise WheelError("Could not register wheel console exit handler")

    def close(self):
        if self.registered:
            if not self.set_handler(self.callback, 0):
                raise WheelError("Could not unregister wheel console exit handler")
            self.registered = False


class WheelIo(threading.Thread):
    def __init__(self, settings, device_id=1, *, reader_factory=None, feeder_factory=None, write_output=None):
        super().__init__(name="wheel-io", daemon=True)
        self.settings = settings
        self.reader_factory = reader_factory or (lambda: PhysicalWheel(settings))
        self.feeder_factory = feeder_factory or (lambda: VJoyFeeder(device_id))
        self.write_output = write_output or OutputGate().write_output
        self.reset = threading.Event()
        self.stop_epoch = 0
        self.lock, self.ready, self.stopping = threading.Lock(), threading.Event(), threading.Event()
        self.commands, self.output, self.physical = Commands(), Output(), Physical()
        self.available, self.error, self.map_request = False, None, False
        self.name_text, self.map_status, self.pending = "No wheel", "", {}
        self.times, self.latencies = deque(maxlen=RELAY_HZ), deque(maxlen=RELAY_HZ)
        self.gap_max, self.ticks, self.hooks = 0.0, 0, []

    def run(self):
        reader = feeder = None
        previous_interval = sys.getswitchinterval()
        previous, written_at, last_tick, warned_at = None, 0.0, None, time.monotonic()
        machine = Override(self.settings.rim_range_deg)
        try:
            reader = self.reader_factory()
            if not reader.open():
                return
            feeder = self.write_output(lambda _: self.feeder_factory())
            self.available = True
            sys.setswitchinterval(SWITCH_INTERVAL_S)
            self.ready.set()
            deadline = time.monotonic()
            while not self.stopping.is_set():
                now = time.monotonic()
                with self.lock:
                    commands, mapping = self.commands, self.map_request
                    self.map_request = False
                    reset, epoch = self.reset.is_set(), self.stop_epoch
                    self.reset.clear()
                try:
                    p = reader.read(now)
                    if mapping and not machine.engaged and p.ok:
                        reader.map_pedals(now)
                        p = replace(p, throttle=None, clutch=None)
                    machine.rim_range_deg = self.settings.rim_range_deg
                    if reset:
                        machine = Override(self.settings.rim_range_deg)
                        machine.output = player_output(p)
                    out = machine.step(now, p, commands)
                    if reset or epoch != self.stop_epoch or out != previous or now - written_at >= RELAY_REFRESH_S:
                        def emit(_):
                            with self.lock:
                                actual = (Output() if self.stopping.is_set() else
                                          player_output(p) if epoch != self.stop_epoch else out)
                                feeder.write(actual)
                                return actual
                        out = self.write_output(emit) or Output()
                        finished = time.monotonic()
                        if p.ok:
                            self.latencies.append(max(0, finished - p.sampled_at) * 1000)
                        previous, written_at = out, now
                    for hook in self.hooks:
                        hook(now)
                except Exception as e:
                    LOG.warning("Wheel I/O: %s", e)
                    self.error, p = str(e), Physical()
                    out = machine.step(now, p, commands)
                    # Never keep the previous throttle after a reader/writer error.
                    try:
                        self.write_output(lambda _: feeder.write(Output()))
                    except Exception:
                        break
                    previous = None
                with self.lock:
                    self.physical = p
                    self.output = player_output(p) if epoch != self.stop_epoch else out
                    self.name_text, self.map_status = reader.name, reader.status
                    if reader.instance_proven:
                        # Save cleared/rebuilt axes before the new identity.
                        # A restart must never apply the previous instance's map.
                        self.pending.update(wheel_axes=dict(reader.axes), wheel_instance=reader.instance_proven)
                    if reader.mapper and reader.mapper.done:
                        self.pending["wheel_axes"] = dict(reader.axes)
                    if last_tick is not None:
                        self.gap_max = max(self.gap_max, (now - last_tick) * 1000)
                    self.times.append(now)
                    self.ticks += 1
                last_tick = now
                if now - warned_at >= 1:
                    hz = (len(self.times) - 1) / max(1e-6, now - self.times[0])
                    if hz < RELAY_MIN_HZ:
                        LOG.warning("Wheel relay %.0f Hz (target %s)", hz, RELAY_HZ)
                    warned_at = now
                deadline = max(deadline + 1 / RELAY_HZ, now)
                # Python 3.11's Windows sleep uses a high-resolution timer;
                # Event.wait uses a different timeout path. Stop is at most
                # one 2 ms period away here.
                time.sleep(max(0.0, deadline - time.monotonic()))
        except Exception as e:
            self.error = str(e)
        finally:
            try:
                if reader:
                    reader.close()
            finally:
                try:
                    if feeder:
                        self.write_output(lambda _: feeder.close(), cleanup=True)
                finally:
                    self.output, self.physical = Output(reason="lost" if self.error else "closed"), Physical()
                    sys.setswitchinterval(previous_interval)
                    self.ready.set()


class PassthroughWheel:
    def __init__(self, settings, device_id=1, **factories):
        self.settings, self.device_id = settings, device_id
        self._lifecycle_lock = threading.RLock()
        self._refused_reason = ""
        self._console = None
        self._released = False
        self.io = WheelIo(settings, device_id, **factories)
        self.io.start()
        if not self.io.ready.wait(2):
            self.close()
            raise WheelError("Wheel I/O startup timed out")
        if self.io.error:
            self.close()
            raise WheelError(self.io.error)
        if sys.platform == "win32":
            try:
                self._console = ConsoleExit(self.close)
            except Exception:
                self.close()
                raise
        atexit.register(self.close)

    @property
    def available(self):
        return self.io.available

    @property
    def engaged(self):
        return not self.io.stopping.is_set() and self.io.output.engaged

    @property
    def physical(self):
        return self.io.physical

    @property
    def source(self):
        return self.io.output.source

    @property
    def last_reason(self):
        return self._refused_reason or self.io.output.reason

    @property
    def pedal_status(self):
        return self.io.map_status

    @property
    def passthrough_live(self):
        return self.io.is_alive() and self.physical.ok and self.source == "PLAYER"

    @property
    def raw(self):
        return None  # Never expose a second vJoy writer to callers.

    def _request(self, request):
        with self.io.lock:
            if self.io.stopping.is_set():
                return False
            cmd = self.io.commands
            if request == "engage":
                self._refused_reason = "map pedals" if self.io.map_request else engage_block(self)
                if self._refused_reason:
                    return False
                cmd = replace(cmd, steer=self.io.output.steer, throttle=0.0, brake=0.0)
            self._refused_reason = ""
            self.io.commands = replace(cmd, serial=cmd.serial + 1, request=request,
                                       ai_at=time.monotonic(), forced={})
            return True

    def engage(self):
        return self._request("engage")

    def disengage(self, reason="loop"):
        self._request(reason)

    def stop_output(self):
        """Drop AI authority immediately while keeping physical controls live."""
        with self.io.lock:
            self.io.reset.set()
            self.io.stop_epoch += 1
            self.io.commands = Commands(serial=self.io.commands.serial + 1)
            self.io.output = player_output(self.io.physical)

    def resume_output(self):
        """Physical controls already remain live after an AI stop."""

    def toggle(self):
        desired = not self.engaged
        accepted = self._request("engage" if desired else "insert")
        return desired and accepted

    def center(self):
        with self.io.lock:
            self.io.commands = replace(self.io.commands, forced={})

    def _axes(self, values, force):
        now = time.monotonic()
        try:
            require_finite("wheel", **values)
        except NonFiniteError:
            self.stop_output()
            raise
        values = {k: clamp(v, -1 if k == "steer" else 0, 1) for k, v in values.items()}
        with self.io.lock:
            if self.io.stopping.is_set():
                return
            cmd = self.io.commands
            if self.engaged:
                self.io.commands = replace(cmd, **values, ai_at=now)
            elif force:
                overrides = {**cmd.forced, **{k: (v, now + OVERRIDE_HOLD_S) for k, v in values.items()}}
                self.io.commands = replace(cmd, forced=overrides)

    def set_steering(self, x, force=False):
        self._axes({"steer": x}, force)

    def set_throttle_brake(self, throttle, brake, force=False):
        self._axes({"throttle": throttle, "brake": brake}, force)

    def map_pedals(self):
        with self.io.lock:
            if self.engaged or not self.physical.ok:
                return False
            self.io.map_request = True
            return True

    def stats(self, *, consume=False):
        with self._lifecycle_lock:
            with self.io.lock:
                times, latency = tuple(self.io.times), tuple(self.io.latencies)
                result = dict(relay_hz=(len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0,
                              latency_p95_ms=percentile(latency, .95), latency_p99_ms=percentile(latency, .99),
                              latency_max_ms=max(latency, default=0), relay_gap_max_ms=self.io.gap_max,
                              name=self.io.name_text, pedal_status=self.io.map_status)
                pending, self.io.pending = self.io.pending, {}
                if consume:
                    self.io.gap_max = 0.0
            edits = {k: v for k, v in pending.items() if getattr(self.settings, k) != v}
            if edits:
                self.settings.set_persistent(**edits)
                self.settings.save()
            return result

    def wiggle(self, kind, duration_each_s=1.0):
        if kind not in ("steering", "throttle", "brake") or self.engaged:
            return
        key = "steer" if kind == "steering" else kind
        for value, seconds in ((1, duration_each_s), (0, .2), (-1 if key == "steer" else 0, duration_each_s)):
            end = time.monotonic() + seconds
            while time.monotonic() < end and not self.io.stopping.is_set():
                self._axes({key: value}, True)
                self.io.stopping.wait(.05)
        self.center()

    def close(self):
        # Signal neutralization before waiting for a concurrent settings save.
        self.io.stopping.set()
        with self._lifecycle_lock:
            if self._released:
                return
            self.io.join(.45)
            if self.io.is_alive():
                raise WheelError("Wheel I/O did not stop")
            try:
                self.stats()
            finally:
                if self._console:
                    self._console.close()
                atexit.unregister(self.close)
                self._released = True
