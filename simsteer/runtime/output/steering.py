"""Paced live steering writer, serialized with every output safety path."""
from __future__ import annotations

import math
import threading

from simsteer.core.steering import STEERING_PERIOD_S, SteeringExecutor, SteeringTarget


class SteeringOutput:
    def __init__(self, pad, clock, allowed, *, threaded=True):
        self.pad = pad
        self.clock = clock
        self.allowed = allowed
        self.executor = SteeringExecutor()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._device = None
        self.last_axis = None
        self.fault = ""
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="steering-output", daemon=True)
            self._thread.start()

    def reset(self):
        with self._lock:
            self.executor.reset()
            self.last_axis = None
            self._device = None

    def submit(self, target: SteeringTarget, now: float):
        with self._lock:
            self._device = self.pad.device
            self.executor.submit(target, now)

    def tick(self, now: float):
        with self._lock:
            if self.executor.target is None:
                return

            def write(device):
                try:
                    if (device is not self._device or not device.engaged
                            or self.pad.binding_active or not self.allowed()):
                        self.reset()
                        return
                    axis = self.executor.tick(now)
                    if axis is None:
                        self.fault = "Steering plan expired"
                        device.disengage()
                        self.reset()
                    else:
                        device.set_steering(axis)
                        self.last_axis = axis
                except Exception:
                    self.fault = "Steering executor failed"
                    device.disengage()
                    self.reset()

            # Includes the closed/shutdown latch and device identity check.
            self.pad.write_output(write)

    def take_fault(self):
        with self._lock:
            fault, self.fault = self.fault, ""
            return fault

    def _run(self):
        due = self.clock.now() + STEERING_PERIOD_S
        while not self._stop.wait(max(0., due - self.clock.now())):
            self.tick(self.clock.now())
            due += STEERING_PERIOD_S
            now = self.clock.now()
            if now > due:
                due += math.ceil((now - due) / STEERING_PERIOD_S) * STEERING_PERIOD_S

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.reset()
