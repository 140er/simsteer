"""Non-blocking binding pulses, advanced only by the driving loop."""

from __future__ import annotations

import math

from simsteer.calib.steering import OutputDirection


class Binding:
    def __init__(self):
        self.request = None
        self.started = None
        self.button = None

    def cancel(self, manager):
        if self.request is not None:
            backend = self.request[1]
            if self.button is not None:
                if manager.is_wheel:
                    backend._press_button(self.button, False)
                else:
                    backend.raw.release_button(button=self.button)
            backend.center()
        self.request = self.started = self.button = None

    def tick(self, manager, now, speed, manual_override):
        if self.request is None:
            return False
        kind, backend = self.request
        if (manager.engaged or manager.device is not backend or manual_override
                or speed is None or not math.isfinite(speed)
                or abs(speed) >= OutputDirection.STOP_SPEED_MPS):
            self.cancel(manager)
            return False
        if self.started is None:
            self.started = now
            backend.center()
        elapsed = now - self.started - 4.
        if elapsed < 0:
            return True
        button = kind.startswith("button_")
        pedal = "trigger" in kind or kind in ("throttle", "brake")
        duration = 1.2 if pedal else 1. if button else 2.3
        if elapsed >= duration:
            self.cancel(manager)
            return True
        value = 1. if elapsed < 1. or pedal or button else 0. if elapsed < 1.3 else -1.
        if manager.is_wheel:
            if kind == "steering":
                backend.set_steering(value, force=True)
            elif kind in ("throttle", "brake"):
                backend.set_throttle_brake(value if kind == "throttle" else 0.,
                                           value if kind == "brake" else 0., force=True)
            elif button:
                self.button = int(kind.split("_", 1)[1])
                backend._press_button(self.button, True)
        else:
            raw = backend.raw
            if kind == "left_stick_x":
                backend.set_steering(value, force=True)
            elif kind == "right_stick_x":
                raw.right_joystick_float(x_value_float=value, y_value_float=0.)
                raw.update()
            elif pedal:
                backend.set_throttle_brake(value if kind == "right_trigger" else 0.,
                                           value if kind == "left_trigger" else 0., force=True)
            elif button:
                import vgamepad as vg
                suffix = {"lb": "LEFT_SHOULDER", "rb": "RIGHT_SHOULDER"}.get(
                    kind[7:], kind[7:].upper())
                self.button = getattr(vg.XUSB_BUTTON, "XUSB_GAMEPAD_" + suffix)
                raw.press_button(button=self.button)
                raw.update()
        return True
