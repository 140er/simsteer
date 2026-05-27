"""Virtual Xbox 360 controller via ViGEm.

We only drive the left-stick X axis (steering). Throttle and brake stay
with the player / cruise control for v0; adding them later is a two-line
change to `set_throttle_brake`.

Safety: the gamepad starts disengaged and centered. Calling `set_steering`
when disengaged is a no-op — the only place that flips engagement is the
main loop's hotkey handler.

Requires the ViGEm Bus driver to be installed system-wide; if it's not,
constructing the gamepad raises a clear error.
"""

from __future__ import annotations


class GamepadError(RuntimeError):
    pass


class Gamepad:
    def __init__(self) -> None:
        try:
            import vgamepad as vg
        except ImportError as e:
            raise GamepadError(f"vgamepad not installed: {e}") from e

        try:
            self._pad = vg.VX360Gamepad()
        except Exception as e:
            raise GamepadError(
                "could not create virtual Xbox 360 pad — is the ViGEm Bus "
                "driver installed? https://github.com/nefarius/ViGEmBus/releases"
            ) from e

        self._engaged = False
        self.center()

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(self) -> None:
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False
        self.center()

    def toggle(self) -> bool:
        if self._engaged:
            self.disengage()
        else:
            self.engage()
        return self._engaged

    def center(self) -> None:
        """Force all controls to neutral and flush."""
        self._pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self._pad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self._pad.left_trigger_float(value_float=0.0)
        self._pad.right_trigger_float(value_float=0.0)
        self._pad.update()

    def set_steering(self, x: float, force: bool = False) -> None:
        """x in [-1, 1]. No-op when disengaged, unless `force=True` —
        used by manual override so the tuner steer slider drives the
        pad even when the AI is disengaged."""
        if not self._engaged and not force:
            return
        x = max(-1.0, min(1.0, float(x)))
        self._pad.left_joystick_float(x_value_float=x, y_value_float=0.0)
        self._pad.update()

    def set_throttle_brake(self, throttle: float, brake: float,
                           force: bool = False) -> None:
        """Right trigger = throttle, left trigger = brake; each in [0, 1].
        No-op when disengaged, unless `force=True` — used by manual
        override so the tuner throttle/brake sliders drive the pad even
        when the AI is disengaged."""
        if not self._engaged and not force:
            return
        throttle = max(0.0, min(1.0, float(throttle)))
        brake = max(0.0, min(1.0, float(brake)))
        self._pad.right_trigger_float(value_float=throttle)
        self._pad.left_trigger_float(value_float=brake)
        self._pad.update()

    @property
    def raw(self):
        """Underlying vgamepad.VX360Gamepad — used by the bind-wiggle hotkeys
        in main.py so they share the same pad (and therefore the same XInput
        slot) as the steering output."""
        return self._pad
