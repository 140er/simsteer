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

import ctypes
import time


class GamepadError(RuntimeError):
    pass


class Gamepad:
    def __init__(self) -> None:
        self._engaged = False
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

        try:
            try:
                self._wait_for_ready()
                # ViGEm's boot packet differs from its all-zero cache, so
                # repeated neutral reports can be discarded. Change only
                # one raw LSB of the unused right stick to force delivery.
                self._pad.reset()
                self._pad.right_joystick(x_value=1, y_value=0)
                self._pad.update()
            finally:
                self.center()
        except Exception as e:
            # vgamepad has no public close. Detach now: a traceback can
            # keep the transport alive until well after construction fails.
            # Its destructor still owns freeing the detached target.
            try:
                from vgamepad.win import vigem_client
                vigem_client.vigem_target_remove(self._pad._busp, self._pad._devicep)
            finally:
                self._pad = None
                raise GamepadError("could not initialize a neutral virtual gamepad") from e

    def _wait_for_ready(self) -> None:
        """Wait for this target's XInput slot before submitting reports.

        Reports sent before USB startup finishes can be lost even though
        the transport accepts them, especially when recreating a pad.
        """
        from vgamepad.win import vigem_client, vigem_commons

        errors = vigem_commons.VIGEM_ERRORS
        index = ctypes.c_ulong()
        state = ctypes.create_string_buffer(16)  # XINPUT_STATE
        get_state = ctypes.WinDLL("xinput1_4.dll").XInputGetState
        get_state.argtypes = (ctypes.c_ulong, ctypes.c_void_p)
        get_state.restype = ctypes.c_ulong
        deadline = time.monotonic() + 2.0
        while True:
            error = vigem_client.vigem_target_x360_get_user_index(
                self._pad._busp, self._pad._devicep, ctypes.byref(index))
            if error == errors.VIGEM_ERROR_NONE:
                if index.value < 4 and get_state(index.value, state) == 0:
                    return
            elif error != errors.VIGEM_ERROR_XUSB_USERINDEX_OUT_OF_RANGE:
                raise GamepadError(f"could not read virtual gamepad slot: {error:#x}")
            if time.monotonic() >= deadline:
                raise GamepadError("virtual gamepad did not become ready")
            time.sleep(0.01)

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
        self._pad.reset()
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
