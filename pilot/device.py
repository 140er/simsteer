"""Output-device manager.

Holds whichever virtual controller is currently driving the truck —
either a `Gamepad` (ViGEm Xbox 360, XInput) or a `Wheel` (vJoy
DirectInput). Same public interface as either underlying device, so
the main loop and tuner only need a single `pad: DeviceManager`
reference and don't have to reach into the implementation.

Live swap: `set_kind("wheel")` releases the current device and tries
to acquire a vJoy. If that fails (driver missing, device busy) the
manager stays on whatever was active and reports the failure — never
leaves the caller with a dead pad.

The two underlying devices differ subtly:
  - Gamepad uses XInput sticks/triggers. ETS2 routes it through the
    gamepad rack with speed-sensitive assist (the `b · v²` term in
    LiveParams comes from this).
  - Wheel uses vJoy axes. ETS2 treats it as a real wheel and skips
    the rack-assist nonlinearity. Set ETS2 "Steering non-linearity =
    0" in controller options to get a fully linear axis -> wheel.

When you change device kind, the LiveParams `a` / `b` fit you have
will be wrong for the new device — the rack response curve changed.
Reset the fit (`Reset RLS` button in tuner) after a swap.
"""
from __future__ import annotations

from typing import Literal

from pilot.gamepad import Gamepad, GamepadError
from pilot.wheel import Wheel, WheelError

DeviceKind = Literal["gamepad", "wheel"]


class DeviceManager:
    """Single point of contact for whichever virtual controller is
    active. Public interface mirrors `Gamepad` / `Wheel` so callers
    can use a `DeviceManager` anywhere either was previously used."""

    def __init__(self, initial_kind: DeviceKind = "gamepad",
                 vjoy_device_id: int = 1) -> None:
        self._vjoy_id = vjoy_device_id
        self._kind: DeviceKind | None = None
        self._device: Gamepad | Wheel | None = None
        self.last_error: str | None = None
        # Best effort — if initial_kind fails (e.g. no vJoy driver), the
        # caller can see `self.kind is None` and react.
        self.set_kind(initial_kind)

    # ----- selection -----

    @property
    def kind(self) -> DeviceKind | None:
        """Currently active device kind, or None if construction failed."""
        return self._kind

    @property
    def is_gamepad(self) -> bool:
        return self._kind == "gamepad"

    @property
    def is_wheel(self) -> bool:
        return self._kind == "wheel"

    def set_kind(self, kind: DeviceKind) -> bool:
        """Switch to a different output device. Disengages and releases
        the current one before acquiring the new one. Returns True on
        success; on failure leaves whatever was active untouched and
        sets `self.last_error`."""
        if kind == self._kind:
            return True
        # Save was-engaged state so a swap mid-drive doesn't drop control.
        was_engaged = bool(self._device and self._device.engaged)
        # Tear down current device first — vJoy needs exclusive access,
        # ViGEm doesn't care but it's cleaner.
        if self._device is not None:
            try:
                self._device.disengage()
            except Exception:
                pass
            self._device = None
            self._kind = None
        # Acquire new device.
        try:
            if kind == "gamepad":
                self._device = Gamepad()
            elif kind == "wheel":
                self._device = Wheel(device_id=self._vjoy_id)
            else:
                self.last_error = f"unknown device kind: {kind}"
                return False
        except (GamepadError, WheelError) as e:
            self.last_error = str(e)
            return False
        self._kind = kind
        self.last_error = None
        if was_engaged:
            self._device.engage()
        return True

    # ----- delegate the controller API -----

    @property
    def engaged(self) -> bool:
        return bool(self._device and self._device.engaged)

    def engage(self) -> None:
        if self._device is not None:
            self._device.engage()

    def disengage(self) -> None:
        if self._device is not None:
            self._device.disengage()

    def toggle(self) -> bool:
        if self._device is None:
            return False
        return self._device.toggle()

    def center(self) -> None:
        if self._device is not None:
            self._device.center()

    def set_steering(self, x: float, force: bool = False) -> None:
        if self._device is not None:
            self._device.set_steering(x, force=force)

    def set_throttle_brake(self, throttle: float, brake: float,
                           force: bool = False) -> None:
        if self._device is not None:
            self._device.set_throttle_brake(throttle, brake, force=force)

    @property
    def raw(self):
        """Underlying device — used by the bind-wiggle hotkeys for the
        gamepad path. Wheel's `raw` returns the vJoy device but the
        wiggle helper doesn't know how to drive it."""
        return self._device.raw if self._device is not None else None

    @property
    def device(self) -> Gamepad | Wheel | None:
        """The actual underlying object — useful for `isinstance` checks."""
        return self._device
