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
from threading import RLock

from simsteer.runtime.output.gamepad import Gamepad, GamepadError
from simsteer.runtime.output.wheel import Wheel, WheelError
from simsteer.runtime.output.safety import OutputSafety, output_locked

DeviceKind = Literal["gamepad", "wheel"]


class DeviceManager(OutputSafety):
    """Single point of contact for whichever virtual controller is
    active. Public interface mirrors `Gamepad` / `Wheel` so callers
    can use a `DeviceManager` anywhere either was previously used."""

    def __init__(self, initial_kind: DeviceKind = "gamepad",
                 vjoy_device_id: int = 1, *, passthrough=False, settings=None) -> None:
        if settings is None and passthrough:
            from simsteer.app.settings import Settings
            settings = Settings()
        self._vjoy_id = vjoy_device_id
        self._passthrough, self._settings = passthrough, settings
        self._kind: DeviceKind | None = None
        self._device: Gamepad | Wheel | None = None
        self.last_error: str | None = None
        self._lifecycle_lock = RLock()
        self._shutdown_requested = False
        self._init_safety()
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
        # Never hold the output lock while waiting for the relay owner.
        with self._lifecycle_lock:
            return self._set_kind(kind)

    def _set_kind(self, kind: DeviceKind) -> bool:
        """Switch to a different output device. Disengages and releases
        the current one before acquiring the new one. Returns True on
        success; on failure leaves whatever was active untouched and
        sets `self.last_error`."""
        if self._closed or self._shutdown_requested:
            return False
        if kind == self._kind:
            return True
        self.cancel_bind()
        # Save was-engaged state so a swap mid-drive doesn't drop control.
        was_engaged = bool(self._device and self._device.engaged)
        # Tear down current device first — vJoy needs exclusive access,
        # ViGEm doesn't care but it's cleaner.
        if self._device is not None:
            self.release_device()
        # Acquire new device.
        try:
            if kind == "gamepad":
                self._device = self.write_output(lambda _: Gamepad())
            elif kind == "wheel":
                if self._passthrough:
                    from simsteer.runtime.output.passthrough import PassthroughWheel
                    self._device = PassthroughWheel(self._settings, self._vjoy_id,
                                                     write_output=self.write_output)
                    if not self._device.available:
                        self._device.close()
                        self._device = None
                        print("No Fanatec wheel; using plain vJoy")
                if self._device is None:
                    self._device = self.write_output(lambda _: Wheel(device_id=self._vjoy_id))
            else:
                self.last_error = f"unknown device kind: {kind}"
                return False
        except (GamepadError, WheelError) as e:
            self.last_error = str(e)
            return False
        self._kind = kind
        self.last_error = None
        if was_engaged and not self.passthrough:
            self.engage()
        return True

    # ----- delegate the controller API -----

    @property
    def engaged(self) -> bool:
        return bool(self._device and self._device.engaged)

    @output_locked
    def engage(self) -> bool | None:
        self.cancel_bind()
        if self._device is not None:
            return self._device.engage()

    @output_locked
    def disengage(self) -> None:
        self.cancel_bind()
        if self.passthrough:
            self._device.stop_output()
        elif self._device is not None:
            self._device.disengage()

    @output_locked
    def insert_disengage(self) -> None:
        if not self.engaged:
            return
        self.cancel_bind()
        if self.passthrough:
            self._device.disengage("insert")
        else:
            self.disengage()

    @property
    def passthrough(self):
        return hasattr(self._device, "passthrough_live")

    @property
    def passthrough_live(self):
        return getattr(self._device, "passthrough_live", False)

    @property
    def physical(self):
        return getattr(self._device, "physical", None)

    @property
    def last_reason(self):
        return getattr(self._device, "last_reason", "")

    @property
    def pedal_status(self):
        return getattr(self._device, "pedal_status", "")

    @property
    def source(self):
        return getattr(self._device, "source", None)

    def stats(self, **kwargs):
        return self._device.stats(**kwargs) if self.passthrough else {}

    def map_pedals(self):
        return self._device.map_pedals() if self.passthrough else False

    def release_device(self):
        """Release for a swap; permanent exit uses shutdown_output()."""
        with self._lifecycle_lock:
            if self._device:
                if self.passthrough:
                    self._device.close()
                else:
                    self.write_output(lambda dev: getattr(dev, "close", dev.disengage)())
                with self._output_lock:
                    self._device, self._kind = None, None

    def shutdown_output(self):
        with self._lifecycle_lock:
            with self._output_lock:
                self._shutdown_requested = True
                if self.passthrough:
                    self._device.io.stopping.set()
            if self.passthrough:
                self._device.close()
            super().shutdown_output()

    @output_locked
    def resume_output(self):
        if self.passthrough:
            self._device.resume_output()

    @output_locked
    def toggle(self) -> bool:
        self.cancel_bind()
        if self._device is None:
            return False
        return self._device.toggle()

    @output_locked
    def center(self) -> None:
        if self._device is not None:
            self._device.center()

    @output_locked
    def set_steering(self, x: float, force: bool = False) -> None:
        if self._device is not None:
            self._device.set_steering(x, force=force)

    @output_locked
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
