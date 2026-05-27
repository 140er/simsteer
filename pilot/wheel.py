"""Virtual steering wheel via vJoy. Also handles bind-wiggling for
ETS2's controller setup wizard — it ignores any axis that isn't
actively moving, so we drive the requested axis up/down for a few
seconds while the user clicks the binding slot in-game.


Why: ETS2's gamepad input pipeline applies speed-sensitive rack assist
(the truck rack stiffens at speed, which we modelled with the `b · v²`
term in LiveParams). A real steering wheel skips this — ETS2 treats
wheel-mode axes as a direct linear mapping. Same for most sims that
distinguish "wheel" vs "gamepad" input categories.

By presenting as a vJoy DirectInput device the game sees us as a
wheel, the speed-stiffness disappears, and `axis · scale = wheel_angle`
becomes (close to) linear. The whole `b` term in LiveParams drops to
near zero and the AI can actually reach full lock.

Setup the user must do once:
  1. Install the vJoy driver: https://sourceforge.net/projects/vjoystick/
  2. Run `Configure vJoy`. Enable device #1, give it at least 3 axes
     (X, Y, Z) and 4 buttons.
  3. `pip install pyvjoy`
  4. In the game, bind the vJoy device. ETS2: detect it as a wheel
     and set steering linearity to ZERO. AC: same.

Then run with `--device wheel`.

Axis layout (matches typical sim-wheel + pedals):
  - X axis: steering (0 = full left, 0x8000 = center, 0xFFFF = full right)
    Internally we use [-1, +1] like the gamepad.
  - Y axis: throttle (0 = no throttle, 0x8000 = full)
  - Z axis: brake    (0 = no brake,    0x8000 = full)

Same interface as `pilot.gamepad.Gamepad` so main.py can swap them.
"""
from __future__ import annotations


# vJoy axis values are 16-bit unsigned. Range 0x0001..0x8000, with
# 0x4000 as center for a bidirectional axis like steering. Some
# pyvjoy builds use 0..0x8000 (32768) — we stick to that range.
VJOY_AXIS_MIN = 0x0001
VJOY_AXIS_MAX = 0x8000
VJOY_AXIS_CTR = 0x4000


class WheelError(RuntimeError):
    pass


class Wheel:
    """Virtual steering wheel + pedals via vJoy. Matches the Gamepad
    interface so the main loop can use either."""

    def __init__(self, device_id: int = 1) -> None:
        try:
            import pyvjoy
        except ImportError as e:
            raise WheelError(
                "pyvjoy not installed. Run `pip install pyvjoy` and install "
                "the vJoy driver from https://sourceforge.net/projects/vjoystick/"
            ) from e

        try:
            self._dev = pyvjoy.VJoyDevice(device_id)
        except Exception as e:
            raise WheelError(
                f"could not acquire vJoy device #{device_id} — is the vJoy "
                f"driver installed and is device #{device_id} enabled in "
                f"`Configure vJoy`?"
            ) from e

        self._pyvjoy = pyvjoy
        self._engaged = False
        self.center()

    # ----- engagement -----

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

    # ----- output -----

    def center(self) -> None:
        """Steering centered, pedals released."""
        self._dev.set_axis(self._pyvjoy.HID_USAGE_X, VJOY_AXIS_CTR)
        self._dev.set_axis(self._pyvjoy.HID_USAGE_Y, VJOY_AXIS_MIN)
        self._dev.set_axis(self._pyvjoy.HID_USAGE_Z, VJOY_AXIS_MIN)

    def set_steering(self, x: float, force: bool = False) -> None:
        """x in [-1, +1]. Maps to the wheel's X axis with linear units:
        no deadzone, no curve — the entire point of using a wheel.
        No-op when disengaged, unless `force=True` — used by manual
        override so the tuner steer slider drives the wheel even when
        the AI is disengaged."""
        if not self._engaged and not force:
            return
        x = max(-1.0, min(1.0, float(x)))
        # Linear map [-1, +1] -> [VJOY_AXIS_MIN, VJOY_AXIS_MAX]
        v = int(round(VJOY_AXIS_CTR + x * (VJOY_AXIS_CTR - VJOY_AXIS_MIN)))
        v = max(VJOY_AXIS_MIN, min(VJOY_AXIS_MAX, v))
        self._dev.set_axis(self._pyvjoy.HID_USAGE_X, v)

    def set_throttle_brake(self, throttle: float, brake: float,
                           force: bool = False) -> None:
        """Each in [0, 1]. Y = throttle pedal, Z = brake pedal — sim
        wheels typically expose these as separate analog axes. No-op
        when disengaged, unless `force=True` — used by manual override
        so the tuner pedal sliders drive the wheel even when the AI is
        disengaged."""
        if not self._engaged and not force:
            return
        throttle = max(0.0, min(1.0, float(throttle)))
        brake = max(0.0, min(1.0, float(brake)))
        t = VJOY_AXIS_MIN + int(round(throttle * (VJOY_AXIS_MAX - VJOY_AXIS_MIN)))
        b = VJOY_AXIS_MIN + int(round(brake * (VJOY_AXIS_MAX - VJOY_AXIS_MIN)))
        self._dev.set_axis(self._pyvjoy.HID_USAGE_Y, t)
        self._dev.set_axis(self._pyvjoy.HID_USAGE_Z, b)

    # ----- compatibility -----

    @property
    def raw(self):
        """Underlying VJoyDevice. The bind-wiggle hotkeys in main.py
        target the Gamepad's XInput buttons — they don't work against
        a vJoy device. Wheel users bind via the game's own controller
        config screen instead. Returning the underlying device anyway
        so callers can poke at it directly if needed."""
        return self._dev

    # ----- bind-wiggle (ETS2 / AC controller wizard) -----

    # Vocabulary the tuner uses to label the buttons. Maps to specific
    # vJoy axes / button indices.
    WIGGLE_INPUTS: tuple[tuple[str, str], ...] = (
        ("steering", "Steering (X axis — full left ↔ full right)"),
        ("throttle", "Throttle (Y axis — 0 ↔ full)"),
        ("brake",    "Brake (Z axis — 0 ↔ full)"),
        ("button_1", "Button 1"),
        ("button_2", "Button 2"),
        ("button_3", "Button 3"),
        ("button_4", "Button 4"),
    )

    def _press_button(self, idx: int, down: bool) -> None:
        # pyvjoy's set_button is 1-indexed.
        self._dev.set_button(idx, 1 if down else 0)

    def wiggle(self, kind: str, duration_each_s: float = 1.0) -> None:
        """Drive the named input on/off so the game's binding wizard
        detects it. Bypasses the engagement gate — this is for setup,
        not live driving. Caller should run this in a background
        thread (it blocks for up to ~3 s).

        Axes wiggle bipolar (full +, then full -) so ETS2 sees an
        analog range and binds correctly. Pedals wiggle 0 -> full -> 0.
        Buttons just press and release once.
        """
        import time as _t
        if kind == "steering":
            self._dev.set_axis(self._pyvjoy.HID_USAGE_X, VJOY_AXIS_MAX)
            _t.sleep(duration_each_s)
            self._dev.set_axis(self._pyvjoy.HID_USAGE_X, VJOY_AXIS_CTR)
            _t.sleep(0.2)
            self._dev.set_axis(self._pyvjoy.HID_USAGE_X, VJOY_AXIS_MIN)
            _t.sleep(duration_each_s)
            self._dev.set_axis(self._pyvjoy.HID_USAGE_X, VJOY_AXIS_CTR)
            return
        if kind == "throttle":
            self._dev.set_axis(self._pyvjoy.HID_USAGE_Y, VJOY_AXIS_MAX)
            _t.sleep(duration_each_s)
            self._dev.set_axis(self._pyvjoy.HID_USAGE_Y, VJOY_AXIS_MIN)
            return
        if kind == "brake":
            self._dev.set_axis(self._pyvjoy.HID_USAGE_Z, VJOY_AXIS_MAX)
            _t.sleep(duration_each_s)
            self._dev.set_axis(self._pyvjoy.HID_USAGE_Z, VJOY_AXIS_MIN)
            return
        if kind.startswith("button_"):
            try:
                idx = int(kind.split("_", 1)[1])
            except ValueError:
                return
            self._press_button(idx, True)
            _t.sleep(duration_each_s)
            self._press_button(idx, False)
            return
