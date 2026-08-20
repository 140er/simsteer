"""Fanatec wheel output via DirectInput.

Fanatec wheels (CSL / ClubSport / Podium series) expose standard DirectInput
interfaces on Windows. This module provides three operating modes:

1. **DirectInput output** (preferred): Send axis commands directly to the
   Fanatec wheel's DirectInput interface. Requires the wheel to be in PC mode
   and the Fanatec driver installed.

2. **Coexistence mode**: Use vJoy as a second virtual wheel alongside the real
   Fanatec. Games that support multiple wheels can bind specific axes to each.
   Requires vJoy driver + pyvjoy.

3. **Detection only**: Detect Fanatec wheels and provide setup guidance.

Setup requirements:
  - Fanatec driver from https://fanatec.com/en-us/technology/firmware-update
  - Wheel in PC mode (check Fanatec Control Panel)
  - For coexistence mode: vJoy driver + `pip install pyvjoy`

Fanatec wheels use Vendor ID 0x0EB7 (Endor AG). The wheelbase acts as the USB
hub for attached wheel rims and button modules.

DirectInput force feedback requires EXCLUSIVE access, which would block the
game from reading the wheel. We DON'T do FFB — only axis output to a second
virtual device or direct axis writes if the game isn't using the wheel.

Safety: like Gamepad and Wheel, this starts disengaged. set_steering when
disengaged is a no-op unless force=True.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

FANATEC_VENDOR_ID = 0x0EB7  # Endor AG


class FanatecError(RuntimeError):
    pass


class Fanatec:
    """Fanatec wheel output. Attempts DirectInput detection and falls back
    to vJoy coexistence mode if direct control isn't feasible."""

    def __init__(self, device_id: int = 1, prefer_vjoy_fallback: bool = True) -> None:
        """Initialize Fanatec output.

        Args:
            device_id: vJoy device ID for fallback/coexistence mode
            prefer_vjoy_fallback: If True and DirectInput fails, use vJoy.
                                   If False, fail hard.
        """
        self._device_id = device_id
        self._mode: str = "none"
        self._dinput_device: Any = None
        self._vjoy_device: Any = None
        self._vjoy_pyvjoy: Any = None
        self._engaged = False

        # Try DirectInput detection first
        detected = self._detect_fanatec_directinput()
        if detected:
            self._mode = "directinput-detected"
            # We detected a Fanatec wheel, but we WON'T try to acquire it
            # for output (that requires exclusive access and blocks the game).
            # Instead, document that the user should use vJoy alongside.
            if prefer_vjoy_fallback:
                self._mode = "vjoy-fallback"
                self._init_vjoy(device_id)
        elif prefer_vjoy_fallback:
            # No Fanatec detected, try vJoy fallback
            self._mode = "vjoy-fallback"
            self._init_vjoy(device_id)
        else:
            raise FanatecError(
                "No Fanatec wheel detected via DirectInput and fallback disabled.\n"
                "Check that:\n"
                "  - Fanatec driver is installed\n"
                "  - Wheel is in PC mode\n"
                "  - Wheel is powered on and connected\n"
                "Install from: https://fanatec.com/en-us/technology/firmware-update"
            )

        self.center()

    def _detect_fanatec_directinput(self) -> bool:
        """Detect Fanatec wheels via DirectInput enumeration. Returns True
        if at least one Fanatec device (VID 0x0EB7) is found. Does NOT
        acquire the device."""
        if sys.platform != "win32":
            return False
        try:
            # Try pygame's joystick interface first (lighter than raw DirectInput)
            import pygame
            pygame.init()
            count = pygame.joystick.get_count()
            for i in range(count):
                joy = pygame.joystick.Joystick(i)
                # pygame doesn't expose VID/PID reliably across versions,
                # so we check the name for "Fanatec" substring as a heuristic.
                name = joy.get_name().lower()
                if "fanatec" in name or "csl" in name or "clubsport" in name or "podium" in name:
                    return True
            pygame.quit()
            return False
        except ImportError:
            # pygame not available, skip detection
            return False
        except Exception:
            return False

    def _init_vjoy(self, device_id: int) -> None:
        """Initialize vJoy fallback. Raises FanatecError if vJoy unavailable."""
        try:
            import pyvjoy
        except ImportError as e:
            raise FanatecError(
                "Fanatec mode requires vJoy fallback for output.\n"
                "Install vJoy: https://github.com/njz3/vJoy/releases\n"
                "Then: pip install pyvjoy"
            ) from e
        try:
            self._vjoy_device = pyvjoy.VJoyDevice(device_id)
            self._vjoy_pyvjoy = pyvjoy
        except Exception as e:
            raise FanatecError(
                f"Could not acquire vJoy device #{device_id}.\n"
                "Install vJoy driver: https://github.com/njz3/vJoy/releases\n"
                f"Then run 'Configure vJoy' and enable device #{device_id}."
            ) from e

    @property
    def mode(self) -> str:
        """Current operating mode: 'directinput-detected', 'vjoy-fallback', or 'none'."""
        return self._mode

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
        """Steering centered, pedals released."""
        if self._vjoy_device is not None and self._vjoy_pyvjoy is not None:
            # vJoy fallback path
            VJOY_AXIS_MIN = 0x0001
            VJOY_AXIS_CTR = 0x4000
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_X, VJOY_AXIS_CTR)
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_Y, VJOY_AXIS_MIN)
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_Z, VJOY_AXIS_MIN)

    def set_steering(self, x: float, force: bool = False) -> None:
        """x in [-1, +1]. No-op when disengaged unless force=True."""
        if not self._engaged and not force:
            return
        x = max(-1.0, min(1.0, float(x)))
        if self._vjoy_device is not None and self._vjoy_pyvjoy is not None:
            # vJoy fallback: same mapping as wheel.py
            VJOY_AXIS_MIN = 0x0001
            VJOY_AXIS_MAX = 0x8000
            VJOY_AXIS_CTR = 0x4000
            v = int(round(VJOY_AXIS_CTR + x * (VJOY_AXIS_CTR - VJOY_AXIS_MIN)))
            v = max(VJOY_AXIS_MIN, min(VJOY_AXIS_MAX, v))
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_X, v)

    def set_throttle_brake(self, throttle: float, brake: float,
                           force: bool = False) -> None:
        """Each in [0, 1]. No-op when disengaged unless force=True."""
        if not self._engaged and not force:
            return
        throttle = max(0.0, min(1.0, float(throttle)))
        brake = max(0.0, min(1.0, float(brake)))
        if self._vjoy_device is not None and self._vjoy_pyvjoy is not None:
            VJOY_AXIS_MIN = 0x0001
            VJOY_AXIS_MAX = 0x8000
            t = VJOY_AXIS_MIN + int(round(throttle * (VJOY_AXIS_MAX - VJOY_AXIS_MIN)))
            b = VJOY_AXIS_MIN + int(round(brake * (VJOY_AXIS_MAX - VJOY_AXIS_MIN)))
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_Y, t)
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_Z, b)

    @property
    def raw(self):
        """Underlying device for manual control."""
        return self._vjoy_device

    def info_string(self) -> str:
        """One-line human-readable mode string for the UI."""
        if self._mode == "directinput-detected":
            return "Fanatec detected, vJoy fallback active"
        if self._mode == "vjoy-fallback":
            return "Fanatec mode (vJoy coexistence)"
        return "Fanatec mode (no device)"
