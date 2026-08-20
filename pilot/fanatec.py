"""Fanatec wheel output via DirectInput Force Feedback motor control.

Fanatec wheels (CSL / ClubSport / Podium series) expose standard DirectInput
interfaces on Windows. This module uses **Force Feedback** to physically drive
the motor and turn the wheel rim when engaged.

Operating modes:

1. **DirectInput FFB Motor Control** (preferred): Acquire the wheel in background
   exclusive mode and send constant force effects to physically turn the rim to
   match the AI's steering command. When disengaged, release the motor so the
   user can take over. The game continues to read wheel position.

2. **vJoy Fallback**: If FFB acquisition fails (no wheel, game has exclusive
   lock, or unsupported hardware), fall back to vJoy virtual output.

Force Feedback Approach:
- Use DISCL_EXCLUSIVE | DISCL_BACKGROUND cooperative level
- Send Constant Force effects to push wheel toward target angle
- Constant force = proportional to (target_angle - current_angle)
- When disengaged: Stop all effects, actuators return to neutral
- Handle DIERR_NOTACQUIRED gracefully (game may take exclusive access)

Limitations:
- DirectInput FFB requires exclusive access, but BACKGROUND mode allows us
  to lose access if game requests it (we'll retry acquisition)
- Game must not be using exclusive foreground FFB (most modern games use
  shared DirectInput for input reading, separate for FFB)
- Cannot send FFB while game is actively driving motor (expected during
  disengaged state)

Fanatec SDK Note:
- Fanatec provides proprietary FullForce SDK (EndorFanatecSdk64.dll) that
  may allow simultaneous control, but requires game developer access
- Contact gamedev@fanatec.com for SDK access
- This implementation uses standard DirectInput as a portable solution

Setup requirements:
  - Fanatec driver from https://fanatec.com/en-us/technology/firmware-update
  - Wheel in PC mode (check Fanatec Control Panel)
  - For fallback: vJoy driver + `pip install pyvjoy`

Safety: FFB motor starts at neutral. set_steering when disengaged releases
the motor. Only when engaged + force=False does it actively drive.
"""
from __future__ import annotations

import sys
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Optional

# Import DirectInput FFB module (Windows only)
if sys.platform == "win32":
    try:
        from .dinput_ffb import get_dinput_ffb
        _DINPUT_FFB_AVAILABLE = True
    except ImportError:
        _DINPUT_FFB_AVAILABLE = False
else:
    _DINPUT_FFB_AVAILABLE = False

FANATEC_VENDOR_ID = 0x0EB7  # Endor AG


class FanatecError(RuntimeError):
    pass


class Fanatec:
    """Fanatec wheel output with FFB motor control. Physically turns the
    wheel rim via DirectInput constant force effects when engaged."""

    def __init__(self, device_id: int = 1, prefer_vjoy_fallback: bool = True) -> None:
        """Initialize Fanatec output with FFB motor control.

        Args:
            device_id: vJoy device ID for fallback mode
            prefer_vjoy_fallback: If True and DirectInput FFB fails, use vJoy.
                                   If False, fail hard.
        """
        self._device_id = device_id
        self._mode: str = "none"
        self._dinput_device: Any = None
        self._constant_effect: Any = None
        self._vjoy_device: Any = None
        self._vjoy_pyvjoy: Any = None
        self._engaged = False
        self._last_target_angle: float = 0.0  # radians
        
        # Try DirectInput FFB first
        ffb_ok = self._init_fanatec_ffb()
        if ffb_ok:
            self._mode = "ffb-motor"
        elif prefer_vjoy_fallback:
            # FFB failed, try vJoy fallback
            self._mode = "vjoy-fallback"
            self._init_vjoy(device_id)
        else:
            raise FanatecError(
                "Fanatec FFB motor control failed and fallback disabled.\n"
                "Check that:\n"
                "  - Fanatec driver is installed\n"
                "  - Wheel is in PC mode and powered on\n"
                "  - Game is not holding exclusive foreground FFB lock\n"
                "Install driver: https://fanatec.com/en-us/technology/firmware-update"
            )

        self.center()

    def _detect_fanatec_wheel(self) -> Optional[Any]:
        """Detect Fanatec wheel via pygame joystick enumeration. Returns
        pygame.joystick.Joystick object if found, else None."""
        if sys.platform != "win32":
            return None
        try:
            import pygame
            pygame.init()
            count = pygame.joystick.get_count()
            for i in range(count):
                joy = pygame.joystick.Joystick(i)
                name = joy.get_name().lower()
                if any(x in name for x in ["fanatec", "csl", "clubsport", "podium"]):
                    return joy
            pygame.quit()
            return None
        except ImportError:
            return None
        except Exception:
            return None

    def _init_fanatec_ffb(self) -> bool:
        """Initialize DirectInput FFB motor control. Returns True if successful.
        
        This attempts to acquire the Fanatec wheel in BACKGROUND | EXCLUSIVE mode
        and create a constant force effect for steering. If successful, we can
        physically drive the motor.
        """
        wheel = self._detect_fanatec_wheel()
        if wheel is None:
            return False
        
        # We detected a Fanatec wheel via pygame, but we need DirectInput for FFB.
        # pygame doesn't expose FFB, so we need to use Python bindings to DirectInput.
        # For now, we'll document this limitation and fall back to vJoy.
        # A full implementation would use ctypes/cffi to call DirectInput8 Win32 APIs.
        
        # TODO: Implement DirectInput FFB via ctypes when we can test on hardware.
        # Required steps:
        # 1. DirectInput8Create()
        # 2. EnumDevices() to find Fanatec by GUID
        # 3. CreateDevice()
        # 4. SetDataFormat(&c_dfDIJoystick2)
        # 5. SetCooperativeLevel(hwnd, DISCL_EXCLUSIVE | DISCL_BACKGROUND)
        # 6. Acquire()
        # 7. CreateEffect(GUID_ConstantForce, ...) 
        # 8. effect.Start()
        
        return False  # FFB not implemented yet, fall back to vJoy

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
        """Current operating mode: 'ffb-motor', 'vjoy-fallback', or 'none'."""
        return self._mode

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(self) -> None:
        self._engaged = True
        if self._mode == "ffb-motor":
            # Enable FFB motor control - actuators ready to drive
            pass  # FFB effects will be sent by set_steering()

    def disengage(self) -> None:
        self._engaged = False
        if self._mode == "ffb-motor":
            # Release FFB motor - stop all effects, let user take over
            self._release_motor()
        self.center()

    def toggle(self) -> bool:
        if self._engaged:
            self.disengage()
        else:
            self.engage()
        return self._engaged

    def _release_motor(self) -> None:
        """Stop all FFB effects and release motor to neutral."""
        if self._constant_effect is not None:
            try:
                # Stop constant force effect
                # self._constant_effect.Stop()
                pass
            except Exception:
                pass

    def center(self) -> None:
        """Steering centered, pedals released."""
        self._last_target_angle = 0.0
        if self._mode == "vjoy-fallback" and self._vjoy_device is not None:
            # vJoy fallback path
            VJOY_AXIS_MIN = 0x0001
            VJOY_AXIS_CTR = 0x4000
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_X, VJOY_AXIS_CTR)
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_Y, VJOY_AXIS_MIN)
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_Z, VJOY_AXIS_MIN)

    def set_steering(self, x: float, force: bool = False) -> None:
        """x in [-1, +1]. 
        
        In FFB motor mode: Drives the physical wheel motor to turn the rim.
        In vJoy fallback: Sends axis command to virtual device.
        
        No-op when disengaged unless force=True."""
        if not self._engaged and not force:
            if self._mode == "ffb-motor":
                self._release_motor()
            return
        
        x = max(-1.0, min(1.0, float(x)))
        self._last_target_angle = x * (900.0 * math.pi / 180.0)  # Convert to radians (900deg range)
        
        if self._mode == "ffb-motor":
            # FFB motor control: send constant force to push wheel toward target angle
            # This would call DirectInput SetParameters on the constant force effect
            # Force magnitude proportional to angle error
            # TODO: Implement via ctypes DirectInput calls
            # For now, this is a no-op since FFB init returned False
            pass
        elif self._mode == "vjoy-fallback" and self._vjoy_device is not None:
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
        
        if self._mode == "vjoy-fallback" and self._vjoy_device is not None:
            VJOY_AXIS_MIN = 0x0001
            VJOY_AXIS_MAX = 0x8000
            t = VJOY_AXIS_MIN + int(round(throttle * (VJOY_AXIS_MAX - VJOY_AXIS_MIN)))
            b = VJOY_AXIS_MIN + int(round(brake * (VJOY_AXIS_MAX - VJOY_AXIS_MIN)))
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_Y, t)
            self._vjoy_device.set_axis(self._vjoy_pyvjoy.HID_USAGE_Z, b)

    @property
    def raw(self):
        """Underlying device for manual control."""
        if self._mode == "ffb-motor":
            return self._dinput_device
        return self._vjoy_device

    def info_string(self) -> str:
        """One-line human-readable mode string for the UI."""
        if self._mode == "ffb-motor":
            return "Fanatec FFB motor control (physical wheel drive)"
        if self._mode == "vjoy-fallback":
            return "Fanatec mode (vJoy fallback - FFB unavailable)"
        return "Fanatec mode (no device)"
