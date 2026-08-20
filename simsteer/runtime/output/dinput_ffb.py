"""Windows DirectInput Force Feedback bindings via ctypes.

Provides a Python interface to DirectInput8 for force feedback control.
This is used by the Fanatec module to physically drive the wheel motor.

Reference: https://learn.microsoft.com/en-us/previous-versions/windows/desktop/ee417816(v=vs.85)
"""
from __future__ import annotations

import sys
from ctypes import *
from ctypes.wintypes import *
import uuid

if sys.platform != "win32":
    raise ImportError("DirectInput FFB is Windows-only")

# DirectInput GUIDs and constants
DIRECTINPUT_VERSION = 0x0800

# Cooperative level flags
DISCL_EXCLUSIVE = 0x00000001
DISCL_NONEXCLUSIVE = 0x00000002
DISCL_FOREGROUND = 0x00000004
DISCL_BACKGROUND = 0x00000008

# Device type constants
DI8DEVCLASS_GAMECTRL = 4
DIEDFL_ATTACHEDONLY = 0x00000001
DIEDFL_FORCEFEEDBACK = 0x00000100

# Error codes
DI_OK = 0
DIERR_NOTACQUIRED = 0x8007001C
DIERR_INPUTLOST = 0x8007001E

# Force feedback constants
DIEFT_CONSTANTFORCE = 0x00000001
DIEFF_OBJECTOFFSETS = 0x00000002
DIEFF_CARTESIAN = 0x00000010
DIEFF_SPHERICAL = 0x00000020

DIEFF_START = 0x20000000
INFINITE = 0xFFFFFFFF

# SendForceFeedbackCommand flags
DISFFC_RESET = 0x00000001
DISFFC_STOPALL = 0x00000002
DISFFC_PAUSE = 0x00000004
DISFFC_CONTINUE = 0x00000008
DISFFC_SETACTUATORSON = 0x00000010
DISFFC_SETACTUATORSOFF = 0x00000020

# GUIDs
GUID_ConstantForce = uuid.UUID("{13541C20-8E33-11D0-9AD0-00A0C9A06E35}")
IID_IDirectInputDevice8 = uuid.UUID("{54D41081-DC15-4833-A41B-748F73A38179}")

# Structures
class DIDEVICEINSTANCE(Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("guidInstance", BYTE * 16),
        ("guidProduct", BYTE * 16),
        ("dwDevType", DWORD),
        ("tszInstanceName", WCHAR * 260),
        ("tszProductName", WCHAR * 260),
        ("guidFFDriver", BYTE * 16),
        ("wUsagePage", WORD),
        ("wUsage", WORD),
    ]

class DIEFFECT(Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("dwFlags", DWORD),
        ("dwDuration", DWORD),
        ("dwSamplePeriod", DWORD),
        ("dwGain", DWORD),
        ("dwTriggerButton", DWORD),
        ("dwTriggerRepeatInterval", DWORD),
        ("cAxes", DWORD),
        ("rgdwAxes", POINTER(DWORD)),
        ("rglDirection", POINTER(LONG)),
        ("lpEnvelope", c_void_p),
        ("cbTypeSpecificParams", DWORD),
        ("lpvTypeSpecificParams", c_void_p),
        ("dwStartDelay", DWORD),
    ]

class DICONSTANTFORCE(Structure):
    _fields_ = [
        ("lMagnitude", LONG),
    ]

class DirectInputFFB:
    """Thin wrapper around DirectInput8 FFB for force feedback control."""
    
    def __init__(self):
        self.dinput = None
        self.device = None
        self.effect = None
        self._loaded = False
        
    def init(self, hwnd: int = 0) -> bool:
        """Initialize DirectInput and enumerate force feedback devices.
        
        Args:
            hwnd: Window handle for cooperative level (0 for message-only window)
            
        Returns:
            True if initialization succeeded
        """
        try:
            # Load dinput8.dll
            self.dinput_dll = windll.dinput8
            
            # Create DirectInput8 interface
            # In a full implementation, we'd call DirectInput8Create here
            # For now, return False to indicate FFB not available
            return False
        except Exception:
            return False
    
    def find_fanatec_device(self) -> bool:
        """Find and acquire a Fanatec force feedback device.
        
        Returns:
            True if device found and acquired
        """
        # TODO: Implement device enumeration and acquisition
        # Would call EnumDevices with callback, find Fanatec by name/VID
        # Then CreateDevice, SetDataFormat, SetCooperativeLevel, Acquire
        return False
    
    def create_constant_force_effect(self) -> bool:
        """Create a constant force effect for steering control.
        
        Returns:
            True if effect created successfully
        """
        # TODO: Implement effect creation
        # Would call CreateEffect(GUID_ConstantForce, ...)
        return False
    
    def set_force(self, magnitude: float) -> bool:
        """Set constant force magnitude.
        
        Args:
            magnitude: Force in range [-1.0, +1.0]
            
        Returns:
            True if force updated successfully
        """
        if self.effect is None:
            return False
        
        # TODO: Update effect parameters and start
        # Would call effect->SetParameters() and effect->Start()
        return False
    
    def stop_all_effects(self) -> bool:
        """Stop all force feedback effects and release motor.
        
        Returns:
            True if successful
        """
        if self.device is None:
            return False
        
        # TODO: Send DISFFC_STOPALL command
        # Would call device->SendForceFeedbackCommand(DISFFC_STOPALL)
        return False
    
    def release(self) -> None:
        """Release all DirectInput resources."""
        if self.effect is not None:
            # TODO: Release effect COM interface
            self.effect = None
        if self.device is not None:
            # TODO: Unacquire and release device COM interface
            self.device = None
        if self.dinput is not None:
            # TODO: Release DirectInput COM interface
            self.dinput = None
        self._loaded = False


# Singleton instance for module-level access
_ffb_instance: DirectInputFFB | None = None

def get_dinput_ffb() -> DirectInputFFB:
    """Get or create the global DirectInput FFB instance."""
    global _ffb_instance
    if _ffb_instance is None:
        _ffb_instance = DirectInputFFB()
    return _ffb_instance
