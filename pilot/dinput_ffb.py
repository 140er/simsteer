"""Windows DirectInput Force Feedback implementation via ctypes COM.

Provides a Python interface to DirectInput8 for force feedback control.
This is used by the Fanatec module to physically drive the wheel motor.

Reference: https://learn.microsoft.com/en-us/previous-versions/windows/desktop/ee417816(v=vs.85)

Architecture:
- DirectInput8 COM interface via ctypes
- Enumerate devices, find Fanatec by VID 0x0EB7 or name
- Acquire BACKGROUND|EXCLUSIVE for FFB
- Create constant force effect
- Send force commands per-frame
- Handle DIERR_NOTACQUIRED gracefully

Thread safety: All DirectInput calls must happen on the same thread (COM apartment rules).
"""
from __future__ import annotations

import sys
import ctypes
from ctypes import *
from ctypes.wintypes import *
import uuid as _uuid

if sys.platform != "win32":
    raise ImportError("DirectInput FFB is Windows-only")

# DirectInput constants
DIRECTINPUT_VERSION = 0x0800
FANATEC_VENDOR_ID = 0x0EB7  # Endor AG

# Cooperative level flags
DISCL_EXCLUSIVE = 0x00000001
DISCL_NONEXCLUSIVE = 0x00000002
DISCL_FOREGROUND = 0x00000004
DISCL_BACKGROUND = 0x00000008

# Device enumeration
DI8DEVCLASS_GAMECTRL = 4
DIEDFL_ATTACHEDONLY = 0x00000001
DIEDFL_FORCEFEEDBACK = 0x00000100

# HRESULT values
S_OK = 0
S_FALSE = 1
E_FAIL = 0x80004005
DIERR_NOTACQUIRED = 0x8007001C
DIERR_INPUTLOST = 0x8007001E
DIERR_OTHERAPPHASPRIO = 0x80070005

# Effect flags
DIEFF_OBJECTOFFSETS = 0x00000002
DIEFF_CARTESIAN = 0x00000010
DIEFF_SPHERICAL = 0x00000020
DIEFF_POLAR = 0x00000020

# Effect control
DIEFF_START = 0x20000000
DIES_SOLO = 0x00000001
DIES_NODOWNLOAD = 0x80000000
INFINITE = 0xFFFFFFFF

# SendForceFeedbackCommand
DISFFC_RESET = 0x00000001
DISFFC_STOPALL = 0x00000002
DISFFC_PAUSE = 0x00000004
DISFFC_CONTINUE = 0x00000008
DISFFC_SETACTUATORSON = 0x00000010
DISFFC_SETACTUATORSOFF = 0x00000020

# GUIDs
GUID_XAxis = _uuid.UUID("{A36D02E0-C9F3-11CF-BFC7-444553540000}")
GUID_ConstantForce = _uuid.UUID("{13541C20-8E33-11D0-9AD0-00A0C9A06E35}")

# GUID structure for ctypes
class GUID(Structure):
    _fields_ = [
        ("Data1", c_ulong),
        ("Data2", c_ushort),
        ("Data3", c_ushort),
        ("Data4", c_ubyte * 8),
    ]

    @classmethod
    def from_uuid(cls, u: _uuid.UUID):
        g = cls()
        g.Data1 = (u.int >> 96) & 0xffffffff
        g.Data2 = (u.int >> 80) & 0xffff
        g.Data3 = (u.int >> 64) & 0xffff
        for i in range(8):
            g.Data4[i] = (u.int >> (56 - i * 8)) & 0xff
        return g

# Device instance
class DIDEVICEINSTANCE(Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("guidInstance", GUID),
        ("guidProduct", GUID),
        ("dwDevType", DWORD),
        ("tszInstanceName", WCHAR * 260),
        ("tszProductName", WCHAR * 260),
        ("guidFFDriver", GUID),
        ("wUsagePage", WORD),
        ("wUsage", WORD),
    ]

# Effect structures
class DICONSTANTFORCE(Structure):
    _fields_ = [("lMagnitude", LONG)]

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

# COM interface definitions
class IUnknown(Structure):
    pass

class IDirectInput8(Structure):
    pass

class IDirectInputDevice8(Structure):
    pass

class IDirectInputEffect(Structure):
    pass

# Function prototypes
LPDIENUMDEVICESCALLBACK = CFUNCTYPE(c_int, POINTER(DIDEVICEINSTANCE), c_void_p)

# COM vtable method signatures
IDirectInput8_EnumDevices = CFUNCTYPE(c_long, c_void_p, DWORD, c_void_p, c_void_p, DWORD)
IDirectInput8_CreateDevice = CFUNCTYPE(c_long, c_void_p, POINTER(GUID), POINTER(c_void_p), c_void_p)

IDirectInputDevice8_SetDataFormat = CFUNCTYPE(c_long, c_void_p, c_void_p)
IDirectInputDevice8_SetCooperativeLevel = CFUNCTYPE(c_long, c_void_p, HWND, DWORD)
IDirectInputDevice8_Acquire = CFUNCTYPE(c_long, c_void_p)
IDirectInputDevice8_Unacquire = CFUNCTYPE(c_long, c_void_p)
IDirectInputDevice8_CreateEffect = CFUNCTYPE(c_long, c_void_p, POINTER(GUID), POINTER(DIEFFECT), POINTER(c_void_p), c_void_p)
IDirectInputDevice8_SendForceFeedbackCommand = CFUNCTYPE(c_long, c_void_p, DWORD)

IDirectInputEffect_SetParameters = CFUNCTYPE(c_long, c_void_p, POINTER(DIEFFECT), DWORD)
IDirectInputEffect_Start = CFUNCTYPE(c_long, c_void_p, DWORD, DWORD)
IDirectInputEffect_Stop = CFUNCTYPE(c_long, c_void_p)

# Data format (we'll use c_dfDIJoystick2 from dinput8.lib, but define a minimal one)
class DIOBJECTDATAFORMAT(Structure):
    _fields_ = [
        ("pguid", POINTER(GUID)),
        ("dwOfs", DWORD),
        ("dwType", DWORD),
        ("dwFlags", DWORD),
    ]

class DIDATAFORMAT(Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("dwObjSize", DWORD),
        ("dwFlags", DWORD),
        ("dwDataSize", DWORD),
        ("dwNumObjs", DWORD),
        ("rgodf", POINTER(DIOBJECTDATAFORMAT)),
    ]

# Load dinput8.dll
try:
    _dinput8 = windll.dinput8
    _DirectInput8Create = _dinput8.DirectInput8Create
    _DirectInput8Create.argtypes = [HINSTANCE, DWORD, POINTER(GUID), POINTER(c_void_p), c_void_p]
    _DirectInput8Create.restype = c_long
except Exception:
    _dinput8 = None
    _DirectInput8Create = None


class DirectInputFFB:
    """DirectInput8 Force Feedback implementation via COM/ctypes."""
    
    def __init__(self):
        self.dinput = None
        self.device = None
        self.effect = None
        self.hwnd = None
        self._device_guid = None
        self._last_magnitude = 0
        
    def init(self) -> bool:
        """Initialize DirectInput8. Returns True if successful."""
        if _DirectInput8Create is None:
            return False
        
        try:
            # Create DirectInput8 interface
            iid = GUID.from_uuid(_uuid.UUID("{BF798031-483A-4DA2-AA99-5D64ED369700}"))  # IID_IDirectInput8W
            dinput_ptr = c_void_p()
            hr = _DirectInput8Create(
                windll.kernel32.GetModuleHandleW(None),
                DIRECTINPUT_VERSION,
                byref(iid),
                byref(dinput_ptr),
                None
            )
            
            if hr != S_OK:
                return False
            
            self.dinput = dinput_ptr
            return True
        except Exception:
            return False
    
    def find_fanatec_device(self) -> bool:
        """Find and acquire a Fanatec force feedback device. Returns True if found."""
        if self.dinput is None:
            return False
        
        # Get vtable for IDirectInput8
        vtable = cast(self.dinput, POINTER(c_void_p)).contents
        vtable_ptr = cast(vtable, POINTER(c_void_p))
        
        # EnumDevices is at index 4 in vtable
        enum_devices_func = cast(vtable_ptr[4], IDirectInput8_EnumDevices)
        
        # Callback to find Fanatec
        found_guid = [None]
        
        @LPDIENUMDEVICESCALLBACK
        def enum_callback(lpddi, pvRef):
            instance = lpddi.contents
            name = instance.tszProductName.lower()
            
            # Extract VID from guidProduct (VID is lower 16 bits of Data1)
            # DirectInput product GUID format: {VID_PID-0000-0000-0000-504944564944}
            vid = instance.guidProduct.Data1 & 0xFFFF
            
            # Match Fanatec by VID 0x0EB7 or name heuristic
            if (vid == FANATEC_VENDOR_ID or 
                any(x in name for x in ["fanatec", "csl", "clubsport", "podium"])):
                # Store GUID
                found_guid[0] = GUID()
                memmove(byref(found_guid[0]), byref(instance.guidInstance), sizeof(GUID))
                return 0  # DIENUM_STOP
            return 1  # DIENUM_CONTINUE
        
        # Enumerate force feedback game controllers
        hr = enum_devices_func(
            self.dinput,
            DI8DEVCLASS_GAMECTRL,
            enum_callback,
            None,
            DIEDFL_ATTACHEDONLY | DIEDFL_FORCEFEEDBACK
        )
        
        if hr != S_OK or found_guid[0] is None:
            return False
        
        self._device_guid = found_guid[0]
        
        # Create device
        create_device_func = cast(vtable_ptr[3], IDirectInput8_CreateDevice)
        device_ptr = c_void_p()
        hr = create_device_func(
            self.dinput,
            byref(self._device_guid),
            byref(device_ptr),
            None
        )
        
        if hr != S_OK:
            return False
        
        self.device = device_ptr
        
        # Get device vtable for remaining calls
        dev_vtable = cast(self.device, POINTER(c_void_p)).contents
        dev_vtable_ptr = cast(dev_vtable, POINTER(c_void_p))
        
        # Set data format - define minimal c_dfDIJoystick structure
        # GUID_XAxis for axis object type
        guid_xaxis = GUID.from_uuid(GUID_XAxis)
        
        # Minimal data format for joystick (just need axis 0 for steering)
        obj_fmt = DIOBJECTDATAFORMAT()
        obj_fmt.pguid = cast(byref(guid_xaxis), POINTER(GUID))
        obj_fmt.dwOfs = 0
        obj_fmt.dwType = 0x80000000 | 0x00000001  # DIDFT_AXIS | DIDFT_ANYINSTANCE
        obj_fmt.dwFlags = 0
        
        data_fmt = DIDATAFORMAT()
        data_fmt.dwSize = sizeof(DIDATAFORMAT)
        data_fmt.dwObjSize = sizeof(DIOBJECTDATAFORMAT)
        data_fmt.dwFlags = 0x00000001  # DIDF_ABSAXIS
        data_fmt.dwDataSize = 4  # sizeof(LONG) for one axis
        data_fmt.dwNumObjs = 1
        data_fmt.rgodf = cast(byref(obj_fmt), POINTER(DIOBJECTDATAFORMAT))
        
        # SetDataFormat at index 11
        set_fmt_func = cast(dev_vtable_ptr[11], IDirectInputDevice8_SetDataFormat)
        hr = set_fmt_func(self.device, byref(data_fmt))
        
        if hr != S_OK:
            return False
        
        # Create message-only window for cooperative level
        self.hwnd = windll.user32.CreateWindowExW(
            0, "Message", None, 0, 0, 0, 0, 0,
            -3,  # HWND_MESSAGE
            None, None, None
        )
        
        if not self.hwnd:
            return False
        
        # SetCooperativeLevel at index 13
        set_coop_func = cast(dev_vtable_ptr[13], IDirectInputDevice8_SetCooperativeLevel)
        hr = set_coop_func(
            self.device,
            self.hwnd,
            DISCL_EXCLUSIVE | DISCL_BACKGROUND
        )
        
        if hr != S_OK:
            return False
        
        # Acquire at index 7
        acquire_func = cast(dev_vtable_ptr[7], IDirectInputDevice8_Acquire)
        hr = acquire_func(self.device)
        
        if hr != S_OK and hr != S_FALSE:
            return False
        
        return True
    
    def create_constant_force_effect(self) -> bool:
        """Create a constant force effect. Returns True if successful."""
        if self.device is None:
            return False
        
        try:
            # Setup effect
            axes = (DWORD * 1)(0)  # X axis
            direction = (LONG * 1)(0)  # Direction in Cartesian
            cf = DICONSTANTFORCE(lMagnitude=0)
            
            effect = DIEFFECT()
            effect.dwSize = sizeof(DIEFFECT)
            effect.dwFlags = DIEFF_CARTESIAN | DIEFF_OBJECTOFFSETS
            effect.dwDuration = INFINITE
            effect.dwSamplePeriod = 0
            effect.dwGain = 10000  # 100%
            effect.dwTriggerButton = 0xFFFFFFFF  # No trigger
            effect.dwTriggerRepeatInterval = 0
            effect.cAxes = 1
            effect.rgdwAxes = cast(axes, POINTER(DWORD))
            effect.rglDirection = cast(direction, POINTER(LONG))
            effect.lpEnvelope = None
            effect.cbTypeSpecificParams = sizeof(DICONSTANTFORCE)
            effect.lpvTypeSpecificParams = cast(byref(cf), c_void_p)
            effect.dwStartDelay = 0
            
            # CreateEffect at index 18 (not 14)
            dev_vtable = cast(self.device, POINTER(c_void_p)).contents
            dev_vtable_ptr = cast(dev_vtable, POINTER(c_void_p))
            create_effect_func = cast(dev_vtable_ptr[18], IDirectInputDevice8_CreateEffect)
            
            effect_ptr = c_void_p()
            guid_cf = GUID.from_uuid(GUID_ConstantForce)
            hr = create_effect_func(
                self.device,
                byref(guid_cf),
                byref(effect),
                byref(effect_ptr),
                None
            )
            
            if hr != S_OK:
                return False
            
            self.effect = effect_ptr
            return True
        except Exception:
            return False
    
    def set_force(self, magnitude: float) -> bool:
        """Set constant force magnitude. magnitude in [-1.0, +1.0].
        Returns True if successful."""
        if self.effect is None:
            return False
        
        try:
            # Scale to DirectInput range [-10000, +10000]
            mag = int(magnitude * 10000)
            mag = max(-10000, min(10000, mag))
            
            if mag == self._last_magnitude:
                return True  # No change
            
            self._last_magnitude = mag
            
            # Update effect parameters
            cf = DICONSTANTFORCE(lMagnitude=mag)
            axes = (DWORD * 1)(0)
            direction = (LONG * 1)(mag)  # Direction = sign of magnitude
            
            effect_params = DIEFFECT()
            effect_params.dwSize = sizeof(DIEFFECT)
            effect_params.dwFlags = DIEFF_CARTESIAN | DIEFF_OBJECTOFFSETS
            effect_params.cAxes = 1
            effect_params.rgdwAxes = cast(axes, POINTER(DWORD))
            effect_params.rglDirection = cast(direction, POINTER(LONG))
            effect_params.cbTypeSpecificParams = sizeof(DICONSTANTFORCE)
            effect_params.lpvTypeSpecificParams = cast(byref(cf), c_void_p)
            
            # SetParameters at index 3
            eff_vtable = cast(self.effect, POINTER(c_void_p)).contents
            eff_vtable_ptr = cast(eff_vtable, POINTER(c_void_p))
            set_params_func = cast(eff_vtable_ptr[3], IDirectInputEffect_SetParameters)
            
            hr = set_params_func(
                self.effect,
                byref(effect_params),
                DIEFF_CARTESIAN | DIEFF_OBJECTOFFSETS
            )
            
            if hr != S_OK:
                return False
            
            # Start effect at index 4
            start_func = cast(eff_vtable_ptr[4], IDirectInputEffect_Start)
            hr = start_func(self.effect, 1, 0)  # iterations=1 (infinite), flags=0
            
            return hr == S_OK
        except Exception:
            return False
    
    def stop_all_effects(self) -> bool:
        """Stop all force feedback effects. Returns True if successful."""
        if self.device is None:
            return False
        
        try:
            # SendForceFeedbackCommand at index 22 (not 33)
            dev_vtable = cast(self.device, POINTER(c_void_p)).contents
            dev_vtable_ptr = cast(dev_vtable, POINTER(c_void_p))
            send_cmd_func = cast(dev_vtable_ptr[22], IDirectInputDevice8_SendForceFeedbackCommand)
            
            hr = send_cmd_func(self.device, DISFFC_STOPALL)
            return hr == S_OK
        except Exception:
            return False
    
    def reacquire(self) -> bool:
        """Try to reacquire device after losing access. Returns True if successful."""
        if self.device is None:
            return False
        
        try:
            dev_vtable = cast(self.device, POINTER(c_void_p)).contents
            dev_vtable_ptr = cast(dev_vtable, POINTER(c_void_p))
            acquire_func = cast(dev_vtable_ptr[7], IDirectInputDevice8_Acquire)
            hr = acquire_func(self.device)
            return hr == S_OK or hr == S_FALSE
        except Exception:
            return False
    
    def release(self) -> None:
        """Release all DirectInput resources."""
        if self.effect is not None:
            try:
                # Release COM interface
                eff_vtable = cast(self.effect, POINTER(c_void_p)).contents
                eff_vtable_ptr = cast(eff_vtable, POINTER(c_void_p))
                release_func = cast(eff_vtable_ptr[2], CFUNCTYPE(c_ulong, c_void_p))
                release_func(self.effect)
            except Exception:
                pass
            self.effect = None
        
        if self.device is not None:
            try:
                # Unacquire
                dev_vtable = cast(self.device, POINTER(c_void_p)).contents
                dev_vtable_ptr = cast(dev_vtable, POINTER(c_void_p))
                unacquire_func = cast(dev_vtable_ptr[8], IDirectInputDevice8_Unacquire)
                unacquire_func(self.device)
                
                # Release COM interface
                release_func = cast(dev_vtable_ptr[2], CFUNCTYPE(c_ulong, c_void_p))
                release_func(self.device)
            except Exception:
                pass
            self.device = None
        
        if self.dinput is not None:
            try:
                # Release COM interface
                vtable = cast(self.dinput, POINTER(c_void_p)).contents
                vtable_ptr = cast(vtable, POINTER(c_void_p))
                release_func = cast(vtable_ptr[2], CFUNCTYPE(c_ulong, c_void_p))
                release_func(self.dinput)
            except Exception:
                pass
            self.dinput = None
        
        if self.hwnd is not None:
            try:
                windll.user32.DestroyWindow(self.hwnd)
            except Exception:
                pass
            self.hwnd = None


# Singleton instance
_ffb_instance: DirectInputFFB | None = None

def get_dinput_ffb() -> DirectInputFFB:
    """Get or create the global DirectInput FFB instance."""
    global _ffb_instance
    if _ffb_instance is None:
        _ffb_instance = DirectInputFFB()
    return _ffb_instance
