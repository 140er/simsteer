"""Wheel button listener for engage/disengage binding.

Uses pygame to detect button presses on physical DirectInput wheels
(Fanatec, Logitech, Thrustmaster, etc.) and provides an edge-triggered
interface similar to GlobalKeys.

Thread safety: All pygame calls must happen on the same thread.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

# Only available on Windows with pygame
_PYGAME_AVAILABLE = False
if sys.platform == "win32":
    try:
        import pygame
        import pygame.joystick
        _PYGAME_AVAILABLE = True
    except ImportError:
        pass


class WheelButtonListener:
    """Edge-triggered wheel button listener.
    
    Polls pygame joystick state each frame and reports button transitions
    from up to down. Works with any DirectInput device (wheels, HOTAS, etc.).
    """
    
    def __init__(self) -> None:
        self._initialized = False
        self._joysticks: list = []  # List of pygame.joystick.Joystick
        self._button_states: dict[tuple[int, int], bool] = {}  # (joy_id, btn_idx) -> pressed
        
        if not _PYGAME_AVAILABLE:
            return
        
        try:
            # Initialize pygame joystick subsystem only (not video/audio)
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            
            # Initialize all connected joysticks
            count = pygame.joystick.get_count()
            for i in range(count):
                try:
                    joy = pygame.joystick.Joystick(i)
                    joy.init()
                    self._joysticks.append(joy)
                    # Initialize button states to False
                    num_buttons = joy.get_numbuttons()
                    for btn_idx in range(num_buttons):
                        self._button_states[(i, btn_idx)] = False
                except Exception:
                    continue
            
            self._initialized = True
        except Exception:
            self._initialized = False
    
    def poll(self) -> list[tuple[str, int]]:
        """Poll joystick state and return button press events.
        
        Returns:
            List of (device_name, button_index) tuples for buttons that
            transitioned from up to down since the previous poll.
        """
        if not self._initialized:
            return []
        
        pressed: list[tuple[str, int]] = []
        
        try:
            # Process pygame events to update joystick state
            pygame.event.pump()
            
            # Check each joystick's button state
            for joy_id, joy in enumerate(self._joysticks):
                if not joy.get_init():
                    continue
                
                device_name = joy.get_name()
                num_buttons = joy.get_numbuttons()
                
                for btn_idx in range(num_buttons):
                    key = (joy_id, btn_idx)
                    current_state = bool(joy.get_button(btn_idx))
                    previous_state = self._button_states.get(key, False)
                    
                    # Edge trigger: 0 -> 1 transition
                    if current_state and not previous_state:
                        pressed.append((device_name, btn_idx))
                    
                    self._button_states[key] = current_state
        except Exception:
            # Joystick disconnected or pygame error - return empty
            pass
        
        return pressed
    
    def get_device_names(self) -> list[str]:
        """Return list of connected device names for UI display."""
        if not self._initialized:
            return []
        
        names = []
        for joy in self._joysticks:
            try:
                if joy.get_init():
                    names.append(joy.get_name())
            except Exception:
                continue
        return names
    
    def is_available(self) -> bool:
        """True if pygame is available and at least one joystick is connected."""
        return self._initialized and len(self._joysticks) > 0
    
    def __del__(self):
        """Cleanup pygame joysticks on destruction."""
        if self._initialized:
            try:
                for joy in self._joysticks:
                    if joy.get_init():
                        joy.quit()
                if pygame.joystick.get_init():
                    pygame.joystick.quit()
            except Exception:
                pass


def parse_button_bind(bind_str: str) -> tuple[Optional[str], Optional[int]]:
    """Parse wheel_button_bind setting string.
    
    Args:
        bind_str: Format "device_name:button_index" e.g. "Fanatec CSL Elite:12"
    
    Returns:
        (device_name, button_index) or (None, None) if invalid/empty
    """
    if not bind_str or ":" not in bind_str:
        return None, None
    
    try:
        parts = bind_str.rsplit(":", 1)  # Split from right in case device name has ":"
        device_name = parts[0]
        button_index = int(parts[1])
        return device_name, button_index
    except (ValueError, IndexError):
        return None, None


def format_button_bind(device_name: str, button_index: int) -> str:
    """Format wheel button bind for settings storage.
    
    Args:
        device_name: Joystick device name from pygame
        button_index: Button index (0-based)
    
    Returns:
        Formatted string "device_name:button_index"
    """
    return f"{device_name}:{button_index}"
