"""Global keyboard polling via Win32 `GetAsyncKeyState`.

`cv2.waitKey` only fires when the OpenCV overlay has focus, which is
useless in normal operation — the game has to be focused for our
virtual gamepad to steer it. This module polls the OS keyboard state
each frame and reports edge-triggered key presses (transitions from
up to down). The caller can then handle them with the same logic it
uses for OpenCV-focused keypresses.

The returned key codes are lowercase ASCII bytes (e.g. `ord('a')` for
A, `ord(' ')` for space, `ord('[')` for `[`) so the consumer can
treat polled keys exactly like `cv2.waitKey() & 0xFF` returns.

Quirks worth knowing:
  - `GetAsyncKeyState` reads the *global* key state and doesn't care
    whether our window has focus. That's the entire point.
  - It does NOT eat the keystroke — the game still receives it. So
    you don't want to bind hotkeys onto keys the game also uses. The
    current bindings (A/D/Q/SPACE/etc.) overlap with ETS2 controls,
    but ETS2 ignores keyboard while a gamepad is driving steering,
    so in practice they don't collide. If they ever do, the user
    should rebind in-game or remove the conflict here.
  - No admin / hook required. This is just a polled read of an OS
    bitfield, so it works on any Windows session.
"""
from __future__ import annotations

import ctypes

# VK codes for keys that don't match their ASCII char.
_VK_SPACE = 0x20
_VK_OEM_1 = 0xBA        # ';:' on US layouts
_VK_OEM_COMMA = 0xBC
_VK_OEM_MINUS = 0xBD    # not used today; here for symmetry
_VK_OEM_PERIOD = 0xBE
_VK_OEM_4 = 0xDB        # '['
_VK_OEM_6 = 0xDD        # ']'
_VK_OEM_7 = 0xDE        # '\''

# Arrow keys. The raw VK codes (0x25..0x28) collide with printable
# ASCII characters in our watch-set (notably VK_RIGHT == 0x27 == ord
# "'"), so the public identifiers are biased into the 0x1xx range —
# guaranteed to be outside any single-byte cv2 keypress, and the
# low byte still carries the VK code for the actual GetAsyncKeyState
# call (extracted in _ascii_to_vk).
_SPECIAL = 0x100
KEY_LEFT = _SPECIAL | 0x25
KEY_UP = _SPECIAL | 0x26
KEY_RIGHT = _SPECIAL | 0x27
KEY_DOWN = _SPECIAL | 0x28
KEY_INSERT = _SPECIAL | 0x2D
KEY_DELETE = _SPECIAL | 0x2E
KEY_HOME = _SPECIAL | 0x24
KEY_END = _SPECIAL | 0x23
KEY_PAGEUP = _SPECIAL | 0x21
KEY_PAGEDOWN = _SPECIAL | 0x22
# Numpad digits — dedicated VK codes (VK_NUMPAD0..9 = 0x60..0x69).
# These fire even with Num Lock OFF on the numpad keys themselves
# (they're distinct from the digit row at the top of the keyboard).
KEY_NUMPAD4 = _SPECIAL | 0x64
KEY_NUMPAD6 = _SPECIAL | 0x66
KEY_NUMPAD5 = _SPECIAL | 0x65
KEY_NUMPAD8 = _SPECIAL | 0x68
KEY_NUMPAD2 = _SPECIAL | 0x62
# F1..F12 — VK_F1=0x70 .. VK_F12=0x7B. Used by NavManager hotkeys.
KEY_F1 = _SPECIAL | 0x70
KEY_F2 = _SPECIAL | 0x71
KEY_F3 = _SPECIAL | 0x72
KEY_F4 = _SPECIAL | 0x73

# Map from the lowercase ASCII byte we want to report -> the VK code
# we read from the OS. For letters/digits/space the VK code IS the
# uppercase ASCII byte, so we just upper() at lookup time.
_OEM_KEYS: dict[int, int] = {
    ord("["): _VK_OEM_4,
    ord("]"): _VK_OEM_6,
    ord(","): _VK_OEM_COMMA,
    ord("."): _VK_OEM_PERIOD,
    ord(";"): _VK_OEM_1,
    ord("'"): _VK_OEM_7,
    ord(" "): _VK_SPACE,
}


def _ascii_to_vk(ch: int) -> int:
    # Sentinel-tagged arrow / function keys: the low byte is the
    # actual VK code.
    if ch & _SPECIAL:
        return ch & 0xFF
    if ch in _OEM_KEYS:
        return _OEM_KEYS[ch]
    # A-Z / 0-9: VK codes match the uppercase ASCII byte. Letters
    # arrive as lowercase from `ord('a')` etc., so upper() them.
    c = chr(ch)
    if c.isalpha():
        return ord(c.upper())
    return ch  # digits already match


class GlobalKeys:
    """Edge-triggered global key reader.

    Construct with the set of ASCII bytes to watch (use lowercase for
    letters, e.g. `ord('q')`, and the literal punctuation byte for
    OEM keys like `ord('[')`). Each call to `poll()` returns the
    subset of watched keys that transitioned from up to down since
    the previous call.
    """

    def __init__(self, watch: set[int]) -> None:
        self._user32 = ctypes.windll.user32
        # Map watched ASCII bytes -> VK codes, cached.
        self._vk: dict[int, int] = {ch: _ascii_to_vk(ch) for ch in watch}
        self._down: dict[int, bool] = {ch: False for ch in watch}

    def poll(self) -> list[int]:
        """Return the list of ASCII bytes whose VK transitioned to
        the pressed state since the previous poll."""
        pressed: list[int] = []
        for ch, vk in self._vk.items():
            # High bit of the returned short = key is currently down.
            down = bool(self._user32.GetAsyncKeyState(vk) & 0x8000)
            if down and not self._down[ch]:
                pressed.append(ch)
            self._down[ch] = down
        return pressed
