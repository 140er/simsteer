"""Locate a game's window so capture can crop to exactly what it renders.

Desktop duplication grabs the whole primary monitor. A windowed or
borderless game that does not fill that monitor leaves desktop around
the view, which shifts the road off the model's optical centre. This
finds the game's client area in physical pixels, relative to the
primary monitor, so `Capture` can crop each frame to it.

Windows only; elsewhere every lookup returns None.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass

# Smallest client area worth cropping to; anything smaller is a splash
# screen, a minimised stub or a helper window.
MIN_WIDTH = 320
MIN_HEIGHT = 180

_PER_MONITOR_AWARE_V2 = -4
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_MONITOR_DEFAULTTONULL = 0
_MONITORINFOF_PRIMARY = 1


@dataclass(frozen=True)
class WindowRect:
    """Client area in pixels relative to the primary monitor's top left."""
    left: int
    top: int
    right: int
    bottom: int
    on_primary: bool

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def crop_to_window(frame, rect: WindowRect | None):
    """Crop a primary-monitor frame to the window, or return it unchanged
    when there is no usable window on that monitor."""
    if rect is None or not rect.on_primary:
        return frame
    h, w = frame.shape[:2]
    left, top = max(0, rect.left), max(0, rect.top)
    right, bottom = min(w, rect.right), min(h, rect.bottom)
    if right - left < MIN_WIDTH or bottom - top < MIN_HEIGHT:
        return frame
    return frame[top:bottom, left:right]


def find_client_rect(process_names: tuple[str, ...]) -> WindowRect | None:
    """The largest visible, non-minimised top-level window of a process
    named in `process_names` (lowercase exe names), or None."""
    if sys.platform != "win32" or not process_names:
        return None
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    wanted = {name.lower() for name in process_names}

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    # Windows 10 1607+; older systems skip the DPI context.
    set_dpi = getattr(user32, "SetThreadDpiAwarenessContext", None)
    if set_dpi is not None:
        set_dpi.restype = ctypes.c_void_p
        set_dpi.argtypes = [ctypes.c_void_p]
    user32.MonitorFromRect.restype = ctypes.c_void_p
    user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFO)]
    kernel32.OpenProcess.restype = wintypes.HANDLE

    def exe_name(hwnd) -> str:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return ""
            return os.path.basename(buf.value).lower()
        finally:
            kernel32.CloseHandle(handle)

    candidates: list[tuple[int, wintypes.RECT]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        if exe_name(hwnd) not in wanted:
            return True
        client = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(client)):
            return True
        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return True
        rect = wintypes.RECT(origin.x, origin.y,
                             origin.x + client.right, origin.y + client.bottom)
        area = (rect.right - rect.left) * (rect.bottom - rect.top)
        if rect.right - rect.left >= MIN_WIDTH and rect.bottom - rect.top >= MIN_HEIGHT:
            candidates.append((area, rect))
        return True

    # Physical pixels, matching what desktop duplication captures,
    # whatever DPI awareness the rest of the process runs with.
    previous = set_dpi(_PER_MONITOR_AWARE_V2) if set_dpi is not None else None
    try:
        user32.EnumWindows(visit, 0)
        if not candidates:
            return None
        rect = max(candidates, key=lambda c: c[0])[1]
        monitor = user32.MonitorFromRect(ctypes.byref(rect), _MONITOR_DEFAULTTONULL)
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        mon = info.rcMonitor
        return WindowRect(rect.left - mon.left, rect.top - mon.top,
                          rect.right - mon.left, rect.bottom - mon.top,
                          bool(info.dwFlags & _MONITORINFOF_PRIMARY))
    finally:
        if previous:
            set_dpi(previous)
