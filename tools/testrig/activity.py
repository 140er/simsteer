"""Owner keyboard/mouse activity, without recording keys or pointer positions."""

import ctypes as C
from ctypes import wintypes as W
import threading
import time
from collections import deque

from . import RigError
from .windows import dll, signature


def real_input(flags, *, keyboard):
    return not flags & (0x12 if keyboard else 0x03)


class InputHeartbeat:
    """Match LastInputInfo ticks to hook events, including injected events."""
    def __init__(self, initial, *, clock=time.monotonic):
        self.clock = clock
        self.seen = deque([initial], maxlen=512)
        self.pending = None
        self.since = None

    def observe(self, tick):
        self.seen.append(tick)

    def missed(self, tick):
        if tick in self.seen:
            self.pending = self.since = None
            return False
        if self.since is None:
            self.pending, self.since = tick, self.clock()
        return self.clock() - self.since >= 1


class OwnerInput:
    """Dedicated hook pump ignores injected automation and cannot stall the cover."""
    def __init__(self):
        self.triggered = threading.Event()
        self.stopped = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.failure = None
        self.last_pump = time.monotonic()
        user = dll("user32")
        class LastInput(C.Structure):
            _fields_ = [("size", W.UINT), ("time", W.DWORD)]
        self.info = LastInput(C.sizeof(LastInput), 0)
        self.get_input = signature(user, "GetLastInputInfo", [C.POINTER(LastInput)], W.BOOL)
        if not self.get_input(C.byref(self.info)):
            raise RigError("Owner activity check unavailable")
        self.heartbeat = InputHeartbeat(self.info.time)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(5) or self.error:
            self.close()
            raise RigError("Owner input monitor unavailable: " + str(self.error or "startup timed out"))

    def _run(self):
        hooks = []
        try:
            user = dll("user32")
            callback_type = C.WINFUNCTYPE(W.LPARAM, C.c_int, W.WPARAM, W.LPARAM)
            install = signature(user, "SetWindowsHookExW", [C.c_int, callback_type, W.HINSTANCE, W.DWORD], W.HANDLE)
            following = signature(user, "CallNextHookEx", [W.HANDLE, C.c_int, W.WPARAM, W.LPARAM], W.LPARAM)
            remove = signature(user, "UnhookWindowsHookEx", [W.HANDLE], W.BOOL)
            peek = signature(user, "PeekMessageW", [C.POINTER(W.MSG), W.HWND, W.UINT, W.UINT, W.UINT], W.BOOL)
            signature(user, "TranslateMessage", [C.POINTER(W.MSG)], W.BOOL)
            signature(user, "DispatchMessageW", [C.POINTER(W.MSG)], W.LPARAM)
            callbacks = []
            for kind, keyboard, offset in ((13, True, 8), (14, False, 12)):
                def observe(code, message, data, keyboard=keyboard, offset=offset):
                    if code >= 0:
                        self.heartbeat.observe(C.c_uint32.from_address(data + offset + 4).value)
                        if real_input(C.c_uint32.from_address(data + offset).value, keyboard=keyboard):
                            self.triggered.set()
                    return following(None, code, message, data)
                callback = callback_type(observe)
                callbacks.append(callback)
                module = signature(dll("kernel32"), "GetModuleHandleW", [W.LPCWSTR], W.HMODULE)(None)
                handle = install(kind, callback, module, 0)
                if not handle:
                    raise RigError("Cannot install owner input hook")
                hooks.append(handle)
            self.ready.set()
            message = W.MSG()
            while not self.stopped.is_set():
                self.last_pump = time.monotonic()
                while peek(C.byref(message), None, 0, 0, 1):
                    user.TranslateMessage(C.byref(message))
                    user.DispatchMessageW(C.byref(message))
                self.stopped.wait(0.02)
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            for handle in hooks:
                remove(handle)

    def active(self):
        if self.error:
            raise RigError("Owner input monitor failed: " + str(self.error))
        if not self.get_input(C.byref(self.info)):
            raise RigError("Owner activity check unavailable")
        if self.heartbeat.missed(self.info.time) or time.monotonic() - self.last_pump > 1:
            self.failure = "Owner input hook heartbeat lost; treating the owner as present"
            self.triggered.set()
        return self.triggered.is_set()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
