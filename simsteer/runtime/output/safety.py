"""Exit neutralization and serialized binding requests for the output manager.

This is a minimum stop path; it does not supply a general command watchdog.
"""

from __future__ import annotations

import atexit
import ctypes
from functools import wraps
from threading import RLock

from simsteer.runtime.output.binding import Binding


def output_locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._output_lock:
            if self._closed:
                return None
            return method(self, *args, **kwargs)
    return call


class OutputSafety:
    def _init_safety(self):
        self._output_lock = RLock()
        self._closed = False
        self._binding = Binding()
        self._console_handler = None
        self._console_registered = False
        atexit.register(self.close)
        # WINFUNCTYPE exists only on Windows. Keep the callback alive until
        # it has been unregistered; ctypes callbacks otherwise get collected.
        if hasattr(ctypes, "WINFUNCTYPE"):
            self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self._console_handler = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(
                self._console_stop)
            if not self._kernel.SetConsoleCtrlHandler(self._console_handler, True):
                self._console_handler = None
                raise OSError("could not install output stop handler")
            self._console_registered = True

    def _console_stop(self, event):
        if event in (0, 1, 2, 5, 6):  # interrupt, break, close, logoff, shutdown
            self.close()
        # Preserve normal interrupt/exit processing after neutralization.
        return False

    def close(self):
        with self._output_lock:
            self._closed = True
            try:
                self._binding.cancel(self)
            finally:
                try:
                    if self.device is not None:
                        self.device.disengage()
                finally:
                    atexit.unregister(self.close)
                    if self._console_registered:
                        self._kernel.SetConsoleCtrlHandler(self._console_handler, False)
                        self._console_registered = False
                        # Keep the ctypes trampoline alive even if close() is
                        # currently executing inside this very callback.

    @output_locked
    def request_bind(self, kind):
        if self.device is None or self.engaged:
            return False
        self._binding.cancel(self)
        self._binding.request = (kind, self.device)
        return True

    @output_locked
    def cancel_bind(self):
        self._binding.cancel(self)

    @output_locked
    def tick_bind(self, now, speed, manual_override=False):
        return self._binding.tick(self, now, speed, manual_override)
