"""Exit neutralization and serialized binding requests for the output manager.

Every device writer must hold the manager's output lock. Public output
methods, binding ticks and relay threads submit
through write_output(), including neutral/handback writes. Direct backend
or raw-device writes outside that lock are forbidden.

This is a minimum stop path; it does not supply a general command watchdog.
"""

from __future__ import annotations

import atexit
import ctypes
from functools import wraps
from threading import RLock, get_ident

from simsteer.runtime.output.binding import Binding


def output_locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._output_lock:
            if self._closed or getattr(self, "_shutdown_requested", False):
                return None
            return self.write_output(lambda _: method(self, *args, **kwargs))
    return call


class OutputGate:
    """The shared writer path, also usable by a standalone relay."""
    def __init__(self):
        self._output_lock = RLock()
        self._closed = False

    def write_output(self, writer, *, cleanup=False):
        """Serialize output; cleanup permits the owner's final neutral/release."""
        with self._output_lock:
            io = getattr(getattr(self, "device", None), "io", None)
            final_release = cleanup and io is not None and io.stopping.is_set() and io.ident == get_ident()
            if not self._closed or final_release:
                return writer(getattr(self, "device", None))


class OutputSafety(OutputGate):
    def _init_safety(self):
        OutputGate.__init__(self)
        self._binding = Binding()
        self._console_handler = None
        self._console_registered = False
        atexit.register(self.shutdown_output)
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
            self.shutdown_output()
        # Preserve normal interrupt/exit processing after neutralization.
        return False

    def shutdown_output(self, *, close_device=True):
        with self._output_lock:
            if self._closed:
                return
            try:
                def finish(_):
                    try:
                        self._binding.cancel(self)
                    finally:
                        if close_device and self.device is not None:
                            self.device.disengage()
                            close = getattr(self.device, "close", None)
                            if close is not None:
                                close()
                self.write_output(finish)
            finally:
                self._closed = True
                atexit.unregister(self.shutdown_output)
                if self._console_registered:
                    self._kernel.SetConsoleCtrlHandler(self._console_handler, False)
                    self._console_registered = False
                    # Keep the ctypes trampoline alive inside this callback.

    @output_locked
    def request_bind(self, kind):
        if self.device is None or self.engaged:
            return False
        self._binding.cancel(self)
        self._binding.request = (kind, self.device)
        self._binding.status = "Binding in 4 s"
        # An explicit manual bind remains usable in paused/static menus.
        getattr(self.device, "resume_output", lambda: None)()
        return True

    @property
    def binding_active(self):
        return self._binding.request is not None

    @property
    def binding_status(self):
        return self._binding.status

    @output_locked
    def cancel_bind(self):
        self._binding.cancel(self)

    @output_locked
    def tick_bind(self, now, speed, manual_override=False):
        return self._binding.tick(self, now, speed, manual_override)
