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


def hook_timeout(*, registry=None):
    """Read, never change, the configured timeout; unknown values stay unknown."""
    try:
        if registry is None:
            import winreg as registry
        with registry.OpenKey(registry.HKEY_CURRENT_USER, r"Control Panel\Desktop") as key:
            raw = registry.QueryValueEx(key, "LowLevelHooksTimeout")[0]
        value = int(raw)
        if isinstance(raw, bool) or value < 0:
            raise ValueError("invalid LowLevelHooksTimeout")
        # Supported Windows versions cap this at 1000ms. This only labels
        # diagnostics; neither of the existing one-second stop checks changes.
        effective = min(value, 1000)
        return {"registry_ms": value, "effective_ms": effective, "classification_ms": effective, "source": "registry"}
    except (OSError, ValueError, TypeError) as exc:
        return {"registry_ms": None, "effective_ms": None, "classification_ms": 1000,
                "source": "unknown; classification uses Windows upper bound", "error": str(exc)}


class InputHeartbeat:
    """Match LastInputInfo ticks to hook events, including injected events."""
    def __init__(self, initial, *, clock=time.monotonic, wall=time.time, ticks=lambda: None, timeout=None):
        self.clock, self.wall, self.ticks = clock, wall, ticks
        self.timeout = timeout or {"effective_ms": None, "classification_ms": 1000, "source": "unknown"}
        self.seen = deque([initial], maxlen=512)
        self.pending = None
        self.since = None
        self.unmatched = {}
        self.last_tick, self.input_tick, self.last_event = initial, initial, None
        self.last_pump = clock()
        self.last_pump_stamp = self.stamp()
        self.max_pump_gap = {"milliseconds": 0, "start": self.last_pump_stamp, "end": self.last_pump_stamp}
        self.counts = {kind: {"real": 0, "injected": 0} for kind in ("keyboard", "mouse")}
        self.max_delivery_lag, self.lag_at_timeout, self.delivery_lag_stamp = 0, 0, None
        self.checks = {"unobserved_input_tick": False, "pump_stall": False}
        self.failure = None

    def stamp(self):
        return {"monotonic": self.clock(), "wall_time": self.wall(), "tick": self.ticks()}

    def observe(self, tick, *, keyboard=None, flags=0, delivered=None):
        self.seen.append(tick)
        self.last_tick, self.last_event = tick, self.clock()
        if keyboard is not None:
            self.counts["keyboard" if keyboard else "mouse"]["real" if real_input(flags, keyboard=keyboard) else "injected"] += 1
        if delivered is not None:
            lag = (delivered - tick) & 0xffffffff
            if lag >= self.max_delivery_lag:
                self.max_delivery_lag = lag
                self.delivery_lag_stamp = {**self.stamp(), "tick": delivered, "event_tick": tick}
            if lag >= self.timeout["classification_ms"]:
                self.lag_at_timeout += 1
        if tick in self.unmatched:
            self.unmatched[tick]["observed_at"] = self.clock()

    def missed(self, tick):
        if tick in self.seen:
            self.pending = self.since = None
            return False
        if self.since is None:
            self.since = self.clock()
        self.pending = tick  # a changing unmatched tick must not reset since
        self.unmatched.setdefault(tick, {"tick": tick, "first_poll": self.clock(), "first_poll_time": self.stamp()})
        return self.clock() - self.since >= 1

    def pump_gap(self):
        stamp = self.stamp()
        milliseconds = (stamp["monotonic"] - self.last_pump) * 1000
        if milliseconds > self.max_pump_gap["milliseconds"]:
            # No native call between comparison and assignment: the hook pump
            # and active() must never replace a larger recorded gap with less.
            self.max_pump_gap = {"milliseconds": milliseconds, "start": self.last_pump_stamp, "end": stamp}
        return milliseconds

    def pump(self):
        self.pump_gap()
        self.last_pump, self.last_pump_stamp = self.clock(), self.stamp()

    def check(self, tick):
        self.input_tick = tick
        missed, age = self.missed(tick), self.pump_gap()  # always evaluate both
        self.checks = {"unobserved_input_tick": missed, "pump_stall": age > 1000}
        if (missed or age > 1000) and self.failure is None:
            cause = ("hook-pump-stall" if self.max_pump_gap["milliseconds"] >= self.timeout["classification_ms"]
                     else "unobserved-input-tick")
            self.failure = {"cause": cause, "checks": self.checks.copy(), "detected_at": self.stamp(),
                            "last_input_tick": tick, "last_hook_tick": self.last_tick,
                            "pump_age_ms": age}
        return missed or age > 1000

    def diagnostics(self):
        seen = set(self.seen)
        return {"timeout": self.timeout, "effective_timeout_ms": self.timeout["effective_ms"],
                "checks": self.checks.copy(), "failure": self.failure,
                "last_input_tick": self.input_tick, "last_hook_tick": self.last_tick,
                "tick_gap_ms": (self.input_tick - self.last_tick) & 0xffffffff,
                "last_hook_age_ms": None if self.last_event is None else (self.clock() - self.last_event) * 1000,
                "pump_age_ms": (self.clock() - self.last_pump) * 1000, "max_pump_gap": self.max_pump_gap,
                "max_delivery_lag_ms": self.max_delivery_lag, "max_delivery_lag_at": self.delivery_lag_stamp,
                "delivery_lag_at_timeout_count": self.lag_at_timeout,
                "event_counts": {key: value.copy() for key, value in self.counts.items()},
                "unmatched_ticks": [{**row, "seen_at_last_poll": tick in seen,
                                     "status": "late" if "observed_at" in row or tick in seen else "never-observed"}
                                    for tick, row in self.unmatched.copy().items()]}


class OwnerInput:
    """Dedicated hook pump ignores injected automation and cannot stall the cover."""
    def __init__(self):
        self.triggered = threading.Event()
        self.stopped = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.failure = None
        user = dll("user32")
        self.get_tick = signature(dll("kernel32"), "GetTickCount", [], C.c_uint32)
        class LastInput(C.Structure):
            _fields_ = [("size", W.UINT), ("time", W.DWORD)]
        self.info = LastInput(C.sizeof(LastInput), 0)
        self.get_input = signature(user, "GetLastInputInfo", [C.POINTER(LastInput)], W.BOOL)
        if not self.get_input(C.byref(self.info)):
            raise RigError("Owner activity check unavailable")
        self.heartbeat = InputHeartbeat(self.info.time, ticks=self.get_tick, timeout=hook_timeout())
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
                        delivered = self.get_tick()  # includes the wait for Python callback entry
                        flags = C.c_uint32.from_address(data + offset).value
                        self.heartbeat.observe(C.c_uint32.from_address(data + offset + 4).value,
                                               keyboard=keyboard, flags=flags, delivered=delivered)
                        if real_input(flags, keyboard=keyboard):
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
                self.heartbeat.pump()
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
        if self.heartbeat.check(self.info.time):
            row = self.diagnostics()
            if self.failure is None:
                self.failure = (f"Owner input hook heartbeat lost: {row['failure']['cause']}; "
                    f"checks={row['checks']}, last_input_tick={row['last_input_tick']}, last_hook_tick={row['last_hook_tick']}, "
                    f"tick_gap_ms={row['tick_gap_ms']}, last_hook_age_ms={row['last_hook_age_ms']}, "
                    f"pump_age_ms={row['pump_age_ms']:.1f}, max_pump_gap_ms={row['max_pump_gap']['milliseconds']:.1f}, "
                    f"effective_timeout_ms={row['effective_timeout_ms']}, max_delivery_lag_ms={row['max_delivery_lag_ms']}; "
                    "treating the owner as present")
            self.triggered.set()
        return self.triggered.is_set()

    def diagnostics(self):
        return self.heartbeat.diagnostics()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
