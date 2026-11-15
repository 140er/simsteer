"""Screen capture for the game window.

dxcam is the fastest path on Windows (uses DXGI desktop duplication) and
grabs the primary monitor. With `window_processes` set, each frame is
cropped to that game's client area, so a windowed game that does not fill
the monitor still reaches the model centred. Without a usable window on
the primary monitor the full screen is used, as before.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import math
from queue import Empty, Full, Queue
from threading import Event, Thread
import sys
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from simsteer.runtime.window import WindowRect, crop_to_window, find_client_rect

if TYPE_CHECKING:
    import dxcam


@dataclass
class CaptureConfig:
    region: tuple[int, int, int, int] | None = None  # (left, top, right, bottom)
    target_fps: int = 20
    output_color: str = "BGR"  # dxcam: "BGR", "RGB", "BGRA", "RGBA", "GRAY"
    wait_s: float = 0.1
    max_age_s: float = 0.25
    unchanged_ticks: int = 5
    # Lowercase exe names whose window to crop to, e.g. ("eurotrucks2.exe",).
    window_processes: tuple[str, ...] = ()
    window_refresh_s: float = 1.0


class Capture:
    """Pulls frames from the screen on demand.

    Use as a context manager or call .start() / .stop() yourself. .grab()
    returns the most recent frame (numpy HxWxC, uint8) or None if no new
    frame has arrived since the last call.
    """

    def __init__(self, cfg: CaptureConfig | None = None,
                 find_window: Callable[[tuple[str, ...]], WindowRect | None] = find_client_rect,
                 ) -> None:
        self.cfg = cfg or CaptureConfig()
        self._find_window = find_window
        self._window: WindowRect | None = None
        self._window_checked = -math.inf
        # Last capture-area message, printed once per change.
        self.window_status = ""
        self._cam: "dxcam.DXCamera | None" = None
        # A screen never runs out; `ReplaySource` sets this at clip end.
        self.done = False
        self._frames = Queue(maxsize=1)
        self._stop = Event()
        self._reader = None
        self._previous = None
        self._unchanged = 0
        self._timer = None

    def __enter__(self) -> "Capture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._cam is not None:
            return
        # Imported here so the loop and replay import on hosts without dxcam.
        import dxcam
        if sys.platform == "win32":
            timer = ctypes.WinDLL("winmm")
            if timer.timeBeginPeriod(1) != 0:
                raise OSError("could not request capture timer resolution")
            self._timer = timer
        try:
            self._cam = dxcam.create(output_color=self.cfg.output_color)
            self._cam.start(region=self.cfg.region, target_fps=self.cfg.target_fps,
                            video_mode=True)
        except Exception:
            self.stop()
            raise
        self._stop = Event()
        self._frames = Queue(maxsize=1)
        self._previous, self._unchanged = None, 0
        self._reader = Thread(target=self._read, args=(self._cam, self._frames, self._stop), daemon=True)
        self._reader.start()

    def _read(self, cam, frames, stop):
        period = 1. / self.cfg.target_fps
        next_due = time.monotonic()
        while not stop.is_set():
            try:
                frame = cam.get_latest_frame()
                sample = (time.monotonic(), None if frame is None else frame.copy())
            except Exception:
                sample = (time.monotonic(), None)
            try:
                frames.put_nowait(sample)
            except Full:
                try:
                    frames.get_nowait()
                except Empty:
                    pass
                frames.put_nowait(sample)
            # Charge read/copy time to this tick. Keep the deadline phase
            # even when a wait rounds up, and skip missed ticks when late.
            next_due += period
            now = time.monotonic()
            if now > next_due:
                next_due += math.ceil((now - next_due) / period) * period
            if stop.wait(max(0., next_due - now)):
                break

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._cam is not None:
                self._cam.stop()
            if self._reader is not None:
                self._reader.join(timeout=self.cfg.wait_s)
        finally:
            self._cam = None
            self._previous = None
            if self._timer is not None:
                self._timer.timeEndPeriod(1)
                self._timer = None

    def grab(self) -> np.ndarray | None:
        assert self._cam is not None, "call start() first"
        try:
            captured_at, frame = self._frames.get(timeout=self.cfg.wait_s)
        except Empty:
            return None
        if frame is None or time.monotonic() - captured_at > self.cfg.max_age_s:
            return None
        h, w = frame.shape[:2]
        small = frame[::max(1, math.ceil(h / 32)), ::max(1, math.ceil(w / 64))]
        if self._previous is not None and np.array_equal(small, self._previous):
            self._unchanged += 1
        else:
            # The reader already owns frame. Retain only a small sample for
            # freshness so neither a full copy nor a full compare is needed.
            self._previous = small.copy()
            self._unchanged = 0
        if self._unchanged >= self.cfg.unchanged_ticks:
            return None
        return crop_to_window(frame, self._current_window())

    def _current_window(self) -> WindowRect | None:
        if not self.cfg.window_processes:
            return None
        now = time.monotonic()
        if now - self._window_checked >= self.cfg.window_refresh_s:
            self._window_checked = now
            try:
                self._window = self._find_window(self.cfg.window_processes)
            except Exception:  # a failed lookup must never break capture
                self._window = None
            rect = self._window
            if rect is None:
                status = "capture: game window not found; using the full primary screen"
            elif not rect.on_primary:
                status = ("capture: game window is not on the primary monitor; "
                          "using the full primary screen (move the game to the main monitor)")
            else:
                status = (f"capture: game window {rect.width}x{rect.height} "
                          f"at ({rect.left},{rect.top})")
            if status != self.window_status:
                self.window_status = status
                print(status, flush=True)
        return self._window
