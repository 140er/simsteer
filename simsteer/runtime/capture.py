"""Screen capture for the ETS2 game window.

dxcam is the fastest path on Windows (uses DXGI desktop duplication). We
default to grabbing the primary monitor's full screen; if you want to
restrict to a specific window region, pass `region=(x1, y1, x2, y2)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
from typing import TYPE_CHECKING

import numpy as np

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


class Capture:
    """Pulls frames from the screen on demand.

    Use as a context manager or call .start() / .stop() yourself. .grab()
    returns the most recent frame (numpy HxWxC, uint8) or None if no new
    frame has arrived since the last call.
    """

    def __init__(self, cfg: CaptureConfig | None = None) -> None:
        self.cfg = cfg or CaptureConfig()
        self._cam: "dxcam.DXCamera | None" = None
        # A screen never runs out; `ReplaySource` sets this at clip end.
        self.done = False
        self._frames = Queue(maxsize=1)
        self._stop = Event()
        self._reader = None
        self._previous = None
        self._unchanged = 0

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
        self._cam = dxcam.create(output_color=self.cfg.output_color)
        self._cam.start(region=self.cfg.region, target_fps=self.cfg.target_fps,
                        video_mode=True)
        self._stop = Event()
        self._frames = Queue(maxsize=1)
        self._previous, self._unchanged = None, 0
        self._reader = Thread(target=self._read, args=(self._cam, self._frames, self._stop), daemon=True)
        self._reader.start()

    def _read(self, cam, frames, stop):
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
            # Some capture versions return a repeated frame immediately.
            # Pace these reads so a static image cannot spin the worker.
            if stop.wait(1. / self.cfg.target_fps):
                break

    def stop(self) -> None:
        if self._cam is None:
            return
        self._stop.set()
        self._cam.stop()
        if self._reader is not None:
            self._reader.join(timeout=self.cfg.wait_s)
        self._cam = None
        self._previous = None

    def grab(self) -> np.ndarray | None:
        assert self._cam is not None, "call start() first"
        try:
            captured_at, frame = self._frames.get(timeout=self.cfg.wait_s)
        except Empty:
            return None
        if frame is None or time.monotonic() - captured_at > self.cfg.max_age_s:
            return None
        if self._previous is not None and np.array_equal(frame, self._previous):
            self._unchanged += 1
        else:
            self._previous = frame.copy()
            self._unchanged = 0
        if self._unchanged >= self.cfg.unchanged_ticks:
            return None
        return frame
