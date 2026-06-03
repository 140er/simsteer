"""Screen capture for the ETS2 game window.

dxcam is the fastest path on Windows (uses DXGI desktop duplication). We
default to grabbing the primary monitor's full screen; if you want to
restrict to a specific window region, pass `region=(x1, y1, x2, y2)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import dxcam
import numpy as np


@dataclass
class CaptureConfig:
    region: tuple[int, int, int, int] | None = None  # (left, top, right, bottom)
    target_fps: int = 20
    output_color: str = "BGR"  # dxcam: "BGR", "RGB", "BGRA", "RGBA", "GRAY"


class Capture:
    """Pulls frames from the screen on demand.

    Use as a context manager or call .start() / .stop() yourself. .grab()
    returns the most recent frame (numpy HxWxC, uint8) or None if no new
    frame has arrived since the last call.
    """

    def __init__(self, cfg: CaptureConfig | None = None) -> None:
        self.cfg = cfg or CaptureConfig()
        self._cam: dxcam.DXCamera | None = None

    def __enter__(self) -> "Capture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._cam is not None:
            return
        self._cam = dxcam.create(output_color=self.cfg.output_color)
        self._cam.start(region=self.cfg.region, target_fps=self.cfg.target_fps,
                        video_mode=True)

    def stop(self) -> None:
        if self._cam is None:
            return
        self._cam.stop()
        del self._cam
        self._cam = None

    def grab(self) -> np.ndarray | None:
        assert self._cam is not None, "call start() first"
        return self._cam.get_latest_frame()
