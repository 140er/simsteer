"""Frame -> driving_vision input tensors (narrow + wide).

The vision ONNX takes TWO uint8 inputs of shape (1, 12, 128, 256):
    `img`     — the standard road camera (narrow virtual FOV ~31°)
    `big_img` — the wide road camera     (wide   virtual FOV ~59°)

In real openpilot these come from two physically distinct cameras
mounted side-by-side; modeld warps each into a fixed virtual camera
(medmodel for `img`, sbigmodel for `big_img`). We only have one screen
capture, so we apply both warps to the same source — pulling a tight
forward view for the narrow input and a wider context view for the
wide input. The two inputs are NOT bit-identical: their virtual cameras
have different focal lengths (910 vs 455), so the model sees the same
scene at two different effective zooms, mirroring what the parallax
pair provides on the comma 3.

Each frame's 12-channel tensor is 2 stacked 6-channel YUV frames at
t and t-4 (`frame_skip=4`). Each YUV6 = 4 strided Y planes + half-res
U + half-res V.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pilot.constants import FRAME_SKIP, MODEL_H, MODEL_W, SRC_H, SRC_W
from pilot.warp import warp_to_model

if TYPE_CHECKING:
    from pilot.calibration import Calibration


def make_warp_matrix(src_w: int, src_h: int) -> np.ndarray:
    """Legacy stretch-fit used only by `bgr_to_yuv6` when given a frame
    that isn't already the model's canonical (512, 256). The new
    `crop_for_model` pipeline always warps to (512, 256) first, so this
    is only exercised by smoke tests that bypass the warp."""
    sx = SRC_W / src_w
    sy = SRC_H / src_h
    return np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float32)


def bgr_to_yuv6(bgr: np.ndarray) -> np.ndarray:
    """BGR HxWx3 (uint8) -> (6, MODEL_H, MODEL_W) uint8."""
    h, w = bgr.shape[:2]
    if (w, h) != (SRC_W, SRC_H):
        M = make_warp_matrix(w, h)
        bgr = cv2.warpAffine(bgr, M, (SRC_W, SRC_H), flags=cv2.INTER_LINEAR)

    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)  # planar Y..U..V
    y_plane = yuv[:SRC_H, :SRC_W]
    u_plane = yuv[SRC_H:SRC_H + SRC_H // 4].reshape(SRC_H // 2, SRC_W // 2)
    v_plane = yuv[SRC_H + SRC_H // 4:].reshape(SRC_H // 2, SRC_W // 2)

    out = np.empty((6, MODEL_H, MODEL_W), dtype=np.uint8)
    out[0] = y_plane[0::2, 0::2]
    out[1] = y_plane[1::2, 0::2]
    out[2] = y_plane[0::2, 1::2]
    out[3] = y_plane[1::2, 1::2]
    out[4] = u_plane
    out[5] = v_plane
    return out


def crop_for_model(bgr: np.ndarray,
                   calib: "Calibration | None") -> tuple[np.ndarray, np.ndarray]:
    """Return (narrow_bgr, wide_bgr) — the two views the vision model wants.

    Each output is reprojected onto a fixed virtual camera using
    `pilot.warp.warp_to_model`:
      - narrow: medmodel  (fl=910, cy=47.6) -> tight forward view (~31° HFOV)
      - wide:   sbigmodel (fl=455, cy=152)  -> wider context view (~59° HFOV)

    Both outputs are (256, 512, 3) BGR — the model's canonical input
    size. The mount pitch / yaw stored on `calib` is folded into the
    warp, so the model sees a perfectly forward-mounted camera by
    construction (openpilot's `model_transform_main` /
    `model_transform_extra` do the same thing).
    """
    narrow = warp_to_model(bgr, calib, bigmodel=False)
    wide = warp_to_model(bgr, calib, bigmodel=True)
    return narrow, wide


def yuv6_to_bgr(yuv6: np.ndarray) -> np.ndarray:
    """Reverse bgr_to_yuv6 for visualization. yuv6 is (6, MODEL_H, MODEL_W)
    uint8; returns BGR (SRC_H, SRC_W, 3) uint8. Y is exact, UV is half-res
    (same as what the model actually receives)."""
    h, w = MODEL_H, MODEL_W
    H, W = SRC_H, SRC_W
    y = np.empty((H, W), dtype=np.uint8)
    y[0::2, 0::2] = yuv6[0]
    y[1::2, 0::2] = yuv6[1]
    y[0::2, 1::2] = yuv6[2]
    y[1::2, 1::2] = yuv6[3]
    yuv_i420 = np.empty((H * 3 // 2, W), dtype=np.uint8)
    yuv_i420[:H] = y
    yuv_i420[H:H + H // 4] = yuv6[4].reshape(H // 4, W)
    yuv_i420[H + H // 4:] = yuv6[5].reshape(H // 4, W)
    return cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_I420)


class FrameQueue:
    """Holds the last 5 YUV6 frames per camera (narrow + wide) and emits
    the (1, 12, 128, 256) tensors the vision model wants — frames at
    t and t-4 stacked.

    Each `push` returns `(narrow_tensor, wide_tensor)`.

    `last_yuv_narrow` / `last_yuv_wide` are the most recent (6, 128, 256)
    frames after crop+warp+YUV. Pass either to `yuv6_to_bgr` to see
    exactly what the model is looking at. `last_yuv` aliases the narrow
    one for backward-compat with overlay code.
    """

    BUF_LEN = FRAME_SKIP + 1  # 5

    def __init__(self) -> None:
        self._buf_narrow: deque[np.ndarray] = deque(maxlen=self.BUF_LEN)
        self._buf_wide: deque[np.ndarray] = deque(maxlen=self.BUF_LEN)
        self.last_yuv_narrow: np.ndarray | None = None
        self.last_yuv_wide: np.ndarray | None = None
        # Shapes captured each frame so the overlay can compute the model's
        # effective FOV.
        self.last_captured_shape: tuple[int, int] | None = None
        self.last_narrow_shape: tuple[int, int] | None = None
        self.last_wide_shape: tuple[int, int] | None = None

    @property
    def last_yuv(self) -> np.ndarray | None:
        """Backward-compat alias — overlay code expects this name."""
        return self.last_yuv_narrow

    @property
    def last_cropped_shape(self) -> tuple[int, int] | None:
        """Backward-compat alias for code that wants 'the' cropped shape."""
        return self.last_narrow_shape

    def push(self, bgr: np.ndarray,
             calib: "Calibration | None" = None,
             ) -> tuple[np.ndarray, np.ndarray]:
        self.last_captured_shape = bgr.shape[:2]
        narrow_bgr, wide_bgr = crop_for_model(bgr, calib)
        self.last_narrow_shape = narrow_bgr.shape[:2]
        self.last_wide_shape = wide_bgr.shape[:2]

        narrow_yuv = bgr_to_yuv6(narrow_bgr)
        wide_yuv = bgr_to_yuv6(wide_bgr)
        self.last_yuv_narrow = narrow_yuv
        self.last_yuv_wide = wide_yuv

        # Pre-fill both buffers with copies of the first frame so we can
        # produce a valid stacked input from frame 1.
        while len(self._buf_narrow) < self.BUF_LEN:
            self._buf_narrow.append(narrow_yuv.copy())
            self._buf_wide.append(wide_yuv.copy())
        self._buf_narrow.append(narrow_yuv)
        self._buf_wide.append(wide_yuv)

        narrow_stack = np.concatenate(
            [self._buf_narrow[0], self._buf_narrow[-1]], axis=0)
        wide_stack = np.concatenate(
            [self._buf_wide[0], self._buf_wide[-1]], axis=0)
        return narrow_stack[np.newaxis, ...], wide_stack[np.newaxis, ...]
