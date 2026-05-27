"""Camera-mount calibration warp using openpilot's model frame.

The driving model was trained on a virtual camera with FIXED intrinsics
(focal length, principal point, image size). openpilot's modeld doesn't
feed raw frames — it warps each frame so the model always sees the
same forward-looking virtual view, regardless of where the physical
camera is mounted or how wide its FOV is.

This module reproduces that warp:

    warp = K_camera · view_from_device · device_from_calib(rpy)
                                       · view_from_device^-1 · K_model^-1

With identity rpy the warp is a pure intrinsic swap: source pixels at
the captured FOV get remapped to the model's virtual camera (medmodel:
~31° HFOV, ~16° VFOV, horizon at the top 18.6% of the frame). The
model's output (plan, lane lines, pose) is then in the calibrated
frame by construction.

Two virtual cameras (constants from openpilot's
`common/transformations/model.py`):
    medmodel  (narrow input "img"):     fl=910, cx=256, cy=47.6  in 512x256
    sbigmodel (wide   input "big_img"): fl=455, cx=256, cy=152   in 512x256

Frame conventions (openpilot's `common/transformations/camera.py`):
    device frame = FRD (Forward, Right, Down)  — vehicle-mounted basis
    view frame   = RDF (Right,   Down, Forward) — OpenCV camera basis
    rpy euler    = ZYX intrinsic [roll, pitch, yaw], radians, in device frame
    pitch > 0    = camera nose pitched DOWN
    yaw   > 0    = camera nose pitched LEFT
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pilot.calibration import Calibration


# --- Model virtual-camera intrinsics (openpilot common/transformations/model.py) ---

MEDMODEL_INPUT_SIZE = (512, 256)   # (W, H)
MEDMODEL_FL = 910.0
MEDMODEL_CX = MEDMODEL_INPUT_SIZE[0] / 2
MEDMODEL_CY = 47.6                  # horizon at top 18.6% of the frame

SBIGMODEL_INPUT_SIZE = (512, 256)
SBIGMODEL_FL = 455.0
SBIGMODEL_CX = SBIGMODEL_INPUT_SIZE[0] / 2
SBIGMODEL_CY = 152.0                # horizon at top 59% — more sky than medmodel

MEDMODEL_INTRINSICS = np.array([
    [MEDMODEL_FL, 0.0, MEDMODEL_CX],
    [0.0, MEDMODEL_FL, MEDMODEL_CY],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

SBIGMODEL_INTRINSICS = np.array([
    [SBIGMODEL_FL, 0.0, SBIGMODEL_CX],
    [0.0, SBIGMODEL_FL, SBIGMODEL_CY],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


# --- Axis permutations (openpilot common/transformations/camera.py) ---

# device frame (Forward, Right, Down)  ->  view frame (Right, Down, Forward).
DEVICE_FROM_VIEW = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)
VIEW_FROM_DEVICE = DEVICE_FROM_VIEW.T


# Precomputed: pixel-in-model -> ray-in-calib (the model's virtual camera
# is mounted with zero calibration offset by definition).
_CALIB_FROM_MEDMODEL = np.linalg.inv(MEDMODEL_INTRINSICS @ VIEW_FROM_DEVICE)
_CALIB_FROM_SBIGMODEL = np.linalg.inv(SBIGMODEL_INTRINSICS @ VIEW_FROM_DEVICE)


def rot_from_euler(rpy: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """ZYX intrinsic Euler -> rotation matrix `device_from_calib`.
    Matches openpilot's `common.transformations.orientation.rot_from_euler`.
    Input is `[roll, pitch, yaw]` in radians."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def euler_from_rot(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `rot_from_euler` for ZYX intrinsic. Returns
    `(roll, pitch, yaw)` in radians. No gimbal-lock handling — fine
    for the small mount angles calibration deals with."""
    pitch = math.asin(-float(R[2, 0]))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return (roll, pitch, yaw)


def camera_intrinsics(image_w: int, image_h: int, fov_h_deg: float,
                      cx: float | None = None,
                      cy: float | None = None) -> np.ndarray:
    """K matrix for the captured frame. Assumes square pixels.

    `fov_h_deg`: the in-game horizontal FOV setting.
    `cx, cy`: principal-point overrides (pixels). Default to image
    center — true for game-engine renders. After cropping the source
    top/bottom the original optical center is no longer at the cropped
    frame's vertical mid-line; pass the shifted cy so the warp's
    vertical math (horizon position, above/below split) stays
    geometrically correct."""
    f = (image_w / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
    if cx is None:
        cx = image_w / 2.0
    if cy is None:
        cy = image_h / 2.0
    return np.array([
        [f, 0.0, float(cx)],
        [0.0, f, float(cy)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def get_warp_matrix(rpy_calib: np.ndarray | tuple[float, float, float],
                    camera_K: np.ndarray,
                    bigmodel: bool = False) -> np.ndarray:
    """Homography that maps a pixel in the MODEL input back to its source
    pixel in the CAMERA frame. Pass to `cv2.warpPerspective(...,
    flags=WARP_INVERSE_MAP)` (or use directly with backward-warp
    sampling) — the matrix is dest-to-source.

    Mirror of openpilot's `common.transformations.model.get_warp_matrix`.
    Layout:
        warp = K_cam · view_from_device · R(rpy) · calib_from_model
    where `calib_from_model = inv(K_model · view_from_device)` is
    precomputed at module load.
    """
    calib_from_model = _CALIB_FROM_SBIGMODEL if bigmodel else _CALIB_FROM_MEDMODEL
    device_from_calib = rot_from_euler(rpy_calib)
    camera_from_calib = camera_K @ VIEW_FROM_DEVICE @ device_from_calib
    return (camera_from_calib @ calib_from_model).astype(np.float32)


def _crop_and_resize(bgr: np.ndarray,
                     calib: "Calibration | None",
                     bigmodel: bool) -> np.ndarray:
    """Simple, no-perspective-transform path: honor the top/bottom
    crop, take the central horizontal FOV slice that matches the
    model's expected HFOV, then vertical-crop SO THE HORIZON LANDS AT
    THE MODEL'S EXPECTED CY ROW, resize to 512 x 256.

    The "horizon alignment" step is what was missing in the first
    version of this function. Centered cropping put the source's
    horizon at the output's vertical midline, but medmodel was
    trained with horizon at row 47.6 (18.6% from top) and sbigmodel
    at row 152 (59.4%). With horizon mis-aligned, the model
    interpreted road pixels as sky and emitted garbage pose, which
    LiveCalib then rejected wholesale.

    Source horizon defaults to the cropped frame's vertical midline
    (correct for level cameras). pitch_deg shifts that — positive
    pitch (camera looking down) moves the horizon up in the source,
    so we sample further up to land it on the model's CY.

    Used in 'simple_warp' mode (default). The expensive perspective
    warp doesn't earn its complexity for screen-captured game frames
    — there's no fisheye to undo, no real-camera mount offset to
    rotate around.
    """
    out_size = SBIGMODEL_INPUT_SIZE if bigmodel else MEDMODEL_INPUT_SIZE
    out_w, out_h = out_size
    out_aspect = out_w / out_h        # 2.0
    h_full, w_full = bgr.shape[:2]

    if calib is None:
        return cv2.resize(bgr, out_size, interpolation=cv2.INTER_AREA)

    # 1. Top/bottom crop — hide cab roof / hood. Clamped to 0..0.45 each.
    top_pct = max(0.0, min(0.45, float(calib.crop_top_pct)))
    bot_pct = max(0.0, min(0.45, float(calib.crop_bottom_pct)))
    if top_pct > 0.0 or bot_pct > 0.0:
        top_px = int(round(h_full * top_pct))
        bot_px = int(round(h_full * bot_pct))
        bgr = bgr[top_px:h_full - bot_px, :]

    h, w = bgr.shape[:2]

    # 2. Horizontal crop to the model's expected HFOV (only if the
    # source is wider — otherwise we'd be inventing pixels).
    model_fl = SBIGMODEL_FL if bigmodel else MEDMODEL_FL
    model_cy = SBIGMODEL_CY if bigmodel else MEDMODEL_CY
    model_hfov_deg = math.degrees(2.0 * math.atan((out_w / 2.0) / model_fl))
    src_hfov_deg = float(calib.fov_h_deg)
    if src_hfov_deg > model_hfov_deg:
        crop_w = int(round(
            w * math.tan(math.radians(model_hfov_deg / 2.0))
              / math.tan(math.radians(src_hfov_deg / 2.0))))
        x0 = max(0, (w - crop_w) // 2)
        bgr = bgr[:, x0:x0 + crop_w]
        w = bgr.shape[1]

    # 3. Locate source horizon. For a level camera (pitch=0) the
    # horizon sits at the cropped frame's vertical midline. Positive
    # pitch (camera angled DOWN) shifts the horizon UP in the image,
    # so we adjust horizon_row upward. Pitch is in degrees; convert
    # to pixels using the source focal length (square pixels →
    # same fy as fx, fx = (w/2)/tan(hfov/2)).
    fx = (w / 2.0) / math.tan(math.radians(src_hfov_deg / 2.0))
    pitch_rad = math.radians(float(calib.pitch_deg))
    horizon_row = h / 2.0 - fx * math.tan(pitch_rad)

    # 4. Vertical crop sized to maintain 2:1 aspect with the
    # horizontal crop, POSITIONED so the source horizon lands at
    # `model_cy` in the output (= same fraction in our intermediate
    # crop). If the source doesn't have enough content above / below
    # horizon, pad with black — visually obvious "missing input".
    target_h = int(round(w / out_aspect))
    above = target_h * (model_cy / out_h)
    y0 = int(round(horizon_row - above))
    y1 = y0 + target_h

    pad_top = max(0, -y0)
    pad_bot = max(0, y1 - h)
    y0c = max(0, y0)
    y1c = min(h, y1)
    sliced = bgr[y0c:y1c, :]
    if pad_top > 0 or pad_bot > 0:
        sliced = cv2.copyMakeBorder(sliced, pad_top, pad_bot, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 5. Final resize to canonical model input.
    return cv2.resize(sliced, out_size, interpolation=cv2.INTER_LINEAR)


def warp_to_model(bgr: np.ndarray,
                  calib: "Calibration | None",
                  bigmodel: bool = False) -> np.ndarray:
    """Reproject a captured BGR frame onto the model's virtual camera.

    Output is always 256 x 512 BGR — the model-input canonical size.

    Two paths:
      - simple (default, `calib.simple_warp = True`): crop the source
        to the model's expected HFOV, aspect-correct vertical crop,
        resize. No perspective transform, no rotation. Faster, easier
        to reason about, and matches what you'd intuitively expect
        when feeding a screen-captured game frame to a model — the
        model sees the central forward view at the right zoom.
      - perspective (`calib.simple_warp = False`): the full openpilot-
        style warp: K_cam · view_from_device · R(rpy) · K_model^-1.
        Folds in LiveCalib's pitch/yaw rotation. Pre-existing, kept
        for cases where rotation correction matters more than image
        cleanliness.

    Source cropping (`calib.crop_top_pct`, `calib.crop_bottom_pct`)
    applies in both paths.

    Out-of-frame samples use BORDER_CONSTANT(black) so it's obvious
    when calibration is asking for content the source doesn't have.
    """
    out_size = SBIGMODEL_INPUT_SIZE if bigmodel else MEDMODEL_INPUT_SIZE
    h_full, w_full = bgr.shape[:2]
    if calib is None:
        if (w_full, h_full) != out_size:
            bgr = cv2.resize(bgr, out_size, interpolation=cv2.INTER_AREA)
        return bgr

    if getattr(calib, "simple_warp", True):
        return _crop_and_resize(bgr, calib, bigmodel)

    # Perspective path. Apply top/bottom crop, then full warp.
    top_pct = max(0.0, min(0.45, float(calib.crop_top_pct)))
    bot_pct = max(0.0, min(0.45, float(calib.crop_bottom_pct)))
    if top_pct > 0.0 or bot_pct > 0.0:
        top_px = int(round(h_full * top_pct))
        bot_px = int(round(h_full * bot_pct))
        bgr = bgr[top_px:h_full - bot_px, :]
        h, w = bgr.shape[:2]
        cy_shifted = (h_full * 0.5) - top_px
        K_cam = camera_intrinsics(w, h, calib.fov_h_deg, cy=cy_shifted)
    else:
        h, w = h_full, w_full
        K_cam = camera_intrinsics(w, h, calib.fov_h_deg)

    rpy = calib.rpy_calib_rad()
    M_dst_to_src = get_warp_matrix(rpy, K_cam, bigmodel=bigmodel)
    return cv2.warpPerspective(
        bgr, M_dst_to_src, out_size,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
