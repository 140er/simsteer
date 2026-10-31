"""The model's trained time intervals, expressed in seconds."""

import math

from simsteer.core.constants import MODEL_CONTEXT_FREQ, MODEL_RUN_FREQ

FRAME_PERIOD_S = 1.0 / MODEL_RUN_FREQ
CONTEXT_PERIOD_S = 1.0 / MODEL_CONTEXT_FREQ
MIN_HEALTHY_FPS = 8.0
DISCONTINUITY_GAP_S = 2 / MIN_HEALTHY_FPS
PAIR_SPAN_TOL_S = 0.06
TIME_EPSILON_S = 1e-6

assert DISCONTINUITY_GAP_S <= CONTEXT_PERIOD_S + PAIR_SPAN_TOL_S


def frame_gap(previous: float | None, current: float) -> bool:
    return (not math.isfinite(current)
            or (previous is not None
                and (not math.isfinite(previous) or current <= previous
                     or current - previous > DISCONTINUITY_GAP_S + TIME_EPSILON_S)))


def frame_geometry(calib, shape) -> tuple:
    """Snapshot values: calibration objects can be edited in place."""
    fields = ("image_w", "image_h", "fov_h_deg", "cx_frac", "cy_frac", "crop_top_pct",
              "crop_bottom_pct", "simple_warp")
    return (tuple(shape[:2]), *(getattr(calib, key, None) for key in fields))


def frame_pose(calib) -> tuple:
    """Tag the pose used to warp a frame, including in-place refinements."""
    return tuple(getattr(calib, key, None) for key in ("pitch_deg", "yaw_deg", "height_m"))
