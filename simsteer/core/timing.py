"""The model's trained time intervals, expressed in seconds."""

from simsteer.core.constants import MODEL_CONTEXT_FREQ, MODEL_RUN_FREQ

FRAME_PERIOD_S = 1.0 / MODEL_RUN_FREQ
CONTEXT_PERIOD_S = 1.0 / MODEL_CONTEXT_FREQ
MAX_FRAME_GAP_S = 2 * FRAME_PERIOD_S
TIME_EPSILON_S = 1e-6


def frame_gap(previous: float | None, current: float) -> bool:
    return (previous is not None
            and (current <= previous
                 or current - previous > MAX_FRAME_GAP_S + TIME_EPSILON_S))


def frame_geometry(calib, shape) -> tuple:
    """Snapshot values: calibration objects can be edited in place."""
    fields = ("image_w", "image_h", "fov_h_deg", "pitch_deg", "yaw_deg",
              "height_m", "cx_frac", "cy_frac", "crop_top_pct",
              "crop_bottom_pct", "simple_warp")
    return (tuple(shape[:2]), *(getattr(calib, key, None) for key in fields))
