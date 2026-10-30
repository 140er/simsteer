"""`Calibration.road_to_image` against closed-form and independent
matrix references, and its agreement with the model warp."""
from __future__ import annotations

import math

import numpy as np
import pytest

from simsteer.core.calibration import Calibration
from simsteer.core.warp import (
    MEDMODEL_INTRINSICS,
    VIEW_FROM_DEVICE,
    camera_intrinsics,
    get_warp_matrix,
    rot_from_euler,
)

TOL_PX = 1e-6
# The warp matrices are float32 to match cv2.
TOL_PX_F32 = 1e-3


def _calib(pitch_deg=0.0, yaw_deg=0.0, height_m=1.5, fov_h_deg=70.0,
           image_w=1280, image_h=720, lateral_sign=1.0) -> Calibration:
    return Calibration(image_w=image_w, image_h=image_h, fov_h_deg=fov_h_deg,
                       height_m=height_m, pitch_deg=pitch_deg,
                       yaw_deg=yaw_deg, lateral_sign=lateral_sign)


def _project(calib: Calibration, x: float, y: float, z: float = 0.0):
    pts = calib.road_to_image(np.array([[x, y, z]]))
    return float(pts[0, 0]), float(pts[0, 1])


def test_zero_rpy_is_pure_pinhole():
    calib = _calib()
    X, Y = 50.0, 1.85
    want = (calib.cx + calib.fx * Y / X, calib.cy + calib.fy * calib.height_m / X)
    assert _project(calib, X, Y) == pytest.approx(want, abs=TOL_PX)


def test_pitch_only_matches_closed_form():
    calib = _calib(pitch_deg=5.0)
    X = 40.0
    p = math.radians(5.0)
    h = calib.height_m
    depth = math.cos(p) * X + math.sin(p) * h
    want = (calib.cx, calib.cy + calib.fy * (math.cos(p) * h - math.sin(p) * X) / depth)
    assert _project(calib, X, 0.0) == pytest.approx(want, abs=TOL_PX)


def test_yaw_only_matches_closed_form():
    calib = _calib(yaw_deg=3.0)
    X, Y = 40.0, 1.85
    y = math.radians(3.0)
    depth = math.cos(y) * X - math.sin(y) * Y
    want = (calib.cx + calib.fx * (math.sin(y) * X + math.cos(y) * Y) / depth,
            calib.cy + calib.fy * calib.height_m / depth)
    assert _project(calib, X, Y) == pytest.approx(want, abs=TOL_PX)


def test_pitch_and_yaw_match_matrix_reference():
    # The old 2D-decomposed projection got this combined case wrong.
    calib = _calib(pitch_deg=5.0, yaw_deg=3.0)
    X, Y = 40.0, 1.85
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1.0]])
    R = rot_from_euler((0.0, math.radians(5.0), math.radians(3.0)))
    h = K @ VIEW_FROM_DEVICE @ R @ np.array([X, Y * calib.lateral_sign, calib.height_m])
    assert _project(calib, X, Y) == pytest.approx((h[0] / h[2], h[1] / h[2]), abs=TOL_PX)


def test_lateral_sign_mirrors_u_only():
    base = _calib(pitch_deg=2.0, lateral_sign=+1.0)
    flip = _calib(pitch_deg=2.0, lateral_sign=-1.0)
    u_a, v_a = _project(base, 30.0, 1.85)
    assert _project(flip, 30.0, 1.85) == pytest.approx((2 * base.cx - u_a, v_a), abs=TOL_PX)


def test_projection_agrees_with_model_warp():
    calib = _calib(pitch_deg=2.5, yaw_deg=-1.0, height_m=1.22,
                   image_w=1928, image_h=1208, fov_h_deg=40.0)
    K_cam = camera_intrinsics(calib.image_w, calib.image_h, calib.fov_h_deg)
    rpy = (0.0, math.radians(2.5), math.radians(-1.0))
    M_dst_to_src = get_warp_matrix(rpy, K_cam, bigmodel=False)

    X, Y = 30.0, 1.0
    model_h = MEDMODEL_INTRINSICS @ VIEW_FROM_DEVICE @ np.array([X, Y, calib.height_m])
    captured = M_dst_to_src @ np.array([model_h[0] / model_h[2], model_h[1] / model_h[2], 1.0])

    want = (captured[0] / captured[2], captured[1] / captured[2])
    assert _project(calib, X, Y) == pytest.approx(want, abs=TOL_PX_F32)


def test_point_behind_camera_is_nan():
    u, v = _project(_calib(pitch_deg=2.0, yaw_deg=1.0), -5.0, 0.0)
    assert math.isnan(u) and math.isnan(v)
