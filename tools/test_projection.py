"""Numerical checks for `Calibration.road_to_image` after the switch
from the 2D-decomposed formulation to openpilot's single-matrix
projection chain. Run with:

    python -m tools.test_projection

Each case prints PASS/FAIL with the actual vs expected pixel coords.

Cases (in order of increasing strictness):

1. rpy = 0 — pure pinhole. A point at (X, ±Y_off, 0) lands at
   `cx ± fx · Y_off / X`. Confirms the K matrix and frame conversions
   are wired correctly.

2. pitch-only — yaw = 0, height set. The vertical coordinate matches
   the closed-form `v = cy + fy · (cp·h − sp·X) / (cp·X + sp·h)`
   (perspective + pitch-rotated camera). u stays at cx.

3. yaw-only — pitch = 0, height set. Same idea but for the lateral.

4. Combined pitch + yaw — this is the regression case. The old code
   was off here by `fx · sin(yaw) · sin(pitch) · h_d / (depth^2)`
   terms. We check against an independent matrix-multiplication
   reference (computed inline) rather than against the closed-form,
   so the test isn't just restating the implementation.

5. Warp <-> projection round-trip — a point at the medmodel's optical
   center (cx_model, cy_model in the warped frame) should map through
   `warp_to_captured = inv(warp_dst_to_src)` to a captured pixel, and
   `road_to_image` on that point's 3D back-projection should land at
   the same pixel. This is the property that lets the overlay align
   when calibration is correct.
"""
from __future__ import annotations

import math

import numpy as np

from pilot.calibration import Calibration
from pilot.warp import (
    MEDMODEL_INTRINSICS,
    VIEW_FROM_DEVICE,
    camera_intrinsics,
    get_warp_matrix,
    rot_from_euler,
)


TOL_PX = 1e-6  # we expect float-precision agreement, not approximation
TOL_PX_F32 = 1e-3  # round-trip via cv2-compatible float32 matrices


def _calib(pitch_deg: float = 0.0, yaw_deg: float = 0.0,
           height_m: float = 1.5,
           fov_h_deg: float = 70.0,
           image_w: int = 1280, image_h: int = 720,
           lateral_sign: float = 1.0) -> Calibration:
    """A Calibration with explicit values. `lateral_sign=+1` so the
    input Y matches the FRD right-positive convention used in the
    closed-form references below."""
    return Calibration(
        image_w=image_w, image_h=image_h,
        fov_h_deg=fov_h_deg, height_m=height_m,
        pitch_deg=pitch_deg, yaw_deg=yaw_deg,
        lateral_sign=lateral_sign,
    )


def _check(name: str, got: tuple[float, float],
           want: tuple[float, float], tol: float = TOL_PX) -> bool:
    du = got[0] - want[0]
    dv = got[1] - want[1]
    ok = abs(du) < tol and abs(dv) < tol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: got ({got[0]:.4f}, {got[1]:.4f})  "
          f"want ({want[0]:.4f}, {want[1]:.4f})  err=({du:+.2e}, {dv:+.2e})")
    return ok


def case_1_zero_rpy() -> bool:
    """Pure pinhole: rpy=0. A point at (X, Y, 0) in the road frame is
    at FRD (X, Y, +h) — projection is just K · view_from_device · pt
    which collapses to u = cx + fx·Y/X, v = cy + fy·h/X."""
    print("\nCase 1: rpy = 0 (pure pinhole)")
    calib = _calib(pitch_deg=0.0, yaw_deg=0.0, height_m=1.5)
    X, Y, Z = 50.0, 1.85, 0.0
    pts = calib.road_to_image(np.array([[X, Y, Z]]))
    u, v = float(pts[0, 0]), float(pts[0, 1])
    u_want = calib.cx + calib.fx * Y / X
    v_want = calib.cy + calib.fy * calib.height_m / X
    return _check("forward + right lane", (u, v), (u_want, v_want))


def case_2_pitch_only() -> bool:
    """Yaw = 0. Closed form for the vertical:
        cp = cos(p), sp = sin(p)
        depth = cp·X + sp·h    (z_view component)
        v = cy + fy · (cp·h − sp·X) / depth"""
    print("\nCase 2: pitch only (yaw = 0)")
    p_deg = 5.0
    calib = _calib(pitch_deg=p_deg, yaw_deg=0.0, height_m=1.5)
    X, Y, Z = 40.0, 0.0, 0.0
    pts = calib.road_to_image(np.array([[X, Y, Z]]))
    u, v = float(pts[0, 0]), float(pts[0, 1])
    p = math.radians(p_deg)
    cp, sp = math.cos(p), math.sin(p)
    h = calib.height_m
    depth = cp * X + sp * h
    u_want = calib.cx + calib.fx * 0.0 / depth       # Y=0
    v_want = calib.cy + calib.fy * (cp * h - sp * X) / depth
    return _check("forward, Y=0, pitch=5 deg", (u, v), (u_want, v_want))


def case_3_yaw_only() -> bool:
    """Pitch = 0. Closed form:
        cy_t = cos(yaw), sy = sin(yaw)
        depth = cy_t·X − sy·Y
        u = cx + fx · (sy·X + cy_t·Y) / depth
        v = cy + fy · h / depth"""
    print("\nCase 3: yaw only (pitch = 0)")
    y_deg = 3.0
    calib = _calib(pitch_deg=0.0, yaw_deg=y_deg, height_m=1.5)
    X, Y, Z = 40.0, 1.85, 0.0
    pts = calib.road_to_image(np.array([[X, Y, Z]]))
    u, v = float(pts[0, 0]), float(pts[0, 1])
    y_rad = math.radians(y_deg)
    cy_t, sy = math.cos(y_rad), math.sin(y_rad)
    depth = cy_t * X - sy * Y
    u_want = calib.cx + calib.fx * (sy * X + cy_t * Y) / depth
    v_want = calib.cy + calib.fy * calib.height_m / depth
    return _check("forward, Y=+1.85, yaw=3 deg", (u, v), (u_want, v_want))


def case_4_pitch_and_yaw() -> bool:
    """Both non-zero. Independent reference: compute the matrix
    M = K · view_from_device · device_from_calib by hand, project the
    point manually, and compare. This catches a regression to the
    2D-decomposed formulation (which gets this case wrong)."""
    print("\nCase 4: pitch + yaw (the regression case)")
    p_deg, y_deg = 5.0, 3.0
    calib = _calib(pitch_deg=p_deg, yaw_deg=y_deg, height_m=1.5)
    X, Y, Z = 40.0, 1.85, 0.0
    pts = calib.road_to_image(np.array([[X, Y, Z]]))
    u, v = float(pts[0, 0]), float(pts[0, 1])

    # Reference: same matrix the implementation builds, but inline.
    K = np.array([[calib.fx, 0, calib.cx],
                  [0, calib.fy, calib.cy],
                  [0, 0, 1.0]], dtype=np.float64)
    R = rot_from_euler((0.0, math.radians(p_deg), math.radians(y_deg)))
    M = K @ VIEW_FROM_DEVICE @ R
    pt_frd = np.array([X, Y * calib.lateral_sign, calib.height_m - Z])
    h = M @ pt_frd
    u_want, v_want = float(h[0] / h[2]), float(h[1] / h[2])
    return _check("forward, Y=+1.85, pitch=5 deg, yaw=3 deg",
                  (u, v), (u_want, v_want))


def case_5_lateral_sign() -> bool:
    """lateral_sign = −1 should mirror the result of lateral_sign = +1
    across the optical axis (in u only). v should be unchanged."""
    print("\nCase 5: lateral_sign sanity check")
    base = _calib(pitch_deg=2.0, yaw_deg=0.0, lateral_sign=+1.0)
    flip = _calib(pitch_deg=2.0, yaw_deg=0.0, lateral_sign=-1.0)
    X, Y, Z = 30.0, +1.85, 0.0
    a = base.road_to_image(np.array([[X, Y, Z]]))
    b = flip.road_to_image(np.array([[X, Y, Z]]))
    u_a, v_a = float(a[0, 0]), float(a[0, 1])
    u_b, v_b = float(b[0, 0]), float(b[0, 1])
    # b's u should be 2·cx − a's u (reflection across vertical line at cx).
    u_b_want = 2 * base.cx - u_a
    return _check("u(Y=+1.85, lsign=-1) == 2*cx - u(lsign=+1)",
                  (u_b, v_b), (u_b_want, v_a))


def case_6_warp_projection_roundtrip() -> bool:
    """For any 3D road point P, two routes should give the same
    captured pixel:

      Route A:  road_to_image(P)
                  = K_cam · view_from_device · R(rpy) · P_frd

      Route B:  warp(model_pix(P))
                  = warp_mat · (K_model · view_from_device · P_frd)
                  = K_cam · view_from_device · R(rpy)
                    · inv(K_model · view_from_device)
                    · K_model · view_from_device · P_frd
                  = K_cam · view_from_device · R(rpy) · P_frd

    Algebraically they're identical, so this catches any drift between
    the warp and the projection — exactly the property that makes the
    overlay align when calibration is correct.
    """
    print("\nCase 6: warp <-> projection round-trip (rpy non-zero)")
    p_deg, y_deg = 2.5, -1.0
    calib = _calib(pitch_deg=p_deg, yaw_deg=y_deg, height_m=1.22,
                   image_w=1928, image_h=1208, fov_h_deg=40.0,
                   lateral_sign=+1.0)
    K_cam = camera_intrinsics(calib.image_w, calib.image_h, calib.fov_h_deg)
    rpy = (0.0, math.radians(p_deg), math.radians(y_deg))
    M_dst_to_src = get_warp_matrix(rpy, K_cam, bigmodel=False)

    # A road point 30 m ahead, 1 m to the right.
    X, Y, Z = 30.0, 1.0, 0.0
    pt_frd = np.array([X, Y, calib.height_m - Z])

    # Route A: road_to_image.
    a = calib.road_to_image(np.array([[X, Y, Z]]))[0]

    # Route B: project to model pixel, then warp.
    K_model_view = MEDMODEL_INTRINSICS @ VIEW_FROM_DEVICE
    model_h = K_model_view @ pt_frd
    model_pix = np.array([model_h[0] / model_h[2],
                          model_h[1] / model_h[2], 1.0])
    captured_h = M_dst_to_src @ model_pix
    captured_pix = captured_h[:2] / captured_h[2]

    return _check("road_to_image == warp(K_model * P)",
                  (float(a[0]), float(a[1])),
                  (float(captured_pix[0]), float(captured_pix[1])),
                  tol=TOL_PX_F32)


def case_7_behind_camera_nan() -> bool:
    """A point behind the camera (negative X) must project to NaN, so
    the overlay caller can filter it before drawing."""
    print("\nCase 7: behind-camera point -> NaN")
    calib = _calib(pitch_deg=2.0, yaw_deg=1.0)
    pts = calib.road_to_image(np.array([[-5.0, 0.0, 0.0]]))
    u, v = float(pts[0, 0]), float(pts[0, 1])
    ok = math.isnan(u) and math.isnan(v)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] X=-5 -> got ({u}, {v})  want (nan, nan)")
    return ok


def main() -> None:
    cases = [
        case_1_zero_rpy,
        case_2_pitch_only,
        case_3_yaw_only,
        case_4_pitch_and_yaw,
        case_5_lateral_sign,
        case_6_warp_projection_roundtrip,
        case_7_behind_camera_nan,
    ]
    results = [(f.__name__, f()) for f in cases]
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    print(f"  {passed} / {len(results)} cases passed")
    if passed < len(results):
        print("  FAILED:", [n for n, ok in results if not ok])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
