"""Overlay renderer: lane lines, plan path, leads, and tuning HUD.

Pure drawing surface — given a captured frame, the model's `decoded`
output, and a `Calibration`, it projects the model's road-frame
predictions back onto the image and draws them. It holds no state and
runs no model; `runtime/loop.py` owns the capture/inference loop and
calls these functions each frame.

The projections are exact ports of v1's tuned geometry (horizon-row
fix, device-Z sign flip, lag markers); do not adjust the math here —
calibration is what makes them correct, not the renderer.
"""

from __future__ import annotations

import cv2
import numpy as np

from simsteer.core.calibration import Calibration, polyline_to_image
from simsteer.core.preprocess import yuv6_to_bgr

LANE_COLORS = [
    (180, 220, 255),  # outer left  (light blue/white — adjacent lane)
    (80, 255, 255),   # left lane   (yellow — current lane)
    (80, 255, 255),   # right lane  (yellow — current lane)
    (180, 220, 255),  # outer right (light blue/white — adjacent lane)
]
ROAD_EDGE_COLOR = (40, 40, 220)   # red — same as openpilot's UI

# Visualization caps. The model's X_IDXS go to 192 m / its plan covers
# 10 s, but very-far points project to within a few pixels of the
# horizon — small angle errors there become huge pixel jumps. We let
# the plan + path wedge run further than the lanes so you can see what
# the model intends across the full corner-anticipation horizon, while
# keeping the lane polylines tighter to where they still look clean.
MAX_DRAW_DIST_M = 120.0           # plan center line + path wedge
INNER_LANE_DRAW_DIST_M = 90.0     # current-lane yellow lines
OUTER_LANE_DRAW_DIST_M = 60.0     # adjacent-lane light-blue lines
ROAD_EDGE_DRAW_DIST_M = 60.0      # red road edges

PATH_HALF_WIDTH_M = 1.8   # half-width of the green path wedge; full lane ~3.6 m
PATH_FILL_BGR = (40, 200, 40)
PATH_ALPHA = 0.35

# Desire labels — openpilot's `desire_state` softmax dimension order. Used
# for the "DESIRE" indicator in the top-left.
DESIRE_LABELS = [
    "none", "turn-L", "turn-R", "lane-L", "lane-R", "keep-L", "keep-R", "null",
]

WINDOW_NAME = "SimSteer overlay"


def setup_window(initial_w: int | None = None, initial_h: int | None = None) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if initial_w and initial_h:
        cv2.resizeWindow(WINDOW_NAME, initial_w, initial_h)


def show_scaled(img: np.ndarray, max_width: int) -> None:
    """Downscale the framebuffer for display when wider than max_width.
    The window itself is resizable, but we cap the underlying buffer so a
    3440-wide capture doesn't try to render at native size into a 2560-wide
    monitor before the user has dragged it.
    """
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW_NAME, img)


def _polyline_pts(calib: Calibration, xs: np.ndarray, ys: np.ndarray,
                  zs_device: np.ndarray | None = None,
                  max_x: float = MAX_DRAW_DIST_M) -> np.ndarray | None:
    """Convert (X, Y[, Z]) road points to a Nx1x2 int32 polyline, or None
    if everything is behind the camera / above the horizon. Points beyond
    `max_x` meters are dropped — they project to almost the horizon and any
    angle error blows up to huge pixel offsets that look terrible.

    `zs_device` is the model's Z in device frame (+down, FRD); we flip
    it to the +up convention `polyline_to_image` expects so the path
    follows hills/bumps."""
    mask = xs < max_x
    xs = xs[mask]
    ys = ys[mask]
    zs = None
    if zs_device is not None:
        zs = _device_z_to_road_z(zs_device[mask])
    img = polyline_to_image(calib, xs, ys, zs)
    valid = ~np.isnan(img[:, 0])
    if not valid.any():
        return None
    pts = img[valid].astype(np.int32).reshape(-1, 1, 2)
    return pts if len(pts) >= 2 else None


def _accel_to_color(a: float) -> tuple[int, int, int]:
    """Map m/s² to a BGR color. Braking (negative) = red; coasting = green;
    accelerating = blue. Sharp braking saturates at 5 m/s²."""
    a_clip = max(-5.0, min(5.0, a))
    if a_clip <= 0.0:
        # 0 → green, -5 → red. Lerp green to red.
        t = -a_clip / 5.0
        b = int(40 + 0 * t)
        g = int(200 * (1 - t) + 60 * t)
        r = int(40 * (1 - t) + 240 * t)
        return (b, g, r)
    else:
        # 0 → green, +5 → cyan/blue. Lerp green to blue.
        t = a_clip / 5.0
        b = int(40 * (1 - t) + 240 * t)
        g = int(200)
        r = int(40 * (1 - t) + 80 * t)
        return (b, g, r)


def _device_z_to_road_z(z_device: np.ndarray) -> np.ndarray:
    """Convert the model's device-frame Z (+down, FRD) to the
    Calibration.road_to_image Z convention (+up, height above road).
    A road point ahead at the same elevation as the camera has the
    model emitting z ≈ +height_m (point is height_m below the camera
    in FRD); we want this to land as Z = 0 in the projection
    (on the road plane). So we negate AND don't subtract camera
    height — the projection itself already does `height_m - Z`,
    and the model already encodes elevation from-road in its
    near-field samples (z[0] ≈ 0 on flat ground). The only thing
    we need is the sign flip from +down to +up."""
    return -z_device


def _draw_path_wedge(out: np.ndarray, calib: Calibration,
                     plan_xyz: np.ndarray,
                     plan_accel: np.ndarray | None = None) -> None:
    """Fill the path wedge between (center ± half_width), with each
    longitudinal segment colored by the plan's commanded acceleration
    at that distance. Braking sections turn red, accelerating sections
    blue, cruise green — directly visualizes the model's longitudinal
    intent along the path.

    `plan_xyz` is the model's predicted trajectory in device frame —
    (X forward, Y lateral, Z down per FRD). We carry the Z through to
    the projection so the wedge follows hills and bumps instead of
    sitting on a fixed-pitch ground plane (matches openpilot's path
    rendering — they always project with per-frame Z).

    `plan_accel` is the (33,) longitudinal accel column from the plan.
    If None, falls back to a single flat green wedge.
    """
    xs = plan_xyz[:, 0]
    ys = plan_xyz[:, 1]
    zs = _device_z_to_road_z(plan_xyz[:, 2]) if plan_xyz.shape[1] >= 3 else None
    mask = xs < MAX_DRAW_DIST_M
    xs = xs[mask]
    ys = ys[mask]
    if zs is not None:
        zs = zs[mask]
    if plan_accel is not None:
        plan_accel = plan_accel[mask]
    if len(xs) < 2:
        return

    left_img = polyline_to_image(calib, xs, ys + PATH_HALF_WIDTH_M, zs)
    right_img = polyline_to_image(calib, xs, ys - PATH_HALF_WIDTH_M, zs)
    valid = ~np.isnan(left_img[:, 0]) & ~np.isnan(right_img[:, 0])
    if valid.sum() < 2:
        return

    left = left_img[valid].astype(np.int32)
    right = right_img[valid].astype(np.int32)
    if plan_accel is None:
        # Legacy flat-green path — single fillPoly.
        poly = np.vstack([left, right[::-1]])
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], PATH_FILL_BGR, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, PATH_ALPHA, out, 1.0 - PATH_ALPHA, 0, dst=out)
        return

    # Color each quad between i and i+1 by the avg accel of those samples.
    a_valid = plan_accel[valid]
    overlay = out.copy()
    for i in range(len(left) - 1):
        quad = np.array([left[i], left[i + 1], right[i + 1], right[i]],
                        dtype=np.int32)
        a_avg = float((a_valid[i] + a_valid[i + 1]) / 2.0)
        color = _accel_to_color(a_avg)
        cv2.fillPoly(overlay, [quad], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, PATH_ALPHA, out, 1.0 - PATH_ALPHA, 0, dst=out)


# Times at which we drop a labeled marker along the plan path. Chosen
# to be visually distinct without crowding. The 10 s endpoint lines up
# with the plan's own horizon, so a missing 10 s marker means the plan
# at that time is past `MAX_DRAW_DIST_M` (i.e. the truck is moving
# fast enough that 10 s of plan is further than we draw).
PLAN_TIMESTEP_MARKERS_S = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)


def _draw_timestep_markers(out: np.ndarray, calib: Calibration,
                           plan: np.ndarray) -> None:
    """Drop a small filled circle + label at fixed future times along
    the plan path so you can see 'where the model thinks we'll be in 1s,
    2s, 3s, 5s, 7s, 10s.'"""
    from simsteer.core.constants import T_IDXS
    t_idxs = np.asarray(T_IDXS, dtype=np.float32)
    xs = plan[:, 0]
    ys = plan[:, 1]
    zs_dev = plan[:, 2] if plan.shape[1] >= 3 else None
    for t_mark in PLAN_TIMESTEP_MARKERS_S:
        x_at_t = float(np.interp(t_mark, t_idxs, xs))
        y_at_t = float(np.interp(t_mark, t_idxs, ys))
        if x_at_t > MAX_DRAW_DIST_M or x_at_t < 1.0:
            continue
        z_at_t = (-float(np.interp(t_mark, t_idxs, zs_dev))
                  if zs_dev is not None else 0.0)
        img = polyline_to_image(
            calib,
            np.asarray([x_at_t], dtype=np.float32),
            np.asarray([y_at_t], dtype=np.float32),
            np.asarray([z_at_t], dtype=np.float32))
        if np.isnan(img[0, 0]):
            continue
        px, py = int(img[0, 0]), int(img[0, 1])
        cv2.circle(out, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 5, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(out, f"{t_mark:g}s", (px + 8, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, f"{t_mark:g}s", (px + 8, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_plan_charts(out: np.ndarray, decoded) -> None:
    """Bottom-left mini-charts: planned velocity (m/s) and acceleration
    (m/s²) over the plan horizon. Lets the user see at a glance whether
    the model is planning to brake / accelerate, and where in time.

    Shares the same x-axis (plan time, 0..10 s).
    """
    from simsteer.core.constants import T_IDXS
    H, W = out.shape[:2]
    chart_w, chart_h = 240, 80
    pad = 6
    # Stack two charts vertically in the lower-left, above the
    # calibration HUD which sits at the bottom edge.
    x0 = 10
    y_top_chart = H - 220
    y_bot_chart = y_top_chart + chart_h + 8

    t_idxs = np.asarray(T_IDXS, dtype=np.float32)
    v = decoded.plan[:, 3].astype(np.float64)
    a = decoded.plan[:, 6].astype(np.float64)

    def _draw_axes(x, y, w, h, title, y_min, y_max, color_zero_line=False):
        # Background panel.
        cv2.rectangle(out, (x - 2, y - 2), (x + w + 2, y + h + 2),
                      (30, 30, 30), -1)
        cv2.rectangle(out, (x - 2, y - 2), (x + w + 2, y + h + 2),
                      (200, 200, 200), 1)
        # Title + y-axis bounds.
        cv2.putText(out, title, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(out, f"{y_max:+.1f}", (x + w - 28, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160),
                    1, cv2.LINE_AA)
        cv2.putText(out, f"{y_min:+.1f}", (x + w - 28, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160),
                    1, cv2.LINE_AA)
        if color_zero_line and y_min < 0 < y_max:
            y_zero = y + int(h * (y_max / (y_max - y_min)))
            cv2.line(out, (x, y_zero), (x + w, y_zero),
                     (110, 110, 110), 1, cv2.LINE_AA)

    def _plot_series(values, x, y, w, h, y_min, y_max, color):
        n = len(values)
        if n < 2:
            return
        pts = []
        for i, val in enumerate(values):
            px = x + int(w * i / (n - 1))
            v_clip = max(y_min, min(y_max, float(val)))
            py = y + int(h * (1 - (v_clip - y_min) / (y_max - y_min)))
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(out, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    # Velocity chart: 0..40 m/s
    v_lo, v_hi = 0.0, max(40.0, float(v.max()) * 1.1)
    _draw_axes(x0, y_top_chart, chart_w, chart_h, "plan v (m/s)",
               v_lo, v_hi)
    _plot_series(v, x0, y_top_chart, chart_w, chart_h, v_lo, v_hi,
                 (80, 220, 80))

    # Acceleration chart: -5..+3 m/s²
    a_lo = min(-5.0, float(a.min()) - 0.5)
    a_hi = max(3.0, float(a.max()) + 0.5)
    _draw_axes(x0, y_bot_chart, chart_w, chart_h, "plan a (m/s^2)",
               a_lo, a_hi, color_zero_line=True)
    # Color the accel line per sample so it visually matches the path.
    n = len(a)
    for i in range(1, n):
        px0 = x0 + int(chart_w * (i - 1) / (n - 1))
        px1 = x0 + int(chart_w * i / (n - 1))
        a0 = max(a_lo, min(a_hi, float(a[i - 1])))
        a1 = max(a_lo, min(a_hi, float(a[i])))
        py0 = y_bot_chart + int(chart_h * (1 - (a0 - a_lo) / (a_hi - a_lo)))
        py1 = y_bot_chart + int(chart_h * (1 - (a1 - a_lo) / (a_hi - a_lo)))
        seg_color = _accel_to_color(0.5 * (a[i - 1] + a[i]))
        cv2.line(out, (px0, py0), (px1, py1), seg_color, 2, cv2.LINE_AA)


def _draw_horizon_line(out: np.ndarray, calib: Calibration) -> None:
    """Thin cyan horizontal line at v_horizon = cy - fy*tan(pitch).
    Lets the user visually verify whether their FOV/pitch matches the
    actual road horizon. If this line doesn't sit on the visible road's
    vanishing point, the calibration is off."""
    import math
    p = math.radians(calib.pitch_deg)
    v = int(round(calib.cy - calib.fy * math.tan(p)))
    h, w = out.shape[:2]
    if 0 <= v < h:
        cv2.line(out, (0, v), (w, v), (255, 200, 0), 1, cv2.LINE_AA)


LEAD_MIN_PROB = 0.3   # below this, don't draw a chevron — too uncertain
LEAD_COLORS = [
    (60, 60, 230),    # lead 0 — red (highest-prob)
    (60, 180, 230),   # lead 1 — orange
    (60, 230, 230),   # lead 2 — yellow
]


def _draw_lead_indicator(out: np.ndarray, decoded) -> None:
    """Top-center bar of three lead probabilities."""
    H, W = out.shape[:2]
    probs = decoded.lead_prob.astype(np.float64)
    bar_w, bar_h = 40, 8
    gap = 6
    total_w = 3 * bar_w + 2 * gap
    x0 = W // 2 - total_w // 2
    y0 = 14
    for i, p in enumerate(probs):
        x = x0 + i * (bar_w + gap)
        cv2.rectangle(out, (x, y0), (x + bar_w, y0 + bar_h),
                      (60, 60, 60), -1)
        fill = int(round(bar_w * float(p)))
        if fill > 0:
            color = LEAD_COLORS[i] if p > LEAD_MIN_PROB else (60, 200, 220)
            cv2.rectangle(out, (x, y0), (x + fill, y0 + bar_h), color, -1)
        cv2.rectangle(out, (x, y0), (x + bar_w, y0 + bar_h),
                      (200, 200, 200), 1)
        cv2.putText(out, f"L{i+1} {p:.2f}", (x, y0 + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, f"L{i+1} {p:.2f}", (x, y0 + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_lead_chevrons(out: np.ndarray, calib: Calibration, decoded) -> None:
    """Triangle marker on the road for each lead with prob > threshold.

    Reads `decoded.leads` shape (3, 6, 4): for each of 3 leads, 6 future
    timesteps of (x, y, v, a) in vehicle frame. We draw a chevron at
    the lead's current position (timestep 0) and label it with its
    distance + relative velocity. Chevron size encodes confidence.
    """
    if not hasattr(decoded, "leads"):
        return
    H, W = out.shape[:2]
    for i in range(decoded.leads.shape[0]):
        p = float(decoded.lead_prob[i])
        if p < LEAD_MIN_PROB:
            continue
        x_m = float(decoded.leads[i, 0, 0])
        y_m = float(decoded.leads[i, 0, 1])
        v_rel = float(decoded.leads[i, 0, 2])
        # Skip negative distance (behind us) or absurd values.
        if x_m <= 1.0 or x_m > 250.0:
            continue
        img = polyline_to_image(calib,
                                np.asarray([x_m], dtype=np.float32),
                                np.asarray([y_m], dtype=np.float32))
        if np.isnan(img[0, 0]):
            continue
        px, py = int(img[0, 0]), int(img[0, 1])
        # Chevron size shrinks with distance (objects look smaller far
        # away) but never below a readable floor. Confidence scales the
        # opacity / fill alpha.
        size = max(10, min(60, int(800 / max(x_m, 4.0))))
        color = LEAD_COLORS[i]
        # Downward-pointing triangle (chevron) centered on (px, py).
        tri = np.array([[px, py - size // 2],
                        [px - size, py + size // 2],
                        [px + size, py + size // 2]], dtype=np.int32)
        alpha = 0.4 + 0.5 * p
        overlay = out.copy()
        cv2.fillPoly(overlay, [tri], color, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, dst=out)
        cv2.polylines(out, [tri], True, color, 2, cv2.LINE_AA)
        # Distance + relative velocity label above the chevron.
        label = f"L{i+1}  {x_m:5.1f} m  {v_rel:+.1f} m/s"
        ty = max(py - size // 2 - 10, 14)
        cv2.putText(out, label, (px - 60, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (px - 60, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)




def _draw_desire_indicator(out: np.ndarray, decoded) -> None:
    """Top-left readout of the policy's softmax over desires. Most-likely
    desire highlighted; tail probability shown for non-trivial alternates."""
    desires = decoded.desire_state.astype(np.float64)
    top = int(desires.argmax())
    label = DESIRE_LABELS[top] if 0 <= top < len(DESIRE_LABELS) else f"d{top}"
    color = (80, 255, 80) if top != 0 else (200, 200, 200)
    txt = f"DESIRE: {label} ({desires[top]:.2f})"
    cv2.putText(out, txt, (10, out.shape[0] - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (10, out.shape[0] - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_lanes_and_edges(out: np.ndarray, calib: Calibration, decoded) -> None:
    """Draw the model's 4 lane lines + 2 road edges as polylines that
    ride the road surface the model sees.

    Each line is 33 points at the fixed `X_IDXS`; column 0 is the
    lateral offset (y). Column 1 is NOT usable as height (noisy, ranges
    to ~-14 m), so for elevation we sample the PLAN's clean Z profile
    (decoded.plan[:, 2], the road height the model sees) at each lane
    x-position via interpolation. That puts the lanes on the SAME
    surface as the green path (which follows plan Z) — including over
    hills/dips — instead of a flat plane. Inner lane lines (1, 2) bound
    the current lane and draw further out than the outer adjacent pair
    (0, 3); low-confidence lines are skipped."""
    from simsteer.core.constants import X_IDXS
    xs = np.asarray(X_IDXS, dtype=np.float64)
    # Road elevation (device Z, +down) at each lane x, taken from the
    # plan's clean Z so lanes/edges follow the same surface as the path.
    plan_x = decoded.plan[:, 0].astype(np.float64)
    plan_z = decoded.plan[:, 2].astype(np.float64)
    surf_z = np.interp(xs, plan_x, plan_z)
    probs = decoded.lane_lines_prob
    for i in range(decoded.lane_lines.shape[0]):
        if i < len(probs) and float(probs[i]) < 0.2:
            continue
        ys = decoded.lane_lines[i, :, 0].astype(np.float64)
        max_x = INNER_LANE_DRAW_DIST_M if i in (1, 2) else OUTER_LANE_DRAW_DIST_M
        pts = _polyline_pts(calib, xs, ys, zs_device=surf_z, max_x=max_x)
        if pts is not None:
            cv2.polylines(out, [pts], False, LANE_COLORS[i], 2, cv2.LINE_AA)
    for i in range(decoded.road_edges.shape[0]):
        ys = decoded.road_edges[i, :, 0].astype(np.float64)
        pts = _polyline_pts(calib, xs, ys, zs_device=surf_z,
                            max_x=ROAD_EDGE_DRAW_DIST_M)
        if pts is not None:
            cv2.polylines(out, [pts], False, ROAD_EDGE_COLOR, 2, cv2.LINE_AA)


def draw_overlay(frame_bgr: np.ndarray, decoded, calib: Calibration) -> np.ndarray:
    out = frame_bgr.copy()
    calib.update_for_frame(out.shape)

    # Horizon line (cyan) — shows where the projection thinks the road
    # vanishing point is. Compare to where the road actually meets the sky.
    _draw_horizon_line(out, calib)

    # Path wedge (translucent, drawn under the plan center + lanes).
    # Colored by planned longitudinal accel — braking segments turn red,
    # accelerating segments blue.
    plan_xyz = decoded.plan[:, :3].astype(np.float64)
    plan_a = decoded.plan[:, 6].astype(np.float64)
    _draw_path_wedge(out, calib, plan_xyz, plan_accel=plan_a)

    # Plan center line — pass Z so it follows hills/dips.
    plan_pts = _polyline_pts(calib, plan_xyz[:, 0], plan_xyz[:, 1],
                             zs_device=plan_xyz[:, 2])
    if plan_pts is not None:
        cv2.polylines(out, [plan_pts], False, (0, 255, 0), 2, cv2.LINE_AA)

    # Lane lines (4) + road edges (2) — the model's other outputs,
    # drawn over the green path wedge.
    _draw_lanes_and_edges(out, calib, decoded)

    # Timestep markers along the plan path — small dots at 1, 2, 3, 5,
    # 8 seconds into the future. Helps eyeball where the plan thinks
    # we'll be in N seconds and whether that's reasonable.
    _draw_timestep_markers(out, calib, decoded.plan)

    # Lead vehicles — proper road-projected chevrons, sized by distance
    # and faded by confidence.
    _draw_lead_chevrons(out, calib, decoded)

    # Lead probabilities + desire state — top of the frame.
    _draw_lead_indicator(out, decoded)
    _draw_desire_indicator(out, decoded)

    # Bottom-left mini-charts of planned velocity + acceleration. The
    # accel chart is colored to match the path wedge above (red/blue/
    # green by sign), so visually you can correlate a red section of
    # the road wedge with a dip in the accel chart.
    _draw_plan_charts(out, decoded)

    return out


def draw_calibration_hud(img: np.ndarray, calib: Calibration) -> None:
    """Bottom-left readout of the live calibration values."""
    h = img.shape[0]
    lines = [
        f"CAP FOV     : V={calib.fov_v_deg:5.1f}° / H={calib.fov_h_deg:5.1f}°",
        f"PITCH ,/.   : {calib.pitch_deg:+5.1f} deg",
        f"YAW         : {calib.yaw_deg:+5.1f} deg",
        f"HEIGHT ;/'  : {calib.height_m:4.2f} m",
        f"LATERAL m   : {calib.lateral_sign:+.0f}",
        "C save  0 reset  I model input",
    ]
    y = h - 18 * len(lines) - 10
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        y += 18


CALIB_KEYS: dict[int, tuple[str, float]] = {
    ord("["): ("fov_v_deg", -1.0),
    ord("]"): ("fov_v_deg", +1.0),
    ord(","): ("pitch_deg", -0.5),
    ord("."): ("pitch_deg", +0.5),
    ord(";"): ("height_m", -0.05),
    ord("'"): ("height_m", +0.05),
}


def draw_model_input_inset(img: np.ndarray,
                           yuv_narrow: np.ndarray | None,
                           yuv_wide: np.ndarray | None = None,
                           target_w: int = 480) -> None:
    """Draw what the model is actually seeing — narrow on top, wide
    below it — in the top-right corner. Both tiles are labeled so you
    can verify the wide branch is producing a meaningfully different
    image (different framing / FOV). If they look identical the wide
    pipeline is broken upstream."""
    if yuv_narrow is None and yuv_wide is None:
        return
    H, W = img.shape[:2]
    margin = 12
    x0 = W - target_w - margin
    y = margin

    def _blit(yuv: np.ndarray, label: str) -> None:
        nonlocal y
        preview = yuv6_to_bgr(yuv)
        h, w = preview.shape[:2]
        scale = target_w / w
        th = int(round(h * scale))
        preview = cv2.resize(preview, (target_w, th),
                             interpolation=cv2.INTER_AREA)
        img[y:y + th, x0:x0 + target_w] = preview
        cv2.rectangle(img, (x0 - 1, y - 1), (x0 + target_w, y + th),
                      (0, 255, 255), 2)
        cv2.putText(img, label, (x0 + 6, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 6, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        y += th + 6

    if yuv_narrow is not None:
        _blit(yuv_narrow, "narrow (medmodel ~31° HFOV)")
    if yuv_wide is not None:
        _blit(yuv_wide, "wide (sbigmodel ~59° HFOV)")
    cv2.putText(img, "press I to hide", (x0, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "press I to hide", (x0, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def handle_calibration_key(calib: Calibration, key: int) -> str | None:
    """Mutate calib in-place based on key. Returns a status string, or None."""
    if key in CALIB_KEYS:
        attr, delta = CALIB_KEYS[key]
        setattr(calib, attr, getattr(calib, attr) + delta)
        return f"{attr} -> {getattr(calib, attr):+.2f}"
    if key == ord("c"):
        calib.save()
        return "saved calibration.json"
    if key == ord("m"):
        calib.lateral_sign = -calib.lateral_sign
        return f"lateral_sign -> {calib.lateral_sign:+.0f}"
    if key == ord("0"):
        defaults = Calibration()
        for f in Calibration.__dataclass_fields__:
            if f not in ("image_w", "image_h"):
                setattr(calib, f, getattr(defaults, f))
        return "reset to defaults"
    return None
