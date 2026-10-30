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
from simsteer.ui.hud import bar_height
from simsteer.ui.text import draw_text, text_width

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
        draw_text(out, f"{t_mark:g}s", px + 9, py - 6, (0, 0, 0), 12, True)
        draw_text(out, f"{t_mark:g}s", px + 8, py - 7, (255, 255, 255), 12, True)


def _draw_plan_charts(out, decoded, x, y, width, height=110):
    """Two quiet, time-aligned plan charts inside the developer column."""
    chart_h = (height - 34) // 2
    for index, (values, title) in enumerate([
        (decoded.plan[:, 3], "Plan speed, m/s"),
        (decoded.plan[:, 6], "Plan accel, m/s²"),
    ]):
        top = y + index * (chart_h + 17)
        draw_text(out, title, x, top, (164, 164, 164), 10)
        low = 0.0 if index == 0 else min(-5.0, float(values.min()) - .5)
        high = max(40.0, float(values.max()) * 1.1) if index == 0 else max(3.0, float(values.max()) + .5)
        plot_y = top + 17
        draw_text(out, f"{high:+.0f}", x, plot_y - 3, (140, 140, 140), 9)
        draw_text(out, f"{low:+.0f}", x, plot_y + chart_h - 13, (140, 140, 140), 9)
        plot_x, plot_w = x + 30, width - 34
        for val in (low, 0.0, high):
            if low <= val <= high:
                py = plot_y + int(chart_h * (high - val) / (high - low))
                cv2.line(out, (plot_x, py), (plot_x + plot_w, py), (42, 43, 41), 1)
        pts = []
        for i, val in enumerate(values):
            px = plot_x + int(plot_w * i / max(1, len(values) - 1))
            py = plot_y + int(chart_h * (high - np.clip(val, low, high)) / (high - low))
            pts.append((px, py))
        for i in range(1, len(pts)):
            color = (100, 220, 100) if index == 0 else _accel_to_color(float(values[i]))
            cv2.line(out, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)


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
        draw_text(out, label, px - 59, ty - 11, (0, 0, 0), 12, True)
        draw_text(out, label, px - 60, ty - 12, color, 12, True)




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


def _draw_lanes_and_edges(out: np.ndarray, calib: Calibration, decoded,
                          developer: bool = False) -> None:
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
        if not developer and i not in (1, 2):
            continue
        if i < len(probs) and float(probs[i]) < 0.2:
            continue
        ys = decoded.lane_lines[i, :, 0].astype(np.float64)
        max_x = INNER_LANE_DRAW_DIST_M if i in (1, 2) else OUTER_LANE_DRAW_DIST_M
        pts = _polyline_pts(calib, xs, ys, zs_device=surf_z, max_x=max_x)
        if pts is not None:
            cv2.polylines(out, [pts], False, LANE_COLORS[i], 2, cv2.LINE_AA)
    if not developer:
        return
    for i in range(decoded.road_edges.shape[0]):
        ys = decoded.road_edges[i, :, 0].astype(np.float64)
        pts = _polyline_pts(calib, xs, ys, zs_device=surf_z,
                            max_x=ROAD_EDGE_DRAW_DIST_M)
        if pts is not None:
            cv2.polylines(out, [pts], False, ROAD_EDGE_COLOR, 2, cv2.LINE_AA)


def draw_overlay(frame_bgr: np.ndarray, decoded, calib: Calibration, *,
                 engaged: bool = False, mode: str = "user") -> np.ndarray:
    out = frame_bgr.copy()
    calib.update_for_frame(out.shape)
    developer = mode == "dev"
    if developer:
        _draw_horizon_line(out, calib)
    plan_xyz = decoded.plan[:, :3].astype(np.float64)
    if engaged:
        accel = decoded.plan[:, 6].astype(np.float64) if developer else None
        _draw_path_wedge(out, calib, plan_xyz, plan_accel=accel)
    plan_pts = _polyline_pts(calib, plan_xyz[:, 0], plan_xyz[:, 1],
                             zs_device=plan_xyz[:, 2])
    if plan_pts is not None:
        cv2.polylines(out, [plan_pts], False,
                      (100, 230, 100) if engaged else (255, 255, 255), 2, cv2.LINE_AA)
    _draw_lanes_and_edges(out, calib, decoded, developer)
    if developer:
        _draw_timestep_markers(out, calib, decoded.plan)
    _draw_lead_chevrons(out, calib, decoded)
    return out


def _wrap_diagnostics(lines, width, size):
    rows = []
    for text, color in lines:
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and text_width(candidate, size) > width:
                rows.append((current, color))
                current = word
            else:
                current = candidate
        if current:
            rows.append((current, color))
    return rows


def draw_developer_panel(img, decoded, calib, state, yuv_narrow=None, yuv_wide=None):
    """Model inputs, camera, charts, the complete diagnostic dump and keys.

    The two model branches sit side by side so the column leaves room for
    every readout. Its typography adapts to the replay/capture height.
    """
    h, w = img.shape[:2]
    top = bar_height(state, "dev") + 12
    width = min(460, max(280, w // 3 if h >= 650 else int(w * .46)))
    x = w - width - 12
    if top >= h - 40 or x < 0:
        return
    cv2.rectangle(img, (x, top), (w - 12, h - 12), (24, 26, 23), -1)
    x += 12
    width -= 24
    y = top + 10
    inputs = [(yuv_narrow, "Narrow"), (yuv_wide, "Wide")]
    inputs = [(arr, name) for arr, name in inputs if arr is not None]
    if inputs:
        tile_w = (width - 8 * (len(inputs) - 1)) // len(inputs)
        tile_h = min(105 if h >= 650 else 48, tile_w // 2)
        for i, (arr, name) in enumerate(inputs):
            preview = cv2.resize(yuv6_to_bgr(arr), (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            px = x + i * (tile_w + 8)
            if y + tile_h < h - 12:
                img[y:y + tile_h, px:px + tile_w] = preview
                draw_text(img, name, px + 4, y + 3, (255, 255, 255), 10, True)
        y += tile_h + 8
    size = 12 if h >= 650 else 10
    for label, value in [
        ("FOV", f"V {calib.fov_v_deg:.1f}°  H {calib.fov_h_deg:.1f}°"),
        ("Pitch / Yaw", f"{calib.pitch_deg:+.1f}°  {calib.yaw_deg:+.1f}°"),
        ("Height / Lateral", f"{calib.height_m:.2f} m  {calib.lateral_sign:+.0f}"),
    ]:
        draw_text(img, label, x, y, (164, 164, 164), size)
        draw_text(img, value, x + width - text_width(value, size) - 3, y,
                  (235, 235, 235), size)
        y += size + 5
    chart_height = 110 if h >= 650 else 68
    if y + chart_height < h - 50:
        _draw_plan_charts(img, decoded, x, y + 2, width, chart_height)
        y += chart_height + 10
    probs = "  ".join(f"L{i + 1} {float(p):.2f}" for i, p in enumerate(decoded.lead_prob))
    desires = decoded.desire_state
    index = int(desires.argmax())
    desire = DESIRE_LABELS[index] if index < len(DESIRE_LABELS) else str(index)
    draw_text(img, f"{probs}   {desire} {float(desires[index]):.2f}", x, y,
              (164, 164, 164), 10, max_width=width)
    y += 18
    # Only the final fps/version line is identical to the bar readout.
    lines = [(text, color) for text, color in state.dev_lines
             if "SimSteer v" not in text]
    keys_y = h - 51
    for font_size in (12, 11, 10, 9, 8):
        rows = _wrap_diagnostics(lines, width, font_size)
        if len(rows) * (font_size + 3) <= keys_y - y:
            break
    for text, color in rows:
        draw_text(img, text, x, y, color, font_size, max_width=width)
        y += font_size + 3
    if state.calibration and state.calibration.score is not None:
        draw_text(img, f"Score {state.calibration.score:.2f}", x, y, (235, 235, 235), 10)
    draw_text(img, "H  Normal    I  Input    V  View    C  Save    0  Reset", x, keys_y,
              (200, 200, 200), 10, max_width=width)
    draw_text(img, "[/]  FOV    ,/.  Pitch    ;/'  Height    M  Mirror", x, keys_y + 16,
              (200, 200, 200), 10, max_width=width)


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
