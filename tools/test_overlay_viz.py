"""Generate a synthetic Decoded with a corner ahead + a lead vehicle,
render the full overlay, and write to disk so the user can eyeball
that the new visualizations land in plausible image positions.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pilot.calibration import Calibration
from pilot.constants import IDX_N, PLAN_WIDTH, T_IDXS
from pilot.postprocess import Decoded, LEAD_MHP_SELECTION, LEAD_TRAJ_LEN, LEAD_WIDTH
from debug.overlay import draw_overlay


def main() -> None:
    h, w = 720, 1280
    bgr = np.full((h, w, 3), 64, dtype=np.uint8)
    # Sky gradient + road for visual context.
    for y in range(h):
        if y < h // 2:
            shade = int(180 + 40 * y / (h // 2))
            bgr[y, :] = (shade, shade - 20, shade - 60)
        else:
            bgr[y, :] = (90, 90, 95)

    t_idxs = np.asarray(T_IDXS, dtype=np.float32)

    plan = np.zeros((IDX_N, PLAN_WIDTH), dtype=np.float32)
    # Forward x grows quadratically to ~100 m at 5 s.
    plan[:, 0] = t_idxs * 8.0   # rough forward distance scaling
    # Curving right into a corner starting at ~3 s.
    plan[:, 1] = np.where(t_idxs > 3.0, -1.5 * (t_idxs - 3.0), 0.0)
    # Speed dropping from 25 to 12 m/s as the corner approaches.
    plan[:, 3] = np.clip(25.0 - 2.0 * np.maximum(t_idxs - 2.0, 0.0), 12.0, 25.0)
    # Strong braking 2..5 s, then mild accel out.
    plan[:, 6] = np.where(
        (t_idxs > 2.0) & (t_idxs < 5.0), -3.0,
        np.where(t_idxs >= 5.0, 0.5, 0.0))
    # Yaw rate during the corner.
    plan[:, 14] = np.where(t_idxs > 3.0, -0.18, 0.0)

    lane_lines = np.zeros((4, IDX_N, 2), dtype=np.float32)
    for i, base in [(0, 5.25), (1, 1.75), (2, -1.75), (3, -5.25)]:
        lane_lines[i, :, 0] = base
    ll_prob = np.array([0.7, 0.95, 0.95, 0.7], dtype=np.float32)
    road_edges = np.zeros((2, IDX_N, 2), dtype=np.float32)
    road_edges[0, :, 0] = 7.0
    road_edges[1, :, 0] = -7.0

    # Lead 1 directly ahead at 30 m, closing.
    leads = np.zeros((LEAD_MHP_SELECTION, LEAD_TRAJ_LEN, LEAD_WIDTH),
                     dtype=np.float32)
    leads[0, 0] = (30.0, -0.2, -2.0, 0.5)   # x, y, v_rel, a
    leads[1, 0] = (60.0, +1.0, -5.0, 0.0)
    leads_std = np.ones_like(leads) * 0.3

    decoded = Decoded(
        plan=plan, plan_std=np.ones_like(plan) * 0.1,
        lane_lines=lane_lines, lane_lines_prob=ll_prob,
        road_edges=road_edges,
        pose=np.zeros(6), road_transform=np.zeros(6),
        desire_state=np.array([0.9, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        lead_prob=np.array([0.85, 0.55, 0.10], dtype=np.float32),
        leads=leads, leads_std=leads_std,
    )

    calib = Calibration(image_w=w, image_h=h, fov_h_deg=70.0,
                        height_m=1.4, pitch_deg=2.0)
    calib.update_for_frame((h, w, 3))

    out = draw_overlay(bgr, decoded, calib)
    out_path = Path("debug_overlay_test.png")
    cv2.imwrite(str(out_path), out)
    print(f"wrote {out_path}  shape={out.shape}")


if __name__ == "__main__":
    main()
