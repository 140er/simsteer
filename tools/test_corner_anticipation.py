"""Smoke test for longitudinal corner-anticipation (max_lat_accel limit)."""
from __future__ import annotations

import numpy as np

from pilot.constants import IDX_N, PLAN_WIDTH, T_IDXS
from pilot.controller import ControllerConfig, LongitudinalController
from pilot.postprocess import Decoded


def make(v_arr, a_arr, yaw_rate_arr):
    plan = np.zeros((IDX_N, PLAN_WIDTH), dtype=np.float32)
    plan[:, 3] = v_arr
    plan[:, 6] = a_arr
    plan[:, 14] = yaw_rate_arr
    return Decoded(
        plan=plan, plan_std=np.zeros_like(plan),
        lane_lines=np.zeros((4, IDX_N, 2)), lane_lines_prob=np.zeros(4),
        road_edges=np.zeros((2, IDX_N, 2)),
        pose=np.zeros(6), road_transform=np.zeros(6),
        desire_state=np.zeros(8), lead_prob=np.zeros(3))


def main() -> None:
    cfg = ControllerConfig.load()
    cfg.max_lat_accel_mps2 = 3.0
    cfg.corner_scan_horizon_s = 3.0
    cfg.max_speed_mps = 30.0
    lc = LongitudinalController(cfg)
    t_idxs = np.asarray(T_IDXS, dtype=np.float32)

    # 1) Straight road, plan says go 25 m/s, no yaw_rate -> no corner brake.
    v = np.full(IDX_N, 25.0)
    a = np.zeros(IDX_N)
    yr = np.zeros(IDX_N)
    d = make(v, a, yr)
    t, b = lc.compute(d, 25.0)
    print(f"straight:        t={t:.2f} b={b:.2f}  "
          f"v_target={lc.last_v_target:.2f}  v_safe_corner=inf "
          f"  (expect 0/0)")

    # 2) Tight corner at t=2s: yaw_rate = 0.2 rad/s, v=20 m/s -> a_lat = 4 m/s^2
    # κ = 0.2/20 = 0.01 1/m. v_safe = sqrt(3/0.01) = sqrt(300) = 17.3 m/s
    yr = np.zeros(IDX_N)
    yr[t_idxs > 1.5] = 0.2
    yr[t_idxs < 1.5] = 0.0
    v = np.full(IDX_N, 25.0)
    a = np.zeros(IDX_N)
    d = make(v, a, yr)
    t, b = lc.compute(d, 25.0)
    print(f"corner @ 2s:     t={t:.2f} b={b:.2f}  "
          f"v_target={lc.last_v_target:.2f}  "
          f"v_safe_corner={lc.last_v_safe_corner:.2f}  "
          f"k_corner={lc.last_corner_k:.5f}  "
          f"t_corner={lc.last_corner_t:.2f}s  "
          f"(expect v_safe ~17, hard brake)")

    # 3) Very tight corner (hairpin) far ahead. κ=0.05 → v_safe=sqrt(60)=7.7 m/s
    yr = np.zeros(IDX_N)
    yr[t_idxs > 2.5] = 0.4
    v = np.full(IDX_N, 25.0)
    a = np.zeros(IDX_N)
    d = make(v, a, yr)
    t, b = lc.compute(d, 25.0)
    print(f"hairpin @ 3s:    t={t:.2f} b={b:.2f}  "
          f"v_target={lc.last_v_target:.2f}  "
          f"v_safe_corner={lc.last_v_safe_corner:.2f}  "
          f"(expect hard brake to ~8 m/s)")

    # 4) Disabled when max_lat_accel = 0.
    cfg.max_lat_accel_mps2 = 0.0
    t, b = lc.compute(d, 25.0)
    print(f"disabled:        t={t:.2f} b={b:.2f}  "
          f"v_target={lc.last_v_target:.2f}  "
          f"v_safe_corner={lc.last_v_safe_corner}  "
          f"(expect no override, plan v=25)")

    # 5) Already slow approaching corner — no brake needed.
    cfg.max_lat_accel_mps2 = 3.0
    yr = np.zeros(IDX_N)
    yr[t_idxs > 1.5] = 0.2
    d = make(v, a, yr)
    t, b = lc.compute(d, 12.0)
    print(f"slow approach:   t={t:.2f} b={b:.2f}  "
          f"v_target={lc.last_v_target:.2f}  "
          f"v_safe_corner={lc.last_v_safe_corner:.2f}  "
          f"(v_ego=12, target~17, expect mild throttle)")


if __name__ == "__main__":
    main()
