"""Longitudinal corner anticipation caps speed at sqrt(a_lat_max / kappa)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from simsteer.core.constants import IDX_N, PLAN_WIDTH, T_IDXS
from simsteer.core.control import ControllerConfig, LongitudinalController
from simsteer.core.postprocess import Decoded

T = np.asarray(T_IDXS, dtype=np.float32)
V_PLAN = 25.0
A_LAT_MAX = 3.0


def _decoded(yaw_rate: np.ndarray) -> Decoded:
    plan = np.zeros((IDX_N, PLAN_WIDTH), dtype=np.float32)
    plan[:, 3] = V_PLAN
    plan[:, 14] = yaw_rate
    return Decoded(
        plan=plan, plan_std=np.zeros_like(plan),
        lane_lines=np.zeros((4, IDX_N, 2)), lane_lines_prob=np.zeros(4),
        road_edges=np.zeros((2, IDX_N, 2)),
        pose=np.zeros(6), pose_std=np.zeros(6),
        road_transform=np.zeros(6), road_transform_std=np.zeros(6),
        wide_from_device_euler=np.zeros(3),
        desire_state=np.zeros(8), lead_prob=np.zeros(3),
        leads=np.zeros((3, 6, 4)), leads_std=np.zeros((3, 6, 4)))


@pytest.fixture
def lc() -> LongitudinalController:
    cfg = ControllerConfig()
    cfg.max_lat_accel_mps2 = A_LAT_MAX
    cfg.corner_scan_horizon_s = 3.0
    cfg.max_speed_mps = 30.0
    return LongitudinalController(cfg)


def _v_safe(yaw_rate: float) -> float:
    return math.sqrt(A_LAT_MAX / (yaw_rate / V_PLAN))


def test_straight_road_keeps_plan_speed(lc):
    throttle, brake = lc.compute(_decoded(np.zeros(IDX_N)), 25.0)
    assert (throttle, brake) == (0.0, 0.0)
    assert lc.last_v_target == pytest.approx(V_PLAN)


@pytest.mark.parametrize("onset_s,yaw_rate", [(1.5, 0.2), (2.5, 0.4)], ids=["corner", "hairpin"])
def test_corner_ahead_brakes_to_safe_speed(lc, onset_s, yaw_rate):
    throttle, brake = lc.compute(_decoded(np.where(T > onset_s, yaw_rate, 0.0)), 25.0)
    assert lc.last_v_safe_corner == pytest.approx(_v_safe(yaw_rate), rel=1e-4)
    assert lc.last_v_target == pytest.approx(_v_safe(yaw_rate), rel=1e-4)
    assert lc.last_corner_t > onset_s
    assert throttle == 0.0 and brake > 0.0


def test_tighter_corner_brakes_harder(lc):
    _, gentle = lc.compute(_decoded(np.where(T > 1.5, 0.2, 0.0)), 25.0)
    _, hairpin = lc.compute(_decoded(np.where(T > 2.5, 0.4, 0.0)), 25.0)
    assert hairpin > gentle


def test_zero_lat_accel_limit_disables_it(lc):
    lc.cfg.max_lat_accel_mps2 = 0.0
    throttle, brake = lc.compute(_decoded(np.where(T > 2.5, 0.4, 0.0)), 25.0)
    assert brake == 0.0
    assert lc.last_v_safe_corner == math.inf
    assert lc.last_v_target == pytest.approx(V_PLAN)


def test_slow_approach_does_not_brake(lc):
    throttle, brake = lc.compute(_decoded(np.where(T > 1.5, 0.2, 0.0)), 12.0)
    assert brake == 0.0 and throttle > 0.0
    assert lc.last_v_target == pytest.approx(_v_safe(0.2), rel=1e-4)
