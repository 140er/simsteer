"""Bounds, calibration isolation and deterministic longitudinal measurements."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.core.calibration import Calibration
from simsteer.core.control import ControllerConfig, LongitudinalController
from simsteer.runtime.fov import FovSolver, ESTIMATE_SAMPLES, solve_from_ratio
from simsteer.runtime.speedscale import SpeedScale
from tools.speed_scale_bench import benchmark, decoded_plan


@pytest.fixture
def parts():
    cfg = ControllerConfig()
    flow = SimpleNamespace(calibrated=True, camera_learning=False,
                           calib=Calibration(), model_identity={})
    tel = SimpleNamespace(available=True, speed_mps=lambda: 20.)
    return cfg, flow, tel


def learn(learner, flow, tel, *, pose=17.6, frames=600):
    for _ in range(frames):
        learner.update(flow, tel, pose, .05)


def test_deterministic_before_after():
    before, after = benchmark(corrected=False), benchmark(corrected=True)
    assert before["cruise_mps"] == pytest.approx(19.642857, abs=0.001)
    assert after["cruise_mps"] == pytest.approx(25., abs=0.001)
    assert after["half_plan_fraction"] == pytest.approx(.5, abs=1e-5)
    assert after["slowdown_accel_mps2"] == pytest.approx(-1., abs=1e-5)
    assert after["slowdown_brake"] > 0 and after["lead_brake"] > 0
    assert after["aeb_brake"] == 1.


def test_unity_ceiling_is_exact_noop(parts):
    cfg, flow, tel = parts
    cfg.speed_scale_max = 1.
    learner = SpeedScale("ets2", cfg)
    plain, scaled = LongitudinalController(cfg), LongitudinalController(cfg)
    scaled.speed_scale = 1.3
    for fraction, accel in [(1., 0.), (.5, -1.), (1.5, 1.)]:
        decoded = decoded_plan(20., 20., fraction=fraction, accel=accel)
        assert scaled.compute(decoded, 20.) == plain.compute(decoded, 20.)
        assert learner.update(flow, tel, 17.6, .05) == 1.
    learner.save()
    assert not learner.path.exists()


def test_scale_targets_only_and_preserves_corner_and_aeb(parts):
    cfg, flow, tel = parts
    controller = LongitudinalController(cfg)
    controller.speed_scale = 1.3
    decoded = decoded_plan(20., 20., accel=-1.)
    decoded.plan[:, 14] = .8
    original_plan, original_pose = decoded.plan.copy(), decoded.pose.copy()
    controller.compute(decoded, 20.)
    plain = LongitudinalController(cfg)
    plain.compute(decoded, 20.)
    assert controller.last_v_safe_corner == plain.last_v_safe_corner
    assert controller.last_v_target == plain.last_v_target
    np.testing.assert_array_equal(decoded.plan, original_plan)
    np.testing.assert_array_equal(decoded.pose, original_pose)
    decoded.lead_prob[0] = .99
    decoded.leads[0, 0, :3] = [10., 0., 0.]
    assert controller.compute(decoded, 20.) == (0., 1.)
    assert controller.last_aeb


def test_fov_uses_raw_pose_and_unity_residual_stays_unity(parts):
    cfg, flow, tel = parts
    learner = SpeedScale("ets2", cfg)
    solver = FovSolver()
    flow.calibrated = False
    for _ in range(ESTIMATE_SAMPLES):
        assert learner.update(flow, tel, 17.6, .05) == 1.
        solver.update(None, flow.calib, 17.6, 20., 0.)
    assert solver.result == pytest.approx(solve_from_ratio(90., .88))
    flow.calib.fov_h_deg = solver.result
    flow.calibrated = True
    learn(learner, flow, tel, pose=20.)
    assert learner.value == 1.
    learner.save()
    assert SpeedScale("ets2", cfg).update(flow, tel, 20., .05) == 1.


@pytest.mark.parametrize("field,value", [("fov_h_deg", 80.), ("height_m", 2.),
                                        ("pitch_deg", 1.), ("crop_top_pct", .1),
                                        ("simple_warp", False)])
def test_camera_change_cannot_reuse_old_scale(parts, field, value):
    cfg, flow, tel = parts
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel)
    learner.save()
    setattr(flow.calib, field, value)
    assert learner.update(flow, tel, 20., .05) == 1.
    fresh = SpeedScale("ets2", cfg)
    assert fresh.update(flow, tel, 20., .05) == 1.


def test_other_game_and_changed_model_are_isolated(parts):
    cfg, flow, tel = parts
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel)
    learner.save()
    assert SpeedScale("ats", cfg).update(flow, tel, 20., .05) == 1.
    flow.model_identity = {"changed": True}
    assert SpeedScale("ets2", cfg).update(flow, tel, 20., .05) == 1.


@pytest.mark.parametrize("pose,v,dt", [(0., 20., .05), (3., 20., .05),
                                     (17.6, 5., .05), (4., 20., .05),
                                     (float("nan"), 20., .05), (17.6, None, .05),
                                     (17.6, 20., 1.), (17.6, 20., 0.)])
def test_ineligible_sample_cannot_train_or_save(parts, pose, v, dt):
    cfg, flow, tel = parts
    tel.speed_mps = lambda: v
    learner = SpeedScale("ets2", cfg)
    for _ in range(600):
        assert learner.update(flow, tel, pose, dt) == 1.
    learner.save()
    assert not learner.path.exists()


def test_unconverged_session_does_not_overwrite_good_snapshot(parts):
    cfg, flow, tel = parts
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel)
    learner.save()
    saved = learner.path.read_bytes()
    learn(learner, flow, tel, pose=16., frames=10)
    learner.save()
    assert learner.path.read_bytes() == saved


@pytest.mark.parametrize("scale", [0., 4., "1.2", True, None])
def test_invalid_saved_value_cannot_poison_warm_start(parts, scale):
    cfg, flow, tel = parts
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel)
    learner.save()
    data = json.loads(learner.path.read_text())
    data["vehicles"][""]["scale"] = scale
    learner.path.write_text(json.dumps(data))
    fresh = SpeedScale("ets2", cfg)
    assert fresh.update(flow, tel, 20., .05) == 1.


def test_unhealthy_frame_holds_scale_without_learning(parts):
    cfg, flow, tel = parts
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel)
    value = learner.value
    assert value > 1.1
    for _ in range(40):
        # A wildly different pose must not train while unhealthy.
        assert learner.update(flow, tel, 10., .05, healthy=False) == value
    assert learner.value == value
    assert learner.update(flow, tel, 17.6, .05) == pytest.approx(value, abs=.0001)


def test_scale_raises_targets_without_corner_or_lead(parts):
    cfg, _, _ = parts
    decoded = decoded_plan(20., 20., accel=.5)
    plain, scaled = LongitudinalController(cfg), LongitudinalController(cfg)
    scaled.speed_scale = 1.2
    plain.compute(decoded, 20.)
    scaled.compute(decoded, 20.)
    assert plain.last_v_safe_corner > 100.  # no corner limit binds
    assert scaled.last_v_target == pytest.approx(1.2 * plain.last_v_target)
    assert scaled.last_a_target == pytest.approx(1.2 * plain.last_a_target)


def test_disabled_or_calibrating_freezes_and_suspends_saved_scale(parts):
    cfg, flow, tel = parts
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel)
    value = learner.value
    flow.camera_learning = True
    for _ in range(600):
        assert learner.update(flow, tel, 20., .05) == 1.
    assert learner.value == value
    flow.camera_learning = False
    assert learner.update(flow, tel, 17.6, .05) == pytest.approx(value, abs=.0001)
    cfg.speed_scale_max = 1.
    assert learner.update(flow, tel, 17.6, .05) == 1.
    assert learner.value == pytest.approx(value, abs=.0001)


def test_reset_keeps_controller_calibration():
    controller = LongitudinalController()
    controller.speed_scale = 1.14
    controller.reset()
    assert controller.speed_scale == 1.14


def test_parking_before_exit_keeps_last_converged_estimate(parts):
    cfg, flow, tel = parts
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel)
    tel.speed_mps = lambda: 0.
    learn(learner, flow, tel, pose=0.)
    learner.save()
    tel.speed_mps = lambda: 20.
    assert SpeedScale("ets2", cfg).update(flow, tel, 17.6, .05) == pytest.approx(1 / .88)


def test_lower_ceiling_clamps_saved_value(parts):
    cfg, flow, tel = parts
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel, pose=16.)
    learner.save()
    cfg.speed_scale_max = 1.1
    assert SpeedScale("ets2", cfg).update(flow, tel, 16., .05) == 1.1


def test_controller_legacy_state_loads_with_default_ceiling(data_dir):
    path = data_dir / "controller_ets2.json"
    path.write_text('{"speed_p_gain": 0.4}')
    cfg = ControllerConfig.load("ets2")
    assert cfg.speed_p_gain == .4 and cfg.speed_scale_max == 1.3


@pytest.mark.parametrize("ceiling", [0.9, 1.5, True, "1.3"])
def test_invalid_ceiling_in_saved_controller_is_rejected(data_dir, ceiling):
    (data_dir / "controller_ets2.json").write_text(json.dumps({"speed_scale_max": ceiling}))
    assert ControllerConfig.load("ets2").speed_scale_max == 1.3


def test_longitudinal_option_does_not_revoke_steering_certificate(monkeypatch):
    from test_speed_scale_regression import session

    def inspect(st, manager):
        assert "speed_scale_max" not in st.flow._context()["controls"]["controller"]

    session(monkeypatch, frames=1, on_grab=inspect)


@pytest.mark.parametrize("available", [False, True])
def test_missing_telemetry_never_uses_fallback_speed(parts, available):
    cfg, flow, tel = parts
    tel.available = available
    tel.speed_mps = lambda: None
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel)
    assert learner.value == 1.
    learner.save()
    assert not learner.path.exists()


def test_reader_returning_no_vehicle_uses_game_scope(parts):
    cfg, flow, tel = parts
    tel.vehicle_id = lambda: None
    learner = SpeedScale("ets2", cfg)
    learn(learner, flow, tel)
    learner.save()
    assert SpeedScale("ets2", cfg).update(flow, tel, 17.6, .05) == pytest.approx(1 / .88)


@pytest.mark.parametrize("raw", ['{"unfinished":', '[]', '{"version":1,"vehicles":[]}'])
def test_corrupt_longitudinal_state_recovers(parts, raw, data_dir, caplog):
    cfg, flow, tel = parts
    (data_dir / "longitudinal_ets2.json").write_text(raw)
    learner = SpeedScale("ets2", cfg)
    assert learner.update(flow, tel, 20., .05) == 1.
    assert "using defaults" in caplog.text
