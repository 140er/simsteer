"""Certification must describe what the check actually demonstrated."""

import json
import math

import numpy as np
import pytest

from simsteer import calib as calib_api
from simsteer.calib.flow import Step, flow_path
from simsteer.calib.score import CheckPolicy, CheckStats, Verdict, evaluate
from simsteer.core.learners.livecalib import CalStatus
from simsteer.core.learners.liveparams import LiveParams
from test_calib_flow import Driver, FakeLiveCalib, FakeLiveParams, _to_check, new_flow
from test_calib_score import _drive
from test_liveparams import _drive as train_rack


def test_axis_only_drive_has_insufficient_evidence():
    stats = CheckStats(seconds=120.0)
    for _ in range(2400):
        stats.add_axis(0.1)
    result = evaluate(stats).to_dict()
    assert not result["passed"]
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["measurements"]["lane_offset_spread"]["samples"] == 0


def test_constant_one_metre_lane_bias_fails():
    stats = _drive()
    stats.off_n, stats.off_mean, stats.off_m2 = 2400, 1.0, 0.0
    result = evaluate(stats)
    assert not result.passed
    assert result.to_dict()["status"] == "FAIL"
    assert "bias" in result.reason


def test_out_of_lane_samples_are_failures_and_keep_the_denominator():
    stats = _drive()
    for _ in range(300):
        stats.add_offset(2.5)
    result = evaluate(stats).to_dict()
    assert not result["passed"]
    lane = result["measurements"]["lane_excursions"]
    assert lane["samples"] == lane["denominator"] == 2700
    assert lane["value"] == pytest.approx(300 / 2700, abs=0.0001)


def test_required_sparse_measurement_cannot_disappear():
    stats = _drive()
    stats.ratio_n, stats.ratio_log_sum = 20, 20 * math.log(0.5)
    result = evaluate(stats).to_dict()
    assert not result["passed"]
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["measurements"]["speed_ratio_error"]["samples"] == 20


def test_check_uses_held_out_frames_without_refitting():
    lp = LiveParams(lag_frames=1)
    train_rack(lp, a=8.0, b=0.0, frames=range(1000))
    flow = new_flow(lp=lp, persist=False)
    flow._enter(Step.CHECK)
    coefficients, covariance, samples = lp.x.copy(), lp.P.copy(), lp.samples
    train_rack(lp, a=7.0, b=0.0, frames=range(1000, 1200))
    assert np.array_equal(lp.x, coefficients)
    assert np.array_equal(lp.P, covariance)
    assert lp.samples == samples
    assert lp.predictions > flow._pred_seen


@pytest.mark.parametrize("one_sided", [False, True], ids=["single-speed", "one-sided"])
def test_sample_count_without_excitation_is_not_trusted(one_sided):
    lp = LiveParams(lag_frames=0)
    for i in range(1000):
        wheel = 0.02 if one_sided else 0.02 * math.sin(i / 40)
        speed = 15 + 5 * math.sin(i / 100) if one_sided else 22.0
        lp.update(10 * wheel, speed, wheel)
    assert lp.samples >= 100
    assert not lp.trusted()


def test_yaw_derived_wheel_is_qualified_as_dependent_evidence():
    stats = _drive()
    stats.fit_source = "yaw-derived"
    result = evaluate(stats).to_dict()
    assert not result["passed"]
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["measurements"]["steering_fit_error"]["source"] == "yaw-derived"


@pytest.mark.parametrize("field, value", [("fov_h_deg", 80.0), ("pitch_deg", 2.0),
                                         ("yaw_deg", 1.0), ("lateral_sign", 1.0)])
def test_certified_camera_or_binding_change_revokes_pass(field, value):
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    assert flow.calibrated
    setattr(flow.calib, field, value)
    driver.frames(1)
    assert not flow.calibrated
    assert flow.result["status"] == "INSUFFICIENT_EVIDENCE"
    assert flow.step is Step.CHECK
    assert flow.check.seconds == 0


def test_old_saved_pass_requires_check_again():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    _to_check(flow, lc, lp)
    state = flow.to_dict()
    state.update(version=3, step="done", result={"passed": True, "score": 1.0})
    flow_path("ets2", "gamepad").write_text(json.dumps(state))
    assert not calib_api.status("ets2", "gamepad").calibrated
    resumed = new_flow(lc, lp, cal=flow.calib)
    assert not resumed.calibrated
    assert resumed.step is Step.CHECK
    assert resumed.result["status"] == "INSUFFICIENT_EVIDENCE"
    assert "Check again" in resumed.redo_note


def test_unengaged_diagnostic_drive_cannot_certify_lane_keeping():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    driver.frames(2401, engaged=False)
    assert not flow.calibrated
    assert flow.result["status"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("field, value", [("controller_gain", 2.0), ("binding", "other-axis"),
                                         ("device_id", 2), ("schema", 2)])
def test_control_device_binding_and_schema_changes_revoke_immediately(field, value):
    context = dict(controller_gain=1.0, binding="steering-axis", device_id=1, schema=1)
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp, certification_context=lambda: context.copy())
    driver = _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    assert flow.calibrated and flow.ai_steering_ready
    context[field] = value
    assert not flow.calibrated and flow.device_axis(0.3) == 0.0
    driver.frames(1)
    assert not flow.ai_steering_ready and flow.engage_block()
    flow.confirm_check()
    assert flow.ai_steering_ready and not flow.calibrated


def test_camera_invalidation_revokes_output_without_a_pose_change():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    lc.cal_status = CalStatus.INVALID
    assert flow.device_axis(0.3) == 0.0
    driver.frames(1)
    assert flow.step is Step.CAMERA and not flow.ai_steering_ready
    assert flow.result["status"] == "INSUFFICIENT_EVIDENCE"


def test_capabilities_declare_the_required_measurements():
    stats = _drive()
    stats.fit_n = stats.ratio_n = 0
    full = evaluate(stats, CheckPolicy.for_capabilities(True))
    limited = evaluate(stats, CheckPolicy.for_capabilities(False))
    assert full.status == "INSUFFICIENT_EVIDENCE"
    assert limited.status == "PASS"
    assert not next(m for m in limited.measurements if m.name == "steering_fit_error").required


def test_training_innovations_cannot_validate_the_same_fit():
    stats = _drive()
    stats.fit_n = stats.fit_total = 0
    for _ in range(1000):
        stats.add_fit(0.0, 0.1, held_out=False)
    result = evaluate(stats).to_dict()
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["measurements"]["steering_fit_error"]["samples"] == 0
    assert result["measurements"]["steering_fit_error"]["denominator"] == 1000


def test_missing_and_nonfinite_lane_samples_keep_the_denominator():
    stats = _drive()
    for offset in [None, float("nan"), float("inf")] * 1000:
        stats.add_offset(offset)
    result = evaluate(stats).to_dict()
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    lane = result["measurements"]["lane_offset_spread"]
    assert lane["samples"] == 2400 and lane["denominator"] == 5400
    json.dumps(result, allow_nan=False)


def test_trust_coverage_survives_reload_and_reset_clears_it():
    lp = LiveParams(game="ets2", device_kind="gamepad", lag_frames=0)
    for speed in (12.0, 24.0):
        for i in range(400):
            wheel = 0.02 * math.sin(i / 30)
            lp.update(10 * wheel, speed, wheel)
    assert lp.trusted()
    lp.save()
    loaded = LiveParams(game="ets2", device_kind="gamepad")
    assert loaded.trusted() and loaded.coverage == lp.coverage
    loaded.reset()
    assert not loaded.trusted() and loaded.coverage == {}


def test_old_sample_count_without_coverage_is_not_trusted(data_dir):
    (data_dir / "liveparams_ets2_gamepad.json").write_text(json.dumps(
        dict(a=10.0, b=0.0, c=0.0, samples=2000)))
    lp = LiveParams(game="ets2", device_kind="gamepad")
    assert lp.samples == 2000 and not lp.trusted()


def test_check_resume_keeps_the_original_validation_model():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    driver.frames(200)
    model = lp.validation_model
    flow.save()
    other = FakeLiveParams(samples=100)
    other.a_linear = 5.0
    resumed = new_flow(lc, other, cal=flow.calib)
    assert resumed.step is Step.CHECK and resumed.check.seconds == flow.check.seconds
    assert tuple(other.validation_model) == tuple(model)


def test_new_saved_pass_survives_reload_with_matching_context():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    driver.until(Step.DONE)
    assert flow.calibrated
    resumed = new_flow(lc, lp, cal=flow.calib)
    assert resumed.calibrated and calib_api.status("ets2", "gamepad").calibrated


def test_manual_prefix_does_not_shorten_the_engaged_check():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    driver.frames(500, engaged=False)
    assert flow.progress == 0.0 and flow.eta_s() == 120.0
    driver.frames(2400)
    assert flow.calibrated
    assert flow.result["coverage"]["engaged_seconds"] == 120.0


def test_retry_log_reports_the_new_attempt_counts():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    driver.frames(2400, engaged=False)
    assert flow.result["status"] == "INSUFFICIENT_EVIDENCE"
    flow.confirm_check()
    driver.frames(20)
    assert flow.log_fields()["calib_check"]["coverage"]["engaged"] == 20


def test_failed_state_waits_for_restart_when_camera_is_reset():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    flow.attempts = 1
    flow._finish_check(Verdict(False, 0.0, "steering", "bad fit"))
    lc.reset()
    driver.frames(1)
    assert flow.step is Step.FAILED and flow.engage_block()


def test_missing_fit_predictions_keep_the_frame_denominator():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    # Call tick directly so the default Driver does not invent observations.
    from simsteer.calib.flow import Obs
    for i in range(2400):
        if i < 100:
            lp.predict(0.001, 0.1)
        flow.tick(Obs(now=driver.t + (i + 1) * 0.05, dt=0.05, v=22.0,
                      yaw_rate=0.0, vx_model=22.0, lane_offset=0.02,
                      axis=0.1 * math.sin(i / 10), driving=True, engaged=True))
    result = flow.result
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["measurements"]["steering_fit_error"]["denominator"] == 2400
