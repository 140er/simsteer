"""Regressions for certification review findings at the runtime boundaries."""

import json
import math

import pytest

from simsteer.calib.flow import Step, flow_path
from simsteer.calib.score import CheckStats, evaluate
from simsteer.core.learners.liveparams import LiveParams
from test_calib_flow import Driver, FakeLiveCalib, FakeLiveParams, _to_check, new_flow
from test_calib_score import _drive


def _saturating_drive():
    # Real validation gates reject saturated commands, starving fit coverage.
    lp = LiveParams(game="ets2", device_kind="gamepad", clock=lambda: 100.0)
    lp.samples = 500
    lp.begin_validation()
    stats = CheckStats()
    seen = lp.predictions
    for i in range(2400):
        t = i * 0.05
        saturated = (i // 20) % 10 < 3
        sign = 1 if (i // 200) % 2 else -1
        axis = sign * (0.95 if saturated else 0.15 + 0.02 * math.sin(t))
        lp.update(axis, 22.0, axis / lp.a_linear, commanded_axis=axis)
        stats.add_axis(axis, 0.05)
        stats.add_speed(22.0, 22.0, 0.0)
        stats.add_offset(0.05 * math.sin(t))
        if lp.predictions > seen:
            stats.add_fit(lp.last_innovation, lp.last_actual, held_out=lp.validating)
        else:
            stats.fit_total += 1
        seen = lp.predictions
    return stats


def test_clear_lane_failure_beats_missing_speed_evidence():
    stats = _drive()
    stats.off_mean = 1.0
    stats.ratio_n, stats.ratio_total = 20, 2000
    verdict = evaluate(stats)
    assert verdict.status == "FAIL" and verdict.redo == "steering"
    assert "speed_ratio_error" in verdict.missing
    assert "bias" in verdict.reason


def test_saturation_fails_despite_rejected_fit_samples():
    verdict = evaluate(_saturating_drive())
    measurements = {m.name: m for m in verdict.measurements}
    assert measurements["steering_fit_error"].samples < 0.8 * 2400
    assert measurements["steering_saturation"].samples == 2400
    assert verdict.status == "FAIL" and "full lock" in verdict.reason


def test_two_saturating_checks_reach_failed():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    _to_check(flow, lc, lp)
    verdict = evaluate(_saturating_drive())
    flow._finish_check(verdict)
    assert flow.attempts == 1
    flow._finish_check(verdict)
    assert flow.step is Step.FAILED and flow.attempts == 2
    assert flow.engage_block() and not flow.ai_steering_ready


def test_highway_sized_steering_can_establish_trust():
    lp = LiveParams(game="ets2", device_kind="gamepad", lag_frames=0)
    for i in range(4400):
        t = i * 0.05
        wheel = 0.0045 * math.sin(t / 8)
        speed = 22.0 + 5.0 * math.sin(t / 30)
        axis = (4.2 + 0.001 * speed * speed) * wheel
        lp.update(axis, speed, wheel)
    assert lp.samples >= 100
    assert lp.trusted()


def test_missing_coverage_does_not_show_complete_steering():
    lp = LiveParams(game="ets2", device_kind="gamepad")
    lp.samples = 500
    lc = FakeLiveCalib()
    lc.converge()
    flow = new_flow(lc, lp)
    flow._finish_fov("kept")
    Driver(flow).verify_direction()
    flow.prompt_idx = 99
    assert flow.step is Step.STEERING
    assert flow.progress < 1.0
    assert flow.hint(22.0, True).startswith("Need ")


def test_legacy_fit_gets_a_check_without_claiming_coverage(data_dir):
    (data_dir / "liveparams_ets2_gamepad.json").write_text(json.dumps(
        dict(a=4.2, b=0.001, c=0.0, samples=500)))
    lp = LiveParams(game="ets2", device_kind="gamepad")
    lp.save()
    lp = LiveParams(game="ets2", device_kind="gamepad")
    assert lp.legacy_fit
    lc = FakeLiveCalib()
    lc.converge()
    flow = new_flow(lc, lp)
    flow._finish_fov("kept")
    Driver(flow).verify_direction()
    assert not lp.trusted() and not flow.calibrated
    assert flow.step is Step.CHECK
    flow._finish_check(evaluate(_drive()))
    assert flow.calibrated and flow.ai_steering_ready
    assert not lp.trusted()  # A passed Check does not invent training coverage.
    failed = _drive()
    failed.off_mean = 1.0
    flow._finish_check(evaluate(failed))
    assert flow.step is Step.STEERING and not lp.legacy_fit
    assert lp.samples == 0 and not flow.calibrated


def test_new_empty_coverage_cannot_turn_into_a_legacy_fit_on_reload():
    lp = LiveParams(game="ets2", device_kind="gamepad")
    lp.samples = 500
    lp.save()
    resumed = LiveParams(game="ets2", device_kind="gamepad")
    assert not resumed.legacy_fit and not resumed.trusted()


@pytest.mark.parametrize("field, increment", [("pitch_deg", 0.01), ("yaw_deg", 0.01),
                                            ("height_m", 0.001)])
def test_perspective_check_finishes_with_small_camera_learning_updates(field, increment):
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    flow.calib.simple_warp = False
    driver = _to_check(flow, lc, lp)
    for _ in range(24):
        setattr(flow.calib, field, getattr(flow.calib, field) + increment)
        driver.frames(100)
    assert flow.calibrated and flow.camera_learning


def test_perspective_pose_tolerance_is_relative_to_the_certified_pose():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    flow.calib.simple_warp = False
    driver = _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    for _ in range(5):
        flow.calib.pitch_deg += 0.1
        driver.frames(1)
        assert flow.calibrated
    flow.calib.pitch_deg += 0.01
    assert not flow.calibrated and flow.device_axis(0.3) == 0
    driver.frames(1)
    assert not flow.ai_steering_ready


def test_perspective_pose_tolerance_survives_resume():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    flow.calib.simple_warp = False
    _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    flow.calib.pitch_deg += 0.01
    resumed = new_flow(lc, lp, cal=flow.calib)
    assert resumed.calibrated


def test_uncommanded_lane_reassignment_counts_as_an_excursion():
    stats = _drive()
    # The model re-references ego lanes as the car crosses the line.
    stats.add_offset(1.799)
    stats.add_offset(-1.799)
    verdict = evaluate(stats)
    assert verdict.status == "FAIL"
    assert stats.off_excursions > 0
    assert "cross" in verdict.reason


def test_commanded_lane_change_does_not_count_as_a_crossing():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    driver.frames(1, lane_offset=1.799)
    driver.frames(1, lane_offset=-1.799, lane_change=True)
    driver.frames(1, lane_offset=-1.7)
    assert flow.check.off_excursions == 0


def test_frame_logging_is_compact_with_periodic_full_evidence():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    logs = []
    for _ in range(100):
        driver.frames(1)
        logs.append(flow.log_fields())
    assert max(len(json.dumps(row["calib_check"])) for row in logs) < 500
    snapshots = [row for row in logs if "calib_check_result" in row]
    assert 1 <= len(snapshots) <= 6
    assert "measurements" in snapshots[0]["calib_check_result"]


def test_migration_retains_legacy_result_fields():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    _to_check(flow, lc, lp)
    old = dict(passed=True, score=0.87, date="2026-09-01", fov_source="config",
               measurements={"lane_offset_spread": dict(value=0.12, score=0.9, samples=2000)})
    state = flow.to_dict()
    state.update(version=3, step="done", result=old)
    flow_path("ets2", "gamepad").write_text(json.dumps(state))
    resumed = new_flow(lc, lp, cal=flow.calib)
    resumed.save()
    saved = json.loads(flow_path("ets2", "gamepad").read_text())
    assert saved.get("legacy_result") == old
    assert saved["result"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_invalidation_hint_shows_its_reason():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    flow.calib.height_m += 0.1
    driver.frames(1)
    assert "calibration changed" in flow.hint(22.0, False)


def test_output_device_id_change_requires_output_verification():
    context = {"device_id": 1}
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp, certification_context=lambda: context.copy())
    driver = _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    context["device_id"] = 2
    driver.frames(1)
    assert not flow.output_direction.ready
    assert flow.step is Step.STEERING and not flow.ai_steering_ready


def test_saved_output_device_id_change_requires_output_verification():
    context = {"device_id": 1}
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp, certification_context=lambda: context.copy())
    _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    context["device_id"] = 2
    resumed = new_flow(lc, lp, cal=flow.calib, certification_context=lambda: context.copy())
    assert not resumed.output_direction.ready and resumed.step is Step.STEERING


def test_insufficient_evidence_has_no_score():
    stats = CheckStats(seconds=120)
    for _ in range(2400):
        stats.add_axis(0.1)
    verdict = evaluate(stats)
    assert verdict.status == "INSUFFICIENT_EVIDENCE"
    assert verdict.score is None and verdict.to_dict()["score"] is None


def test_model_identity_is_part_of_the_certificate():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    driver = _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    context = flow.to_dict()["check_context"]
    assert "model" in context
    assert context["model"]["vision_sha256"] and context["model"]["policy_sha256"]
    flow.model_identity = {**context["model"], "adapter_version": "changed"}
    assert not flow.calibrated and flow.device_axis(0.3) == 0
    driver.frames(1)
    assert flow.result["status"] == "INSUFFICIENT_EVIDENCE"
