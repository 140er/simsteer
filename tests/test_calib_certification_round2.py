"""Certification regressions for saved context and short lane-reference gaps."""

import json
import math

import pytest

from simsteer.calib.flow import Step, flow_path
from simsteer.calib.score import CheckStats, evaluate
from simsteer.core import model_identity
from simsteer.core.learners.liveparams import LiveParams, TRUSTED_MIN_SPEED_BIN_SAMPLES
from test_calib_flow import Driver, FakeLiveCalib, FakeLiveParams, _to_check, new_flow
from test_calib_score import _drive


@pytest.mark.parametrize("step", ["check", "done"])
@pytest.mark.parametrize("context", [{}, {"device_id": 1, "device": "wheel"}])
def test_legacy_context_does_not_invent_an_output_device_change(step, context):
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp, certification_context=lambda: dict(context))
    _to_check(flow, lc, lp)
    state = flow.to_dict()
    for key in ("check_context", "check_model", "check_requested", "legacy_result"):
        state.pop(key, None)
    state.update(version=3, step=step, result=dict(passed=True, score=0.87))
    flow_path("ets2", "gamepad").write_text(json.dumps(state))
    resumed = new_flow(lc, lp, cal=flow.calib, certification_context=lambda: dict(context))
    assert resumed.step is Step.CHECK and resumed.output_direction.ready
    assert resumed.redo_note == "Check again: saved evidence is outdated"
    assert not resumed.calibrated and not resumed.ai_steering_ready
    assert not resumed._check_requested


def test_camera_reset_before_first_check_keeps_output_verification():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp, certification_context=lambda: {"device_id": 1})
    lc.converge()
    flow._finish_fov("kept")
    driver = Driver(flow)
    driver.verify_direction()
    assert flow.step is Step.STEERING and flow.output_direction.ready
    assert flow._check_context is None
    lc.reset()
    driver.frames(1)
    assert flow.step is Step.CAMERA and flow.output_direction.ready
    assert "device changed" not in flow.redo_note.lower()
    assert not flow.ai_steering_ready


def test_perspective_probe_keeps_small_drift_but_revokes_cumulative_drift():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    flow.calib.simple_warp = False
    driver = _to_check(flow, lc, lp)
    for _ in range(40):
        flow.calib.pitch_deg += 0.01
        driver.frames(1)
    assert flow.step is Step.CHECK and flow._check_requested
    flow._finish_check(evaluate(_drive()))
    assert flow.calibrated
    flow.calib.pitch_deg += 0.2
    driver.frames(1)
    assert flow.step is Step.CHECK and not flow.calibrated
    assert not flow.ai_steering_ready


@pytest.mark.parametrize("transition", [[1.7, -1.7], [1.7, None, None, -1.7],
                                       [1.7, 0.6, -0.6, -1.7]],
                         ids=["instant", "two-invalid", "three-frame"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_crossing_probe_fails_across_short_gaps_and_smoothing(transition, sign):
    stats = _drive()
    lane = CheckStats()
    base = [0.1 * math.sin(i * 0.05) for i in range(2400)]
    offsets = base[:1200] + [sign * o if o is not None else None for o in transition] + base[1200:]
    for offset in offsets:
        lane.add_offset(offset)
    for key, value in lane.to_dict().items():
        if key.startswith("off_"):
            setattr(stats, key, value)
    verdict = evaluate(stats)
    assert stats.off_crossings == stats.off_excursions == 1
    assert stats.off_total == len(offsets)
    assert stats.off_n == len(offsets) - transition.count(None)
    assert verdict.status == "FAIL" and "crossing" in verdict.reason


def test_crossing_window_expires_and_a_commanded_change_clears_it():
    for commanded in (False, True):
        stats = CheckStats()
        stats.add_offset(1.7)
        for _ in range(6 if not commanded else 2):
            stats.add_offset(None, lane_change=commanded)
        stats.add_offset(-1.7)
        assert stats.off_crossings == 0


def test_crossing_is_counted_once_and_the_window_survives_resume():
    stats = CheckStats()
    stats.add_offset(1.7)
    stats.add_offset(None)
    stats = CheckStats.from_dict(stats.to_dict())
    for offset in (-0.6, -1.0, -1.5, -1.7):
        stats.add_offset(offset)
    assert stats.off_crossings == stats.off_excursions == 1


@pytest.mark.parametrize("other_speed", [18.9, 25.1])
def test_speed_span_requires_a_populated_bin_and_survives_resume(other_speed):
    lp = LiveParams(game="ets2", device_kind="gamepad")
    lp.samples = 500
    lp.coverage = {"7": [40, 40]}
    lp.coverage_speed_ranges = {"7": [22.0, 22.0]}
    lp.coverage_speed_min = lp.coverage_speed_max = 22.0
    assert not lp.trusted()
    key = str(int(other_speed // 3))
    lp.coverage_speed_ranges[key] = [other_speed, other_speed]
    lp.coverage_speed_min = min(22.0, other_speed)
    lp.coverage_speed_max = max(22.0, other_speed)
    for count in (1, TRUSTED_MIN_SPEED_BIN_SAMPLES - 1):
        lp.coverage[key] = [count, 0]
        assert not lp.trusted() and lp.coverage_speed_span == 0.0
        assert lp.trust_need == "Need varied speed"
    lp.coverage[key] = [TRUSTED_MIN_SPEED_BIN_SAMPLES, 0]
    assert lp.trusted() and lp.coverage_speed_span == pytest.approx(3.1)
    lp.save()
    resumed = LiveParams(game="ets2", device_kind="gamepad")
    assert resumed.trusted()
    assert resumed.coverage_speed_ranges == lp.coverage_speed_ranges
    resumed.reset()
    assert not resumed.coverage_speed_ranges and not resumed.trusted()


def test_legacy_speed_extremes_cannot_bypass_the_bin_minimum():
    lp = LiveParams(game="ets2", device_kind="gamepad")
    lp.samples = 500
    lp.coverage = {"7": [40, 40], "8": [1, 0]}
    lp.coverage_speed_min, lp.coverage_speed_max = 22.0, 25.1
    assert not lp.trusted()
    lp.coverage = {"5": [10, 0], "7": [40, 40]}
    # Old bins carry no per-bin extrema; their inner edges prove a 3 m/s span.
    assert lp.trusted() and lp.coverage_speed_span == 3.0


def test_adapter_comments_do_not_change_identity_but_version_does(tmp_path, monkeypatch):
    monkeypatch.setattr(model_identity, "__file__", str(tmp_path / "model_identity.py"))
    for name in ("constants.py", "preprocess.py", "postprocess.py"):
        (tmp_path / name).write_text("VALUE = 1\n")
    original = model_identity.pinned_model_identity()
    (tmp_path / "postprocess.py").write_text("# Clarify the unchanged adapter.\nVALUE = 1\n")
    assert model_identity.pinned_model_identity() == original
    monkeypatch.setattr(model_identity, "ADAPTER_VERSION", model_identity.ADAPTER_VERSION + 1)
    assert model_identity.pinned_model_identity() != original


def test_adapter_version_change_invalidates_a_saved_certificate(monkeypatch):
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    assert flow.calibrated
    monkeypatch.setattr(model_identity, "ADAPTER_VERSION", model_identity.ADAPTER_VERSION + 1)
    resumed = new_flow(lc, lp, cal=flow.calib)
    assert not resumed.calibrated and not resumed.ai_steering_ready
    assert resumed.output_direction.ready
