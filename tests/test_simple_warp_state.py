"""Old crop-warp pose and Check evidence must not survive the geometry change."""
from __future__ import annotations

from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from simsteer.calib.flow import CalibFlow, Step, flow_path
from simsteer.core.calibration import Calibration, SIMPLE_WARP_VERSION
from simsteer.core.learners.livecalib import BLOCK_SIZE, INPUTS_NEEDED, CalStatus, LiveCalib
from simsteer.core.learners.liveparams import LiveParams
from simsteer.paths import state_path


def saved_camera():
    lc = LiveCalib(game="ets2")
    lc._rpys.extend([np.array([0.0, math.radians(2.0), 0.0])] * INPUTS_NEEDED)
    lc._heights.extend([1.4] * INPUTS_NEEDED)
    lc.samples = BLOCK_SIZE * INPUTS_NEEDED
    lc._refresh_cal_status()
    assert lc.cal_status == CalStatus.CALIBRATED
    lc.save_state("ets2")


def flow(calib, lc=None):
    return CalibFlow("ets2", "gamepad", calib=calib,
                     live_calib=lc if lc is not None else LiveCalib(game="ets2"),
                     live_params=LiveParams(game="ets2", device_kind="gamepad"),
                     steering_fit=False)


def seed_legacy(*, simple=True, step="done", legacy_camera=True,
                legacy_flow=True, fallback=False):
    cal = Calibration(fov_h_deg=75.0, pitch_deg=2.5, yaw_deg=-0.7,
                      crop_top_pct=0.18, crop_bottom_pct=0.04,
                      simple_warp=simple)
    current = flow(cal)
    data = current.to_dict()
    data.update(step=step, attempts=1, result={"passed": True, "score": 1.0})
    data["fov"]["source"] = "config"
    data["check"]["seconds"] = 120.0
    if legacy_flow:
        data.pop("simple_warp_version")
    flow_path("ets2", "gamepad").write_text(json.dumps(data))
    # The flow initially starts Camera from zero; seed the old saved pose
    # explicitly, independently of its current constructor behavior.
    camera = asdict(cal)
    camera.update(pitch_deg=2.5, yaw_deg=-0.7)
    if legacy_camera:
        camera.pop("simple_warp_version")
    state_path("calibration", None if fallback else "ets2").write_text(json.dumps(camera))
    state_path("liveparams", "ets2_gamepad").write_text(json.dumps(
        {"a": 4.2, "b": 0.001, "c": 0.01, "samples": 500, "axis_sign": 1}))
    saved_camera()


@pytest.mark.parametrize("step", ["fov", "camera", "steering", "check", "done", "failed"])
@pytest.mark.parametrize("fallback", [False, True], ids=["per-game", "legacy-file"])
def test_old_simple_camera_is_measured_again(step, fallback):
    seed_legacy(step=step, fallback=fallback)
    cal = Calibration.load("ets2")
    assert cal.pitch_deg == cal.yaw_deg == 0.0
    assert cal.simple_warp_version != SIMPLE_WARP_VERSION
    lc = LiveCalib(game="ets2")
    assert lc.cal_status == CalStatus.CALIBRATED
    current = flow(cal, lc)
    assert current.step is (Step.FOV if step == "fov" else Step.CAMERA)
    assert lc.samples == lc.blocks == lc._block_n == 0
    assert lc.cal_status == CalStatus.UNCALIBRATED
    assert not state_path("livecalib_state", "ets2").exists()
    assert current.result is None
    assert current.attempts == current.check.seconds == 0
    assert current.fov_source == "config"
    assert cal.fov_h_deg == 75.0
    assert (cal.crop_top_pct, cal.crop_bottom_pct) == (0.18, 0.04)
    assert current.lp.a_linear == pytest.approx(4.2)
    assert current.lp.b_quad == pytest.approx(0.001)
    assert current.lp.c_bias == pytest.approx(0.01)
    assert current.lp.samples == 500
    assert Calibration.load("ets2").simple_warp_version == SIMPLE_WARP_VERSION
    assert current.to_dict()["simple_warp_version"] == SIMPLE_WARP_VERSION


def test_old_device_flow_cannot_reuse_done_with_a_new_camera_file():
    seed_legacy(legacy_camera=False)
    current = flow(Calibration.load("ets2"))
    assert current.step is Step.CAMERA
    assert current.lc.cal_status == CalStatus.UNCALIBRATED
    assert current.result is None


def test_simple_warp_migration_discards_all_saved_certificate_fields():
    seed_legacy()
    path = flow_path("ets2", "gamepad")
    data = json.loads(path.read_text())
    data.update(check_context={"camera": {"pitch_deg": 2.5}},
                check_model=[4.2, 0.001, 0.01], check_requested=True,
                legacy_result={"passed": True, "score": 0.87})
    path.write_text(json.dumps(data))
    current = flow(Calibration.load("ets2"))
    migrated = current.to_dict()
    assert migrated["check_context"] is None
    assert migrated["check_model"] is None
    assert migrated["legacy_result"] is None
    assert not migrated["check_requested"]
    assert not current.calibrated and not current.ai_steering_ready
    assert json.loads(path.read_text()) == migrated


def test_old_camera_without_flow_clears_learner():
    seed_legacy()
    flow_path("ets2", "gamepad").unlink()
    current = flow(Calibration.load("ets2"))
    assert current.step is Step.FOV
    assert current.lc.samples == 0
    assert current.calib.pitch_deg == current.calib.yaw_deg == 0.0


def test_new_camera_progress_survives_restart_after_migration():
    seed_legacy()
    current = flow(Calibration.load("ets2"))
    current.lc._block_n = current.lc.samples = 40
    current.lc._block_rpy_sum[:] = [0.0, 0.4, 0.0]
    current.lc._block_height_sum = 56.0
    current.lc.save_state("ets2")
    current.save()
    resumed = flow(Calibration.load("ets2"))
    assert resumed.step is Step.CAMERA
    assert resumed.lc.samples == resumed.lc._block_n == 40
    assert resumed.lc._block_rpy_sum == pytest.approx([0.0, 0.4, 0.0])
    assert resumed.progress == current.progress


def test_current_simple_pose_survives_but_uncertified_done_requires_check():
    seed_legacy(legacy_camera=False, legacy_flow=False)
    current = flow(Calibration.load("ets2"))
    # Current geometry preserves the pose, but an old Done has no certificate.
    assert current.step is Step.CHECK
    assert current.result["status"] == "INSUFFICIENT_EVIDENCE"
    assert current.legacy_result["passed"]
    assert not current.calibrated and not current.ai_steering_ready
    assert current.calib.pitch_deg == 2.5
    assert current.lc.cal_status == CalStatus.CALIBRATED


def test_perspective_pose_survives_but_uncertified_done_requires_check():
    seed_legacy(simple=False)
    cal = Calibration.load("ets2")
    current = flow(cal)
    # Perspective geometry needs no migration; certification still needs proof.
    assert current.step is Step.CHECK
    assert current.result["status"] == "INSUFFICIENT_EVIDENCE"
    assert current.legacy_result["passed"]
    assert not current.calibrated and not current.ai_steering_ready
    assert cal.pitch_deg == 2.5
    assert cal.yaw_deg == -0.7
    assert current.lc.cal_status == CalStatus.CALIBRATED
