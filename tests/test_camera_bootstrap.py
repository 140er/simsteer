"""Camera bootstrap, bounded waiting, and diagnostics without output devices."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from simsteer.calib.flow import CAMERA_NO_PROGRESS_S, CalibFlow, CameraOutcome, Obs, Step
from simsteer.core.calibration import Calibration
from simsteer.core.finite import NonFiniteError
from simsteer.core.learners.livecalib import (
    BLOCK_SIZE, CERTAINTY_BOOTSTRAP_BLOCKS, CalStatus, CameraBlockedReason, LiveCalib,
)
from simsteer.core.learners.liveparams import LiveParams
from simsteer.ui.hud import HudRenderer, HudState
from simsteer.ui.state import calibration_view
from tools.camera_replay import replay


def sample(learner, calib, **changes):
    values = dict(pose=np.array([10., 0., 0.]), road_transform=np.array([0., 0., 1.2]),
                  actual_yaw_rate=0., v_ego=10., pose_std=np.array([.01, .02, .01]),
                  road_transform_std=np.full(3, .01), game_steer=0.)
    learner.update(calib, **(values | changes))


@pytest.mark.parametrize("uncertainty,reason", [
    ({"pose_std": np.array([.01, 5., .01])}, CameraBlockedReason.LATERAL_UNCERTAINTY),
    ({"road_transform_std": np.array([.01, .01, 1.])}, CameraBlockedReason.HEIGHT_UNCERTAINTY),
])
def test_pinned_upstream_bootstrap_lasts_five_blocks_not_three(uncertainty, reason):
    # calibrationd.py at the driving-model pin: INPUTS_NEEDED=5 (line 28),
    # certain_if_calib bypasses both gates below that count (line 178).
    assert CERTAINTY_BOOTSTRAP_BLOCKS == 5
    learner, calib = LiveCalib(), Calibration()
    for _ in range(5 * BLOCK_SIZE):
        sample(learner, calib, **uncertainty)
        assert learner.last_rejection_reason is None
    assert learner.samples == 500 and learner.blocks == 5
    assert learner.cal_status is CalStatus.CALIBRATED
    sample(learner, calib, **uncertainty)
    assert learner.samples == 500 and learner.last_rejection_reason is reason
    assert learner.blocked_counts == {reason: 1}


@pytest.mark.parametrize("changes,reason", [
    ({"pose": None}, CameraBlockedReason.MISSING_POSE),
    ({"v_ego": None}, CameraBlockedReason.MISSING_SPEED),
    ({"v_ego": 1.}, CameraBlockedReason.SPEED),
    ({"actual_yaw_rate": .1}, CameraBlockedReason.TURNING),
    ({"actual_yaw_rate": .02, "v_ego": 100.}, CameraBlockedReason.LATERAL_ACCELERATION),
    ({"game_steer": .2}, CameraBlockedReason.HUMAN_STEERING),
    ({"pose": np.array([1., 0., 0.])}, CameraBlockedReason.MODEL_SPEED),
    ({"pose": np.array([10., 0., -5.])}, CameraBlockedReason.PITCH_BOUNDS),
    ({"pose": np.array([10., 2., 0.])}, CameraBlockedReason.YAW_BOUNDS),
    ({"road_transform": np.array([0., 0., .1])}, CameraBlockedReason.HEIGHT_BOUNDS),
])
def test_bootstrap_preserves_other_gates_and_counts_missing_inputs(changes, reason):
    learner, calib = LiveCalib(), Calibration()
    learner._last_game_steer = 0.
    sample(learner, calib, **changes)
    assert learner.samples == 0 and learner.rejected == 1
    assert learner.last_rejection_reason is reason
    assert learner.blocked_counts == {reason: 1}


def make_flow(*, persist=False, simple=True, clock=lambda: 0.):
    learner = LiveCalib(game="ets2" if persist else None, clock=clock)
    flow = CalibFlow("ets2", "gamepad", live_calib=learner, live_params=LiveParams(),
                     calib=Calibration(simple_warp=simple), persist=persist, model_identity={})
    if flow.step is Step.FOV:
        flow._enter(Step.CAMERA)
    return flow


def reject_tick(flow, *, dt=.25, **observation):
    sample(flow.lc, flow.calib, pose=np.array([1., 0., 0.]))
    values = dict(now=flow._now + dt, dt=dt, v=10., telemetry_ok=True,
                  driving=False, engaged=False)
    flow.tick(Obs(**(values | observation)))


@pytest.mark.parametrize("simple", [True, False])
def test_three_eligible_minutes_end_in_no_progress_and_disable_learning(simple):
    flow = make_flow(simple=simple)
    for _ in range(719):
        reject_tick(flow)
    assert flow.step is Step.CAMERA
    assert flow._camera_no_progress_s == CAMERA_NO_PROGRESS_S - .25
    reject_tick(flow)
    assert flow.step is Step.FAILED and flow.camera_outcome is CameraOutcome.NO_PROGRESS
    assert not flow.camera_learning and not flow.ai_steering_ready and not flow.calibrated
    assert flow.engage_block()
    assert flow.result["camera_blocked_counts"] == {"model_speed": 720}
    assert "MODEL_SPEED 720" in flow.result["reason"]
    assert "Check the road view and camera FOV." in flow.result["reason"]
    assert flow.log_fields()["camera"]["status"] == "NO_PROGRESS"
    flow.restart("retry")
    assert flow.camera_outcome is CameraOutcome.COLLECTING
    assert flow._camera_no_progress_s == 0 and not flow.lc.blocked_counts


def test_slow_missing_telemetry_and_time_stalls_do_not_consume_three_minutes():
    flow = make_flow()
    for _ in range(800):
        for observation in ({"v": None}, {"v": 0.}, {"telemetry_ok": False}):
            reject_tick(flow, **observation)
    assert flow._camera_no_progress_s == 0
    reject_tick(flow, dt=900.)
    assert flow._camera_no_progress_s == .25
    reject_tick(flow, dt=-1.)
    assert flow._camera_no_progress_s == .25


def test_accepted_progress_restarts_the_clock_and_reason_window():
    flow = make_flow()
    for _ in range(719):
        reject_tick(flow)
    sample(flow.lc, flow.calib)
    flow.tick(Obs(now=180., dt=.25, v=10.))
    assert flow._camera_no_progress_s == 0 and not flow._camera_blocked_counts
    reject_tick(flow)
    assert flow.step is Step.CAMERA and flow._camera_no_progress_s == .25


def test_discarded_blocks_cannot_keep_resetting_the_progress_clock():
    flow = make_flow()
    flow.lc._rpys.append(np.zeros(3))
    flow.lc._heights.append(1.2)
    flow.lc._block_n = 99
    flow.lc._block_rpy_sum[1] = 99 * math.radians(4.)
    flow.lc._block_height_sum = 99 * 1.2
    flow.lc.samples = 199
    flow._reset_camera_progress()
    for i in range(720):
        sample(flow.lc, flow.calib, pose=np.array([10., 0., -10 * math.tan(math.radians(4.))]))
        flow.tick(Obs(now=(i + 1) * .25, dt=.25, v=10.))
    assert flow.step is Step.FAILED and flow.lc.blocks == 1
    assert flow.camera_outcome is CameraOutcome.NO_PROGRESS
    assert flow.result["camera_blocked_counts"]["block_jump"] > 0


def test_rejected_only_diagnostics_and_timeout_survive_restart(data_dir):
    now = [0.]
    flow = make_flow(persist=True, clock=lambda: now[0])
    for _ in range(719):
        now[0] += .25
        reject_tick(flow)
    # Rejected-only input saves periodically, without waiting for an accepted block.
    saved = json.loads((data_dir / "livecalib_state_ets2.json").read_text())
    assert saved["blocked_counts"]["model_speed"] >= 700
    flow.lc.save_state("ets2")
    flow.save()
    restored = make_flow(persist=True, clock=lambda: now[0])
    assert restored.lc.last_rejection_reason is CameraBlockedReason.MODEL_SPEED
    assert restored.lc.blocked_counts == {"model_speed": 719}
    assert restored.lc.rej_speed == 719
    assert restored._camera_no_progress_s == 179.75
    reject_tick(restored)
    assert restored.step is Step.FAILED
    again = make_flow(persist=True)
    assert again.camera_outcome is CameraOutcome.NO_PROGRESS and not again.camera_learning


def test_nonfinite_samples_still_raise_before_learner_mutation():
    learner, calib = LiveCalib(), Calibration()
    with pytest.raises(NonFiniteError):
        sample(learner, calib, pose_std=np.array([.01, np.nan, .01]))
    assert learner.samples == learner.rejected == 0 and not learner.blocked_counts


def test_reason_and_no_progress_details_reach_shared_hud_state(monkeypatch):
    flow = make_flow()
    reject_tick(flow)
    view = calibration_view(flow=flow, v_ego=10.)
    assert view.blocked_reason is CameraBlockedReason.MODEL_SPEED
    assert view.blocked_counts == (("model_speed", 1),)
    assert "MODEL_SPEED" in view.instruction
    texts = []
    monkeypatch.setattr("simsteer.ui.hud.draw_text", lambda image, text, *a, **kw: texts.append(text))
    HudRenderer().draw(np.zeros((128, 1280, 3), np.uint8), HudState(calibration=view))
    assert any("MODEL_SPEED" in text for text in texts)
    flow._camera_no_progress_s = 179.75
    reject_tick(flow)
    view = calibration_view(flow=flow, v_ego=10.)
    assert view.step_name == "Camera NO_PROGRESS" and view.step_index == 1
    assert "MODEL_SPEED 2" in view.instruction
    assert "FOV" in view.key_prompt and "Recalibrate" in view.key_prompt


def test_telemetry_only_replay_never_counts_an_unlogged_model_as_accepted(tmp_path):
    path = tmp_path / "frames.jsonl"
    path.write_text("\n".join(json.dumps(dict(calib="camera", t=i / 20,
        telemetry="ok", v_ego=speed, wheel_angle=0., observed_axis=0.))
        for i, speed in enumerate((0., 10.))) + "\n")
    result, frames = replay([path], "frames")
    assert result["accepted"] is None and result["measured_accepted"] == 0
    assert result["reasons"] == {"speed": 1, "model_inputs_unlogged": 1}
    assert len(frames) == result["frames"] == 2


def test_full_replay_cannot_write_user_calibration(data_dir):
    path = data_dir / "samples.jsonl"
    row = dict(pose=[10., 0., 0.], road_transform=[0., 0., 1.2],
               speed=10., actual_yaw=0., pose_std=[.01] * 3,
               road_transform_std=[.01] * 3)
    path.write_text((json.dumps(row) + "\n") * 500)
    result, _ = replay([path], "samples")
    assert result["accepted"] == 500
    assert not (data_dir / "calibration.json").exists()
