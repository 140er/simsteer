"""Calibration diagnostics and explicit pair eligibility."""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.calib.flow import CalibFlow, CameraOutcome, Step
from simsteer.core.calibration import Calibration
from simsteer.core.finite import require_finite
from simsteer.core.learners.livecalib import CalStatus, CameraBlockedReason, LiveCalib
from simsteer.runtime.faults import neutral_decoded
from simsteer.runtime.fov import FovSolver


def test_placeholder_pose_and_uncertainty_are_absent():
    decoded = neutral_decoded()
    assert decoded.pose is None and decoded.pose_std is None
    require_finite("placeholder", **vars(decoded))


@pytest.mark.parametrize("method", [FovSolver.update, CalibFlow.feed_fov])
def test_pair_eligibility_must_be_supplied(method):
    parameter = inspect.signature(method).parameters["pair_ok"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("check", ["prompt", "timeout"])
def test_bad_pairs_replace_a_stale_speed_rejection_in_the_loop(monkeypatch, check):
    import simsteer.calib.flow as flow_module
    from test_output_safety import drive

    monkeypatch.setattr(flow_module, "CAMERA_NO_PROGRESS_S", 10.)
    hints = []

    def prepare(st, manager):
        if st.i == 0:
            st.flow.lc.cal_status = CalStatus.UNCALIBRATED
            st.flow.lc.record_blocked(CameraBlockedReason.SPEED)
            st.flow._enter(Step.CAMERA)
        if st.i == 1:
            hints.append(st.flow.hint(20., engaged=False))

    def unexpected_update(*a, **kw):
        pytest.fail("LiveCalib must not consume a timing-ineligible pair")

    monkeypatch.setattr(LiveCalib, "update", unexpected_update)
    queue = SimpleNamespace(push=lambda *a, **kw: (None, None), ready=True,
                            pair_ok=False, span_s=.2, resets=0, reset=lambda: None)
    st, _, _ = drive(monkeypatch, frames=201, force=False, on_grab=prepare,
                     frame_queue_factory=lambda: queue)
    if check == "prompt":
        assert "FRAME_TIMING" in hints[0] and "SPEED" not in hints[0]
    else:
        assert st.flow.camera_outcome is CameraOutcome.NO_PROGRESS
        assert "FRAME_TIMING" in st.flow.result["reason"]
        assert "SPEED" not in st.flow.result["reason"]
        assert st.flow.result["camera_blocked_counts"] == {"frame_timing": 200}


@pytest.mark.parametrize("initial,estimate,delta", [
    ((3., -2.), (1., 1.), 3.),
    ((3., 0.), (-1., .5), 4.),
])
def test_livecalib_commit_delta_measures_pitch_and_yaw_once(initial, estimate, delta):
    learner = LiveCalib()
    calib = Calibration(pitch_deg=initial[0], yaw_deg=initial[1])
    learner._rpys.extend([np.radians([0., *estimate])] * 3)
    learner._heights.extend([calib.height_m] * 3)
    learner.bootstrap_samples = 300
    learner._refresh_cal_status()
    learner._write_pose(calib)
    assert learner.last_commit_delta_deg == pytest.approx(delta)
    learner._write_pose(calib)
    assert learner.last_commit_delta_deg == 0.
    learner.update(calib, None, None, 0., 20.)
    assert learner.last_commit_delta_deg is None


def test_placeholder_log_does_not_claim_pose_certainty(monkeypatch, tmp_path):
    from test_output_safety import drive

    class MissingFrames:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def grab(self):
            return None

    path = tmp_path / "capture.jsonl"
    drive(monkeypatch, frames=2, source=MissingFrames(), log_path=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["pose_vx"] is None and row["pose_lateral_std"] is None for row in rows)
