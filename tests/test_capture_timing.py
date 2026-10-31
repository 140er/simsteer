"""Capture cadence and pose refinements must not become disengagements."""

import random

import numpy as np
import pytest

from simsteer.core import timing
from simsteer.core.calibration import Calibration
from simsteer.core.finite import require_finite
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue
from simsteer.runtime.fov import FovSolver
from tools.measure_capture_timing import capture_timestamps
from test_time_base import small_views, pixels, flow_views, flow_vision, model_inputs


@pytest.mark.parametrize("hz", [8, 9, 10, 12, 15, 18, 20])
def test_capture_reader_jitter_keeps_history_and_calibration_pairs(small_views, model_inputs, hz):
    queue = FrameQueue()
    model, _ = model_inputs
    stamps = list(capture_timestamps(hz, random.Random(0)))
    good = 0
    for t in stamps:
        inputs = queue.push(pixels(1), Calibration(), timestamp=t)
        model.step(*inputs, timestamp=t)
        if t >= stamps[0] + .2 + 1 / hz:
            assert queue.ready
        good += queue.pair_ok
    assert queue.resets == 0
    assert good / len(stamps) >= .9
    # A model reset would discard the long context despite the warm pair.
    assert model._times[-1] - model._times[0] > 4.7


def test_discontinuity_limit_preserves_the_pair_span_contract():
    assert timing.MIN_HEALTHY_FPS == 8.
    assert timing.DISCONTINUITY_GAP_S == 2 / timing.MIN_HEALTHY_FPS
    assert timing.DISCONTINUITY_GAP_S <= timing.CONTEXT_PERIOD_S + timing.PAIR_SPAN_TOL_S


def test_bad_span_skips_calibration_without_resetting_history(small_views):
    queue = FrameQueue()
    for i in range(6):
        queue.push(pixels(i), timestamp=i / 20)
    queue.push(pixels(9), timestamp=.49)  # 240 ms arrival gap; nearest pair is 240 ms.
    assert queue.ready and queue.pair_ok and queue.resets == 0
    queue.push(pixels(10), timestamp=.55)  # Nearest old frame is 300 ms away.
    assert queue.ready and not queue.pair_ok and queue.resets == 0
    queue.push(pixels(11), timestamp=.6)
    assert queue.ready and not queue.pair_ok and queue.resets == 0
    queue.push(pixels(12), timestamp=.65)
    assert queue.ready and queue.pair_ok and queue.resets == 0


@pytest.mark.parametrize("field", ["pitch_deg", "yaw_deg", "height_m"])
def test_pose_change_tags_each_buffered_frame_without_a_reset(small_views, field):
    queue, calib = FrameQueue(), Calibration(simple_warp=False)
    for i in range(6):
        queue.push(pixels(i), calib, timestamp=i / 20)
    setattr(calib, field, getattr(calib, field) + .01)
    for i in range(6, 10):
        queue.push(pixels(i), calib, timestamp=i / 20)
        assert queue.ready and not queue.pair_ok and queue.resets == 0
    queue.push(pixels(10), calib, timestamp=.5)
    assert queue.ready and queue.pair_ok and queue.resets == 0


def test_bad_main_pair_does_not_stop_candidate_ticks_or_drop_evidence():
    calib = Calibration(simple_warp=False)
    solver = FovSolver(flow_vision, flow_views)
    for i in range(55):
        solver.update(np.array([i / 20]), calib, 22., 22., 0., timestamp=i / 20, pair_ok=True)
    assert solver.stage == "bracket" and any(solver._prev)
    evidence = [list(ratios) for ratios in solver._cand_ratios]
    estimates = list(solver._ratios)
    calib.pitch_deg += .01
    tick = solver._tick
    for i in range(55, 59):
        solver.update(np.array([i / 20]), calib, 22., 22., 0.,
                      timestamp=i / 20, pair_ok=False)
    assert solver._tick == tick + 4
    assert solver._ratios == estimates and solver._cand_ratios == evidence
    assert all(t >= 2.75 for t in solver._prev_times)
    # The main-pair flag never gates the candidates' own four-tick pairs.
    solver.update(np.array([2.95]), calib, 22., 22., 0., timestamp=2.95, pair_ok=False)
    assert sum(map(len, solver._cand_ratios)) == sum(map(len, evidence)) + 1


def test_pose_straddling_pair_is_not_used_for_fov_estimate(small_views):
    queue, calib = FrameQueue(), Calibration(simple_warp=False)
    solver = FovSolver()
    for i in range(6):
        queue.push(pixels(i), calib, timestamp=i / 20)
    solver.update(None, calib, 22., 22., 0., timestamp=.25, pair_ok=True)
    calib.yaw_deg += .01
    for i in range(6, 10):
        queue.push(pixels(i), calib, timestamp=i / 20)
        solver.update(None, calib, 22., 22., 0., timestamp=i / 20, pair_ok=queue.pair_ok)
        assert solver.samples == 1
    queue.push(pixels(10), calib, timestamp=.5)
    solver.update(None, calib, 22., 22., 0., timestamp=.5, pair_ok=queue.pair_ok)
    assert solver.samples == 2


def test_completed_fov_survives_later_geometry_edits():
    solver, calib = FovSolver(), Calibration()
    for i in range(40):
        solver.update(None, calib, 22., 22., 0., timestamp=i / 20, pair_ok=True)
    result = solver.result
    calib.fov_h_deg += 10
    solver.update(None, calib, 20., 22., 0., timestamp=3., pair_ok=True)
    assert solver.done and solver.result == result


def test_zero_span_has_no_pose_or_uncertainty():
    decoded = decode(np.zeros(1576), np.zeros(1000), frame_span_s=0.)
    assert decoded.pose is None and decoded.pose_std is None
    require_finite("decoded", **vars(decoded))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_stamp_clears_temporal_state_before_rejection(small_views, model_inputs, bad):
    from simsteer.core.finite import NonFiniteError
    queue, calib = FrameQueue(), Calibration()
    model, _ = model_inputs
    solver = FovSolver()
    for i in range(10):
        img, big = queue.push(pixels(1), calib, timestamp=i / 20)
        model.step(img, big, timestamp=i / 20)
        solver.update(None, calib, 22., 22., 0., timestamp=i / 20, pair_ok=True)
    with pytest.raises(NonFiniteError):
        queue.push(pixels(1), calib, timestamp=bad)
    with pytest.raises(NonFiniteError):
        model.step(img, big, timestamp=bad)
    with pytest.raises(NonFiniteError):
        solver.update(None, calib, 22., 22., 0., timestamp=bad, pair_ok=True)
    assert queue.resets == 1 and not queue.ready and not queue.pair_ok
    assert not model._times and not model._feat_q and not model._desire_q
    assert solver.samples == 0


def test_user_accepting_a_different_fov_still_resets_frame_history(small_views):
    from test_simple_warp_state import flow
    calib = Calibration()
    current, queue = flow(calib), FrameQueue()
    for i in range(6):
        queue.push(pixels(i), calib, timestamp=i / 20)
    current.fov_proposal = 80.
    assert current.accept_fov()
    queue.push(pixels(6), calib, timestamp=.3)
    assert calib.fov_h_deg == 80.
    assert queue.resets == 1 and not queue.ready and not queue.pair_ok


def test_flow_restart_explicitly_clears_temporal_inputs(monkeypatch, small_views):
    from test_output_safety import drive
    queue = FrameQueue()

    def request(st, manager):
        if st.i == 18:
            st.flow.request_restart("test reset")

    drive(monkeypatch, frames=20, frame_queue_factory=lambda: queue, on_grab=request)
    assert queue.resets == 1 and not queue.ready


def test_livecalib_commits_keep_an_engaged_drive_and_skip_straddling_pairs(
        monkeypatch, small_views, tmp_path):
    import json
    import simsteer.runtime.loop as loop
    from simsteer.core.learners.livecalib import INPUTS_NEEDED, LiveCalib
    from simsteer.runtime.output.gamepad import Gamepad
    from test_output_safety import drive

    state = {"i": -1, "calls": [], "commits": [], "disengages": []}
    queue = FrameQueue()
    monkeypatch.setattr(loop.Calibration, "load", lambda *a, **kw: Calibration(simple_warp=False))

    def update(self, calib, *args, **kwargs):
        self.last_commit_delta_deg = None
        i = state["i"]
        state["calls"].append(i)
        if i not in (8, 20, 32):
            return
        if not self._rpys:
            self._rpys.extend([np.zeros(3)] * INPUTS_NEEDED)
            self._heights.extend([calib.height_m] * INPUTS_NEEDED)
            self.samples = INPUTS_NEEDED * 100
            if hasattr(self, "bootstrap_samples"):
                self.bootstrap_samples = self.samples
        # Exercise the real block-commit writer while retaining the existing
        # certificate's pose tolerances. No device is opened.
        before = calib.pitch_deg
        self._block_n = 100
        self._block_rpy_sum[:] = [0., np.radians(.02) * 100, 0.]
        self._block_height_sum = calib.height_m * 100
        self._commit_block(calib)
        assert calib.pitch_deg != before
        state["commits"].append(i)

    original = Gamepad.disengage

    def disengage(self):
        if self.engaged and 4 < state["i"] < 48:
            state["disengages"].append(state["i"])
        original(self)

    monkeypatch.setattr(LiveCalib, "update", update)
    monkeypatch.setattr(Gamepad, "disengage", disengage)
    path = tmp_path / "drive.jsonl"
    run, _, _ = drive(monkeypatch, frames=48, frame_queue_factory=lambda: queue, log_path=path,
                      on_grab=lambda st, manager: state.update(i=st.i))
    assert state["commits"] == [8, 20, 32]
    assert state["disengages"] == [] and queue.resets == 0
    assert all(engaged for engaged, _ in run.checkpoints[8:])
    for commit in state["commits"]:
        assert not (set(range(commit + 1, commit + 5)) & set(state["calls"]))
        assert commit + 5 in state["calls"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    commits = [i for i, row in enumerate(rows) if row["camera_commit_delta_deg"] is not None]
    assert commits == state["commits"]
    assert all(0. < rows[i]["camera_commit_delta_deg"] < .02 for i in commits)


def test_pitch_hotkey_keeps_engagement_and_excludes_straddling_pairs(monkeypatch, small_views):
    import simsteer.runtime.loop as loop
    from test_output_safety import drive

    pairs = []

    class ObservedQueue(FrameQueue):
        def push(self, *args, **kwargs):
            result = super().push(*args, **kwargs)
            pairs.append(self.pair_ok)
            return result

    queue = ObservedQueue()
    monkeypatch.setattr(loop.Calibration, "load", lambda *a, **kw: Calibration(simple_warp=False))
    # Isolate the user edit from automatic pose writes.
    monkeypatch.setattr(loop.LiveCalib, "update", lambda *a, **kw: None)
    st, _, _ = drive(monkeypatch, frames=18, mode="dev", frame_queue_factory=lambda: queue,
                     keys_at={8: [ord(".")]})
    assert st.flow.calib.pitch_deg == .5
    assert queue.resets == 0
    assert all(engaged for engaged, _ in st.checkpoints[8:])
    assert pairs[8] and pairs[13]
    assert not any(pairs[9:13])
