from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from simsteer import calib as calib_api
from simsteer.calib import fov_source
from simsteer.calib.flow import (
    CHECK_SECONDS, CalibFlow, Obs, Step, flow_path,
)
from simsteer.core.calibration import Calibration
from simsteer.games.profile import FovSource
from simsteer.core.learners.livecalib import BLOCK_SIZE, INPUTS_NEEDED, CalStatus
from simsteer.core.learners.liveparams import TRUSTED_MIN_SAMPLES, LiveParams
from simsteer.runtime import fov
from simsteer.runtime.fov import FovSolver, solve_from_ratio
from test_steering_direction import _probe_output

FIXTURE = Path(__file__).parent / "fixtures" / "ets2_config.cfg"
TRUE_FOV = 75.0


class FakeLiveCalib:
    def __init__(self) -> None:
        self.cal_status = CalStatus.UNCALIBRATED
        self.blocks = 0
        self._block_n = 0
        self.samples = 0
        self.rej_human = 0
        self.last_reset_reason = None
        self.resets: list[str] = []
        self.rescales: list[tuple[float, float]] = []

    def converge(self) -> None:
        self.blocks = INPUTS_NEEDED
        self.cal_status = CalStatus.CALIBRATED

    def reset(self, reason=None) -> None:
        resets, rescales = self.resets, self.rescales
        self.__init__()
        self.resets, self.rescales = resets + [reason], rescales

    def rescale_for_fov(self, calib, old, new) -> None:
        self.rescales.append((old, new))

    def eta_to_calibrated_s(self):
        return None


class FakeLiveParams:
    a_linear, b_quad, c_bias = 3.5, 0.0015, 0.0

    def __init__(self, samples: int = 0) -> None:
        self.lag_frames = 1
        self.axis_sign = None
        self.samples = samples
        self.session_samples = 0
        self.predictions = 0
        self.last_innovation = 0.0
        self.last_actual = 0.0
        self.wipes = 0
        self.validation_model = None

    @property
    def validating(self):
        return self.validation_model is not None

    def begin_validation(self, coefficients=None):
        self.validation_model = coefficients or (self.a_linear, self.b_quad, self.c_bias)

    def end_validation(self):
        self.validation_model = None

    def trusted(self) -> bool:
        return max(self.samples, self.session_samples) >= TRUSTED_MIN_SAMPLES

    def reset(self, wipe_disk: bool = False, clear_axis_sign: bool = False) -> None:
        self.samples = self.session_samples = 0
        if clear_axis_sign:
            self.axis_sign = None
        self.wipes += int(wipe_disk)

    def reset_covariance_only(self) -> None:
        pass

    def set_axis_sign(self, sign: int) -> None:
        self.axis_sign = sign

    def predict(self, error: float, actual: float = 0.1) -> None:
        self.predictions += 1
        self.last_innovation, self.last_actual = error, actual


def make_solver(true_fov: float = TRUE_FOV) -> FovSolver:
    """A solver whose vision reads speed as tan(true/2)/tan(set/2) of the
    real speed. The fake views carry the warp's FOV as pixel values."""
    def views(frame, calib):
        return np.full((6, 1, 1), calib.fov_h_deg), np.zeros((6, 1, 1))

    def vision(img, big):
        prev_fov, cur_fov = float(img[0, 0, 0, 0]), float(img[0, 6, 0, 0])
        assert prev_fov == cur_fov  # both frames warped at one candidate
        return model_vx(cur_fov, true_fov), 0.1
    return FovSolver(vision, views)


def model_vx(set_fov: float, true_fov: float = TRUE_FOV, v: float = 22.0) -> float:
    return v * (math.tan(math.radians(true_fov) / 2)
                / math.tan(math.radians(set_fov) / 2))


def new_flow(lc=None, lp=None, cal=None, **kw) -> CalibFlow:
    kw.setdefault("make_fov_solver", make_solver)
    kw.setdefault("today", lambda: "2026-10-30")
    return CalibFlow("ets2", "gamepad", live_calib=lc or FakeLiveCalib(),
                     live_params=lp or FakeLiveParams(),
                     calib=cal or Calibration(fov_h_deg=90.0), **kw)


class Driver:
    def __init__(self, flow: CalibFlow) -> None:
        self.flow = flow
        self.t = 0.0

    def frames(self, n: int, **obs) -> None:
        for _ in range(n):
            self.t += 0.05
            o = dict(v=22.0, yaw_rate=0.0, wheel=0.0, driving=True, engaged=True,
                     lane_offset=0.02, vx_model=22.0, axis=0.15 * math.sin(self.t))
            o.update(obs)
            if (self.flow.step is Step.CHECK
                    and isinstance(self.flow.lp, FakeLiveParams)
                    and self.flow.lp.predictions == self.flow._pred_seen):
                # Supply independent validation observations. The former
                # fixture passed without any steering-fit observations.
                self.flow.lp.predict(0.005, actual=0.1)
            if self.flow.step is Step.FOV:
                self.flow.feed_fov(np.zeros((2, 2, 3), np.uint8),
                                   model_vx(self.flow.calib.fov_h_deg), o["v"], 0.0)
            self.flow.tick(Obs(now=self.t, dt=0.05, **o))

    def verify_direction(self) -> None:
        # A real manual drive through both bends, independent of AI output.
        for i in range(180):
            if self.flow.direction.ready:
                break
            wheel = 0.015 * math.sin(2 * math.pi * i / 90)
            self.frames(1, wheel=wheel, observed_axis=4.0 * wheel,
                        model_curvature=wheel / 2.65, engaged=False)
        _probe_output(self.flow, -1)
        self.frames(1)

    def until(self, step: Step, limit: int = 20000, **obs) -> None:
        for _ in range(limit):
            if self.flow.step is step:
                return
            self.frames(1, **obs)
        raise AssertionError(f"stuck at {self.flow.step}, wanted {step}")


def steps_seen(flow: CalibFlow, lc: FakeLiveCalib, lp: FakeLiveParams) -> list[Step]:
    seen = [flow.step]

    def note() -> None:
        if flow.step is not seen[-1]:
            seen.append(flow.step)
    d = Driver(flow)
    for _ in range(20000):
        if flow.step is Step.FOV and flow.fov_proposal is not None:
            flow.accept_fov()
            note()
        d.frames(1)
        note()
        if flow.step is Step.CAMERA:
            lc.converge()
        if flow.step is Step.STEERING:
            if not flow.direction.ready:
                d.verify_direction()
            lp.samples = TRUSTED_MIN_SAMPLES
        if flow.step in (Step.DONE, Step.FAILED):
            break
    return seen


# ----- step order -----

def test_full_profile_runs_every_step_in_order():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    assert flow.steps == [Step.FOV, Step.CAMERA, Step.STEERING, Step.CHECK]
    assert steps_seen(flow, lc, lp) == [Step.FOV, Step.CAMERA, Step.STEERING,
                                        Step.CHECK, Step.DONE]
    assert flow.calibrated
    assert flow.result["date"] == "2026-10-30"


def test_generic_profile_skips_steering_and_keeps_its_seed():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp, steering_fit=False)
    assert flow.steps == [Step.FOV, Step.CAMERA, Step.CHECK]
    assert steps_seen(flow, lc, lp) == [Step.FOV, Step.CAMERA, Step.CHECK, Step.DONE]
    assert lp.wipes == 0


def test_converged_learners_still_require_manual_direction_evidence():
    lc, lp = FakeLiveCalib(), FakeLiveParams(samples=500)
    lc.converge()
    flow = new_flow(lc, lp, cal=Calibration(fov_h_deg=75.5),
                    fov_config=lambda aspect: 75.0)
    Driver(flow).frames(1)
    assert flow.step is Step.STEERING and flow.engage_block()
    Driver(flow).verify_direction()
    assert flow.step is Step.CHECK
    assert lc.resets == []


def test_config_fov_change_rescales_the_camera_history():
    lc = FakeLiveCalib()
    lc.converge()
    flow = new_flow(lc, cal=Calibration(fov_h_deg=90.0), fov_config=lambda aspect: 75.0)
    Driver(flow).frames(1)
    assert lc.rescales == [(90.0, 75.0)] and lc.resets == []
    assert flow.step is Step.STEERING


def test_engage_is_blocked_until_the_camera_is_calibrated():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    assert flow.engage_block()
    d = Driver(flow)
    while flow.step is Step.FOV:
        if flow.fov_proposal is not None:
            assert "FOV" in flow.engage_block()
            flow.accept_fov()
        d.frames(1)
    assert flow.step is Step.CAMERA and flow.engage_block()
    lc.converge()
    d.frames(1)
    assert flow.step is Step.STEERING and flow.engage_block()
    d.verify_direction()
    assert flow.engage_block() == ""
    assert flow.probe_wanted and flow.gate_override is not None


def test_simple_warp_measures_the_pose_once_from_zero():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    cal = Calibration(fov_h_deg=75.0, pitch_deg=3.0, yaw_deg=1.0)
    flow = new_flow(lc, lp, cal=cal, fov_config=lambda aspect: 75.0)
    assert (cal.pitch_deg, cal.yaw_deg) == (0.0, 0.0)
    Driver(flow).frames(1)
    assert flow.step is Step.CAMERA
    assert flow.camera_learning
    lc.converge()
    Driver(flow).frames(1)
    assert flow.step is Step.STEERING and not flow.camera_learning

    cal.simple_warp = False
    assert flow.camera_learning


# ----- FOV -----

def test_fov_source_from_config_fixture(tmp_path):
    assert fov_source.read_scs_cfg(FIXTURE, "r_multimon_fov_horizontal") == 68.5
    assert fov_source.read_scs_cfg(FIXTURE, "g_steer_dead_zone") == 0.0
    assert fov_source.read_scs_cfg(FIXTURE, "missing_key") is None
    assert fov_source.read_scs_cfg(tmp_path / "nope.cfg", "x") is None

    source = FovSource(file=str(FIXTURE), key="r_multimon_fov_horizontal")
    reader = fov_source.profile_reader(source)
    assert reader(16 / 9) == 68.5
    lc = FakeLiveCalib()
    cal = Calibration(fov_h_deg=90.0)
    flow = new_flow(lc, cal=cal, fov_config=reader)
    assert flow.step is Step.FOV
    Driver(flow).frames(1)
    assert flow.step is Step.CAMERA
    assert cal.fov_h_deg == 68.5
    assert flow.fov_source == "config"
    assert lc.rescales == [(90.0, 68.5)] and lc.resets == []


def test_profile_reader_converts_vertical_fov(tmp_path, monkeypatch):
    cfg = tmp_path / "game.ini"
    cfg.write_text("[Camera]\nFieldOfView = 60\n")
    monkeypatch.setenv("GAME_DIR", str(tmp_path))
    horizontal = fov_source.profile_reader(
        FovSource(file="$GAME_DIR/game.ini", key="FieldOfView"))
    vertical = fov_source.profile_reader(
        FovSource(file="$GAME_DIR/game.ini", key="FieldOfView", axis="vertical"))
    assert horizontal(16 / 9) == 60.0
    assert vertical(16 / 9) == pytest.approx(91.5, abs=0.1)
    assert fov_source.profile_reader(None) is None


def test_missing_or_odd_config_value_means_measure(tmp_path):
    cfg = tmp_path / "game.cfg"
    cfg.write_text('uset fov "500"\n')
    assert fov_source.config_reader(str(cfg), "fov")(1.8) is None
    assert fov_source.config_reader(str(tmp_path / "gone.cfg"), "fov")(1.8) is None
    flow = new_flow(fov_config=fov_source.config_reader(str(cfg), "fov"))
    d = Driver(flow)
    d.frames(1)
    assert flow.step is Step.FOV
    while flow.fov_proposal is None:
        d.frames(1)
    assert flow.fov_proposal == pytest.approx(TRUE_FOV, abs=0.5)


def test_formula_points_the_right_way():
    # Set too wide, the model reads slow, and the solve narrows the FOV.
    assert solve_from_ratio(95.0, model_vx(95.0) / 22.0) == pytest.approx(TRUE_FOV)
    assert solve_from_ratio(55.0, model_vx(55.0) / 22.0) == pytest.approx(TRUE_FOV)


@pytest.mark.parametrize("start", [55.0, 95.0, 120.0])
def test_fov_measurement_proposes_and_waits_for_the_player(start):
    flow = new_flow(cal=Calibration(fov_h_deg=start))
    d = Driver(flow)
    while flow.fov_proposal is None:
        d.frames(1)
    assert flow.fov_proposal == pytest.approx(TRUE_FOV, abs=0.5)
    d.frames(50)
    assert flow.step is Step.FOV  # never applied on its own
    assert flow.calib.fov_h_deg == start
    assert flow.accept_fov()
    assert flow.calib.fov_h_deg == pytest.approx(TRUE_FOV, abs=0.5)
    assert flow.step is Step.CAMERA and flow.fov_source == "measured"


def test_keep_fov_keeps_the_players_value():
    flow = new_flow(cal=Calibration(fov_h_deg=95.0))
    d = Driver(flow)
    while flow.fov_proposal is None:
        d.frames(1)
    assert flow.keep_fov()
    assert flow.calib.fov_h_deg == 95.0 and flow.fov_source == "kept"


def test_small_fov_correction_applies_without_asking():
    flow = new_flow(cal=Calibration(fov_h_deg=76.0))
    d = Driver(flow)
    d.until(Step.CAMERA)
    assert flow.fov_source == "measured"
    assert flow.calib.fov_h_deg == pytest.approx(TRUE_FOV, abs=0.5)


def test_manual_fov_entry_ends_the_step_and_rescales_the_camera():
    lc = FakeLiveCalib()
    flow = new_flow(lc)
    d = Driver(flow)
    d.frames(5)
    flow.calib.fov_h_deg = 80.0
    d.frames(1)
    assert flow.step is Step.CAMERA and flow.fov_source == "manual"
    assert lc.rescales == [(90.0, 80.0)]
    flow.calib.fov_h_deg = 78.0
    d.frames(1)
    assert lc.rescales[-1] == (80.0, 78.0) and flow.step is Step.CAMERA


def test_solver_without_vision_uses_the_estimate():
    s = FovSolver()
    cal = Calibration(fov_h_deg=95.0)
    out = None
    for _ in range(fov.ESTIMATE_SAMPLES):
        out = s.update(None, cal, model_vx(95.0), 22.0, 0.0) or out
    assert out == pytest.approx(TRUE_FOV)


@pytest.mark.parametrize("resume_at", [None, 95])
def test_candidate_measurement_handles_periodic_motion_and_resume(resume_at):
    phases = {}

    def views(frame, calib):
        pixels = np.zeros((6, 1, 2))
        pixels[:, 0, 0] = calib.fov_h_deg
        pixels[:, 0, 1] = frame.item()
        return pixels, pixels

    def vision(img, big):
        previous, current = img[0, 0, 0, 1], img[0, 6, 0, 1]
        assert current - previous == fov.FRAME_SKIP
        candidate = float(img[0, 6, 0, 0])
        phase = int(current) % fov.FRAME_SKIP
        phases.setdefault(candidate, set()).add(phase)
        bias = (0.72, 1.28, 0.88, 1.12)[phase]
        return model_vx(candidate) * bias, 0.1

    solver = FovSolver(vision, views)
    cal = Calibration(fov_h_deg=95.0)
    for frame in range(500):
        solver.update(np.array([frame]), cal, model_vx(95.0), 22.0, 0.0)
        if frame == resume_at:
            saved = solver.to_dict()
            solver = FovSolver(vision, views)
            solver.load_dict(saved)
        if solver.done:
            break
    assert solver.result == pytest.approx(TRUE_FOV, abs=0.5)
    assert all(seen == {0, 1, 2, 3} for seen in phases.values())


@pytest.mark.parametrize("responses, expected", [
    ([1.15, 1.04, 1.07, 0.85], TRUE_FOV),
    ([1.04, 1.10, 1.02, 0.96], 78.333333),
])
def test_crossing_needs_a_consistent_local_response(responses, expected):
    response_by_fov = dict(zip((69.0, 73.0, 77.0, 81.0), responses))

    def views(frame, calib):
        return np.full((6, 1, 1), calib.fov_h_deg), np.zeros((6, 1, 1))

    def vision(img, big):
        return 22.0 * response_by_fov[float(img[0, 6, 0, 0])], 0.1

    solver = FovSolver(vision, views)
    cal = Calibration(fov_h_deg=95.0)
    for _ in range(500):
        solver.update(np.zeros((1, 1)), cal, model_vx(95.0), 22.0, 0.0)
        if solver.done:
            break
    assert solver.result == pytest.approx(expected)


# ----- resume -----

def test_resume_mid_camera_and_mid_check(tmp_path):
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    d = Driver(flow)
    while flow.step is Step.FOV:
        if flow.fov_proposal is not None:
            flow.accept_fov()
        d.frames(1)
    lc.blocks, lc._block_n = 1, 40
    pct = flow.progress
    flow.save()
    again = new_flow(lc, lp)
    assert again.step is Step.CAMERA
    assert again.progress == pytest.approx(pct)

    lc.converge()
    lp.samples = TRUSTED_MIN_SAMPLES
    d = Driver(again)
    d.verify_direction()
    d.until(Step.CHECK)
    d.frames(1000)
    seconds = again.check.seconds
    again.save()
    third = new_flow(lc, lp)
    assert third.step is Step.CHECK
    assert third.check.seconds == pytest.approx(seconds)
    assert third.progress == pytest.approx(seconds / CHECK_SECONDS)


def test_resume_mid_fov_measurement():
    flow = new_flow(cal=Calibration(fov_h_deg=95.0))
    d = Driver(flow)
    d.frames(fov.ESTIMATE_SAMPLES + 20)
    assert flow.fov_solver.stage == "bracket"
    progress = flow.progress
    flow.save()
    again = new_flow(cal=flow.calib)
    assert again.fov_solver.stage == "bracket"
    assert again.progress == pytest.approx(progress)


def test_other_version_starts_over(tmp_path):
    flow = new_flow()
    flow.save()
    d = json.loads(flow_path("ets2", "gamepad").read_text())
    d["version"] = 0
    d["step"] = "done"
    flow_path("ets2", "gamepad").write_text(json.dumps(d))
    assert new_flow().step is Step.FOV


# ----- check outcome -----

def _to_check(flow, lc, lp) -> Driver:
    d = Driver(flow)
    lp.samples = TRUSTED_MIN_SAMPLES
    for _ in range(20000):
        if flow.step is Step.CHECK:
            return d
        if flow.fov_proposal is not None:
            flow.accept_fov()
        if flow.step is Step.CAMERA:
            lc.converge()
        if flow.step is Step.STEERING and not flow.direction.ready:
            d.verify_direction()
            continue
        d.frames(1)
    raise AssertionError(f"stuck at {flow.step}")


def test_failed_check_redoes_steering_with_a_fresh_fit_then_stops():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    d = _to_check(flow, lc, lp)
    for _ in range(3000):
        lp.predict(error=0.9)
        d.frames(1)
        if flow.step is not Step.CHECK:
            break
    assert flow.step is Step.STEERING
    assert lp.wipes == 1
    assert lp.axis_sign is None
    assert not flow.output_direction.ready
    assert "mispredicts" in flow.redo_note

    lp.samples = TRUSTED_MIN_SAMPLES
    assert flow.engage_block()
    d.verify_direction()
    d.until(Step.CHECK)
    for _ in range(3000):
        lp.predict(error=0.9)
        d.frames(1)
        if flow.step is not Step.CHECK:
            break
    assert flow.step is Step.FAILED
    panel = flow.panel(22.0)
    assert panel.title == "NOT CALIBRATED Steering"
    assert "R restarts" in panel.hint
    st = calib_api.status("ets2", "gamepad")
    assert not st.calibrated and st.redo_step == "steering"


def test_check_only_counts_driven_highway_frames():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    d = _to_check(flow, lc, lp)
    d.frames(100, driving=False)
    d.frames(100, v=10.0)
    d.frames(100, telemetry_ok=False)
    assert flow.check.seconds == 0.0
    d.frames(100)
    assert flow.check.seconds == pytest.approx(5.0)


# ----- restart -----

def test_restart_wipes_fits_and_starts_at_fov():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    d = _to_check(flow, lc, lp)
    d.frames(10)
    flow.request_restart("R pressed")
    assert flow.take_restart_request() == "R pressed"
    assert flow.take_restart_request() is None
    flow.restart("R pressed")
    assert flow.step is Step.FOV
    assert lc.resets[-1] == "R pressed" and lc.blocks == 0
    assert lp.wipes == 1 and not lp.trusted()
    assert flow.result is None


# ----- HUD and status -----

def test_hud_panel_shows_step_percent_and_eta():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    d = Driver(flow)
    d.frames(30)
    panel = flow.panel(22.0)
    assert panel.title.startswith("CALIBRATE 1/4 FOV ")
    assert panel.title.endswith(" s") or panel.title.endswith(" min")
    while flow.step is Step.FOV:
        if flow.fov_proposal is not None:
            flow.accept_fov()
        d.frames(1)
    lc.blocks, lc._block_n = 1, 89
    assert flow.panel(22.0).title.startswith("CALIBRATE 2/4 Camera 63%")
    assert flow.panel(3.0).hint == "Drive faster than 25 km/h"

    lc.converge()
    d.frames(1)
    assert flow.panel(22.0, engaged=False).hint == "Drive gentle left and right bends manually"
    d.verify_direction()
    assert flow.panel(22.0, engaged=False).hint == "Press INSERT and let SimSteer drive"
    lp.samples = TRUSTED_MIN_SAMPLES
    d.until(Step.DONE)
    assert flow.panel(22.0).title.startswith("CALIBRATED score ")
    d.frames(400)
    assert flow.panel(22.0) is None


def test_status_api_and_offline_restart(data_dir):
    assert calib_api.status("ets2", "gamepad").label == "not calibrated"
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp)
    assert calib_api.status("ets2", "gamepad").step == "fov"
    steps_seen(flow, lc, lp)
    st = calib_api.status("ets2", "gamepad")
    assert st.calibrated and st.score > 0.5 and st.date == "2026-10-30"
    assert st.label.startswith("calibrated, score")

    (data_dir / "livecalib_state_ets2.json").write_text("{}")
    (data_dir / "liveparams_ets2_gamepad.json").write_text("{}")
    calib_api.restart("ets2", "gamepad")
    assert not flow_path("ets2", "gamepad").exists()
    assert not (data_dir / "livecalib_state_ets2.json").exists()
    assert not (data_dir / "liveparams_ets2_gamepad.json").exists()
    assert not calib_api.status("ets2", "gamepad").calibrated


def test_livecalib_rescale_matches_the_new_fov():
    from simsteer.core.learners.livecalib import LiveCalib
    lc = LiveCalib()
    cal = Calibration(fov_h_deg=90.0)
    for _ in range(INPUTS_NEEDED):
        lc._rpys.append(np.radians([0.0, 2.42, 1.71]))
        lc._heights.append(1.35)
    lc._block_rpy_sum = np.radians([0.0, 2.42, 1.71]) * 10
    lc._block_n = 10
    lc._refresh_cal_status()
    lc.rescale_for_fov(cal, 90.0, 71.0)
    assert lc.pitch_estimate == pytest.approx(1.73, abs=0.01)
    assert lc.yaw_estimate == pytest.approx(1.22, abs=0.01)
    assert lc.height_estimate == pytest.approx(1.35)
    assert np.degrees(lc._block_rpy_sum[1] / 10) == pytest.approx(1.73, abs=0.01)
    assert lc.cal_status == CalStatus.CALIBRATED


# ----- clocks -----

def test_liveparams_timestamps_use_the_given_clock():
    now = [100.0]
    lp = LiveParams(game="ets2", device_kind="gamepad", clock=lambda: now[0])
    lp._intervention_learn(0.5, 0.05, 400.0)
    assert lp.last_intervened_ts == 100.0
