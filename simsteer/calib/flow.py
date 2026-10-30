"""One guided calibration per game and device.

Steps run in order: FOV, Camera, Steering, Check. Each step ends on its
own predicate, a step whose learner has already converged passes at
once, and the Check scores a two minute drive. A passing Check marks the
game calibrated; a failing one sends the flow back to the step it
blames, and after `MAX_CHECK_ATTEMPTS` failures the flow stops and waits
for R. State saves per game and device, so a restart resumes mid-step.

No step sets a parameter from a signal that depends on it. FOV comes
from the game's config, a one-shot measurement the player accepts, or
manual entry. Camera and Steering are LiveCalib and LiveParams, which
learn from independent observations. The Check only judges.

With the default simple warp, the Camera step measures the mount pose
from zero and writes it once. That warp applies pitch only partly and
yaw not at all, so a LiveCalib that kept composing its residual onto
the written pose would feed on itself (on a synthetic drive with a 2.0
degree pitch and 1.5 degree yaw both crept past 3.5 degrees in three
minutes). The perspective warp applies the whole rotation, so there
LiveCalib keeps learning after the step, as in openpilot.

Steering resets LiveParams whenever the flow restarts (R, Recalibrate)
or a Check blames Steering. A trusted fit adapts too slowly to follow a
different car, so a fresh fit is the only way the Check's verdict can
change.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import Enum

from simsteer.app import audio
from simsteer.calib.fov_source import FovReader
from simsteer.calib.score import CheckStats, Verdict, evaluate
from simsteer.calib.steering import SteeringDirection
from simsteer.core.learners.livecalib import (
    BLOCK_SIZE, INPUTS_NEEDED, MIN_SPEED_FILTER_MPS, CalStatus,
)
from simsteer.core.learners.liveparams import TRUSTED_MIN_SAMPLES
from simsteer.paths import state_path
from simsteer.runtime.fov import (
    CANDIDATE_OFFSETS, CANDIDATE_SAMPLES, ESTIMATE_SAMPLES, FovSolver,
)

CALIB_VERSION = 2


class Step(str, Enum):
    FOV = "fov"
    CAMERA = "camera"
    STEERING = "steering"
    CHECK = "check"
    DONE = "done"
    FAILED = "failed"


TITLES = {Step.FOV: "FOV", Step.CAMERA: "Camera", Step.STEERING: "Steering",
          Step.CHECK: "Check"}

CHECK_SECONDS = 120.0
CHECK_MIN_SPEED_MPS = 15.0
MAX_CHECK_ATTEMPTS = 2
FOV_AUTO_ACCEPT_DEG = 2.0
RESULT_BANNER_S = 10.0
SAVE_EVERY_S = 5.0
MAX_DT_S = 0.25

# LiveParams gates while Steering runs, from the old guided routine:
# admit slalom samples, let `a` leave a wrong seed, adapt faster. Passed
# per call, so the per-game profile stays untouched.
STEERING_GATE_OVERRIDE: dict = {
    "max_axis_spread_for_fit": 0.30,
    "max_innovation_cold": 1.0,
    "Q_diag": (1e-6, 1e-9, 1e-5),
    "min_speed_mps": 3.0,
}


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    timeout_s: float


PROMPTS = (
    Prompt("straight", "Drive a straight stretch", 30.0),
    Prompt("curves", "Take gentle curves", 60.0),
    Prompt("lane_changes", "Change lanes twice", 60.0),
    Prompt("slow_turns", "Take a slow turn", 60.0),
)


@dataclass
class Obs:
    """What the flow sees of one frame."""

    now: float
    dt: float
    v: float | None = None
    yaw_rate: float | None = None
    wheel: float | None = None
    vx_model: float | None = None
    lane_offset: float | None = None
    axis: float = 0.0
    observed_axis: float | None = None
    model_curvature: float | None = None
    # The check scores frames where SimSteer drives, or every frame when
    # there is no output device (replay, overlay only).
    driving: bool = False
    engaged: bool = False
    lane_change: bool = False
    telemetry_ok: bool = True


def steering_fit_for(profile) -> bool:
    """A game without telemetry has no wheel angle to fit against."""
    return profile.telemetry.kind != "none"


@dataclass(frozen=True)
class Panel:
    title: str
    hint: str
    progress: float
    tone: str  # "calibrating" | "passed" | "failed"


class _Rate:
    """Count growth per second over a sliding window of clock time."""

    def __init__(self, window_s: float = 20.0) -> None:
        self.window_s = window_s
        self._pts: deque[tuple[float, float]] = deque()

    def add(self, t: float, count: float) -> None:
        self._pts.append((t, count))
        while self._pts and t - self._pts[0][0] > self.window_s:
            self._pts.popleft()

    def per_s(self) -> float:
        if len(self._pts) < 2:
            return 0.0
        (t0, c0), (t1, c1) = self._pts[0], self._pts[-1]
        return (c1 - c0) / (t1 - t0) if t1 > t0 else 0.0

    def clear(self) -> None:
        self._pts.clear()


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"~{max(5, round(seconds / 5) * 5):.0f} s"
    return f"~{round(seconds / 60):.0f} min"


def flow_path(game: str, device: str):
    return state_path("calibflow", f"{game}_{device}")


class CalibFlow:
    def __init__(self, game: str, device: str, *, live_calib, live_params,
                 calib, steering_fit: bool = True, passive_fit: bool = False,
                 make_fov_solver: Callable[[], FovSolver] = FovSolver,
                 fov_config: FovReader | None = None,
                 persist: bool = True,
                 today: Callable[[], str] = lambda: date.today().isoformat(),
                 ) -> None:
        self.game = game
        self.device = device
        self.lc = live_calib
        self.lp = live_params
        self.calib = calib
        self.steering_fit = steering_fit
        self.passive_fit = passive_fit
        self._make_fov_solver = make_fov_solver
        self._fov_config = fov_config
        self._config_pending = False
        self._persist = persist
        self._today = today
        self._restart_request: str | None = None
        self._lock = threading.Lock()
        self._reset_state()

        if not self._load():
            self._enter(Step.FOV)

    def _reset_state(self) -> None:
        self._config_pending = False
        self.step = Step.FOV
        self.attempts = 0
        self.result: dict | None = None
        self.redo_note = ""
        self.fov_solver = self._make_fov_solver()
        self.fov_proposal: float | None = None
        self.fov_source = ""
        self._fov_reference = float(self.calib.fov_h_deg)
        self.direction = SteeringDirection(self.lp.lag_frames)
        self.prompt_idx = 0
        self.prompt_s = 0.0
        self.regime_s = 0.0
        self.lane_changes = 0
        self._was_lane_change = False
        self.check = CheckStats()
        self._pred_seen = getattr(self.lp, "predictions", 0)
        self._rate = _Rate()
        self._now = 0.0
        self._banner_until = 0.0
        self._last_save = 0.0

    def switch_device(self, device: str) -> None:
        """Keep the tuner's flow reference, but load this device's state."""
        if device == self.device:
            return
        self.save()
        self.device = device
        self._reset_state()
        if not self._load():
            self._enter(Step.FOV)

    # ----- state -----

    @property
    def steps(self) -> list[Step]:
        out = [Step.FOV, Step.CAMERA]
        if self.steering_fit:
            out.append(Step.STEERING)
        out.append(Step.CHECK)
        return out

    @property
    def calibrating(self) -> bool:
        return self.step in TITLES

    @property
    def calibrated(self) -> bool:
        return self.step is Step.DONE and bool(self.result and self.result.get("passed"))

    @property
    def camera_learning(self) -> bool:
        """Whether LiveCalib should take samples this frame. It learns
        during FOV too, and a new FOV rescales what it has."""
        if not getattr(self.calib, "simple_warp", True):
            return True
        if self.step in (Step.FOV, Step.CAMERA):
            return self.lc.cal_status != CalStatus.CALIBRATED
        return False

    def _zero_pose_for_camera(self) -> None:
        """With the simple warp, measure the pose from zero (see above)."""
        if (getattr(self.calib, "simple_warp", True)
                and self.lc.cal_status != CalStatus.CALIBRATED
                and self.lc.blocks == 0 and self.lc._block_n == 0):
            self.calib.pitch_deg = self.calib.yaw_deg = 0.0

    @property
    def gate_override(self) -> dict | None:
        return STEERING_GATE_OVERRIDE if self.step is Step.STEERING else None

    @property
    def probe_wanted(self) -> bool:
        return self.step is Step.STEERING and self.direction.ready

    @property
    def steering_sign(self) -> float:
        """Model curvature to game wheel; independent of HUD projection."""
        if self.steering_fit:
            return float(self.direction.model_sign or 1)
        # Without wheel telemetry, retain the profile's manual convention.
        return self.calib.lateral_sign

    @property
    def progress(self) -> float:
        if self.step is Step.FOV:
            return 1.0 if self.fov_proposal is not None else self.fov_solver.progress
        if self.step is Step.CAMERA:
            return min(1.0, (self.lc.blocks * BLOCK_SIZE + self.lc._block_n)
                       / float(INPUTS_NEEDED * BLOCK_SIZE))
        if self.step is Step.STEERING:
            return min(self.direction.progress,
                       self._lp_samples() / float(TRUSTED_MIN_SAMPLES), 1.0)
        if self.step is Step.CHECK:
            return min(1.0, self.check.seconds / CHECK_SECONDS)
        return 1.0

    def eta_s(self) -> float | None:
        rate = self._rate.per_s()
        if self.step is Step.FOV:
            if self.fov_proposal is not None or rate <= 0:
                return None
            need = ESTIMATE_SAMPLES + len(CANDIDATE_OFFSETS) * CANDIDATE_SAMPLES
            return max(0.0, need - self.fov_solver.samples) / rate
        if self.step is Step.CAMERA:
            return self.lc.eta_to_calibrated_s()
        if self.step is Step.STEERING:
            if not self.direction.ready or rate <= 0:
                return None
            return max(0.0, TRUSTED_MIN_SAMPLES - self._lp_samples()) / rate
        if self.step is Step.CHECK:
            return max(0.0, CHECK_SECONDS - self.check.seconds)
        return None

    def _lp_samples(self) -> int:
        return max(self.lp.samples, self.lp.session_samples)

    def engage_block(self) -> str:
        """Why engaging is not allowed yet, or "" when it is."""
        if self.step is Step.FOV:
            if self.fov_proposal is not None:
                return "accept or keep the measured FOV first"
            return "calibrating FOV: drive manually"
        if self.step is Step.CAMERA:
            return "calibrating camera: drive manually"
        if self.steering_fit and not self.direction.ready:
            return "checking steering direction: drive both left and right bends manually"
        return ""

    # ----- per frame -----

    def tick(self, o: Obs) -> None:
        self._now = o.now
        dt = min(max(o.dt, 0.0), MAX_DT_S)
        if abs(self.calib.fov_h_deg - self._fov_reference) > 0.05:
            # Set by hand (tuner, keys): carry the camera history over, and
            # during the FOV step take it as the player's answer.
            self.lc.rescale_for_fov(self.calib, self._fov_reference,
                                    self.calib.fov_h_deg)
            self._fov_reference = float(self.calib.fov_h_deg)
            if self.step is Step.FOV:
                self.fov_proposal = None
                self._finish_fov("manual")
                return
        if self.step is Step.FOV:
            self._rate.add(o.now, self.fov_solver.samples)
        elif self.step is Step.CAMERA:
            if self.lc.cal_status == CalStatus.CALIBRATED:
                print("calib: camera done")
                self._advance()
        elif self.step is Step.STEERING:
            self._tick_steering(o, dt)
        elif self.step is Step.CHECK:
            self._tick_check(o, dt)
        elif self.lc.cal_status == CalStatus.UNCALIBRATED:
            print("calib: camera calibration was reset, recalibrating camera")
            self.redo_note = "camera calibration was reset"
            self._enter(Step.CAMERA)
        if o.now - self._last_save >= SAVE_EVERY_S:
            self.save()

    def feed_fov(self, frame, vx_model: float | None, v_ego: float | None,
                 yaw_rate: float | None, perf_counter=None) -> None:
        if self.step is not Step.FOV or self.fov_proposal is not None:
            return
        if self._config_pending:
            # Read here, where the frame gives the aspect a vertical FOV needs.
            self._config_pending = False
            h, w = frame.shape[:2]
            fov = self._fov_config(w / h)
            if fov is not None:
                self._apply_fov(fov)
                self._finish_fov("config")
                return
        fov = self.fov_solver.update(frame, self.calib, vx_model, v_ego,
                                     yaw_rate, perf_counter)
        if fov is None:
            return
        current = float(self.calib.fov_h_deg)
        print(f"calib: fov measured {fov:.1f} (set {current:.1f})")
        if abs(fov - current) < FOV_AUTO_ACCEPT_DEG:
            self._apply_fov(fov)
            self._finish_fov("measured")
        else:
            self.fov_proposal = round(fov, 1)
            self.save()

    def accept_fov(self) -> bool:
        if self.step is not Step.FOV or self.fov_proposal is None:
            return False
        fov, self.fov_proposal = self.fov_proposal, None
        self._apply_fov(fov)
        self._finish_fov("measured")
        return True

    def keep_fov(self) -> bool:
        if self.step is not Step.FOV or self.fov_proposal is None:
            return False
        self.fov_proposal = None
        self._finish_fov("kept")
        return True

    def _apply_fov(self, fov: float) -> None:
        old = float(self.calib.fov_h_deg)
        new = round(float(fov), 1)
        if new != old:
            # Camera blocks so far were measured through the old warp.
            self.lc.rescale_for_fov(self.calib, old, new)
        self.calib.fov_h_deg = new
        self._fov_reference = self.calib.fov_h_deg
        if self._persist:
            self.calib.save(game=self.game)

    def _finish_fov(self, source: str) -> None:
        self.fov_source = source
        print(f"calib: fov {self.calib.fov_h_deg:.1f} ({source})")
        self._advance()

    def _tick_steering(self, o: Obs, dt: float) -> None:
        if not self.direction.ready:
            self.direction.update(
                curvature=o.model_curvature, axis=o.observed_axis,
                wheel=o.wheel, speed=o.v,
                manual=not o.engaged and o.telemetry_ok,
                lane_change=o.lane_change)
            if self.direction.ready:
                self.lp.set_axis_sign(self.direction.axis_sign)
                print(f"calib: steering direction verified "
                      f"(model={self.direction.model_sign:+d}, "
                      f"input={self.direction.axis_sign:+d}, "
                      f"r={self.direction.model.correlation:+.3f}/"
                      f"{self.direction.axis.correlation:+.3f})")
                self.save()
            else:
                return
        samples = self._lp_samples()
        self._rate.add(o.now, samples)
        if self.lp.trusted():
            print(f"calib: steering done (a={self.lp.a_linear:.3f} "
                  f"b={self.lp.b_quad:.5f} c={self.lp.c_bias:+.4f})")
            self._advance()
            return
        if self.prompt_idx >= len(PROMPTS):
            return
        prompt = PROMPTS[self.prompt_idx]
        self.prompt_s += dt
        v, w = o.v or 0.0, abs(o.wheel) if o.wheel is not None else None
        if w is not None:
            if prompt.name == "straight" and v >= CHECK_MIN_SPEED_MPS and w < 0.01:
                self.regime_s += dt
            elif prompt.name == "curves" and v >= CHECK_MIN_SPEED_MPS and 0.01 <= w < 0.05:
                self.regime_s += dt
            elif prompt.name == "slow_turns" and v < 12.0 and w >= 0.05:
                self.regime_s += dt
        if o.lane_change and not self._was_lane_change:
            self.lane_changes += 1
        self._was_lane_change = o.lane_change
        done = {"straight": self.regime_s >= 8.0,
                "curves": self.regime_s >= 10.0,
                "lane_changes": self.lane_changes >= 2,
                "slow_turns": self.regime_s >= 3.0}[prompt.name]
        if done or self.prompt_s >= prompt.timeout_s:
            self._next_prompt()

    def _next_prompt(self) -> None:
        self.prompt_idx += 1
        self.prompt_s = self.regime_s = 0.0
        self.lane_changes = 0
        if self.prompt_idx < len(PROMPTS):
            self.lp.reset_covariance_only()

    def _tick_check(self, o: Obs, dt: float) -> None:
        predictions = getattr(self.lp, "predictions", 0)
        new_prediction = predictions > self._pred_seen
        self._pred_seen = predictions
        if not (o.telemetry_ok and o.driving and o.v is not None
                and o.v >= CHECK_MIN_SPEED_MPS):
            return
        self.check.seconds += dt
        self.check.add_axis(o.axis)
        if o.engaged and not o.lane_change:
            # Only SimSteer's own lane keeping says anything about the fit.
            self.check.add_offset(o.lane_offset)
        self.check.add_speed(o.vx_model, o.v, o.yaw_rate)
        if new_prediction:
            self.check.add_fit(self.lp.last_innovation, self.lp.last_actual)
        if self.check.seconds >= CHECK_SECONDS:
            self._finish_check(evaluate(self.check))

    def _finish_check(self, verdict: Verdict) -> None:
        self.result = {**verdict.to_dict(), "date": self._today(),
                       "fov_source": self.fov_source}
        if verdict.passed:
            print(f"calib: check passed, score {verdict.score:.2f}")
            self.attempts = 0
            self.redo_note = ""
            self._banner_until = self._now + RESULT_BANNER_S
            audio.play("ready")
            self._set_step(Step.DONE)
            return
        self.attempts += 1
        print(f"calib: check failed, score {verdict.score:.2f}: "
              f"redo {verdict.redo}: {verdict.reason}")
        if self.attempts >= MAX_CHECK_ATTEMPTS:
            self._set_step(Step.FAILED)
            return
        self.redo_note = verdict.reason
        redo = Step(verdict.redo) if verdict.redo in Step._value2member_map_ else Step.CHECK
        if redo is Step.STEERING:
            self.lp.reset(wipe_disk=True)
            self.direction = SteeringDirection(self.lp.lag_frames)
        self._enter(redo)

    # ----- transitions -----

    def _set_step(self, step: Step) -> None:
        self.step = step
        self._rate.clear()
        self.save()

    def _advance(self) -> None:
        order = [Step.FOV, Step.CAMERA, Step.STEERING, Step.CHECK]
        nxt = order[order.index(self.step) + 1] if self.step in order[:-1] else Step.CHECK
        self._enter(nxt)

    def _enter(self, step: Step) -> None:
        if step is Step.FOV:
            self.fov_solver = self._make_fov_solver()
            self.fov_proposal = None
            self._fov_reference = float(self.calib.fov_h_deg)
            self._zero_pose_for_camera()
            self._config_pending = self._fov_config is not None
            self._set_step(step)
            return
        if step is Step.CAMERA:
            self._set_step(step)
            if self.lc.cal_status == CalStatus.CALIBRATED:
                self._advance()
            else:
                self._zero_pose_for_camera()
            return
        if step is Step.STEERING:
            if not self.steering_fit:
                self._enter(Step.CHECK)
                return
            self.prompt_idx = 0
            self.prompt_s = self.regime_s = 0.0
            self.lane_changes = 0
            self._set_step(step)
            if self.lp.trusted() and self.direction.ready:
                self._advance()
            return
        if step is Step.CHECK:
            self.check = CheckStats()
            self._pred_seen = getattr(self.lp, "predictions", 0)
        self._set_step(step)

    # ----- restart -----

    def request_restart(self, reason: str) -> None:
        """Safe from another thread; the loop applies it on its next frame."""
        with self._lock:
            self._restart_request = reason

    def take_restart_request(self) -> str | None:
        with self._lock:
            reason, self._restart_request = self._restart_request, None
        return reason

    def restart(self, reason: str) -> None:
        """Wipe the camera pose and steering fit and start over at FOV."""
        self.lc.reset(reason=reason)
        self.lp.reset(wipe_disk=True)
        self.direction = SteeringDirection(self.lp.lag_frames)
        self.attempts = 0
        self.result = None
        self.redo_note = ""
        self.fov_source = ""
        print(f"calib: restart ({reason})")
        self._enter(Step.FOV)

    # ----- HUD -----

    def hint(self, v_ego: float | None, engaged: bool = True) -> str:
        if self.step is Step.FOV:
            if self.fov_proposal is not None:
                return (f"FOV {self.fov_proposal:.0f} measured, set "
                        f"{self.calib.fov_h_deg:.0f}: INSERT accepts, "
                        f"BACKSPACE keeps")
            if v_ego is not None and v_ego < 8.0:
                return "Drive faster than 30 km/h on a straight road"
            return self.redo_note or "Drive straight, SimSteer measures the FOV"
        if self.step is Step.CAMERA:
            if self.lc.last_reset_reason and self.lc.samples == 0:
                return f"camera reset: {self.lc.last_reset_reason}"
            if v_ego is not None and v_ego < MIN_SPEED_FILTER_MPS:
                return "Drive faster than 25 km/h"
            if self.lc.samples > 50 and self.lc.rej_human > self.lc.samples * 0.3:
                return "Steer less, small inputs only"
            return self.redo_note or "Drive manually on a highway"
        if self.step is Step.STEERING:
            if not self.direction.ready:
                return "Drive gentle left and right bends manually"
            if self.redo_note and self.prompt_idx == 0 and self.prompt_s < 10.0:
                return f"Redo: {self.redo_note}"
            if not self.passive_fit and not engaged:
                return "Press INSERT and let SimSteer drive"
            if self.prompt_idx < len(PROMPTS):
                return PROMPTS[self.prompt_idx].text
            return "Keep driving"
        if self.step is Step.CHECK:
            if self.redo_note and self.check.seconds < 10.0:
                return f"Check again after: {self.redo_note}"
            if not engaged:
                return "Press INSERT for the check drive"
            if v_ego is not None and v_ego < CHECK_MIN_SPEED_MPS:
                return "Check counts above 54 km/h"
            return "Let SimSteer drive the highway"
        if self.step is Step.FAILED and self.result:
            return f"{self.result.get('reason', '')}. R restarts"
        return ""

    def panel(self, v_ego: float | None, engaged: bool = True) -> Panel | None:
        if self.calibrating:
            order = self.steps
            title = (f"CALIBRATE {order.index(self.step) + 1}/{len(order)} "
                     f"{TITLES[self.step]} {self.progress * 100:.0f}%")
            eta = self.eta_s()
            if eta is not None and self.progress < 1.0:
                title += f" {fmt_eta(eta)}"
            return Panel(title, self.hint(v_ego, engaged), self.progress,
                         "calibrating")
        if self.step is Step.DONE and self._now < self._banner_until and self.result:
            return Panel(f"CALIBRATED score {self.result['score']:.2f}", "", 1.0,
                         "passed")
        if self.step is Step.FAILED and self.result:
            redo = TITLES.get(Step(self.result["redo"]), "") if self.result.get(
                "redo") in Step._value2member_map_ else ""
            return Panel(f"NOT CALIBRATED {redo}".strip(), self.hint(v_ego, engaged),
                         0.0, "failed")
        return None

    def log_fields(self) -> dict:
        return {"calib": self.step.value, "calib_pct": round(self.progress * 100, 1),
                "steering_model_sign": self.direction.model_sign,
                "steering_input_sign": self.direction.axis_sign}

    # ----- persistence -----

    def to_dict(self) -> dict:
        return {
            "version": CALIB_VERSION,
            "game": self.game,
            "device": self.device,
            "step": self.step.value,
            "attempts": self.attempts,
            "result": self.result,
            "redo_note": self.redo_note,
            "fov": {"solver": self.fov_solver.to_dict(),
                    "proposal": self.fov_proposal,
                    "source": self.fov_source},
            "steering": {"prompt": self.prompt_idx,
                         "prompt_s": round(self.prompt_s, 3),
                         "regime_s": round(self.regime_s, 3),
                         "lane_changes": self.lane_changes,
                         "direction": self.direction.to_dict()},
            "check": self.check.to_dict(),
        }

    def save(self) -> None:
        self._last_save = self._now
        if not self._persist:
            return
        try:
            flow_path(self.game, self.device).write_text(
                json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load(self) -> bool:
        if not self._persist:
            return False
        p = flow_path(self.game, self.device)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if (d.get("version") != CALIB_VERSION or d.get("game") != self.game
                or d.get("device") != self.device):
            return False
        try:
            self.step = Step(d["step"])
            self.attempts = int(d.get("attempts", 0))
            self.result = d.get("result")
            self.redo_note = d.get("redo_note", "")
            fov = d.get("fov", {})
            self.fov_solver.load_dict(fov.get("solver", {}))
            self.fov_proposal = fov.get("proposal")
            self.fov_source = fov.get("source", "")
            st = d.get("steering", {})
            self.prompt_idx = int(st.get("prompt", 0))
            self.prompt_s = float(st.get("prompt_s", 0.0))
            self.regime_s = float(st.get("regime_s", 0.0))
            self.lane_changes = int(st.get("lane_changes", 0))
            self.direction.load_dict(st.get("direction", {}))
            self.check = CheckStats.from_dict(d.get("check"))
        except (KeyError, ValueError, TypeError):
            return False
        if self.step is Step.FOV and self.fov_proposal is None:
            self._config_pending = self._fov_config is not None
        if self.step is Step.STEERING and not self.steering_fit:
            self._enter(Step.CHECK)
        elif self.steering_fit:
            if self.direction.ready:
                self.lp.set_axis_sign(self.direction.axis_sign)
            elif self.step in (Step.CHECK, Step.DONE):
                self.result = None
                self._enter(Step.STEERING)
        return True
