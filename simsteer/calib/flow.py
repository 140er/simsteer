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
from zero and writes it once. That warp aligns the forward horizon but
does not rotate off-axis rays or apply yaw, so composing residuals onto
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
from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum

from simsteer.app import audio
from simsteer.calib.fov_source import FovReader
from simsteer.calib.score import CheckPolicy, CheckStats, Verdict, certified_result, evaluate
from simsteer.calib.steering import OutputDirection, SteeringDirection
from simsteer.core.calibration import SIMPLE_WARP_VERSION
from simsteer.core.learners.livecalib import (
    BLOCK_SIZE, INPUTS_NEEDED, MIN_SPEED_FILTER_MPS, CalStatus,
)
from simsteer.core.learners.liveparams import TRUSTED_MIN_SAMPLES
from simsteer.core.model_identity import pinned_model_identity
from simsteer.paths import state_path
from simsteer.runtime.fov import (
    CANDIDATE_OFFSETS, CANDIDATE_SAMPLES, ESTIMATE_SAMPLES, FovSolver,
)

CALIB_VERSION = 5
# Perspective calibration continues to adapt. Compare to the certified pose,
# not the previous frame, so accumulated drift still revokes the certificate.
POSE_ANGLE_TOLERANCE_DEG = 0.5
POSE_HEIGHT_TOLERANCE_M = 0.05
FULL_CHECK_LOG_INTERVAL_S = 1.0


def contexts_match(saved: dict | None, current: dict) -> bool:
    if not saved:
        return False
    old_camera, camera = saved.get("camera", {}), current.get("camera", {})
    if old_camera.get("simple_warp", True) or camera.get("simple_warp", True):
        return saved == current
    old_camera, camera = old_camera.copy(), camera.copy()
    for name, tolerance in (("pitch_deg", POSE_ANGLE_TOLERANCE_DEG),
                            ("yaw_deg", POSE_ANGLE_TOLERANCE_DEG),
                            ("height_m", POSE_HEIGHT_TOLERANCE_M)):
        try:
            if not abs(old_camera.pop(name) - camera.pop(name)) <= tolerance + 1e-9:
                return False
        except (KeyError, TypeError):
            return False
    return {**saved, "camera": old_camera} == {**current, "camera": camera}


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
    # Unengaged replay frames can measure elapsed driving, but cannot
    # certify that SimSteer controlled the vehicle.
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
                 check_policy: CheckPolicy | None = None,
                 model_identity: dict | None = None,
                 certification_context: Callable[[], dict] = lambda: {},
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
        self.check_policy = check_policy or CheckPolicy.for_capabilities(steering_fit)
        self.model_identity = model_identity if model_identity is not None else pinned_model_identity()
        self._context_provider = certification_context
        self._make_fov_solver = make_fov_solver
        self._fov_config = fov_config
        self._config_pending = False
        self._persist = persist
        self._today = today
        self._restart_request: str | None = None
        self._lock = threading.Lock()
        self._reset_state()

        loaded = self._load()
        if (self.calib.simple_warp
                and self.calib.simple_warp_version != SIMPLE_WARP_VERSION):
            self._migrate_simple_warp()
        elif not loaded:
            self._enter(Step.FOV)

    def _migrate_simple_warp(self) -> None:
        """Discard pose and Check evidence measured through the old warp.

        Preserve FOV/crops and the device's steering fit. A completed FOV
        resumes at Camera; an unfinished FOV still has to finish first.
        The version is also saved per device so another device's old Done
        cannot silently reuse its previous Check.
        """
        next_step = Step.FOV if self.step is Step.FOV else Step.CAMERA
        self.lc.reset(reason="simple warp geometry changed")
        self.calib.pitch_deg = self.calib.yaw_deg = 0.0
        self.calib.simple_warp_version = SIMPLE_WARP_VERSION
        self.attempts = 0
        self.result = None
        self.legacy_result = None
        self._check_context = None
        self._check_model = None
        self._check_requested = False
        self.redo_note = ""
        self.check = CheckStats()
        if self._persist:
            self.calib.save(game=self.game)
        self._enter(next_step)

    def _reset_state(self) -> None:
        self._config_pending = False
        self.step = Step.FOV
        self.attempts = 0
        self.result: dict | None = None
        self.legacy_result: dict | None = None
        self.redo_note = ""
        self.fov_solver = self._make_fov_solver()
        self.fov_proposal: float | None = None
        self.fov_source = ""
        self._fov_reference = float(self.calib.fov_h_deg)
        self.direction = SteeringDirection(self.lp.lag_frames)
        self.output_direction = OutputDirection(self.lp.lag_frames,
                                                int(self.calib.lateral_sign))
        self.prompt_idx = 0
        self.prompt_s = 0.0
        self.regime_s = 0.0
        self.lane_changes = 0
        self._was_lane_change = False
        self.check = CheckStats()
        self._pred_seen = getattr(self.lp, "predictions", 0)
        self._check_context: dict | None = None
        self._check_model = None
        self._check_requested = True
        self._rate = _Rate()
        self._now = 0.0
        self._banner_until = 0.0
        self._last_save = 0.0
        self._last_full_check_log = float("-inf")
        self._last_check_log_signature = None

    def switch_device(self, device: str) -> None:
        """Keep the tuner's flow reference, but load this device's state."""
        if device == self.device:
            return
        self.save()
        self.device = device
        self._reset_state()
        if not self._load():
            self._enter(Step.FOV)
        # A device swap can also change the game's active binding. Recheck
        # output even when returning to a device with a saved calibration.
        self.output_direction = OutputDirection(self.lp.lag_frames,
                                                int(self.calib.lateral_sign))
        if self.steering_fit and self.step in (Step.CHECK, Step.DONE):
            self.result = None
            self._enter(Step.STEERING)
        self.save()

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
        return (self.step is Step.DONE and certified_result(self.result, self._check_context)
                and self.lc.cal_status == CalStatus.CALIBRATED
                and contexts_match(self._check_context, self._context()))

    def _context(self) -> dict:
        # Normalize tuples to JSON lists, so a saved certificate compares
        # identically after loading. Include the actual mutable config.
        return json.loads(json.dumps({
            "schema": CALIB_VERSION, "game": self.game, "device": self.device,
            "camera": asdict(self.calib), "policy": asdict(self.check_policy),
            "controls": self._context_provider(),
            "model": self.model_identity,
        }))

    def confirm_check(self) -> None:
        """An explicit engage request acknowledges an invalidated check."""
        if self.step is Step.CHECK:
            self._check_requested = True

    def _invalidate_check(self, reason: str) -> None:
        old_controls = (self._check_context or {}).get("controls", {})
        device_changed = old_controls.get("device_id") != self._context_provider().get("device_id")
        self.redo_note = "Check again: " + reason
        self.result = {**evaluate(CheckStats(), self.check_policy).to_dict(),
                       "reason": self.redo_note}
        self._check_requested = False
        if device_changed and self.steering_fit:
            self.output_direction = OutputDirection(self.lp.lag_frames,
                                                    int(self.calib.lateral_sign))
            self.redo_note = "Output device changed: stop to check steering direction"
            self.result["reason"] = self.redo_note
            self._enter(Step.STEERING)
        else:
            self._enter(Step.CHECK)

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
        return self.step is Step.STEERING and self.ai_steering_ready

    @property
    def ai_steering_ready(self) -> bool:
        return (self.step in (Step.STEERING, Step.CHECK, Step.DONE)
                and self.lc.cal_status == CalStatus.CALIBRATED
                and self._check_requested
                and (self.step is Step.STEERING
                     or contexts_match(self._check_context, self._context()))
                and (not self.steering_fit
                     or (self.direction.ready and self.output_direction.ready)))

    def device_axis(self, game_axis: float) -> float:
        """Map the desired game input to the device only after measurement."""
        if not self.ai_steering_ready:
            return 0.0
        return game_axis * self.output_direction.sign if self.steering_fit else game_axis

    def observe_output(self, **observation) -> bool:
        if not self.steering_fit or not self.direction.ready:
            return False
        if self.output_direction.update(**observation):
            print("calib: output direction contradicted; disengaging")
            self.output_direction = OutputDirection(self.lp.lag_frames,
                                                    int(self.calib.lateral_sign))
            self.result = {**evaluate(CheckStats(), self.check_policy).to_dict(),
                           "reason": "Check again: output direction changed"}
            self.redo_note = "Output direction changed; stop to check again"
            self._enter(Step.STEERING)
            return True
        return False

    def measure_output(self, **observation) -> float:
        was_ready = self.output_direction.ready
        command = self.output_direction.tick_stopped(**observation)
        if self.output_direction.ready and not was_ready:
            if self.redo_note.startswith("Output direction changed"):
                self.redo_note = ""
            print(f"calib: output direction verified "
                  f"(output={self.output_direction.sign:+d}, "
                  f"prior={self.output_direction.prior:+d}, "
                  f"r={self.output_direction.evidence.correlation:+.3f}, "
                  f"seconds={self.output_direction.elapsed:.2f})")
            self.save()
        return command

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
            if self.output_direction.active:
                return min(self.direction.progress, self.output_direction.progress)
            return min(self.direction.progress,
                       self.output_direction.evidence.progress,
                       getattr(self.lp, "trust_progress",
                               self._lp_samples() / float(TRUSTED_MIN_SAMPLES)), 1.0)
        if self.step is Step.CHECK:
            return min(1.0, self.check.engaged_seconds / CHECK_SECONDS)
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
            if not self.ai_steering_ready or rate <= 0:
                return None
            if self._lp_samples() >= TRUSTED_MIN_SAMPLES and not self.lp.trusted():
                return None
            return max(0.0, TRUSTED_MIN_SAMPLES - self._lp_samples()) / rate
        if self.step is Step.CHECK:
            return max(0.0, CHECK_SECONDS - self.check.engaged_seconds)
        return None

    def _lp_samples(self) -> int:
        return max(self.lp.samples, self.lp.session_samples)

    def engage_block(self) -> str:
        """Why engaging is not allowed yet, or "" when it is."""
        if self.step is Step.FAILED:
            return "calibration failed: R restarts"
        if not self._check_requested:
            return "Check again: press INSERT"
        if self.step is Step.FOV:
            if self.fov_proposal is not None:
                return "accept or keep the measured FOV first"
            return "calibrating FOV: drive manually"
        if self.step is Step.CAMERA:
            return "calibrating camera: drive manually"
        if self.steering_fit and not self.direction.ready:
            return "checking steering direction: drive both left and right bends manually"
        if self.steering_fit and not self.output_direction.ready:
            return "stop to check steering direction"
        return ""

    # ----- per frame -----

    def tick(self, o: Obs) -> None:
        self._now = o.now
        dt = min(max(o.dt, 0.0), MAX_DT_S)
        if (self.step in (Step.STEERING, Step.CHECK, Step.DONE)
                and self.lc.cal_status != CalStatus.CALIBRATED):
            self._invalidate_check("camera calibration changed")
            self._enter(Step.CAMERA)
            return
        if self.step in (Step.CHECK, Step.DONE) and not contexts_match(self._check_context, self._context()):
            self._invalidate_check("calibration changed")
            return
        if abs(self.calib.fov_h_deg - self._fov_reference) > 0.05:
            # Set by hand (tuner, keys): carry the camera history over, and
            # during the FOV step take it as the player's answer.
            self.lc.rescale_for_fov(self.calib, self._fov_reference,
                                    self.calib.fov_h_deg)
            self._fov_reference = float(self.calib.fov_h_deg)
            if self._persist:
                self.calib.save(game=self.game)
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
        if not self.output_direction.ready:
            return
        samples = self._lp_samples()
        self._rate.add(o.now, samples)
        if self.lp.trusted() or getattr(self.lp, "legacy_fit", False):
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
        if not self._check_requested:
            return
        self.check.frames += 1
        predictions = getattr(self.lp, "predictions", 0)
        new_prediction = predictions > self._pred_seen
        self._pred_seen = predictions
        if not (o.telemetry_ok and o.driving and o.v is not None
                and o.v >= CHECK_MIN_SPEED_MPS):
            return
        self.check.driving_n += 1
        self.check.seconds += dt
        if o.engaged:
            self.check.add_axis(o.axis, dt)
            self.check.add_speed(o.vx_model, o.v, o.yaw_rate)
        self.check.lane_change_n += int(o.engaged and o.lane_change)
        if o.engaged:
            # Only SimSteer's own lane keeping says anything about the fit.
            self.check.add_offset(o.lane_offset, lane_change=o.lane_change)
        if o.engaged and new_prediction:
            self.check.add_fit(self.lp.last_innovation, self.lp.last_actual,
                               held_out=self.lp.validating,
                               source=self.check_policy.fit_source)
        elif o.engaged:
            self.check.fit_total += 1
        if (self.check.engaged_seconds + 1e-6 >= CHECK_SECONDS
                or (self.check.seconds + 1e-6 >= CHECK_SECONDS and not self.check.sat_n)):
            self._finish_check(evaluate(self.check, self.check_policy))

    def _finish_check(self, verdict: Verdict) -> None:
        self.result = {**verdict.to_dict(), "date": self._today(),
                       "fov_source": self.fov_source, "context": self._check_context}
        if verdict.passed:
            print(f"calib: check passed, score {verdict.score:.2f}")
            self.attempts = 0
            self.redo_note = ""
            self._banner_until = self._now + RESULT_BANNER_S
            audio.play("ready")
            self._set_step(Step.DONE)
            return
        if verdict.status == "INSUFFICIENT_EVIDENCE":
            # Missing evidence is not a failed fit and cannot justify a reset.
            # Keep the diagnostic result; a new Check needs an explicit request.
            self._invalidate_check(verdict.reason.removeprefix("Check again: "))
            # Retain the counts from the completed attempt in the result.
            self.result = {**verdict.to_dict(), "date": self._today(),
                           "fov_source": self.fov_source, "context": self._check_context}
            self.save()
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
            self._reset_steering()
        self._enter(redo)

    # ----- transitions -----

    def _set_step(self, step: Step) -> None:
        if step is not Step.CHECK:
            self.lp.end_validation()
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
            self._check_requested = True
            self.prompt_s = self.regime_s = 0.0
            self.lane_changes = 0
            self._set_step(step)
            if self.lp.trusted() and self.ai_steering_ready:
                self._advance()
            return
        if step is Step.CHECK:
            self.check = CheckStats()
            self.check.fit_source = self.check_policy.fit_source
            self._check_context = self._context()
            self.lp.begin_validation()
            self._check_model = self.lp.validation_model
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
        self._reset_steering()
        self.attempts = 0
        self.result = None
        self._check_requested = True
        self.redo_note = ""
        self.fov_source = ""
        print(f"calib: restart ({reason})")
        self._enter(Step.FOV)

    def _reset_steering(self) -> None:
        self.lp.reset(wipe_disk=True, clear_axis_sign=True)
        self.direction = SteeringDirection(self.lp.lag_frames)
        self.output_direction = OutputDirection(self.lp.lag_frames,
                                                int(self.calib.lateral_sign))

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
            if not self.output_direction.ready:
                output = self.output_direction
                if v_ego is None:
                    return output.reason or "Steering check needs speed telemetry"
                if abs(v_ego) >= output.STOP_SPEED_MPS:
                    return "Stop to check steering direction"
                if output.active:
                    return "Checking steering direction: stay stopped"
                if self.game == "ets2" and output.reason == output.SATURATED_RAW_REASON:
                    return ("ETS2: steering mode likely relative; use Wheel (absolute) "
                            "steering and 0% deadzone")
                return output.reason or "Press INSERT to check steering direction"
            if self.redo_note and self.prompt_idx == 0 and self.prompt_s < 10.0:
                return f"Redo: {self.redo_note}"
            if not self.passive_fit and not engaged:
                return "Press INSERT and let SimSteer drive"
            if not self.lp.trusted() and hasattr(self.lp, "trust_need"):
                return self.lp.trust_need
            if self.prompt_idx < len(PROMPTS):
                return PROMPTS[self.prompt_idx].text
            return "Keep driving"
        if self.step is Step.CHECK:
            if not self._check_requested and self.result:
                if not self.result.get("coverage", {}).get("engaged"):
                    return self.redo_note or self.result.get("reason", "Check again")
                ms = self.result.get("measurements", {})
                counts = " ".join(
                    f"{label} {ms.get(name, {}).get('samples', 0)}/"
                    f"{ms.get(name, {}).get('denominator', 0)}"
                    for label, name in (("Lane", "lane_offset_spread"),
                                        ("Speed", "speed_ratio_error"),
                                        ("Fit", "steering_fit_error"),
                                        ("Axis", "steering_saturation")))
                return f"Check again: {counts}"
            if self.redo_note and self.check.seconds < 10.0:
                return self.redo_note
            if not engaged:
                return "Press INSERT for the check drive"
            if v_ego is not None and v_ego < CHECK_MIN_SPEED_MPS:
                return "Check counts above 54 km/h"
            return (f"Lane {self.check.off_n}/{self.check.off_total} "
                    f"Speed {self.check.ratio_n}/{self.check.ratio_total} "
                    f"Fit {self.check.fit_n}/{self.check.fit_total} "
                    f"Axis {self.check.sat_n}/{self.check.sat_total}")
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
        verdict = (evaluate(self.check, self.check_policy).to_dict()
                   if self.step is Step.CHECK and self._check_requested else
                   self.result or evaluate(self.check, self.check_policy).to_dict())
        measurements = verdict["measurements"]
        fields = {"calib": self.step.value, "calib_pct": round(self.progress * 100, 1),
                "calib_check": {
                    "status": verdict["status"], "score": verdict["score"],
                    "coverage": {k: verdict["coverage"].get(k, 0)
                                 for k in ("engaged", "engaged_seconds")},
                    "samples": {key: [measurements.get(name, {}).get("samples", 0),
                                      measurements.get(name, {}).get("denominator", 0)]
                                for key, name in (("lane", "lane_offset_spread"),
                                                  ("speed", "speed_ratio_error"),
                                                  ("fit", "steering_fit_error"),
                                                  ("axis", "steering_saturation"))}},
                "steering_trusted": self.lp.trusted(),
                "steering_coverage": {
                    "turns": getattr(self.lp, "coverage_turns", (0, 0)),
                    "speed_span": round(getattr(self.lp, "coverage_speed_span", 0.0), 3)},
                "steering_model_sign": self.direction.model_sign,
                "steering_input_sign": self.direction.axis_sign,
                "steering_output_sign": self.output_direction.sign}
        signature = (self.step, verdict["status"], verdict["reason"], self.attempts)
        if (signature != self._last_check_log_signature
                or self._now - self._last_full_check_log >= FULL_CHECK_LOG_INTERVAL_S):
            fields["calib_check_result"] = verdict
            self._last_full_check_log = self._now
            self._last_check_log_signature = signature
        return fields

    # ----- persistence -----

    def to_dict(self) -> dict:
        return {
            "version": CALIB_VERSION,
            "simple_warp_version": self.calib.simple_warp_version,
            "game": self.game,
            "device": self.device,
            "step": self.step.value,
            "attempts": self.attempts,
            "result": self.result,
            "legacy_result": self.legacy_result,
            "redo_note": self.redo_note,
            "fov": {"solver": self.fov_solver.to_dict(),
                    "proposal": self.fov_proposal,
                    "source": self.fov_source},
            "steering": {"prompt": self.prompt_idx,
                         "prompt_s": round(self.prompt_s, 3),
                         "regime_s": round(self.regime_s, 3),
                         "lane_changes": self.lane_changes,
                         "direction": self.direction.to_dict(),
                         "output_direction": self.output_direction.to_dict()},
            "check": self.check.to_dict(),
            "check_context": self._check_context,
            "check_model": self._check_model,
            "check_requested": self._check_requested,
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
        if (d.get("version") not in (2, 3, 4, CALIB_VERSION) or d.get("game") != self.game
                or d.get("device") != self.device):
            return False
        try:
            self.step = Step(d["step"])
            self.attempts = int(d.get("attempts", 0))
            self.result = d.get("result")
            self.legacy_result = d.get("legacy_result")
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
            self.output_direction.load_dict(st.get("output_direction", {}))
            self.check = CheckStats.from_dict(d.get("check"))
            self._check_context = d.get("check_context")
            self._check_model = d.get("check_model")
            self._check_requested = bool(d.get("check_requested", True))
        except (KeyError, ValueError, TypeError):
            return False
        if (self.calib.simple_warp
                and (self.calib.simple_warp_version != SIMPLE_WARP_VERSION
                     or d.get("simple_warp_version", 1) != SIMPLE_WARP_VERSION)):
            self._migrate_simple_warp()
            return True
        if self.step is Step.FOV and self.fov_proposal is None:
            self._config_pending = self._fov_config is not None
        if self.step in (Step.CHECK, Step.DONE):
            if (d.get("version") != CALIB_VERSION or not contexts_match(self._check_context, self._context())
                    or self._check_model is None
                    or (self.step is Step.DONE
                        and not certified_result(self.result, self._check_context))):
                if self.result and self.legacy_result is None:
                    self.legacy_result = self.result
                self._invalidate_check("saved evidence is outdated")
            else:
                if self.step is Step.CHECK:
                    self.lp.begin_validation(self._check_model)
        if self.step is Step.STEERING and not self.steering_fit:
            self._enter(Step.CHECK)
        elif self.steering_fit:
            if self.direction.ready:
                self.lp.set_axis_sign(self.direction.axis_sign)
            if (self.step in (Step.CHECK, Step.DONE) and not self.lp.trusted()
                    and not getattr(self.lp, "legacy_fit", False)):
                self._invalidate_check("steering coverage missing")
                self._enter(Step.STEERING)
            if (not (self.direction.ready and self.output_direction.ready)
                    and self.step in (Step.CHECK, Step.DONE)):
                self._enter(Step.STEERING)
        return True
