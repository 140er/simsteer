"""Plan -> gamepad axes (steering + throttle/brake).

LATERAL — feed-forward + slow steady-state trim:

    desired_k    = desired_curvature_from_plan(v_ego, lookahead_s)
    target_wheel = wheel_sign * atan(desired_k * wheelbase * authority * tire_ff)
    axis_ff      = LiveParams.axis_for_wheel_angle(target_wheel, v_ego)
    axis_trim    = leaky_integrator(LPF(target_wheel - actual_wheel))
    axis         = clip(axis_ff + axis_trim + axis_bias, ±steer_max)

The model's plan trajectory is trained to go through lane center; the
controller just follows it. No explicit lane-keep — if the truck hugs
a wall, the fix is upstream (calibration / camera mount / model input
quality), not a band-aid in this controller. `steer_authority` is a
flat multiplier on the commanded curvature; openpilot doesn't have
this knob.

`LiveParams` learns the inverse rack mapping `axis = a*wheel +
b*wheel*v² + c` from telemetry (with a speed-stiffness term so the
same axis produces a different wheel angle at parking vs highway,
matching ETS2's variable-ratio rack). `axis_for_wheel_angle(target,
v_ego)` returns the FF axis directly. Calibration supplies `wheel_sign`
from model curvature versus measured wheel angle. LiveParams absorbs
the separate sign between input axis and actual wheel response. The
display's projection sign does not multiply either control output.

The `axis_trim` term is a leaky integrator on the LPF'd wheel-angle
error. It corrects steady-state offsets that the FF stack alone
can't see (LiveParams bias, residual rack-fit error, model plan
offset). An earlier PID was removed for fighting LiveParams during
transients — this trim is built to specifically *not* re-create
that failure mode: filtered, bounded bias correction + 60 s leak τ,
frozen during every transient regime
(corners, lane changes, saturation, intervention, uncalibrated steering, low
speed), and hard-clipped to ±wheel_trim_clip so it can only nudge.
The separate tire-slip feed-forward starts neutral and learns only
from sustained measured wheel/yaw curvature. It never changes the rack.

LONGITUDINAL — accel feed-forward + light P on speed error, mapped to
trigger pedals:

    v_target      = plan velocity at long_lookahead_s
    a_target      = plan accel at long_lookahead_s
    a_cmd         = a_target + speed_p_gain * (v_target - v_ego)
    throttle/brake = a_cmd mapped to [0,1] via max_accel/decel

Lookahead is longer than the steering one because braking input takes
seconds to bleed off speed — we need to react to upcoming curves before
we're in them. The model's plan already encodes "slow down for the bend"
in its velocity prediction; we just follow it.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from simsteer.core.finite import require_finite

from simsteer.core.constants import T_IDXS
from simsteer.core.learners.liveparams import LiveParams
from simsteer.paths import load_with_fallback, state_path
from simsteer.state import snapshot_exists, atomic_save_json, dataclass_values, read_json_object, state_warning
from simsteer.core.postprocess import Decoded, desired_curvature_lag_adjusted

CONFIG_PATH = state_path("controller")


@dataclass
class ControllerConfig:
    # Cap on the gamepad axis after the inversion. ETS2 saturates the
    # wheel well before axis=1 anyway; lower this if the truck still
    # over-steers (e.g. 0.5 to halve the response).
    steer_max: float = 1.0
    # Static steer-actuator delay (s). Equivalent to openpilot's per-car
    # STEER_ACTUATOR_DELAY — the controller reads the model's plan this
    # far ahead so the commanded curvature lines up with the wheel
    # response by the time it actually arrives. Default 0.3 s is in
    # the openpilot range (most cars 0.10-0.40 s). Tune from the
    # Lateral section; openpilot ships per-car constants, not online
    # estimation.
    lookahead_s: float = 0.3
    # Used when SCS telemetry is unavailable (no plugin). Doesn't drive
    # the truck, just keeps the curvature math sane.
    default_speed: float = 22.0
    # Below this speed the curvature math blows up (k = yaw / v). Command
    # zero instead of garbage.
    min_speed: float = 1.0
    # Truck wheelbase (m), used by the bicycle-model conversion. EU
    # tractors are ~3.5-4.2; default 4.0. A constant error here just
    # rolls into LiveParams.scale, so tuning it precisely doesn't help.
    wheelbase_m: float = 4.0
    # Constant axis bias added to the controller's output. Use to
    # cancel persistent lateral drift the AI can't account for itself
    # (model training-data bias, residual mounting offset, AC alignment
    # quirks, etc). Positive nudges right, negative nudges left.
    # Typical useful range: ±0.1.
    axis_bias: float = 0.0
    # Flat multiplier on the desired curvature (plan + lane-keep) before
    # the bicycle-model conversion to wheel angle. openpilot does NOT
    # have this knob — they take the model's planned curvature as-is
    # and close the loop with a PID/Torque on lateral-accel error. We
    # keep a flat multiplier as a project-specific knob. Default 1.0
    # (comma-faithful) works best here — higher just over-steers. Tune
    # it live in the tuner's Lateral section if a game ever needs more.
    steer_authority: float = 1.0

    # Anticipation buffer added to `lookahead_s` when computing
    # `desired_curvature_lag_adjusted`. Reads the plan's heading at
    # `lookahead_s + curvature_anticipation_s` into the future.
    # openpilot uses +0.2 s here ("for other delays"), but for the
    # sim's gamepad->game path that anticipates corners too early
    # (the truck turns in before the bend), so we default to 0.0 —
    # turn-in happens later. Lower still (negative) reacts even
    # closer to the present; the loop's k/l keys nudge this live.
    # Negative subtracts from lookahead — the function floors total
    # delay at 1 ms so negative values can't break the math.
    curvature_anticipation_s: float = 0.0
    # Bound the anticipation magnitude to the plan's peak curvature over
    # 1.5 times the effective lookahead. Applied before the jerk limit.
    curvature_anticipation_cap: bool = True
    # Apply the cap only to real curves (1/m), preserving straight-road
    # heading recovery when yaw-rate predictions are zero or near zero.
    curvature_anticipation_cap_min_curvature: float = 0.0015

    # Tire-slip compensation, separate from the measured steering rack.
    # Starts neutral; only sustained wheel/yaw telemetry can teach it.
    understeer_enabled: bool = True
    # Speed ramp (m/s). No boost below lo, full learned boost above hi.
    understeer_v_lo: float = 10.0
    understeer_v_hi: float = 22.0
    # Hard bounds on the learned gradient (s²/m²) and applied multiplier.
    understeer_gradient_max: float = 0.003
    understeer_cap: float = 1.8
    # Seconds of steady curvature before learning, and estimator time constant.
    understeer_steady_s: float = 1.0
    understeer_learn_s: float = 8.0
    # Ignore near-straights; reject variation above this fraction in the window.
    understeer_min_curvature: float = 0.0015
    understeer_steady_fraction: float = 0.15

    # Steering contribution to lateral jerk (v² * curvature rate).
    # None retains the legacy speed-interpolated curvature rate table.
    lat_jerk_max_mps3: float | None = 5.0

    # Speed-dependent steering response (variable-ratio rack in ETS2
    # and most games) is handled inside LiveParams now — it fits a
    # `b · wheel · v²` stiffness term, so `axis_for_wheel_angle(...,
    # v_ego)` already compensates. No knob here.

    # Closed-loop trim on wheel-angle error. Re-added after the
    # earlier PID was removed for fighting LiveParams during
    # transients — this version filters error, bounds output and freezes
    # during every transient regime (corners, lane changes,
    # saturation, intervention, uncalibrated steering, low speed). Output is
    # hard-clipped to ±wheel_trim_clip so even if it goes wrong it
    # can only nudge — not swing the axis the wrong way like the
    # old PID could.
    #
    # The pipeline is now:
    #     axis = clip(axis_ff + axis_trim + axis_bias, ±steer_max)
    # where axis_trim is the dynamic correction and axis_bias is the
    # static user knob — both serve their own role.
    wheel_trim_enabled: bool = True
    # Axis per (rad·s) of LPF'd error. Real residuals are milliradians:
    # 0.005 took minutes to cancel them. 0.5 responds in seconds, still
    # well below the actuator bandwidth; the curve gate prevents windup.
    wheel_trim_gain: float = 0.5
    # Learn bias on straights only, including gentle sweepers in the
    # freeze region. Curves use the independent tire-slip feed-forward.
    wheel_trim_max_curvature: float = 0.0015
    # Minimum speed for feedback learning (m/s); avoid parking dynamics.
    feedback_min_speed: float = 8.0
    # Hard clip on the integrator output. ±0.05 axis is enough to
    # cancel a typical steady-state offset without ever dominating
    # the FF axis.
    wheel_trim_clip: float = 0.05
    # Leak time constant — the integrator forgets stale trim over
    # this many seconds when the gate is open. 60 s means a stale
    # value decays by ~63% per minute. Prevents the trim from
    # latching onto a value that was valid in one stretch of road
    # but stops being so after a long pause.
    wheel_trim_leak_s: float = 60.0
    # LPF α on the wheel-angle error before it feeds the integrator.
    # α=0.05 at 20 Hz ≈ 20-frame averaging, equivalent τ ≈ 1 s. The
    # LPF (combined with the slow integrator gain) is what stops
    # transient swings from translating into integrator action.
    wheel_trim_error_lpf_alpha: float = 0.05

    # Confidence threshold on the model's desire_state for lane-change
    # indices (3 = lane-L, 4 = lane-R) above which we consider the
    # controller to be mid-lane-change for trim and calibration gating.
    lane_change_desire_threshold: float = 0.3
    # How long SimSteer treats a manual request as mid-lane-change for
    # steering, calibration and HUD state, unless the model finishes first.
    # The model receives one rising-edge pulse per request.
    lane_change_hold_s: float = 2.5

    # ----- LONGITUDINAL -----
    # Extra anticipation (seconds) on top of `lookahead_s` for the
    # longitudinal controller. Pedal commands take much longer to act
    # on speed than steering does on heading — we have to see the bend
    # coming before we're in it. Effective long-action-time =
    # lookahead_s + long_anticipation_s. lookahead_s is the static
    # steerActuatorDelay tunable from the tuner.
    long_anticipation_s: float = 1.0
    # m/s^2 of *commanded* acceleration that maps to a full-pressed
    # throttle (axis=1.0). Lower = more aggressive throttle for the
    # same target. Trucks typically accelerate at ~1-2 m/s^2; cars at
    # 3-5. Tune until a "cruise to target speed" feels natural.
    max_accel_mps2: float = 1.5
    # Same on the brake side. Lower = more aggressive brake. Comfortable
    # braking is 3-4 m/s^2; emergency is 6-9.
    max_decel_mps2: float = 4.0
    # P term on velocity error (m/s^2 commanded per m/s of error). Keeps
    # the loop tracking the plan's velocity when the FF accel alone
    # under/overshoots. Small by design — the FF carries most of the
    # work, this is a trim.
    speed_p_gain: float = 0.3
    # Residual telemetry/vision speed correction after camera calibration.
    # 1.0 disables; the hard ceiling also bounds persisted warm starts.
    speed_scale_max: float = 1.3
    # Symmetric deadband around 0 m/s^2 — avoids the actuator hunting
    # between throttle and brake when the desired accel is near zero
    # (cruise on flat road). Below this the controller commands 0/0.
    accel_deadband_mps2: float = 0.10
    # Hard cap on ego speed (m/s). The plan's velocity target is clamped
    # to this before computing accel, and any positive accel command is
    # zeroed once we're at/above the cap. Set high (e.g. 50) to
    # effectively disable. Default 25 m/s ≈ 90 km/h ≈ 56 mph.
    max_speed_mps: float = 25.0
    # Maximum lateral acceleration (m/s^2) allowed through upcoming
    # corners. We scan the plan's curvature over the next few seconds,
    # compute the minimum speed that keeps lateral g below this, and
    # clamp v_target to it. This is what makes the truck actually
    # brake hard for tight bends instead of waiting for the model's
    # (often conservative) planned velocity to drop. 2-3 m/s^2 feels
    # comfortable, 4-5 is firm, 6+ is aggressive. Set 0 to disable.
    # Default 4.5 (firm): 3.0 braked for gentle/noisy curvature
    # ("afraid of its own shadow"). Tune live in the tuner's Long section.
    max_lat_accel_mps2: float = 4.5
    # How far ahead in the plan to scan for the tightest upcoming
    # corner. Should be ≥ long_anticipation_s. Default 6 s covers
    # ~150 m at 25 m/s and ~200 m at 33 m/s — long enough to register
    # a hard bend at race speed in time to brake into it. The plan
    # itself spans 10 s (T_IDXS max), so anything up to 10 s is
    # readable; longer than that just clamps. Bumped from 3 s after
    # the hood-cam-on-AC case where the model registered curves but
    # the scanner truncated before reaching them.
    corner_scan_horizon_s: float = 6.0

    # ----- ACC / lead following -----
    # The model emits lead predictions at offsets 0, +2 and +4 s, with
    # (x, y, v, a) at 6 timesteps each and a probability per offset.
    # When the current lead's probability exceeds `lead_min_prob`, we
    # constrain v_target with a time-headway controller (this matches
    # openpilot's `longitudinal_planner.py` + `longitudinal_mpc_lib`
    # in concept, though we use a simpler closed-form law instead of
    # an MPC).
    #
    # The constraint:
    #     desired_dist = TR * v_ego + min_gap
    #     gap_err      = lead_x - desired_dist          (+ = too far)
    #     v_lead_set   = lead_v + lead_gap_p_gain·gap_err
    #     v_target     = min(v_target, v_lead_set)
    #
    # So at the desired headway we match the lead's speed; closer than
    # that we slow further (v_set drops below v_lead); farther we let
    # the upstream v_target stand. The gain is small on purpose — the
    # rest of the longitudinal loop (FF accel from a_target + speed P)
    # closes out the residual error.
    #
    # AEB-ish: if time-to-collision drops below `lead_ttc_brake_s`
    # we override a_cmd directly to a hard decel, regardless of the
    # plan. Below this TTC the closed-form follower can't keep up.
    lead_follow_enabled: bool = True
    # Confident leads only. 0.5 let phantom leads (shadows, signs,
    # overpasses) trigger ACC/AEB braking on an empty road; 0.7 ignores
    # those. Lower it in the tuner if it ever misses a real lead.
    lead_min_prob: float = 0.7
    lead_time_headway_s: float = 1.5      # TR — comfort default
    lead_min_gap_m: float = 5.0           # bumper-to-bumper minimum
    lead_gap_p_gain: float = 0.3          # closure-rate response
    lead_ttc_brake_s: float = 2.0         # below this TTC, override to max decel

    @classmethod
    def load(cls, game: str | None = None,
             path: Path | None = None,
             seed: Mapping[str, float] | None = None) -> "ControllerConfig":
        """Load this game's controller config. Falls back to the legacy
        single-file `controller.json` if a per-game file doesn't exist
        yet (smooth migration), then to `seed` (the game profile's
        `[controller]` table) over the defaults. Pass `path` to override
        entirely."""
        if path is None:
            path = load_with_fallback("controller", game)
        cfg = cls(**(seed or {}))
        if path is not None and snapshot_exists(path):
            try:
                cfg = cls(**{**asdict(cfg), **dataclass_values(cls, read_json_object(path))})
                if not (cfg.wheelbase_m > 0 and cfg.default_speed > 0
                        and cfg.max_speed_mps > 0 and 0 <= cfg.steer_max <= 1
                        and 0 <= cfg.understeer_v_lo < cfg.understeer_v_hi
                        and 1 <= cfg.understeer_cap <= 2
                        and 0 <= cfg.understeer_gradient_max <= .01
                        and cfg.understeer_steady_s >= 0 and cfg.understeer_learn_s > 0
                        and cfg.understeer_min_curvature > 0
                        and 0 < cfg.understeer_steady_fraction < 1
                        and cfg.feedback_min_speed > 0
                        and cfg.wheel_trim_max_curvature > 0
                        and cfg.curvature_anticipation_cap_min_curvature > 0
                        and 1.0 <= cfg.speed_scale_max <= 1.4):
                    raise ValueError("invalid controller limits")
            except (OSError, ValueError, TypeError, OverflowError):
                state_warning(path, "unreadable; using defaults")
                cfg = cls(**(seed or {}))
        # No actuator-delay floor for sim use: the game applies steering
        # almost instantly, so lookahead_s can sit near 0 ("snap to now"),
        # which is what the user wants here. Only guard against a negative
        # value, which would break the lag-curvature math.
        if cfg.lookahead_s < 0.0:
            cfg.lookahead_s = 0.0
        return cfg

    def save(self, game: str | None = None,
             path: Path | None = None) -> None:
        if path is None:
            path = state_path("controller", game)
        atomic_save_json(path, asdict(self))


class LateralController:
    def __init__(self, cfg: ControllerConfig | None = None,
                 live_params: LiveParams | None = None) -> None:
        self.cfg = cfg or ControllerConfig()
        self.live_params = live_params or LiveParams()
        self.last_curvature = 0.0
        self.last_target_wheel = 0.0
        self.last_axis = 0.0
        self.last_in_lane_change = False
        self.last_authority = 1.0
        # Previous-frame desired curvature (pre-authority, pre-scale).
        # Comma's rate-limit clamps THIS frame's request to a small
        # delta from this value — that's what stops a sudden plan jump
        # from translating to a sudden steer-in. Carried across frames.
        self.last_desired_k_raw = 0.0
        # Closed-loop trim state. `axis_trim_state` is the integrator
        # output (clipped to ±cfg.wheel_trim_clip); `lpf_wheel_error`
        # is the heavy LPF on the wheel-angle error that feeds it.
        # Both reset to zero on engage/disengage transitions.
        self.axis_trim_state = 0.0
        self.lpf_wheel_error = 0.0
        # Last-frame trim freeze reason — for HUD diagnostics. Empty
        # string when the integrator was running.
        self.last_trim_frozen_reason = ""
        self._reset_understeer()

    def _reset_understeer(self) -> None:
        self.last_understeer_gradient = 0.0
        self.last_understeer_ff = 1.0
        self.last_understeer_delivery = 1.0
        self.last_understeer_frozen_reason = "cold"
        self.last_curvature_error = 0.0
        self.last_lookahead_s = max(.001, self.cfg.lookahead_s + self.cfg.curvature_anticipation_s)
        self._feedback_time = 0.0
        self._curve_history: deque[tuple[float, ...]] = deque(maxlen=512)

    def reset(self) -> None:
        """Drop derived state. Call on disengage/re-engage transitions
        so nothing carries across handoffs."""
        self.last_curvature = 0.0
        self.last_target_wheel = 0.0
        self.last_axis = 0.0
        self.last_in_lane_change = False
        self.last_authority = 1.0
        self.last_desired_k_raw = 0.0
        self.axis_trim_state = 0.0
        self.lpf_wheel_error = 0.0
        self.last_trim_frozen_reason = ""
        self._reset_understeer()

    def _feedback_freeze_reason(self, speed: float, axis: float) -> str:
        """Shared transient gates for rack trim and tire-slip learning."""
        if speed < self.cfg.feedback_min_speed:
            return "slow"
        if self.last_in_lane_change:
            return "lc"
        if max(abs(axis), abs(self.last_axis)) >= 0.85 * self.cfg.steer_max:
            return "sat"
        if not self.live_params.trusted():
            return "cold"
        if self.live_params.clock() - self.live_params.last_intervened_ts < 1.0:
            return "user"
        return ""

    def _understeer_factors(self, speed: float) -> tuple[float, float]:
        cfg = self.cfg
        weight = max(0., min(1., (speed - cfg.understeer_v_lo)
                             / max(cfg.understeer_v_hi - cfg.understeer_v_lo, .001)))
        ff = max(1., min(cfg.understeer_cap,
                        1. + weight * self.last_understeer_gradient * speed * speed))
        return weight, ff

    def _understeer(self, curvature: float, speed: float, wheel: float | None,
                   yaw: float | None, dt: float, frozen: str) -> float:
        """Estimate tire slip from simultaneous measured wheel and yaw.

        The rack command is deliberately absent from the regression.
        Magnitudes accommodate telemetry yaw/wheel sign conventions;
        signed curvature and wheel stability still reject turn reversals.
        A wheel synthesized from yaw gives a neutral estimate, not a
        second learner fighting the same rack fit.
        """
        cfg = self.cfg
        self._feedback_time += dt
        weight, ff = self._understeer_factors(speed)
        if not cfg.understeer_enabled:
            frozen = "off"
        elif wheel is None or yaw is None:
            frozen = "no-telemetry"
        elif weight <= 0.:
            frozen = "slow"
        self.last_understeer_ff = 1.0
        self.last_curvature_error = 0.0
        if frozen:
            self._curve_history.clear()
            self.last_understeer_frozen_reason = frozen
            # Hold the estimate through maneuvers and saturation, just
            # like bias trim. Removing/reapplying boost at a saturation
            # threshold would itself create a steering oscillation.
            if frozen in ("lc", "sat", "user"):
                self.last_understeer_ff = ff
            return self.last_understeer_ff

        wheel_k = math.tan(wheel) / cfg.wheelbase_m
        measured_k = abs(yaw) / speed
        self.last_curvature_error = abs(curvature) - measured_k
        self._curve_history.append((self._feedback_time, curvature, wheel_k,
                                    measured_k, speed))
        cutoff = self._feedback_time - max(0., cfg.understeer_steady_s)
        while len(self._curve_history) > 2 and self._curve_history[1][0] <= cutoff:
            self._curve_history.popleft()
        history = np.asarray(self._curve_history)
        steady = (len(history) >= 2 and history[-1, 0] - history[0, 0]
                  >= cfg.understeer_steady_s)
        for column in (1, 2, 3):
            values = history[:, column]
            mean = abs(float(values.mean()))
            steady &= (mean >= cfg.understeer_min_curvature
                       and float(np.ptp(values)) <= cfg.understeer_steady_fraction * mean)
        steady &= float(np.ptp(history[:, 4])) <= cfg.understeer_steady_fraction * speed
        self.last_understeer_frozen_reason = "transient"
        if steady:
            # Solve wheel_k / measured_k = 1 + weight * gradient * v².
            # Divide out the partial speed gate so over-delivery at
            # intermediate speeds can reduce an earlier learned boost.
            ratio = abs(wheel_k) / measured_k
            # Broad telemetry plausibility bounds, independent of the
            # output cap: a low boost cap must still permit learning.
            if .4 <= ratio <= 2.5:
                target = max(0., min(cfg.understeer_gradient_max,
                                     (ratio - 1.) / (weight * speed * speed)))
                alpha = -math.expm1(-dt / max(cfg.understeer_learn_s, .001))
                self.last_understeer_gradient += alpha * (
                    target - self.last_understeer_gradient)
                self.last_understeer_delivery = 1. / ratio
                self.last_understeer_frozen_reason = ""
            else:
                self.last_understeer_frozen_reason = "outlier"
        _, self.last_understeer_ff = self._understeer_factors(speed)
        return self.last_understeer_ff

    def compute(self, decoded: Decoded, v_ego: float,
                actual_wheel_angle: float | None = None,
                lane_change_command_active: bool = False,
                dt: float = 0.05, wheel_sign: float = 1.0,
                actual_yaw_rate: float | None = None) -> float:
        """Plan -> gamepad axis. Wheel angle (rad) feeds straight bias
        trim; wheel plus yaw rate (rad/s) feeds tire-slip estimation.
        Missing telemetry disables the corresponding feedback for this
        frame. `dt` (s) drives curvature limiting and both learners —
        defaults to 0.05 s (20 Hz) for callers that don't measure it."""
        cfg = self.cfg
        require_finite("lateral", plan=decoded.plan, desire=decoded.desire_state,
                       speed=v_ego, wheel=actual_wheel_angle, yaw=actual_yaw_rate, dt=dt,
                       sign=wheel_sign, config=vars(cfg), previous=self.last_desired_k_raw,
                       trim=self.axis_trim_state, error=self.lpf_wheel_error,
                       gradient=self.last_understeer_gradient)
        dt = max(0., dt)
        if v_ego < cfg.min_speed:
            self.last_curvature = 0.0
            self.last_target_wheel = 0.0
            self.last_axis = 0.0
            self.last_in_lane_change = False
            self.last_authority = 0.0
            self._curve_history.clear()
            self.last_trim_frozen_reason = "slow"
            self.last_understeer_frozen_reason = "slow"
            self.last_understeer_ff = 1.0
            return 0.0

        # Plan-following via openpilot's `get_lag_adjusted_curvature`:
        # lag-corrected pure-pursuit + rate-limited so a step jump in
        # the model's plan can't translate into a step jump in steer
        # output. The rate-limit is what prevents "the AI steers in
        # too early" — without it, the moment the model first sees
        # an upcoming corner the commanded curvature snaps to its
        # target value; with it, the command ramps in over hundreds
        # of ms even if the plan jumps. The default jerk bound allows
        # faster turn-in at low speed and gentler motion at high speed.
        k_raw = desired_curvature_lag_adjusted(
            decoded.plan, v_ego,
            steer_actuator_delay=cfg.lookahead_s,
            last_desired_curvature=self.last_desired_k_raw,
            extra_buffer_s=cfg.curvature_anticipation_s,
            dt=dt, lat_jerk_max_mps3=cfg.lat_jerk_max_mps3,
            anticipation_cap=cfg.curvature_anticipation_cap,
            anticipation_cap_min_curvature=cfg.curvature_anticipation_cap_min_curvature,
        )
        self.last_lookahead_s = max(.001, cfg.lookahead_s + cfg.curvature_anticipation_s)
        self.last_desired_k_raw = k_raw
        k_total = k_raw

        # Detect lane-change state for trim and calibration gating. Reads
        # the model's own desire_state plus the user-command flag —
        # `lane_change_command_active` catches the initial frames
        # before the model's output reflects the manual request.
        lc_prob = float(decoded.desire_state[3]) + float(decoded.desire_state[4])
        in_lane_change = (lc_prob > cfg.lane_change_desire_threshold
                          or lane_change_command_active)
        self.last_in_lane_change = in_lane_change

        # Authority — flat multiplier on the commanded curvature.
        # openpilot has no such multiplier; we keep a flat one as a
        # project-specific knob. Lane changes use the model's own
        # plan after a rising-edge desire pulse; no separate gain.
        authority = cfg.steer_authority
        k_total *= authority

        self.last_curvature = k_total
        self.last_authority = authority

        # Convert the model target before the rack fit and feedback, so
        # bias and measured wheel angle stay in the game's convention.
        _, previous_ff = self._understeer_factors(v_ego)
        if not cfg.understeer_enabled or actual_wheel_angle is None or actual_yaw_rate is None:
            previous_ff = 1.0
        predicted_wheel = wheel_sign * math.atan(k_total * previous_ff * cfg.wheelbase_m)
        nominal_axis = self.live_params.axis_for_wheel_angle(predicted_wheel, v_ego=v_ego)
        freeze_reason = self._feedback_freeze_reason(
            v_ego, nominal_axis + self.axis_trim_state + cfg.axis_bias)
        ff = self._understeer(k_total, v_ego, actual_wheel_angle,
                             actual_yaw_rate, dt, freeze_reason)
        target_wheel = wheel_sign * math.atan(k_total * ff * cfg.wheelbase_m)
        self.last_target_wheel = target_wheel

        # FF axis from LiveParams inversion. v_ego enters the speed-
        # stiffness term so the inverse demands more axis at speed.
        axis_ff = self.live_params.axis_for_wheel_angle(
            target_wheel, v_ego=v_ego)

        # Closed-loop steady-state trim. Corrects drift the FF stack
        # can't see (rack-fit residual, model plan offset). The LPF
        # runs unconditionally when telemetry is available so the
        # filter is fresh when the gate opens; the integrator only
        # runs while every transient regime is clear (see docstring
        # in the ControllerConfig fields above for the rationale).
        if cfg.wheel_trim_enabled and actual_wheel_angle is not None:
            wheel_error = target_wheel - float(actual_wheel_angle)
            alpha = 1. - (1. - cfg.wheel_trim_error_lpf_alpha) ** (dt / .05)
            self.lpf_wheel_error += alpha * (wheel_error - self.lpf_wheel_error)

            # Gate freeze reasons (first match wins; "" = running).
            # Each maps to a transient regime where wheel-error is
            # dominated by lag, not by the steady-state offset the
            # trim is meant to correct.
            freeze_reason = self._feedback_freeze_reason(
                v_ego, axis_ff + self.axis_trim_state + cfg.axis_bias)
            if not freeze_reason and max(abs(k_total), abs(
                    math.tan(actual_wheel_angle) / cfg.wheelbase_m)
                    ) > cfg.wheel_trim_max_curvature:
                freeze_reason = "corner"

            if not freeze_reason:
                axis_sign = (1.0 if self.live_params.effective_scale_at(v_ego) >= 0
                             else -1.0)
                self.axis_trim_state += (
                    dt * cfg.wheel_trim_gain * self.lpf_wheel_error * axis_sign)
                # Exponential leak — forgets stale trim over
                # wheel_trim_leak_s if the gate stays open.
                self.axis_trim_state *= math.exp(-dt / cfg.wheel_trim_leak_s)
                self.axis_trim_state = max(
                    -cfg.wheel_trim_clip,
                    min(cfg.wheel_trim_clip, self.axis_trim_state))
            self.last_trim_frozen_reason = freeze_reason
        elif not cfg.wheel_trim_enabled:
            # Decay trim to zero quickly when disabled so the tuner
            # toggle is responsive — doesn't leave a stuck trim
            # behind. Same form as the leak path.
            self.axis_trim_state *= math.exp(-dt / 1.0)
            self.last_trim_frozen_reason = "off"
            self.lpf_wheel_error = 0.0
        else:
            self.last_trim_frozen_reason = "no-wheel"

        # Sum FF + trim + manual bias, then clip. Trim and axis_bias
        # serve different roles — bias is a static user knob, trim is
        # the dynamic auto-knob. Both stack additively on top of FF.
        axis = axis_ff + self.axis_trim_state + cfg.axis_bias
        require_finite("lateral", command=axis)
        axis = max(-cfg.steer_max, min(cfg.steer_max, axis))
        self.last_axis = axis
        return axis


class LongitudinalController:
    """Plan -> (throttle, brake) trigger axes.

    Feed-forward on the model's planned acceleration plus a small P term
    on the velocity gap at `long_lookahead_s` into the future. Commands
    in m/s^2 map linearly to throttle / brake via `max_accel_mps2` and
    `max_decel_mps2`. Negative commanded accel becomes brake; positive
    becomes throttle. A symmetric deadband prevents pedal hunting near
    zero.

    The plan tensor's columns 3 and 6 are vel_x and accel_x respectively
    (see `PlanField` in pilot/constants.py). At index 0 they're the
    current-frame estimate; later indices are the model's predicted
    trajectory.
    """

    _T_IDXS = np.asarray(T_IDXS, dtype=np.float32)

    def __init__(self, cfg: ControllerConfig | None = None) -> None:
        self.cfg = cfg or ControllerConfig()
        self.speed_scale = 1.0
        self.last_v_target = 0.0
        self.last_a_target = 0.0
        self.last_a_cmd = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0
        # Corner-anticipation diagnostics for the HUD.
        self.last_v_safe_corner = float("inf")  # inf = no corner near
        self.last_corner_t = 0.0
        self.last_corner_k = 0.0
        # Lead-following diagnostics for the HUD.
        self.last_lead_prob = 0.0
        self.last_lead_x = float("inf")        # inf = no lead engaged
        self.last_lead_v = 0.0
        self.last_lead_v_target = float("inf")  # v_target imposed by lead
        self.last_lead_ttc = float("inf")
        self.last_aeb = False

    def reset(self) -> None:
        self.last_v_target = 0.0
        self.last_a_target = 0.0
        self.last_a_cmd = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_v_safe_corner = float("inf")
        self.last_corner_t = 0.0
        self.last_corner_k = 0.0
        self.last_lead_prob = 0.0
        self.last_lead_x = float("inf")
        self.last_lead_v = 0.0
        self.last_lead_v_target = float("inf")
        self.last_lead_ttc = float("inf")
        self.last_aeb = False

    def compute(self, decoded: Decoded, v_ego: float) -> tuple[float, float]:
        cfg = self.cfg
        require_finite("longitudinal", plan=decoded.plan, speed=v_ego,
                       leads=decoded.leads, probability=decoded.lead_prob, config=vars(cfg),
                       scale=self.speed_scale)

        v_plan = decoded.plan[:, 3]   # vel_x over the 33 plan timesteps
        a_plan = decoded.plan[:, 6]   # accel_x
        yaw_rate_plan = decoded.plan[:, 14]  # yaw rate (rad/s) along plan

        # Effective lookahead = actuator delay (cfg.lookahead_s, the
        # static steerActuatorDelay) + the anticipation buffer for
        # seeing corners coming.
        t = max(0.0, float(cfg.lookahead_s) + float(cfg.long_anticipation_s))
        v_target = float(np.interp(t, self._T_IDXS, v_plan))
        a_target = float(np.interp(t, self._T_IDXS, a_plan))

        # Convert only longitudinal targets, before corner/lead limits.
        # Keep the raw decoded plan/pose available to camera and rack learning.
        scale = min(max(1.0, self.speed_scale), max(1.0, cfg.speed_scale_max), 1.4)
        v_target *= scale
        a_target *= scale

        # Corner anticipation: find the tightest upcoming planned
        # curvature within `corner_scan_horizon_s` and override v_target
        # if its lateral accel would exceed `max_lat_accel_mps2`. This
        # is what makes us brake hard for bends — without it we just
        # follow the model's planned velocity, which is often too
        # conservative on its braking to make a sharp game corner.
        v_safe_corner = float("inf")
        corner_t = 0.0
        corner_k = 0.0
        if cfg.max_lat_accel_mps2 > 0.0:
            scan_horizon = max(t, float(cfg.corner_scan_horizon_s))
            mask = (self._T_IDXS >= 0.0) & (self._T_IDXS <= scan_horizon)
            ts_scan = self._T_IDXS[mask]
            vs_scan = np.maximum(v_plan[mask], 0.5)         # avoid /0
            ks_scan = np.abs(yaw_rate_plan[mask]) / vs_scan  # |κ| = |ω|/v
            # Safe speed at each timestep that keeps |a_lat| = v^2·|κ|
            # ≤ max_lat_accel. sqrt(a_max/|κ|), clipped where |κ| is
            # near zero (straight road → no limit).
            ks_safe = np.maximum(ks_scan, 1e-6)
            v_safe_at_t = np.sqrt(cfg.max_lat_accel_mps2 / ks_safe)
            if v_safe_at_t.size > 0:
                idx_min = int(np.argmin(v_safe_at_t))
                v_safe_corner = float(v_safe_at_t[idx_min])
                corner_t = float(ts_scan[idx_min])
                corner_k = float(ks_scan[idx_min])
                if v_safe_corner < v_target:
                    v_target = v_safe_corner

        # ACC / lead following. Lead indices represent offsets 0, +2
        # and +4 s. Use index 0 (the current lead) to constrain v_target
        # via a time-headway rule. Below `lead_ttc_brake_s`
        # time-to-collision, AEB overrides a_cmd directly to max decel.
        lead_v_target = float("inf")
        lead_ttc = float("inf")
        aeb = False
        lead_prob = float(decoded.lead_prob[0]) if decoded.lead_prob.size > 0 else 0.0
        lead_x = float("inf")
        lead_v = 0.0
        if (cfg.lead_follow_enabled
                and decoded.leads.size > 0
                and lead_prob >= cfg.lead_min_prob):
            # leads[i, t, k] — k = 0:x, 1:y, 2:v, 3:a. Use the t=0
            # trajectory sample for the current lead at offset 0.
            lead_x = float(decoded.leads[0, 0, 0])
            lead_v = float(decoded.leads[0, 0, 2])
            # Closing rate (positive = closing). v_ego - lead_v.
            closure = v_ego - lead_v
            # Time-to-collision: only meaningful when closing.
            if closure > 0.1 and lead_x > 0.0:
                lead_ttc = lead_x / closure
            # Desired bumper-to-bumper distance.
            desired_dist = (cfg.lead_time_headway_s * max(v_ego, 0.0)
                            + cfg.lead_min_gap_m)
            gap_err = lead_x - desired_dist
            # Setpoint: at the desired gap, match lead speed. Inside
            # it, slow further; outside it, the controller doesn't
            # clamp at all (returns inf, never wins the min below).
            lead_v_target = lead_v + cfg.lead_gap_p_gain * gap_err
            # Never command negative speed via the lead law — the
            # follower can stop but not reverse.
            lead_v_target = max(0.0, lead_v_target)
            if lead_v_target < v_target:
                v_target = lead_v_target
            # AEB trigger.
            if lead_ttc < cfg.lead_ttc_brake_s and closure > 0.1:
                aeb = True
        self.last_lead_prob = lead_prob
        self.last_lead_x = lead_x
        self.last_lead_v = lead_v
        self.last_lead_v_target = lead_v_target
        self.last_lead_ttc = lead_ttc
        self.last_aeb = aeb

        # Cap the velocity target. If the model wants to go faster than
        # `max_speed_mps`, we don't follow — the P term then naturally
        # brakes (or eases off throttle) as v_ego approaches the cap.
        v_target = min(v_target, cfg.max_speed_mps)
        a_cmd = a_target + cfg.speed_p_gain * (v_target - v_ego)
        # And belt-and-braces: never command positive accel once we're
        # at/above the cap, even if the plan's a_target was high.
        if v_ego >= cfg.max_speed_mps and a_cmd > 0:
            a_cmd = 0.0
        # AEB override — full brake, ignore the plan and the P term.
        if aeb:
            a_cmd = -cfg.max_decel_mps2

        require_finite("longitudinal", acceleration=a_cmd)
        # Deadband around zero so we don't ping-pong between throttle
        # and brake when the model is happy with current speed.
        if abs(a_cmd) < cfg.accel_deadband_mps2:
            throttle, brake = 0.0, 0.0
        elif a_cmd > 0:
            throttle = min(1.0, a_cmd / max(cfg.max_accel_mps2, 1e-3))
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -a_cmd / max(cfg.max_decel_mps2, 1e-3))

        self.last_v_target = v_target
        self.last_a_target = a_target
        self.last_a_cmd = a_cmd
        self.last_throttle = throttle
        self.last_brake = brake
        self.last_v_safe_corner = v_safe_corner
        self.last_corner_t = corner_t
        self.last_corner_k = corner_k
        require_finite("longitudinal", throttle=throttle, brake=brake)
        return throttle, brake
