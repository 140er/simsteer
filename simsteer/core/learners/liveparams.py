"""Per-game/device steering mapping: axis = a*wheel + b*wheel*v² + c.

The guided Steering step measures a through the origin with b held at its
profile seed and c kept separate. Driving only observes the frozen mapping;
it never trains on the controller's own commands or on interventions.
Legacy fits load frozen as-is and still require the existing direction and
Check evidence. Explicit tuner edits remain available.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simsteer.core.finite import NonFiniteError, require_finite

from simsteer.paths import load_with_fallback, state_path
from simsteer.state import (
    atomic_save_json, bounded_int, bounded_number, delete_state,
    read_json_object, snapshot_exists, state_warning,
)

CONFIG_PATH = state_path("liveparams")

# How many frames the truck's wheel takes to respond to an input change
# (ETS2 input filter + steering rack). ~5 frames at 20 Hz = 250 ms.
LAG_FRAMES_DEFAULT = 5

# Trouble detector: speed collapse implies crash / off-road; skip
# updates until speed recovers.
TROUBLE_SPEED_DROP_FRAC = 0.5
TROUBLE_MIN_PEAK_MPS = 3.0
TROUBLE_WINDOW_S = 5.0

# Interventions are excluded from calibration and pause the trim integrator.
USER_ASSIST_DELTA = 0.15

# Saturation gate: when the controller is commanding near the rail,
# the observation reflects mechanical/slip dynamics, not the linear
# inverse-rack mapping we're fitting. Skip those samples — they are
# the well-known under-steer feedback-loop trigger.
SATURATION_AXIS = 0.9

# Gate profiles, picked by a game profile's `liveparams.gates.profile`.
# "strict" is what worked on ETS2. "loose" is for games whose wheel angle
# is synthesized (Forza, AC), because:
#   - they synthesize wheel angle from yaw rate via bicycle model
#     (instead of reading it directly), so the (axis, wheel) pair has
#     less independent variation;
#   - Forza's signed-byte steer field quantizes to ~0.008 per step;
#   - the player keeps the stick swinging more in racy games — there's
#     no ETS2-style rack smoothing forcing input quiescence.

# ETS2 defaults — the values that worked. KEEP THESE; do not touch
# without per-game opt-in. `lag_frames` is also per-game because the
# rack delay differs: ETS2 smooths input ~250 ms (5 frames @ 20 Hz),
# Forza / AC respond ~50 ms (1 frame), and pairing a too-old axis
# sample with the current wheel reading turns most "steering bursts"
# into a string of rejected-because-axis_lagged-is-near-zero events.
_GATES_DEFAULT = {
    "min_speed_mps": 5.0,
    "min_axis": 0.005,
    "min_wheel": 0.0,         # ETS2 already filters via min_axis
    "max_axis_spread_for_fit": 0.10,
    "max_innovation_cold": 0.60,
    "c_abs_max": 0.10,
    "lag_frames": 5,
    # Retained covariance fields for saved-state and HUD compatibility.
    "P_init": (0.1, 1e-4, 1e-3),
    "Q_diag": (1e-7, 1e-10, 1e-6),
}

# Synthesized wheel angles need a wheel-excitation gate in addition to
# the lag and input-quiescence gates shared with direct telemetry.
_GATES_SYNTH_WHEEL = {
    "min_speed_mps": 3.0,
    "min_axis": 0.0,
    "min_wheel": 0.005,       # ~0.3° — reject straight-driving samples
    "max_axis_spread_for_fit": 0.30,
    "max_innovation_cold": 2.0,
    "c_abs_max": 0.02,
    "lag_frames": 1,
    "P_init": (10.0, 1e-3, 1e-2),
    "Q_diag": (1e-5, 1e-8, 1e-5),
}


GATE_PROFILES = {"strict": _GATES_DEFAULT, "loose": _GATES_SYNTH_WHEEL}


# Back-compat shims — old code reads these as module constants. They
# now reflect the DEFAULT (ETS2) profile. The per-instance values are
# read off `self._gates` inside `update()`.
MIN_SPEED_MPS = _GATES_DEFAULT["min_speed_mps"]
MIN_AXIS = _GATES_DEFAULT["min_axis"]
MAX_AXIS_SPREAD_FOR_FIT = _GATES_DEFAULT["max_axis_spread_for_fit"]

# Outlier gate: reject samples where the wheel angle is past plausible
# road driving. Spin-outs / wall-hits produce huge implied wheel
# angles that drag the fit if accepted.
#
# 0.3 rad ≈ 17°. Real on-road wheel angle (front wheel, not steering
# wheel) almost never exceeds this — most highway is <0.05 rad.
MAX_WHEEL_FOR_FIT = 0.3
MAX_INNOVATION_WARM = 0.30
MAX_INNOVATION_COLD = _GATES_DEFAULT["max_innovation_cold"]

# Bounds on the fitted parameters. The SIGN of `a` is whatever the
# game's axis-to-wheel relationship requires — some setups have a
# positive correlation (axis_right → wheel_right) and some negative
# (axis_right → wheel_left). LiveParams absorbs that sign so the
# controller and `lateral_sign` can remain on a fixed convention.
# Magnitude constraints only: |a| ∈ [A_ABS_MIN, A_ABS_MAX], etc.
#
# Keep the signed mapping and a non-degenerate runtime floor. The fitted
# speed term uses the same direction as a.
A_ABS_MAX = 50.0
A_ABS_MIN_RUNTIME = 0.5
B_ABS_MAX = 0.10
# `c` is a small TRIM bias. Per-game profile decides the clamp —
# ETS2's 0.10 worked, Forza/AC need 0.02 (otherwise c absorbs the
# rack-gain mismatch instead of pushing it into `a`). See
# `GATE_PROFILES`. Module-level shim points at the ETS2 value.
C_ABS_MAX = _GATES_DEFAULT["c_abs_max"]

# Initial guess. a=10 ≈ 1 rad of wheel needs 10 units of axis at low
# speed (so effective scale ≈ 0.1 wheel/axis). b=0 and c=0 are the
# fallbacks when the game profile has no seed for the
# device.
A_INIT = 10.0
B_INIT = 0.0
C_INIT = 0.0

# Profile seeds supply the device's fallback scale, speed term and bias.
DEFAULT_SEED: tuple[float, float, float] = (A_INIT, B_INIT, C_INIT)
Seeds = Mapping[str, tuple[float, float, float]]


def _seed_for(seeds: Seeds, device_kind: str | None) -> tuple[float, float, float]:
    return seeds.get(device_kind, DEFAULT_SEED) if device_kind else DEFAULT_SEED


# At a single speed, wheel and wheel*v² are collinear: arbitrarily many
# samples cannot identify a and b separately. New measurements hold b at
# its seed; speed coverage below is retained for validating older fits.
TRUSTED_MIN_SAMPLES = 100
TRUSTED_MIN_EACH_TURN = 10
# The highway calibration replay peaks below 0.005 rad. At 0.003 rad it
# supplies both turns with variation in speed without requiring tight bends.
TRUSTED_MIN_WHEEL = 0.003
TRUSTED_SPEED_BIN_MPS = 3.0
TRUSTED_MIN_SPEED_SPAN_MPS = 3.0
TRUSTED_MIN_SPEED_BIN_SAMPLES = 10
TRUST_COVERAGE_VERSION = 1

# Bound the collection window so rejected data can age out.
FIT_WINDOW_SAMPLES = 1200
FIT_MAX_RMS_AXIS = 0.035
FIT_MAX_RELATIVE_RMS = 0.25


@dataclass
class _Persisted:
    a: float = A_INIT
    b: float = B_INIT
    c: float = C_INIT
    samples: int = 0
    axis_sign: int | None = None
    coverage: dict[str, list[int]] = field(default_factory=dict)
    coverage_speed_ranges: dict[str, list[float]] = field(default_factory=dict)
    coverage_speed_min: float | None = None
    coverage_speed_max: float | None = None
    legacy_fit: bool = False
    scale_frozen: bool | None = False
    calibration_fit: bool = False


class LiveParams:
    def __init__(self, lag_frames: int | None = None,
                 game: str | None = None,
                 device_kind: str | None = None,
                 seeds: Seeds | None = None,
                 gates: str = "strict",
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.game = game
        self.seeds: Seeds = seeds or {}
        # Timestamps below, and the controller's trim freeze that reads
        # them, use this clock so replay decisions repeat exactly.
        self.clock = clock
        # Per-device persistence: the rack response (and especially
        # the AXIS->WHEEL sign) differs between the ViGEm gamepad and
        # the vJoy wheel in the same game — ETS2 treats them as
        # separate input categories. We persist them separately so
        # switching devices reloads the correct fit instead of using
        # a value that was right for the OTHER device.
        self.device_kind = device_kind
        # Gate profile from the game profile. `lag_frames` is part of
        # it: Forza/AC have a fast rack and need `lag_frames=1`, ETS2
        # needs the full 5 (= 250 ms smoothing).
        self._gates = dict(GATE_PROFILES[gates])
        if lag_frames is None:
            lag_frames = int(self._gates["lag_frames"])
        # Detect cold start (no persisted file) so the startup print
        # below can announce that a/b/c came from the seed table, not
        # a learned fit. Tuner / HUD read this flag to dim or annotate
        # the values.
        resolved = self._resolve_path(game=game, device_kind=device_kind)
        self._was_seeded = (resolved is None)
        loaded = self._load(game=game, device_kind=device_kind,
                            path=resolved, seeds=self.seeds)
        if self._was_seeded:
            print(f"liveparams [{device_kind or 'no device'}]: "
                  f"cold start — seeded for game={game!r}: "
                  f"a={loaded.a:.3f} b={loaded.b:.4f} c={loaded.c:+.4f}")
        # State vector x = [a, b, c]: axis = a·wheel + b·wheel·v² + c.
        self.x = np.array([loaded.a, loaded.b, loaded.c], dtype=np.float64)
        self.axis_sign = loaded.axis_sign
        self.coverage = loaded.coverage
        self.coverage_speed_ranges = loaded.coverage_speed_ranges
        self.coverage_speed_min = loaded.coverage_speed_min
        self.coverage_speed_max = loaded.coverage_speed_max
        self.scale_frozen = False
        self.calibration_fit = loaded.calibration_fit
        self._steering_fit_active = False
        self._fit_samples: deque[tuple[float, float, float]] = deque(maxlen=FIT_WINDOW_SAMPLES)
        self.fit_reason = "Need more steering samples"
        self.legacy_fit = loaded.legacy_fit
        self.validation_model: tuple[float, float, float] | None = None
        # The initial covariance stays as a finite-state invariant only;
        # nothing updates it now that driving no longer trains the fit.
        self.P = np.diag(self._gates["P_init"])
        self.samples = loaded.samples
        self.session_samples = 0
        self._settle_freeze(loaded.scale_frozen)
        self.lag_frames = lag_frames
        # Buffers — lag_frames+1 deep so `[0]` is the value lag_frames
        # samples in the past once full.
        self._axis_buf: deque[float] = deque(maxlen=lag_frames + 1)
        self._commanded_buf: deque[float] = deque(maxlen=lag_frames + 1)
        self._v_buf: deque[float] = deque(maxlen=lag_frames + 1)
        self.last_innovation = 0.0
        self.last_predicted = 0.0
        self.last_actual = 0.0
        self._innov_ema = 0.0
        self._actual_ema = 0.0
        # Rejection counters for HUD diagnostics.
        self.rej_wheel = 0
        self.rej_innovation = 0
        self.rej_trouble = 0
        self.rej_assist = 0
        self.rej_saturated = 0
        self.rej_transient = 0       # mid-ramp — quiescence gate fired
        # Previously-silent gates that just `return`. Counting them
        # makes diagnosing "why isn't liveparams updating?" tractable
        # — especially for Forza/AC where wheel is synthesized and a
        # subtle gate can swallow every sample.
        self.rej_slow = 0            # v_ego below MIN_SPEED_MPS
        self.rej_no_wheel = 0        # wheel_angle_rad is None this frame
        self.rej_small_axis = 0      # |axis_lagged| below MIN_AXIS
        self.rej_small_wheel = 0     # |wheel| below profile min_wheel (Forza/AC)
        self.frozen = False
        self._speed_history: deque[float] = deque(
            maxlen=int(TROUBLE_WINDOW_S * 20))
        # No longer driven (the RLS watchdog is gone); stays 0 so the
        # controller's trim gate reads a valid value.
        self._consecutive_bad_fit = 0
        # Trajectory ring buffer (time, a, b, c) — appended on every
        # installed measurement. Used by a future tuner plot tab; the
        # HUD itself reads `last_update_ts` and the trust_level
        # property for now.
        self.history: deque[tuple[float, float, float, float]] = deque(
            maxlen=600)
        self.last_update_ts: float = 0.0
        # The trim integrator pauses for one second after an intervention.
        self.last_intervened_ts: float = 0.0
        # Frames where the fit predicted the axis, accepted or not. The
        # calibration check reads `last_innovation` when this grows.
        self.predictions = 0

    # ----- raw parameters -----

    @property
    def a_linear(self) -> float:
        """Linear coefficient: axis per radian of wheel at zero speed."""
        return float(self.x[0])

    @a_linear.setter
    def a_linear(self, value: float) -> None:
        v = float(value)
        sign = -1.0 if v < 0 else 1.0
        self.x[0] = sign * min(A_ABS_MAX, abs(v))

    @property
    def b_quad(self) -> float:
        """Speed-quadratic coefficient: extra axis per (rad·v²)."""
        return float(self.x[1])

    @b_quad.setter
    def b_quad(self, value: float) -> None:
        v = float(value)
        sign = -1.0 if v < 0 else 1.0
        self.x[1] = sign * min(B_ABS_MAX, abs(v))

    @property
    def c_bias(self) -> float:
        """Axis offset bias."""
        return float(self.x[2])

    @c_bias.setter
    def c_bias(self, value: float) -> None:
        c_max = self._gates["c_abs_max"]
        self.x[2] = max(-c_max, min(c_max, float(value)))

    # ----- HUD-friendly derived views -----

    @property
    def scale(self) -> float:
        """Wheel-per-axis at zero speed = 1/a. HUD-compatible alias —
        small values (e.g. ≈0.1) mean "needs lots of axis per radian
        of wheel," which is what users intuited from the old fit."""
        a = self.x[0]
        return float(1.0 / a) if abs(a) > 1e-6 else 0.0

    @property
    def offset(self) -> float:
        """Equivalent additive wheel offset = -c/a. HUD-compatible."""
        a = self.x[0]
        return float(-self.x[2] / a) if abs(a) > 1e-6 else 0.0

    def effective_scale_at(self, v_ego: float) -> float:
        """Wheel-per-axis at a given speed: 1 / (a + b·v²).
        Drops with speed when b > 0 — the whole point of fitting b."""
        denom = self.x[0] + self.x[1] * v_ego * v_ego
        return float(1.0 / denom) if abs(denom) > 1e-6 else 0.0

    @property
    def innov_ema(self) -> float:
        """EMA of |innovation| over qualified observations. Same units as
        the regression target (axis). HUD reads this as a quick health
        gauge — should stabilize near zero on a healthy fit."""
        return float(self._innov_ema)

    @property
    def innov_ratio(self) -> float:
        if self._actual_ema < 1e-4:
            return 0.0
        return self._innov_ema / self._actual_ema

    @property
    def auto_frozen(self) -> bool:
        return self.scale_frozen

    @property
    def trust_level(self) -> str:
        """Coarse mapping status for the HUD."""
        if self.locked:
            return "LOCKED"
        if not self.trusted():
            return f"WARMING ({self.samples}/{TRUSTED_MIN_SAMPLES})"
        if self.innov_ratio > 0.30:
            return "UNSTABLE"
        if self.innov_ratio < 0.05:
            return "HEALTHY"
        return "SETTLED"

    @property
    def locked(self) -> bool:
        """Calibration freeze is independent of the tuner's session lock."""
        return self.scale_frozen or bool(self.frozen)

    def _in_trouble(self, v_ego: float) -> bool:
        # Disabled. The detector latched on pause/unpause cycles —
        # while paused, telemetry stalls so the speed history stays
        # full of pre-pause highs, and on resume the low current
        # speed looks like a 50% collapse. Other gates (quiescence,
        # saturation, user-assist, MIN_SPEED_MPS, MAX_WHEEL_FOR_FIT)
        # already filter genuine crash / off-road samples.
        return False

    def trusted(self) -> bool:
        return not self._steering_fit_active and self.trust_progress >= 1.0

    @property
    def coverage_turns(self) -> tuple[int, int]:
        return tuple(sum(counts[i] for counts in self.coverage.values()) for i in (0, 1))

    @property
    def coverage_speed_span(self) -> float:
        lows, highs = [], []
        for key, counts in self.coverage.items():
            if sum(counts) < TRUSTED_MIN_SPEED_BIN_SAMPLES:
                continue
            # Old bins have no per-bin extrema. Their inner edges give a
            # conservative bound; a single old global outlier proves nothing.
            low, high = self.coverage_speed_ranges.get(
                key, [(int(key) + 1) * TRUSTED_SPEED_BIN_MPS,
                      int(key) * TRUSTED_SPEED_BIN_MPS])
            lows.append(low)
            highs.append(high)
        return max(0.0, max(highs) - min(lows)) if lows else 0.0

    @property
    def trust_progress(self) -> float:
        if self.calibration_fit and self.scale_frozen:
            return 1.0
        if self._steering_fit_active:
            data = np.asarray(self._fit_samples)
            if not len(data):
                return 0.0
            wheel = data[:, 0]
            return min(0.99, len(wheel) / TRUSTED_MIN_SAMPLES,
                       np.count_nonzero(wheel < -TRUSTED_MIN_WHEEL) / TRUSTED_MIN_EACH_TURN,
                       np.count_nonzero(wheel > TRUSTED_MIN_WHEEL) / TRUSTED_MIN_EACH_TURN)
        left, right = self.coverage_turns
        return min(1.0, max(self.samples, self.session_samples) / TRUSTED_MIN_SAMPLES,
                   left / TRUSTED_MIN_EACH_TURN, right / TRUSTED_MIN_EACH_TURN,
                   self.coverage_speed_span / TRUSTED_MIN_SPEED_SPAN_MPS)

    @property
    def trust_need(self) -> str:
        if self._steering_fit_active:
            return self.fit_reason
        left, right = self.coverage_turns
        if max(left, right) < TRUSTED_MIN_EACH_TURN:
            return "Need left and right curves"
        if left < TRUSTED_MIN_EACH_TURN:
            return "Need left curves"
        if right < TRUSTED_MIN_EACH_TURN:
            return "Need right curves"
        if self.coverage_speed_span < TRUSTED_MIN_SPEED_SPAN_MPS:
            return "Need varied speed"
        return "Need more steering samples"

    @property
    def validating(self) -> bool:
        return self.validation_model is not None

    def begin_validation(self, coefficients=None) -> None:
        """Freeze a model before admitting any held-out check observations."""
        self.validation_model = tuple(float(v) for v in
                                      (self.x if coefficients is None else coefficients))
        # A lag window spanning training and validation is not held out.
        self._axis_buf.clear()
        self._commanded_buf.clear()
        self._v_buf.clear()

    def end_validation(self) -> None:
        self.validation_model = None

    def axis_for_wheel_angle(self, target_wheel_rad: float,
                             v_ego: float = 0.0) -> float:
        """Invert the inverse-rack model to get the axis that produces
        `target_wheel_rad` at speed `v_ego`:
            axis = a · w + b · w · v² + c
        Explicit tuner edits take effect immediately. Preserve the direction
        and floor |a| so old or manually edited states remain usable.
        """
        self.validate_state()
        require_finite("liveparams", target=target_wheel_rad, speed=v_ego)
        a, b, c = float(self.x[0]), float(self.x[1]), float(self.x[2])
        if 0.0 <= a < A_ABS_MIN_RUNTIME:
            a = A_ABS_MIN_RUNTIME
        elif -A_ABS_MIN_RUNTIME < a < 0.0:
            a = -A_ABS_MIN_RUNTIME
        v2 = v_ego * v_ego
        axis = a * target_wheel_rad + b * target_wheel_rad * v2 + c
        require_finite("liveparams", axis=axis)
        return float(axis)

    @property
    def a_below_floor(self) -> bool:
        """Whether the saved or manually edited gain needs the runtime floor."""
        return abs(float(self.x[0])) < A_ABS_MIN_RUNTIME

    def _intervention_learn(self, target_axis: float, wheel: float,
                            v2: float) -> None:
        """Record an intervention so the trim integrator can pause."""
        require_finite("liveparams.intervention", axis=target_axis, wheel=wheel, v2=v2)
        self.last_intervened_ts = self.clock()

    def _settle_freeze(self, recorded: bool | None) -> None:
        """Files written before the freeze record no flag. Keep such a fit
        frozen only if it already met the trust evidence or is a legacy
        fit the Check has accepted; otherwise the Steering step measures."""
        self.scale_frozen = (recorded if recorded is not None
                             else self.legacy_fit or self.trust_progress >= 1.0)

    def begin_steering_fit(self) -> None:
        """Open collection for the guided Steering step; retain frozen fits."""
        self._steering_fit_active = not self.scale_frozen
        self._axis_buf.clear()
        self._commanded_buf.clear()
        self._v_buf.clear()

    def end_steering_fit(self) -> None:
        self._steering_fit_active = False
        self._fit_samples.clear()

    def finish_steering_fit(self) -> bool:
        """Install a qualified least-squares measurement, then freeze it.

        At one speed a and b are collinear. Hold b at the profile seed's
        magnitude, signed like a, instead of estimating two gains from a
        short drive. c is an independently set bias, subtracted before the
        origin-only solve.
        Rejected windows keep collecting; no candidate changes x.
        """
        if self.scale_frozen:
            return True
        if not self._steering_fit_active or self.validating or self.frozen:
            return False
        self.validate_state()
        if len(self._fit_samples) < TRUSTED_MIN_SAMPLES:
            self.fit_reason = "Need more steering samples"
            return False
        data = np.asarray(self._fit_samples)
        wheel, speed, axis = data.T
        left, right = (np.count_nonzero(wheel < -TRUSTED_MIN_WHEEL),
                       np.count_nonzero(wheel > TRUSTED_MIN_WHEEL))
        if min(left, right) < TRUSTED_MIN_EACH_TURN:
            self.fit_reason = "Need left and right curves"
            return False
        energy = float(wheel @ wheel)
        if energy < len(wheel) * TRUSTED_MIN_WHEEL ** 2:
            self.fit_reason = "Need stronger steering excitation"
            return False
        seed_b = _seed_for(self.seeds, self.device_kind)[1]
        sign = self.axis_sign or (-1 if self.a_linear < 0 else 1)
        b = sign * abs(seed_b)
        target = axis - b * wheel * speed ** 2 - self.c_bias
        a = float(wheel @ target / energy)
        residual = float(np.sqrt(np.mean((target - a * wheel) ** 2)))
        signal = float(np.sqrt(np.mean(target ** 2)))
        if (not A_ABS_MIN_RUNTIME <= abs(a) <= A_ABS_MAX
                or a * sign <= 0):
            self.fit_reason = "Steering fit has implausible gain or direction"
            return False
        if residual > FIT_MAX_RMS_AXIS or residual > FIT_MAX_RELATIVE_RMS * signal:
            self.fit_reason = "Steering fit residual too large; drive steady gentle bends"
            return False
        self.x[:2] = a, b
        self.scale_frozen = self.calibration_fit = True
        self.last_update_ts = self.clock()
        self.history.append((self.last_update_ts, a, b, self.c_bias))
        self.end_steering_fit()
        self.save()
        return True

    def update(self, game_steer: float | None, v_ego: float | None,
               wheel_angle_rad: float | None,
               commanded_axis: float | None = None,
               gate_override: dict | None = None) -> None:
        """Observe a delayed axis/wheel pair without changing coefficients.

        Only an open guided Steering step collects qualified fit samples.

        `gate_override`, when provided, shadows `self._gates` for this
        single call — used by the guided-calibration routine to widen
        gates without mutating the per-game profile. Only the keys
        present override; missing keys fall through to `self._gates`.
        ETS2's strict module-level defaults stay untouched."""
        require_finite("liveparams.sample", axis=game_steer, speed=v_ego,
                       wheel=wheel_angle_rad, command=commanded_axis, gates=gate_override)
        self.validate_state()
        if game_steer is None:
            return
        # The Steering override widens only sample-admission gates.
        gates = (self._gates if gate_override is None
                 else {**self._gates, **gate_override})
        self._axis_buf.append(float(game_steer))
        self._commanded_buf.append(
            float(commanded_axis) if commanded_axis is not None else 0.0)
        self._v_buf.append(float(v_ego) if v_ego is not None else 0.0)
        if wheel_angle_rad is None or v_ego is None:
            self.rej_no_wheel += 1
            return
        if self._in_trouble(v_ego):
            self.rej_trouble += 1
            return
        if v_ego < gates["min_speed_mps"]:
            self.rej_slow += 1
            return
        if len(self._axis_buf) <= self.lag_frames:
            return

        axis_lagged = float(self._axis_buf[0])
        v_lagged = float(self._v_buf[0])
        if abs(axis_lagged) < gates["min_axis"]:
            self.rej_small_axis += 1
            return
        if abs(float(wheel_angle_rad)) > MAX_WHEEL_FOR_FIT:
            self.rej_wheel += 1
            return
        # Synthesized wheels need enough turning signal to exceed noise.
        if abs(float(wheel_angle_rad)) < gates["min_wheel"]:
            self.rej_small_wheel += 1
            return

        # Quiescence gate. Reject samples where the input has been
        # changing fast across the lag window — `axis_lagged` and the
        # current wheel observation aren't actually a matched pair in
        # that case, and the resulting innovation is a lag artifact.
        if len(self._axis_buf) >= 2:
            spread = max(self._axis_buf) - min(self._axis_buf)
            if spread > gates["max_axis_spread_for_fit"]:
                self.rej_transient += 1
                return

        # Reject intervention and saturation samples, including passive rails.
        wheel = float(wheel_angle_rad)
        v2 = v_lagged * v_lagged
        if abs(axis_lagged) >= SATURATION_AXIS:
            self.rej_saturated += 1
            return
        if commanded_axis is not None and len(self._commanded_buf) > self.lag_frames:
            commanded_lagged = float(self._commanded_buf[0])
            if abs(commanded_lagged) >= SATURATION_AXIS:
                self.rej_saturated += 1
                return
            if abs(axis_lagged - commanded_lagged) > USER_ASSIST_DELTA:
                self._intervention_learn(axis_lagged, wheel, v2)
                self.rej_assist += 1   # counts as "intervention" for HUD
                return

        # Predictions remain available to Check and diagnostics.
        H = np.array([wheel, wheel * v2, 1.0], dtype=np.float64)
        predicted = float(H @ (self.validation_model if self.validating else self.x))
        actual = axis_lagged
        innovation = actual - predicted
        require_finite("liveparams.prediction", predicted=predicted, innovation=innovation)
        self.predictions += 1
        if self.validating:
            # These frames are scored, including large errors, but never fit
            # or used in the Steering collection window.
            self.last_innovation = innovation
            self.last_predicted = predicted
            self.last_actual = actual
            return
        self.last_innovation = innovation
        self.last_predicted = predicted
        self.last_actual = actual
        self._innov_ema = 0.95 * self._innov_ema + 0.05 * abs(innovation)
        self._actual_ema = 0.95 * self._actual_ema + 0.05 * abs(actual)
        innov_gate = (MAX_INNOVATION_WARM if self.trusted()
                      else gates["max_innovation_cold"])
        if abs(innovation) > innov_gate:
            self.rej_innovation += 1
            return
        if not self._steering_fit_active or self.locked:
            return
        self._fit_samples.append((wheel, v_lagged, actual))
        self.samples += 1
        self.session_samples += 1
        if abs(wheel) >= TRUSTED_MIN_WHEEL:
            key = str(int(v_lagged // TRUSTED_SPEED_BIN_MPS))
            counts = self.coverage.setdefault(key, [0, 0])
            counts[int(wheel > 0)] += 1
            bounds = self.coverage_speed_ranges.setdefault(key, [v_lagged, v_lagged])
            bounds[0], bounds[1] = min(bounds[0], v_lagged), max(bounds[1], v_lagged)
            self.coverage_speed_min = (v_lagged if self.coverage_speed_min is None
                                       else min(self.coverage_speed_min, v_lagged))
            self.coverage_speed_max = (v_lagged if self.coverage_speed_max is None
                                       else max(self.coverage_speed_max, v_lagged))

    def reset(self, wipe_disk: bool = False, clear_axis_sign: bool = False) -> None:
        """Wipe the measured fit and collection state to the per-(game, device) seed.
        With `wipe_disk=True`, also remove the persisted JSON so the
        next session starts fresh — used by the overlay `R` hotkey
        and the `--reset-calib` CLI flag. Without `wipe_disk`, the
        on-disk fit is preserved until the next `save()` overwrites
        it (useful for tuner-triggered resets that want a known-good
        backup on disk).

        `clear_axis_sign` also discards direction evidence for a new
        calibration; ordinary fit resets retain the measured input sign.

        Uses the per-(game, device) seed for x — the same starting
        point a fresh cold start would use, not the universal
        A_INIT/B_INIT/C_INIT."""
        if clear_axis_sign:
            self.axis_sign = None
        a, b, c = _seed_for(self.seeds, self.device_kind)
        if self.axis_sign is not None:
            a = self.axis_sign * max(abs(a), A_ABS_MIN_RUNTIME)
            b = self.axis_sign * abs(b)
        self.x = np.array([a, b, c], dtype=np.float64)
        self.P = np.diag(self._gates["P_init"])
        self.samples = 0
        self.session_samples = 0
        self.coverage = {}
        self.coverage_speed_ranges = {}
        self.coverage_speed_min = self.coverage_speed_max = None
        self.legacy_fit = False
        self.scale_frozen = self.calibration_fit = False
        self.frozen = False
        self.end_steering_fit()
        self.fit_reason = "Need more steering samples"
        self.end_validation()
        self._axis_buf.clear()
        self._commanded_buf.clear()
        self._v_buf.clear()
        self._speed_history.clear()
        self._consecutive_bad_fit = 0
        self.rej_wheel = 0
        self.rej_innovation = 0
        self.rej_trouble = 0
        self.rej_assist = 0
        self.rej_saturated = 0
        self.rej_transient = 0
        self.rej_slow = 0
        self.rej_no_wheel = 0
        self.rej_small_axis = 0
        self.rej_small_wheel = 0
        self.history.clear()
        self.last_update_ts = 0.0
        if wipe_disk:
            key = self.game
            if self.device_kind and self.game:
                key = f"{self.game}_{self.device_kind}"
            try:
                p = state_path("liveparams", key)
                delete_state(p)
            except OSError:
                pass

    def reset_covariance_only(self) -> None:
        """Reset the diagnostic covariance without changing the mapping."""
        self.P = np.diag(self._gates["P_init"])
        self._consecutive_bad_fit = 0

    def set_axis_sign(self, sign: int) -> None:
        """Use the independently measured input direction as the seed.
        A trusted fit otherwise cannot promptly escape an inverted axis."""
        if sign not in (-1, 1):
            raise ValueError("axis sign must be -1 or 1")
        self.axis_sign = sign
        if not self.scale_frozen and self.a_linear * sign <= 0:
            active = self._steering_fit_active
            self.reset(wipe_disk=True)
            if active:
                self.begin_steering_fit()
        self.save()

    # ----- persistence -----

    @classmethod
    def _resolve_path(cls, game: str | None,
                      device_kind: str | None) -> Path | None:
        """Per-(game, device_kind) → per-game → legacy fallback chain.

        Preferred: `liveparams_<game>_<device>.json`.
        Fallback 1: `liveparams_<game>.json` (pre-device versions).
        Fallback 2: `liveparams.json` (legacy single-file).
        Returns the first existing path, or None if none exist."""
        if device_kind and game:
            key = f"{game}_{device_kind}"
            game_dev_path = state_path("liveparams", key)
            if snapshot_exists(game_dev_path):
                return game_dev_path
        # Per-game (pre-device-suffix) file.
        if game:
            game_path = state_path("liveparams", game)
            if snapshot_exists(game_path):
                return game_path
        # Legacy single-file.
        legacy = state_path("liveparams", None)
        if snapshot_exists(legacy):
            return legacy
        return None

    @classmethod
    def _load(cls, game: str | None = None,
              device_kind: str | None = None,
              path: Path | None = None,
              seeds: Seeds | None = None) -> _Persisted:
        # The "no good data" return value is the per-(game, device)
        # seed, not the universal A_INIT/B_INIT/C_INIT — cold start
        # converges much faster from a per-game ballpark than from a=10.
        def _seeded() -> _Persisted:
            a, b, c = _seed_for(seeds or {}, device_kind)
            return _Persisted(a=a, b=b, c=c, samples=0)

        if path is None:
            path = cls._resolve_path(game, device_kind)
        if path is None or not snapshot_exists(path):
            return _seeded()
        try:
            return cls._decode_state(read_json_object(path))
        except (OSError, ValueError, TypeError, OverflowError):
            state_warning(path, "unreadable; using defaults")
            return _seeded()

    @classmethod
    def _decode_state(cls, data: dict) -> _Persisted:
        for field in ("a", "b", "c", "scale", "offset"):
            if field in data:
                data[field] = bounded_number(data[field], -1e12, 1e12)
        if "samples" in data:
            data["samples"] = bounded_int(data["samples"])
        if "coverage_version" in data:
            bounded_int(data["coverage_version"])
        for flag in ("legacy_fit", "scale_frozen", "calibration_fit"):
            if flag in data and type(data[flag]) is not bool:
                raise ValueError("invalid fit flag")
        if data.get("axis_sign") is not None:
            if type(data["axis_sign"]) not in (int, float) or data["axis_sign"] not in (-1, 1):
                raise ValueError("invalid axis sign")
        # New format: {"a", "b", "c", "samples"}.
        if "a" in data:
            try:
                a = float(data.get("a", A_INIT))
                b = float(data.get("b", B_INIT))
                c = float(data.get("c", C_INIT))
                samples = int(data.get("samples", 0))
            except (ValueError, TypeError, OverflowError):
                raise ValueError("invalid saved fit")
        else:
            # Legacy format: {"scale", "offset", "samples"}. Migrate:
            #   wheel = scale·axis + offset
            # ⇒ axis = (1/scale)·wheel + (-offset/scale)
            # ⇒ a = 1/scale, b = 0, c = -offset/scale.
            #
            # The SIGN of `scale` is preserved — it captures the
            # axis-to-wheel correlation direction in the user's game,
            # which `lateral_sign` is independent of. (We previously
            # abs()'d this and broke users whose legacy fit was
            # negative.)
            #
            # The old sample count cannot certify a different model.
            # Preserve its coefficients frozen without inventing coverage.
            try:
                scale = float(data.get("scale", 0.1))
                offset = float(data.get("offset", 0.0))
            except (ValueError, TypeError):
                raise ValueError("invalid saved fit")
            if abs(scale) < 1e-6:
                raise ValueError("invalid saved fit")
            a = 1.0 / scale
            b = B_INIT
            c = -offset / scale
            samples = 0
        # Bounds check (magnitudes only — signs are user-data). Out-
        # of-range loads were probably written by an older code
        # version against a different model; reset to the per-game
        # seed and samples=0 so the user gets a fresh, well-seeded fit.
        # Use the LARGER (ETS2) c-bound here as the "obviously garbage"
        # threshold — accept files that came from any profile, runtime
        # profile's bias slider retains its tighter clamp.
        in_bounds = (np.isfinite([a, b, c, samples]).all()
                     and abs(a) <= A_ABS_MAX
                     and abs(b) <= B_ABS_MAX
                     and abs(c) <= _GATES_DEFAULT["c_abs_max"])
        if not in_bounds:
            raise ValueError("invalid saved fit")
        axis_sign = data.get("axis_sign")
        if axis_sign not in (-1, 1):
            axis_sign = None
        coverage = data.get("coverage", {})
        if not (isinstance(coverage, dict) and all(
                k.isdigit() and 0 <= int(k) <= 1000 // TRUSTED_SPEED_BIN_MPS
                and isinstance(v, list) and len(v) == 2
                and all(type(n) is int and 0 <= n <= (1 << 31) - 1 for n in v)
                for k, v in coverage.items())):
            raise ValueError("invalid coverage counts")
        speed_min, speed_max = data.get("coverage_speed_min"), data.get("coverage_speed_max")
        for speed in (speed_min, speed_max):
            if speed is not None:
                bounded_number(speed, 0, 1000)
        if (speed_min is None) != (speed_max is None) or (
                speed_min is not None and speed_min > speed_max):
            raise ValueError("invalid speed span")
        ranges = data.get("coverage_speed_ranges", {})
        if not isinstance(ranges, dict):
            raise ValueError("invalid speed ranges")
        for key, values in ranges.items():
            if key not in coverage or not isinstance(values, list) or len(values) != 2:
                raise ValueError("invalid speed range")
            low, high = (bounded_number(v, 0, 1000) for v in values)
            if not (int(key) * TRUSTED_SPEED_BIN_MPS <= low <= high
                    < (int(key) + 1) * TRUSTED_SPEED_BIN_MPS):
                raise ValueError("invalid speed range")
        legacy_fit = data.get("legacy_fit", data.get("coverage_version") is None
                              and samples >= TRUSTED_MIN_SAMPLES and not coverage) is True
        # None = written before the freeze existed; `_settle_freeze`
        # decides once coverage and samples are loaded.
        scale_frozen = data.get("scale_frozen")
        calibration_fit = data.get("calibration_fit", False)
        if calibration_fit and (not scale_frozen or samples < TRUSTED_MIN_SAMPLES):
            raise ValueError("invalid measured fit")
        return _Persisted(a=a, b=b, c=c, samples=samples, axis_sign=axis_sign,
                          coverage=coverage, coverage_speed_ranges=ranges,
                          coverage_speed_min=speed_min,
                          coverage_speed_max=speed_max, legacy_fit=legacy_fit,
                          scale_frozen=scale_frozen, calibration_fit=calibration_fit)

    def validate_state(self) -> None:
        require_finite("liveparams.state", coefficients=self.x, covariance=self.P,
                       axis=list(self._axis_buf), speed=list(self._v_buf),
                       commands=list(self._commanded_buf), samples=self.samples)

    def save(self, path: Path | None = None) -> None:
        if path is None:
            # Always write to the per-(game, device) path so the next
            # load picks up THIS device's fit, not the other one's.
            key = self.game
            if self.device_kind and self.game:
                key = f"{self.game}_{self.device_kind}"
            path = state_path("liveparams", key)
        try:
            self.validate_state()
            atomic_save_json(path, {
                "a": self.a_linear,
                "b": self.b_quad,
                "c": self.c_bias,
                "samples": self.samples,
                "axis_sign": self.axis_sign,
                "coverage": self.coverage,
                "coverage_speed_ranges": self.coverage_speed_ranges,
                "coverage_version": TRUST_COVERAGE_VERSION,
                "coverage_speed_min": self.coverage_speed_min,
                "coverage_speed_max": self.coverage_speed_max,
                "legacy_fit": self.legacy_fit,
                "scale_frozen": self.scale_frozen,
                "calibration_fit": self.calibration_fit,
            })
        except NonFiniteError as exc:
            logging.getLogger(__name__).warning("Not saving non-finite state: %s", exc)
        except OSError:
            pass

    def switch_device(self, new_device_kind: str | None) -> None:
        """Save the current fit (under the OLD device's path) and
        reload from the NEW device's path. Called by the tuner when
        the user toggles between gamepad and wheel — each device has
        a different rack response sign / magnitude, so we keep one
        fit per device-kind instead of trying to share one.

        State that's NOT persisted (session_samples, rejection
        counters, diagnostic covariance) gets reset, matching what would
        happen on a fresh launch with the new device."""
        if new_device_kind == self.device_kind:
            return
        # Persist current state to the OLD device's path.
        self.save()
        # Switch identity and load fresh.
        self.device_kind = new_device_kind
        # If the new device has no persisted file, _load returns the
        # seed — mark the instance as seeded again so HUD/print know.
        resolved = self._resolve_path(game=self.game,
                                      device_kind=new_device_kind)
        self._was_seeded = (resolved is None)
        loaded = self._load(game=self.game, device_kind=new_device_kind,
                            path=resolved, seeds=self.seeds)
        if self._was_seeded:
            print(f"liveparams [{new_device_kind}]: cold start — "
                  f"seeded for game={self.game!r}: "
                  f"a={loaded.a:.3f} b={loaded.b:.4f} c={loaded.c:+.4f}")
        self.x = np.array([loaded.a, loaded.b, loaded.c], dtype=np.float64)
        self.axis_sign = loaded.axis_sign
        self.coverage = loaded.coverage
        self.coverage_speed_ranges = loaded.coverage_speed_ranges
        self.coverage_speed_min = loaded.coverage_speed_min
        self.coverage_speed_max = loaded.coverage_speed_max
        self.scale_frozen = False
        self.calibration_fit = loaded.calibration_fit
        self.frozen = False
        self._steering_fit_active = False
        self._fit_samples: deque[tuple[float, float, float]] = deque(maxlen=FIT_WINDOW_SAMPLES)
        self.fit_reason = "Need more steering samples"
        self.legacy_fit = loaded.legacy_fit
        self.end_validation()
        self.P = np.diag(self._gates["P_init"])
        self.samples = loaded.samples
        self.session_samples = 0
        self._settle_freeze(loaded.scale_frozen)
        self._axis_buf.clear()
        self._commanded_buf.clear()
        self._v_buf.clear()
        self._speed_history.clear()
        self._consecutive_bad_fit = 0
        self._innov_ema = 0.0
        self._actual_ema = 0.0
        # Reset diagnostics so the HUD shows the new device's fresh
        # session, not the previous device's history.
        self.rej_wheel = 0
        self.rej_innovation = 0
        self.rej_trouble = 0
        self.rej_assist = 0
        self.rej_saturated = 0
        self.rej_transient = 0
        self.rej_slow = 0
        self.rej_no_wheel = 0
        self.rej_small_axis = 0
        self.rej_small_wheel = 0
        self.history.clear()
        self.last_update_ts = 0.0
