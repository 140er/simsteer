"""Guided calibration routine for LiveParams.

A 4-step state machine the user invokes from the tuner. While active:
  - LiveParams runs with widened gates + bumped Q (faster adaptation)
    via a per-call `gate_override` dict — the per-game profile in
    LiveParams is NEVER mutated, so the strict ETS2 defaults survive.
  - HUD shows the current step + prompt
  - Each step exits on its own predicate (sample count + stability)
    or its per-step timeout
  - Total cap: TOTAL_TIMEOUT_S; user can cancel anytime
  - On clean completion, saves liveparams. On timeout/partial, leaves
    the fit where it landed but does NOT auto-save — the user can
    re-run or save manually.

Step list and predicates are heuristic; the per-step timeout is the
real safety net so a user who can't reach a regime still moves on.

The routine doesn't drive the car — the human drives, the routine
just collects samples and runs RLS at louder gates while the right
inputs are happening."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from pilot.liveparams import LiveParams


# Hard total cap so a stuck routine doesn't run forever. 90 s is
# enough for a leisurely run through all four phases; a user who
# blows past it is probably stuck and should re-engage manually.
TOTAL_TIMEOUT_S = 90.0

# Override gates while active. Loosens the RLS gates that would
# otherwise reject most calibration-routine samples:
#   - max_axis_spread_for_fit: 0.30 — admits slalom samples that the
#     quiescence gate normally drops.
#   - max_innovation_cold: 1.0 — lets `a` migrate quickly from a
#     wrong seed; the strict 0.60 would reject every cold sample.
#   - Q_diag: 10× ETS2's baseline — faster filter adaptation.
#   - min_speed_mps: 3.0 — lets step 0 start sooner.
# Strict ETS2 profile in liveparams.py is untouched; this dict is
# passed per-call to `LiveParams.update(..., gate_override=...)`.
ROUTINE_GATE_OVERRIDE: dict = {
    "max_axis_spread_for_fit": 0.30,
    "max_innovation_cold": 1.0,
    "Q_diag": (1e-6, 1e-9, 1e-5),
    "min_speed_mps": 3.0,
}


@dataclass
class _StepResult:
    name: str
    duration_s: float = 0.0
    samples: int = 0
    a_final: float = 0.0
    b_final: float = 0.0
    c_final: float = 0.0
    innov_ema_final: float = 0.0
    passed: bool = False


@dataclass
class _Step:
    name: str
    prompt: str
    max_duration_s: float


# Step definitions. `max_duration_s` is the per-step timeout — the
# routine advances even if the exit predicate didn't fire, so the
# user moves through all four even if their driving doesn't perfectly
# match a step's regime.
_STEPS: list[_Step] = [
    _Step("straight", "Drive STRAIGHT at ~70 km/h", 15.0),
    _Step("slalom",   "Gentle SLALOM — ±half lane, smooth", 20.0),
    _Step("corner",   "Take one MODERATE CORNER (sweeper / ramp)", 25.0),
    _Step("highway",  "HIGHWAY at 80+ km/h, straight", 25.0),
]


class CalibrationRoutine:
    """State-machine wrapper around LiveParams driving regimes.

    Public surface:
        active                 — currently running
        current_step           — 0-based step index (== len(_STEPS) when done)
        prompt                 — single-line HUD string
        start()                — begin
        cancel()               — abort
        tick(v_ego, wheel, dt) — call every frame while active
        gate_override()        — dict to pass into LiveParams.update()
    """

    def __init__(self, live_params: LiveParams) -> None:
        self.live_params = live_params
        self.active = False
        self.current_step = -1
        self.routine_started_ts = 0.0
        self.step_started_ts = 0.0
        self.step_samples_start = 0
        # Stability buffers — sampled every tick, not just on accepted
        # RLS updates, so the predicates can converge even when the
        # filter is dropping the current sample on the floor.
        self._a_hist: deque[float] = deque(maxlen=60)
        self._b_hist: deque[float] = deque(maxlen=60)
        self._c_hist: deque[float] = deque(maxlen=60)
        self._corner_seen = False
        self._corner_dwell = 0
        self.results: list[_StepResult] = []
        # Last summary text — shown in the HUD after _finish completes
        # so the user gets a brief on-screen confirmation.
        self.summary_text = ""

    # ----- public API -----

    @property
    def prompt(self) -> str:
        """One-line HUD string. Empty when inactive."""
        if not self.active:
            return self.summary_text
        if self.current_step < 0 or self.current_step >= len(_STEPS):
            return self.summary_text or "Calibration: done"
        step = _STEPS[self.current_step]
        elapsed = time.monotonic() - self.step_started_ts
        return (f"CAL [{self.current_step + 1}/{len(_STEPS)}] "
                f"{step.prompt}  ({elapsed:4.1f}s / {step.max_duration_s:.0f}s)")

    def gate_override(self) -> dict | None:
        """The per-call override dict for LiveParams.update(). None
        when the routine is inactive — LiveParams then uses its stored
        per-game profile unchanged."""
        return ROUTINE_GATE_OVERRIDE if self.active else None

    def start(self) -> None:
        if self.active:
            return
        if self.live_params.locked:
            print("calibration routine: liveparams is LOCKED — unlock first")
            return
        self.active = True
        self.routine_started_ts = time.monotonic()
        self.results = []
        self.summary_text = ""
        print(f"calibration routine: starting (target {TOTAL_TIMEOUT_S:.0f}s total)")
        self._enter_step(0)

    def cancel(self) -> None:
        if not self.active:
            return
        idx = self.current_step
        self.active = False
        self.current_step = len(_STEPS)
        self.summary_text = "Calibration: CANCELLED"
        print(f"calibration routine: cancelled at step {idx + 1}/{len(_STEPS)}")

    def tick(self, v_ego: float | None, wheel_angle_rad: float | None,
             dt: float) -> None:
        """Called every frame while active. Tracks per-step state and
        advances when the exit predicate or timeout hits. `dt` is not
        currently used by predicates but kept on the signature for
        future rate-based gates."""
        _ = dt
        if not self.active or self.current_step < 0:
            return
        now = time.monotonic()
        # Hard total cap — wrap up with whatever we've got.
        if now - self.routine_started_ts > TOTAL_TIMEOUT_S:
            self._finish(reason="total timeout")
            return

        # Snapshot a/b/c every tick. Independent of whether THIS tick
        # produced an accepted RLS sample.
        self._a_hist.append(self.live_params.a_linear)
        self._b_hist.append(self.live_params.b_quad)
        self._c_hist.append(self.live_params.c_bias)

        step = _STEPS[self.current_step]
        step_elapsed = now - self.step_started_ts
        step_samples = self.live_params.samples - self.step_samples_start

        # Per-step exit predicates. Each combines a minimum elapsed
        # time (so we don't bail prematurely after just a few samples)
        # with a minimum-samples threshold and a stability check on
        # the parameter of interest.
        ready = False
        if self.current_step == 0:  # straight — c (bias) should settle
            ready = (step_elapsed > 6.0 and step_samples > 50
                     and self._abs_stable(self._c_hist, 0.005))
        elif self.current_step == 1:  # slalom — a (linear) should settle
            ready = (step_elapsed > 8.0 and step_samples > 60
                     and self._rel_stable(self._a_hist, 0.05))
        elif self.current_step == 2:  # corner — saw a real corner + dwelled
            if (wheel_angle_rad is not None
                    and abs(float(wheel_angle_rad)) > 0.05):
                self._corner_seen = True
            if self._corner_seen:
                self._corner_dwell += 1
            ready = (self._corner_seen and self._corner_dwell > 50)
        elif self.current_step == 3:  # highway — b (stiffness) should settle
            ready = (step_elapsed > 10.0 and step_samples > 80
                     and v_ego is not None and v_ego > 22.0
                     and self._rel_stable(self._b_hist, 0.10))

        timed_out = step_elapsed > step.max_duration_s
        if ready or timed_out:
            self._record_step_result(passed=ready and not timed_out)
            if timed_out and not ready:
                print(f"calibration routine: step {self.current_step + 1} "
                      f"timed out at {step.max_duration_s:.0f}s — moving on")
            self._enter_step(self.current_step + 1)

    # ----- internal -----

    def _enter_step(self, idx: int) -> None:
        if idx >= len(_STEPS):
            self._finish(reason="all steps complete")
            return
        self.current_step = idx
        self.step_started_ts = time.monotonic()
        self.step_samples_start = self.live_params.samples
        self._a_hist.clear()
        self._b_hist.clear()
        self._c_hist.clear()
        self._corner_seen = False
        self._corner_dwell = 0
        # Refresh covariance so this step's samples carry their own
        # weight instead of being diluted by the previous step's
        # accumulated certainty. Keeps the routine snappy as it
        # progresses — important since later steps land on the
        # already-tighter covariance from earlier ones otherwise.
        self.live_params.reset_covariance_only()
        print(f"calibration routine: step {idx + 1}/{len(_STEPS)} — "
              f"{_STEPS[idx].name}: {_STEPS[idx].prompt}")

    def _record_step_result(self, passed: bool) -> None:
        if self.current_step < 0 or self.current_step >= len(_STEPS):
            return
        step = _STEPS[self.current_step]
        self.results.append(_StepResult(
            name=step.name,
            duration_s=time.monotonic() - self.step_started_ts,
            samples=self.live_params.samples - self.step_samples_start,
            a_final=self.live_params.a_linear,
            b_final=self.live_params.b_quad,
            c_final=self.live_params.c_bias,
            innov_ema_final=self.live_params.innov_ema,
            passed=passed,
        ))

    def _finish(self, reason: str) -> None:
        all_passed = bool(self.results) and all(r.passed for r in self.results)
        lines = [f"calibration routine: complete ({reason})"]
        for r in self.results:
            tag = "PASS" if r.passed else "PART"
            lines.append(f"  [{tag}] {r.name:8s}: n={r.samples:3d} "
                         f"a={r.a_final:+.3f} b={r.b_final:+.4f} "
                         f"c={r.c_final:+.4f} innov_ema={r.innov_ema_final:.4f} "
                         f"({r.duration_s:.1f}s)")
        if all_passed:
            self.live_params.save()
            lines.append("  -> all steps passed, liveparams saved")
            self.summary_text = "Calibration: PASS (saved)"
        else:
            lines.append("  -> NOT auto-saved (some steps timed out) — "
                         "save manually if values look good")
            self.summary_text = "Calibration: PARTIAL (review then save)"
        print("\n".join(lines))
        self.active = False
        self.current_step = len(_STEPS)

    # ----- predicates -----

    @staticmethod
    def _abs_stable(hist: deque[float], abs_tol: float) -> bool:
        """True when the last 30 samples span < abs_tol."""
        if len(hist) < 30:
            return False
        recent = list(hist)[-30:]
        return (max(recent) - min(recent)) < abs_tol

    @staticmethod
    def _rel_stable(hist: deque[float], rel_tol: float) -> bool:
        """True when the last 30 samples span < rel_tol * |mean|.
        Returns False when |mean| is near zero (relative stability is
        undefined there)."""
        if len(hist) < 30:
            return False
        recent = list(hist)[-30:]
        mean = sum(recent) / len(recent)
        if abs(mean) < 1e-3:
            return False
        spread = (max(recent) - min(recent)) / abs(mean)
        return spread < rel_tol
