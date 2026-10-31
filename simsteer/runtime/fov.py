"""One-shot FOV measurement for the calibration flow's FOV step.

The warp crops the capture to the model's view using `calib.fov_h_deg`.
A wrong value zooms that view, and the model's forward speed `pose[0]`
drifts from telemetry speed:

    vx_model / v_ego ~= tan(FOV_true / 2) / tan(FOV_set / 2)

A value set too wide magnifies the view, the road looks nearer, and the
model reads slower. Earlier code divided by the ratio instead, so one
solve could push FOV further the wrong way (seen: 90 -> 40 degrees).

On the synthetic clip the relation holds within a few degrees of the
true value but flattens when the set value is far too wide (95 set on a
75 render solves to 70). So the solve has two stages, and the applied
FOV never changes while either runs:

  1. Estimate. The median ratio at the applied FOV, inverted with the
     formula above.
  2. Bracket. The vision model also runs on the same frames warped at
     four candidate FOVs around the estimate, one candidate per frame.
     Each candidate samples all four frame phases so repeating road
     texture cannot bias one candidate's optical flow.
     The answer is where the candidates' median ratios cross 1,
     interpolated between the two neighbours around the crossing. The
     ratio curve bends near the true FOV, so a straight line through all
     four lands degrees off. If all ratios sit on one side of 1, the
     candidates move that way, at most `MAX_SHIFTS` times.
     A crossing next to a response reversal cannot support
     interpolation, so it keeps the coarse estimate.

On the synthetic clips the model reads a few percent slow at the true
FOV, so the answer lands a few degrees narrow there. Whether real game
footage has that bias is for the recorded drive to show.

Each FOV step runs one solve, and the flow asks the player to accept the
result, so nothing here can run away like the removed `LiveFov`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from statistics import median

import numpy as np

from simsteer.core.constants import FRAME_SKIP
from simsteer.core.finite import require_finite
from simsteer.core.timing import (
    CONTEXT_PERIOD_S, FRAME_PERIOD_S, frame_gap, frame_geometry,
)

FOV_MIN_DEG, FOV_MAX_DEG = 40.0, 150.0
MIN_SPEED_MPS = 8.0
MAX_YAW_RAD_S = 0.06
ESTIMATE_SAMPLES = 40
CANDIDATE_PHASE_SAMPLES = 8
CANDIDATE_SAMPLES = CANDIDATE_PHASE_SAMPLES * FRAME_SKIP
# Spacing matches FRAME_SKIP, so each candidate's previous warp is the
# frame the model stacks with the current one.
CANDIDATE_OFFSETS = (-6.0, -2.0, 2.0, 6.0)
MAX_SHIFTS = 2

VisionFn = Callable[[np.ndarray, np.ndarray], tuple[float, float] | None]
ViewsFn = Callable[[np.ndarray, object], tuple[np.ndarray, np.ndarray]]

assert len(CANDIDATE_OFFSETS) == FRAME_SKIP


def solve_from_ratio(fov_set_deg: float, ratio: float) -> float:
    fov = math.degrees(2.0 * math.atan(
        ratio * math.tan(math.radians(fov_set_deg) / 2.0)))
    return _clamp(fov)


def _clamp(fov: float) -> float:
    return max(FOV_MIN_DEG, min(FOV_MAX_DEG, fov))


def _usable(v: float | None, yaw: float | None) -> bool:
    return (v is not None and yaw is not None
            and v >= MIN_SPEED_MPS and abs(yaw) <= MAX_YAW_RAD_S)


class FovSolver:
    """Feed it every frame while the FOV step runs. `result` is set once.

    `vision(img, big_img)` returns `(vx, vx_std)` for a stacked input, and
    `views(bgr, calib)` warps one frame into the (narrow, wide) YUV pair.
    Without `vision` only the estimate stage runs."""

    def __init__(self, vision: VisionFn | None = None,
                 views: ViewsFn | None = None) -> None:
        self.vision = vision
        self.views = views
        self.stage = "estimate"
        self.applied_fov: float | None = None
        self.estimate: float | None = None
        self.candidates: list[float] = []
        self.shifts = 0
        self.result: float | None = None
        self._ratios: list[float] = []
        self._cand_ratios: list[list[float]] = []
        self._prev: list[tuple[np.ndarray, np.ndarray] | None] = []
        self._prev_times: list[float | None] = []
        self._last_timestamp: float | None = None
        self._geometry: tuple | None = None
        self._tick = 0
        self._candidate_phase = 0
        self.vision_s = 0.0

    def reset(self) -> None:
        # Keep the cumulative inference timer so a reset during update does
        # not make the caller's per-frame model-time difference negative.
        vision_s = self.vision_s
        self.__init__(self.vision, self.views)
        self.vision_s = vision_s

    # ----- progress -----

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def progress(self) -> float:
        if self.done:
            return 1.0
        if self.stage == "estimate":
            return 0.25 * min(1.0, len(self._ratios) / ESTIMATE_SAMPLES)
        have = sum(min(len(r), CANDIDATE_SAMPLES) for r in self._cand_ratios)
        need = CANDIDATE_SAMPLES * len(self._cand_ratios)
        return 0.25 + 0.75 * have / need

    @property
    def samples(self) -> int:
        return len(self._ratios) + sum(len(r) for r in self._cand_ratios)

    # ----- feed -----

    def update(self, frame: np.ndarray | None, calib, vx_model: float | None,
               v_ego: float | None, yaw_rate: float | None,
               perf_counter: Callable[[], float] | None = None, *,
               timestamp: float | None = None,
               frame_span_s: float = CONTEXT_PERIOD_S) -> float | None:
        """Use raw vision vx and the measured span of its stacked pair.

        Candidate pairs have separate spans, since each candidate is
        evaluated every fourth frame. Returns a newly solved FOV or None.
        """
        if timestamp is None:
            timestamp = (0.0 if self._last_timestamp is None
                         else self._last_timestamp + FRAME_PERIOD_S)
        require_finite("fov", timestamp=timestamp, frame_span_s=frame_span_s)
        shape = frame.shape if frame is not None else (calib.image_h, calib.image_w)
        geometry = frame_geometry(calib, shape)
        if (frame_gap(self._last_timestamp, timestamp)
                or (self._geometry is not None and geometry != self._geometry)):
            self.reset()
        self._last_timestamp = timestamp
        self._geometry = geometry
        if self.done:
            return None
        if self.applied_fov is None:
            self.applied_fov = float(calib.fov_h_deg)
        if self.stage == "estimate":
            if (_usable(v_ego, yaw_rate) and vx_model is not None
                    and vx_model > 0.1 and frame_span_s > 0):
                self._ratios.append(vx_model / v_ego * CONTEXT_PERIOD_S / frame_span_s)
            if len(self._ratios) >= ESTIMATE_SAMPLES:
                self.estimate = solve_from_ratio(self.applied_fov,
                                                 median(self._ratios))
                if self.vision is None or self.views is None or frame is None:
                    self.result = self.estimate
                    return self.result
                self._start_candidates(self.estimate)
            return None
        self._feed_candidate(frame, calib, v_ego, yaw_rate, perf_counter)
        if all(len(r) >= CANDIDATE_SAMPLES for r in self._cand_ratios):
            return self._solve_candidates()
        return None

    def _start_candidates(self, centre: float) -> None:
        lo = FOV_MIN_DEG - min(CANDIDATE_OFFSETS)
        hi = FOV_MAX_DEG - max(CANDIDATE_OFFSETS)
        centre = max(lo, min(hi, centre))
        self.stage = "bracket"
        self.candidates = [round(centre + o, 3) for o in CANDIDATE_OFFSETS]
        self._cand_ratios = [[] for _ in self.candidates]
        self._prev = [None for _ in self.candidates]
        self._prev_times = [None for _ in self.candidates]
        self._tick = 0
        self._candidate_phase = 0

    def _feed_candidate(self, frame, calib, v_ego, yaw_rate, perf_counter) -> None:
        if frame is None:
            return
        phase = min(len(r) for r in self._cand_ratios) // CANDIDATE_PHASE_SAMPLES
        if phase != self._candidate_phase:
            self._candidate_phase = phase
            # Changing phase breaks the cached pair's four-frame interval.
            self._prev = [None for _ in self.candidates]
            self._prev_times = [None for _ in self.candidates]
        j = (self._tick + phase) % len(self.candidates)
        self._tick += 1
        cur = self.views(frame, replace(calib, fov_h_deg=self.candidates[j]))
        prev, self._prev[j] = self._prev[j], cur
        previous_time, self._prev_times[j] = self._prev_times[j], self._last_timestamp
        if prev is None or previous_time is None or not _usable(v_ego, yaw_rate):
            return
        span = self._last_timestamp - previous_time
        if span <= 0:
            return
        img = np.concatenate([prev[0], cur[0]], axis=0)[np.newaxis, ...]
        big = np.concatenate([prev[1], cur[1]], axis=0)[np.newaxis, ...]
        t0 = perf_counter() if perf_counter else 0.0
        pose = self.vision(img, big)
        if perf_counter:
            self.vision_s += perf_counter() - t0
        if pose is None or pose[0] <= 0.1:
            return
        self._cand_ratios[j].append(pose[0] / v_ego * CONTEXT_PERIOD_S / span)

    def _solve_candidates(self) -> float | None:
        x = self.candidates
        y = [median(r) for r in self._cand_ratios]
        crossings = [x[i] + (x[i + 1] - x[i]) * (y[i] - 1.0) / (y[i] - y[i + 1])
                     for i in range(len(x) - 1)
                     if y[i] >= 1.0 >= y[i + 1] and y[i] > y[i + 1]
                     and (i == 0 or y[i - 1] > y[i])
                     and (i + 2 == len(y) or y[i + 1] > y[i + 2])]
        if crossings:
            # Noise can make two; the one nearest the estimate wins.
            self.result = _clamp(min(crossings, key=lambda c: abs(c - self.estimate)))
            return self.result
        if any(a <= b for a, b in zip(y, y[1:])):
            self.result = self.estimate
            return self.result
        if self.shifts < MAX_SHIFTS and (min(y) > 1.0 or max(y) < 1.0):
            self.shifts += 1
            span = x[-1] - x[0]
            self._start_candidates(x[-1] + span / 2 + 2.0 if min(y) > 1.0
                                   else x[0] - span / 2 - 2.0)
            return None
        self.result = _clamp(min(zip(x, y), key=lambda p: abs(p[1] - 1.0))[0])
        return self.result

    # ----- persistence -----

    def to_dict(self) -> dict:
        return {
            "time_base_version": 1,
            "geometry": self._geometry,
            "stage": self.stage,
            "applied_fov": self.applied_fov,
            "estimate": self.estimate,
            "candidates": self.candidates,
            "shifts": self.shifts,
            "tick": self._tick,
            "result": self.result,
            "ratios": [round(r, 6) for r in self._ratios],
            "cand_ratios": [[round(r, 6) for r in rs] for rs in self._cand_ratios],
        }

    def load_dict(self, d: dict) -> None:
        if d.get("time_base_version") != 1:
            # Old ratios do not describe measured time spans.
            self.reset()
            return
        geometry = d.get("geometry")
        self._geometry = ((tuple(geometry[0]), *geometry[1:])
                          if geometry is not None else None)
        self.stage = d.get("stage", "estimate")
        self.applied_fov = d.get("applied_fov")
        self.estimate = d.get("estimate")
        self.candidates = [float(c) for c in d.get("candidates", [])]
        self.shifts = int(d.get("shifts", 0))
        self._tick = int(d.get("tick", 0))
        self.result = d.get("result")
        self._ratios = [float(r) for r in d.get("ratios", [])]
        self._cand_ratios = [[float(r) for r in rs] for rs in d.get("cand_ratios", [])]
        self._prev = [None for _ in self.candidates]
        self._prev_times = [None for _ in self.candidates]
        self._last_timestamp = None
        if self.stage == "bracket" and len(self._cand_ratios) != len(self.candidates):
            self._cand_ratios = [[] for _ in self.candidates]


def model_vision(model) -> VisionFn | None:
    """Adapt a `DrivingModel` into the solver's vision callable."""
    run = getattr(model, "vision_only", None)
    if run is None:
        return None
    from simsteer.core.postprocess import decode_pose

    def vision(img: np.ndarray, big: np.ndarray) -> tuple[float, float]:
        mu, std = decode_pose(run(img, big))
        return float(mu[0]), float(std[0])
    return vision
