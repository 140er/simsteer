"""Score the calibration check drive.

Four measurements, each mapped linearly from 1 at its `good` value to 0
at its `bad` value. The score is the weakest one, and the check passes
at 0.5 or more. A failing check names the step that owns the weakest
measurement, so the player redoes that step and not the whole flow.

    lane offset spread   std of the distance to the lane centre (m) while
                         SimSteer steers, lane changes left out
                         good 0.15, bad 0.45: a truck in a 3.6 m lane has
                         about 0.55 m each side, so 0.45 means two-sigma
                         swings touch the lines. -> Steering
    speed ratio error    |mean ln(vx_model / v_ego)| on straight, fast road
                         good 0.05, bad 0.20: about 3 and 11 degrees of FOV
                         error near 75 degrees. -> FOV
    steering saturation  share of frames with |axis| >= 0.9
                         good 1 %, bad 10 %. -> Steering
    steering fit error   mean |axis - predicted axis| / mean |axis| from
                         LiveParams on the same frames
                         good 0.10, bad 0.40: LiveParams calls a fit
                         healthy under 0.05 and unstable over 0.30. -> Steering

A measurement with too few samples is left out rather than guessed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

PASS_SCORE = 0.5
SATURATION_AXIS = 0.9
# Beyond this the car is crossing a line or the model swapped lanes.
MAX_LANE_OFFSET_M = 1.8
MAX_LOG_RATIO = 0.5
STRAIGHT_YAW_RAD_S = 0.06
STRAIGHT_MIN_SPEED_MPS = 8.0


@dataclass(frozen=True)
class Rule:
    name: str
    step: str
    good: float
    bad: float
    min_samples: int
    label: str
    unit: str


RULES = (
    Rule("lane_offset_spread", "steering", 0.15, 0.45, 200,
         "car weaves in the lane", "m"),
    Rule("speed_ratio_error", "fov", 0.05, 0.20, 100,
         "model speed disagrees with the game", ""),
    Rule("steering_saturation", "steering", 0.01, 0.10, 200,
         "steering hits full lock", ""),
    Rule("steering_fit_error", "steering", 0.10, 0.40, 50,
         "steering model mispredicts the wheel", ""),
)


@dataclass
class CheckStats:
    """Running sums, so a check can save mid-drive and resume."""

    seconds: float = 0.0
    off_n: int = 0
    off_mean: float = 0.0
    off_m2: float = 0.0
    ratio_n: int = 0
    ratio_log_sum: float = 0.0
    sat_n: int = 0
    sat_hits: int = 0
    fit_n: int = 0
    fit_err_sum: float = 0.0
    fit_act_sum: float = 0.0

    def add_offset(self, offset_m: float | None) -> None:
        if offset_m is None or abs(offset_m) > MAX_LANE_OFFSET_M:
            return
        self.off_n += 1
        d = offset_m - self.off_mean
        self.off_mean += d / self.off_n
        self.off_m2 += d * (offset_m - self.off_mean)

    def add_speed(self, vx_model: float | None, v_ego: float | None,
                  yaw_rate: float | None) -> None:
        if (vx_model is None or v_ego is None or yaw_rate is None
                or v_ego < STRAIGHT_MIN_SPEED_MPS or vx_model <= 0.1
                or abs(yaw_rate) > STRAIGHT_YAW_RAD_S):
            return
        lr = math.log(vx_model / v_ego)
        self.ratio_n += 1
        self.ratio_log_sum += max(-MAX_LOG_RATIO, min(MAX_LOG_RATIO, lr))

    def add_axis(self, axis: float) -> None:
        self.sat_n += 1
        if abs(axis) >= SATURATION_AXIS:
            self.sat_hits += 1

    def add_fit(self, innovation: float, actual: float) -> None:
        self.fit_n += 1
        self.fit_err_sum += abs(innovation)
        self.fit_act_sum += abs(actual)

    def values(self) -> dict[str, tuple[float, int]]:
        """name -> (value, samples)."""
        out: dict[str, tuple[float, int]] = {}
        if self.off_n:
            out["lane_offset_spread"] = (
                math.sqrt(self.off_m2 / self.off_n), self.off_n)
        if self.ratio_n:
            out["speed_ratio_error"] = (
                abs(self.ratio_log_sum / self.ratio_n), self.ratio_n)
        if self.sat_n:
            out["steering_saturation"] = (self.sat_hits / self.sat_n, self.sat_n)
        if self.fit_n and self.fit_act_sum > 1e-6:
            out["steering_fit_error"] = (
                self.fit_err_sum / self.fit_act_sum, self.fit_n)
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "CheckStats":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k):
                setattr(s, k, type(getattr(s, k))(v))
        return s


@dataclass(frozen=True)
class Measurement:
    name: str
    step: str
    value: float
    score: float
    samples: int


@dataclass(frozen=True)
class Verdict:
    passed: bool
    score: float
    redo: str | None
    reason: str
    measurements: tuple[Measurement, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": round(self.score, 3),
            "redo": self.redo,
            "reason": self.reason,
            "measurements": {m.name: {"value": round(m.value, 4),
                                      "score": round(m.score, 3),
                                      "samples": m.samples}
                             for m in self.measurements},
        }


def _sub_score(rule: Rule, value: float) -> float:
    return max(0.0, min(1.0, (rule.bad - value) / (rule.bad - rule.good)))


def _describe(rule: Rule, value: float) -> str:
    if rule.unit == "m":
        shown, limit = f"{value:.2f} m", f"{(rule.good + rule.bad) / 2:.2f} m"
    else:
        shown = f"{value * 100:.0f}%"
        limit = f"{(rule.good + rule.bad) / 2 * 100:.0f}%"
    return f"{rule.label} ({shown}, limit {limit})"


def evaluate(stats: CheckStats) -> Verdict:
    values = stats.values()
    measured: list[tuple[Rule, Measurement]] = []
    for rule in RULES:
        if rule.name not in values:
            continue
        value, n = values[rule.name]
        if n < rule.min_samples:
            continue
        measured.append((rule, Measurement(rule.name, rule.step, value,
                                           _sub_score(rule, value), n)))
    if not measured:
        return Verdict(False, 0.0, "check", "not enough driving was measured")
    rule, worst = min(measured, key=lambda rm: rm[1].score)
    ms = tuple(m for _, m in measured)
    if worst.score >= PASS_SCORE:
        return Verdict(True, worst.score, None, "", ms)
    return Verdict(False, worst.score, worst.step, _describe(rule, worst.value), ms)
