"""Evidence-complete certification of a two-minute check drive.

The weakest required measurement decides the score. Missing measurements
remain visible and produce INSUFFICIENT_EVIDENCE, including a nominally good
score without enough engaged driving. Lane changes are excluded by the caller;
large offsets during lane keeping are excursions, never discarded outliers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

PASS_SCORE = 0.5
SATURATION_AXIS = 0.9
MAX_LANE_OFFSET_M = 1.8
MAX_LOG_RATIO = 0.5
STRAIGHT_YAW_RAD_S = 0.06
STRAIGHT_MIN_SPEED_MPS = 8.0
CHECK_SECONDS = 120.0
MIN_COVERAGE = 0.8
MIN_EACH_TURN = 20


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
    Rule("lane_offset_bias", "steering", 0.10, 0.50, 200,
         "lane bias", "m"),
    Rule("lane_excursions", "steering", 0.0, 0.02, 200,
         "car leaves the lane", ""),
    Rule("speed_ratio_error", "fov", 0.05, 0.20, 100,
         "model speed disagrees with the game", ""),
    Rule("steering_saturation", "steering", 0.01, 0.10, 200,
         "steering hits full lock", ""),
    Rule("steering_fit_error", "steering", 0.10, 0.40, 50,
         "steering model mispredicts the wheel", ""),
)


@dataclass(frozen=True)
class CheckPolicy:
    required: tuple[str, ...] = tuple(r.name for r in RULES)
    fit_source: str = "measured"

    @classmethod
    def for_capabilities(cls, steering_fit: bool, synthetic_wheel: bool = False):
        # Vision-only profiles cannot independently check model speed or rack
        # fit. Their smaller declared contract still requires engaged lanes.
        required = tuple(r.name for r in RULES if steering_fit or r.name not in
                         ("speed_ratio_error", "steering_fit_error"))
        return cls(required, "yaw-derived" if synthetic_wheel else "measured")


def _finite(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


@dataclass
class CheckStats:
    """Running counts, including missing observations, for resumable checks."""

    seconds: float = 0.0
    engaged_seconds: float = 0.0
    frames: int = 0
    driving_n: int = 0
    lane_change_n: int = 0
    left_n: int = 0
    right_n: int = 0
    off_total: int = 0
    off_n: int = 0
    off_mean: float = 0.0
    off_m2: float = 0.0
    off_excursions: int = 0
    ratio_total: int = 0
    ratio_n: int = 0
    ratio_log_sum: float = 0.0
    sat_total: int = 0
    sat_n: int = 0
    sat_hits: int = 0
    fit_total: int = 0
    fit_n: int = 0
    fit_err_sum: float = 0.0
    fit_act_sum: float = 0.0
    fit_source: str = "measured"

    def add_offset(self, offset_m: float | None) -> None:
        self.off_total += 1
        if not _finite(offset_m):
            return
        self.off_n += 1
        self.off_excursions += abs(offset_m) > MAX_LANE_OFFSET_M
        d = offset_m - self.off_mean
        self.off_mean += d / self.off_n
        self.off_m2 += d * (offset_m - self.off_mean)

    def add_speed(self, vx_model: float | None, v_ego: float | None,
                  yaw_rate: float | None) -> None:
        if (_finite(v_ego, yaw_rate) and (v_ego < STRAIGHT_MIN_SPEED_MPS
                                         or abs(yaw_rate) > STRAIGHT_YAW_RAD_S)):
            return
        self.ratio_total += 1
        if not _finite(vx_model, v_ego, yaw_rate) or v_ego <= 0 or vx_model <= 0.1:
            return
        lr = math.log(vx_model / v_ego)
        self.ratio_n += 1
        self.ratio_log_sum += max(-MAX_LOG_RATIO, min(MAX_LOG_RATIO, lr))

    def add_axis(self, axis: float, dt: float = 0.05) -> None:
        self.sat_total += 1
        if not _finite(axis):
            return
        self.sat_n += 1
        self.engaged_seconds += dt
        self.sat_hits += abs(axis) >= SATURATION_AXIS
        self.left_n += axis < -0.01
        self.right_n += axis > 0.01

    def add_fit(self, innovation: float, actual: float, *,
                held_out: bool = False, source: str = "measured") -> None:
        self.fit_total += 1
        self.fit_source = source
        if not held_out or not _finite(innovation, actual):
            return
        self.fit_n += 1
        self.fit_err_sum += abs(innovation)
        self.fit_act_sum += abs(actual)

    def values(self) -> dict[str, tuple[float | None, int, int]]:
        """name -> (value, finite held-out samples, offered observations)."""
        return {
            "lane_offset_spread": (math.sqrt(max(0.0, self.off_m2 / self.off_n))
                                   if self.off_n else None, self.off_n, self.off_total),
            "lane_offset_bias": (abs(self.off_mean) if self.off_n else None,
                                 self.off_n, self.off_total),
            "lane_excursions": (self.off_excursions / self.off_n if self.off_n else None,
                                self.off_n, self.off_total),
            "speed_ratio_error": (abs(self.ratio_log_sum / self.ratio_n)
                                  if self.ratio_n else None, self.ratio_n, self.ratio_total),
            "steering_saturation": (self.sat_hits / self.sat_n if self.sat_n else None,
                                    self.sat_n, self.sat_total),
            "steering_fit_error": (self.fit_err_sum / self.fit_act_sum
                                   if self.fit_n and self.fit_act_sum > 1e-6 else None,
                                   self.fit_n, self.fit_total),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "CheckStats":
        s = cls()
        for k, v in (d or {}).items():
            if k in cls.__dataclass_fields__:
                setattr(s, k, type(getattr(s, k))(v))
        return s


@dataclass(frozen=True)
class Measurement:
    name: str
    step: str
    value: float | None
    score: float | None
    samples: int
    denominator: int
    required_samples: int
    required: bool
    source: str = "measured"


@dataclass(frozen=True)
class Verdict:
    passed: bool
    score: float
    redo: str | None
    reason: str
    measurements: tuple[Measurement, ...] = field(default_factory=tuple)
    status: str = "FAIL"
    missing: tuple[str, ...] = ()
    coverage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "status": self.status,
            "score": round(self.score, 3),
            "redo": self.redo,
            "reason": self.reason,
            "missing": list(self.missing),
            "coverage": self.coverage,
            "measurements": {m.name: {**asdict(m),
                                      "value": round(m.value, 4) if m.value is not None else None,
                                      "score": round(m.score, 3) if m.score is not None else None}
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


def evaluate(stats: CheckStats, policy: CheckPolicy = CheckPolicy()) -> Verdict:
    values = stats.values()
    measured = []
    missing = []
    for rule in RULES:
        value, n, denominator = values[rule.name]
        source = "measured"
        if rule.name == "steering_fit_error":
            source = (policy.fit_source if policy.fit_source != "measured" else stats.fit_source)
        required = rule.name in policy.required
        enough = (value is not None and math.isfinite(value) and n >= rule.min_samples
                  and n >= MIN_COVERAGE * denominator)
        if required and (not enough or source != "measured"):
            missing.append(rule.name)
        measured.append((rule, Measurement(rule.name, rule.step, value,
                                          _sub_score(rule, value) if enough else None,
                                          n, denominator, rule.min_samples, required, source)))
    if stats.engaged_seconds + 1e-6 < CHECK_SECONDS:
        missing.append("engaged_driving")
    if min(stats.left_n, stats.right_n) < MIN_EACH_TURN:
        missing.append("left_right_coverage")
    coverage = {"frames": stats.frames, "driving": stats.driving_n,
                "engaged": stats.sat_n, "engaged_seconds": round(stats.engaged_seconds, 3),
                "left": stats.left_n, "right": stats.right_n,
                "lane_changes": stats.lane_change_n}
    ms = tuple(m for _, m in measured)
    scored = [(r, m) for r, m in measured if m.required and m.score is not None]
    score = min((m.score for _, m in scored), default=0.0)
    if missing:
        return Verdict(False, score, "check", "Check again: " + missing[0].replace("_", " "),
                       ms, "INSUFFICIENT_EVIDENCE", tuple(missing), coverage)
    rule, worst = min(scored, key=lambda rm: rm[1].score)
    if worst.score >= PASS_SCORE:
        return Verdict(True, worst.score, None, "", ms, "PASS", (), coverage)
    return Verdict(False, worst.score, worst.step, _describe(rule, worst.value),
                   ms, "FAIL", (), coverage)


def certified_result(result: dict | None, context: dict | None) -> bool:
    """Legacy booleans and scores alone are not saved certificates."""
    if (not result or not context or result.get("status") != "PASS"
            or result.get("passed") is not True or result.get("missing") != []
            or result.get("context") != context):
        return False
    try:
        policy = CheckPolicy(**context["policy"])
        measurements = result["measurements"]
        for rule in RULES:
            if rule.name not in policy.required:
                continue
            m = measurements[rule.name]
            if (m["samples"] < rule.min_samples or m["score"] < PASS_SCORE
                    or m["samples"] < MIN_COVERAGE * m["denominator"]
                    or m["source"] != "measured"):
                return False
        c = result["coverage"]
        return (c["engaged_seconds"] + 1e-6 >= CHECK_SECONDS
                and min(c["left"], c["right"]) >= MIN_EACH_TURN)
    except (KeyError, TypeError, ValueError):
        return False
