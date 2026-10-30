"""Steering direction from a manual drive, before autonomous output."""

from __future__ import annotations

import math
from collections import deque

MIN_SAMPLES = 80
MIN_EACH_TURN = 20
MIN_WHEEL_RAD = 0.001
MIN_CORRELATION = 0.85
WINDOW = 400


class _Evidence:
    def __init__(self, min_std: float) -> None:
        self.min_std = min_std
        self.pairs: deque[tuple[float, float]] = deque()
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.left = self.right = 0

    def _sum(self, x: float, y: float, weight: int) -> None:
        self.sx += weight * x
        self.sy += weight * y
        self.sxx += weight * x * x
        self.syy += weight * y * y
        self.sxy += weight * x * y
        self.left += weight * int(y < 0)
        self.right += weight * int(y > 0)

    def add(self, x: float, y: float) -> None:
        if len(self.pairs) == WINDOW:
            self._sum(*self.pairs.popleft(), -1)
        self.pairs.append((x, y))
        self._sum(x, y, 1)

    @property
    def correlation(self) -> float:
        n = len(self.pairs)
        if n < 2:
            return 0.0
        xx = self.sxx - self.sx * self.sx / n
        yy = self.syy - self.sy * self.sy / n
        if xx <= 0 or yy <= 0:
            return 0.0
        return (self.sxy - self.sx * self.sy / n) / math.sqrt(xx * yy)

    @property
    def sign(self) -> int | None:
        n = len(self.pairs)
        if n < MIN_SAMPLES or min(self.left, self.right) < MIN_EACH_TURN:
            return None
        variance = max(0.0, self.sxx / n - (self.sx / n) ** 2)
        if variance < self.min_std ** 2 or abs(self.correlation) < MIN_CORRELATION:
            return None
        return 1 if self.correlation > 0 else -1

    @property
    def progress(self) -> float:
        return min(1.0, len(self.pairs) / MIN_SAMPLES,
                   self.left / MIN_EACH_TURN, self.right / MIN_EACH_TURN)

    def load(self, pairs: list) -> None:
        for x, y in pairs[-WINDOW:]:
            x, y = float(x), float(y)
            if math.isfinite(x) and math.isfinite(y) and abs(y) >= MIN_WHEEL_RAD:
                self.add(x, y)


class SteeringDirection:
    """Two independent mappings: model curvature to game wheel, and
    raw input axis to game wheel. Neither uses a commanded AI axis.
    Both turns and centered correlation reject trim bias and noise."""

    def __init__(self, lag_frames: int) -> None:
        self.lag_frames = lag_frames
        self.model = _Evidence(0.00015)
        self.axis = _Evidence(0.005)
        self._axes: deque[float] = deque(maxlen=lag_frames + 1)
        self.model_sign: int | None = None
        self.axis_sign: int | None = None

    @property
    def ready(self) -> bool:
        return self.model_sign is not None and self.axis_sign is not None

    @property
    def progress(self) -> float:
        if self.ready:
            return 1.0
        return min(0.95, self.model.progress, self.axis.progress)

    def update(self, *, curvature: float | None, axis: float | None,
               wheel: float | None, speed: float | None,
               manual: bool, lane_change: bool = False) -> None:
        if self.ready:
            return
        if not manual or axis is None or not math.isfinite(axis):
            self._axes.clear()
            return
        self._axes.append(axis)
        if (curvature is None or wheel is None or speed is None
                or not all(math.isfinite(x) for x in (curvature, wheel, speed))
                or speed < 8.0 or lane_change
                or not MIN_WHEEL_RAD <= abs(wheel) <= 0.15
                or abs(curvature) > 0.02 or abs(axis) >= 0.8):
            return
        self.model.add(curvature, wheel)
        if len(self._axes) > self.lag_frames and max(self._axes) - min(self._axes) < 0.3:
            self.axis.add(self._axes[0], wheel)
        model_sign, axis_sign = self.model.sign, self.axis.sign
        if model_sign is not None and axis_sign is not None:
            self.model_sign, self.axis_sign = model_sign, axis_sign

    def to_dict(self) -> dict:
        return {"model_sign": self.model_sign, "axis_sign": self.axis_sign,
                "model_pairs": list(self.model.pairs),
                "axis_pairs": list(self.axis.pairs)}

    def load_dict(self, d: dict) -> None:
        self.model.load(d.get("model_pairs", []))
        self.axis.load(d.get("axis_pairs", []))
        model_sign, axis_sign = d.get("model_sign"), d.get("axis_sign")
        if model_sign in (-1, 1) and axis_sign in (-1, 1):
            self.model_sign, self.axis_sign = int(model_sign), int(axis_sign)
