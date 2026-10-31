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
    def __init__(self, min_std: float, min_samples: int = MIN_SAMPLES,
                 min_each_turn: int = MIN_EACH_TURN) -> None:
        self.min_std = min_std
        self.min_samples = min_samples
        self.min_each_turn = min_each_turn
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
        if n < self.min_samples or min(self.left, self.right) < self.min_each_turn:
            return None
        variance = max(0.0, self.sxx / n - (self.sx / n) ** 2)
        if variance < self.min_std ** 2 or abs(self.correlation) < MIN_CORRELATION:
            return None
        return 1 if self.correlation > 0 else -1

    @property
    def progress(self) -> float:
        return min(1.0, len(self.pairs) / self.min_samples,
                   self.left / self.min_each_turn, self.right / self.min_each_turn)

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


class OutputDirection:
    """Device command -> observed game input, independent of wheel telemetry.

    Four stationary pulses establish the mapping without engaging AI.
    Average the final second of each three-second hold so input lag and
    smoothing do not depend on the loop's frame rate. Driving observations
    can only revoke this authority, never establish or flip it.
    """

    MIN_AXIS = 0.004
    CONTRADICTION_S = 0.75
    STOP_SPEED_MPS = 0.5
    PLAYER_THRESHOLD = 0.04
    AMPLITUDE = 0.10
    RAW_MAX_GAIN = 1.6
    RAW_SETTLE_S = 0.3
    PULSE_S = 3.0
    SETTLE_S = 2.0
    TIMEOUT_S = 14.0
    METHOD = "stopped-pulses-v1"
    SATURATED_RAW_REASON = "Release steering to check direction: raw input saturated"

    def __init__(self, lag_frames: int, prior: int = -1) -> None:
        self.prior = prior
        self.sign: int | None = None
        self.evidence = _Evidence(0.03, min_samples=4, min_each_turn=2)
        self._commands: deque[float] = deque(maxlen=max(1, lag_frames + 1))
        self._bad_s = 0.0
        self.active = False
        self.reason = ""
        self.elapsed = self._pulse_s = 0.0
        self._pulse = 0
        self._response = self._sample_s = 0.0
        self._raw_response = self._raw_sample_s = 0.0
        self._raw_sign: int | None = None
        self._command = 0.0

    @property
    def ready(self) -> bool:
        return self.sign is not None

    @property
    def progress(self) -> float:
        if self.ready:
            return 1.0
        if self.active:
            return min(0.95, (self._pulse + self._pulse_s / self.PULSE_S) / 4)
        return 0.0

    def _hold_reason(self, speed, observed, user_axis, manual_override) -> str:
        if speed is None or not math.isfinite(speed):
            return "Steering check needs speed telemetry"
        if abs(speed) >= self.STOP_SPEED_MPS:
            return "Stop to check steering direction"
        if observed is None or not math.isfinite(observed):
            return "Steering check needs steering telemetry"
        if manual_override:
            return "Release steering to check direction"
        # Raw game input includes the virtual device, possibly delayed or
        # scaled by the game. Its polarity is unknown until a settled sample.
        raw = observed if user_axis is None else user_axis
        if not math.isfinite(raw):
            return "Steering check needs steering telemetry"
        limit = max(self.PLAYER_THRESHOLD, self.RAW_MAX_GAIN * abs(self._command))
        if abs(raw) > limit:
            if self.active and self._command and abs(raw) >= 0.9 and abs(raw) > 4 * abs(self._command):
                return self.SATURATED_RAW_REASON
            return "Release steering to check direction"
        if (self.active and user_axis is not None and self._command
                and self._pulse_s >= self.RAW_SETTLE_S and abs(raw) >= self.MIN_AXIS):
            raw_sign = 1 if raw * self._command > 0 else -1
            if self._raw_sign is not None and raw_sign != self._raw_sign:
                return "Release steering to check direction"
            self._raw_sign = raw_sign
        return ""

    def start(self, *, speed, observed, user_axis=None, manual_override=False) -> bool:
        self.cancel()
        self.reason = self._hold_reason(speed, observed, user_axis, manual_override)
        if self.ready or self.reason:
            return False
        self.evidence = _Evidence(0.03, min_samples=4, min_each_turn=2)
        self.elapsed = self._pulse_s = 0.0
        self._pulse = 0
        self._response = self._sample_s = 0.0
        self._raw_response = self._raw_sample_s = 0.0
        self._raw_sign = None
        self.active = True
        return True

    def cancel(self, reason: str = "") -> None:
        self.active = False
        self._command = 0.0
        self.reason = reason
        if not self.ready:
            self.evidence = _Evidence(0.03, min_samples=4, min_each_turn=2)

    def tick_stopped(self, *, speed, observed, dt, user_axis=None,
                     manual_override=False) -> float:
        if not self.active:
            return 0.0
        reason = self._hold_reason(speed, observed, user_axis, manual_override)
        if reason:
            self.cancel(reason)
            return 0.0
        if not math.isfinite(dt) or dt <= 0 or dt > 0.5:
            self.cancel("Steering check stopped: frame rate too low")
            return 0.0
        self.elapsed += dt
        if self.elapsed > self.TIMEOUT_S:
            self.cancel("Steering check timed out: press INSERT to retry")
            return 0.0
        if self._command:
            end = self._pulse_s + dt
            sample_s = max(0.0, min(end, self.PULSE_S)
                           - max(self._pulse_s, self.SETTLE_S))
            self._response += observed * sample_s
            self._sample_s += sample_s
            if user_axis is not None:
                self._raw_response += abs(user_axis) * sample_s
                self._raw_sample_s += sample_s
            self._pulse_s = end
            if end + 1e-9 >= self.PULSE_S:
                response = self._response / max(self._sample_s, 1e-9)
                if (abs(response) < self.MIN_AXIS or (self._raw_sample_s > 0
                        and self._raw_response < self.MIN_AXIS * self._raw_sample_s)):
                    self.cancel("No steering response: check binding/deadzone")
                    return 0.0
                self.evidence.add(self._command, response)
                self._pulse += 1
                self._pulse_s = self._response = self._sample_s = 0.0
                self._raw_response = self._raw_sample_s = 0.0
                if self._pulse == 4:
                    self.sign = self.evidence.sign
                    if self.ready:
                        self.active = False
                        self._command = 0.0
                        self.reason = ""
                    else:
                        self.cancel("Steering response unclear: press INSERT to retry")
                    return 0.0
        self._command = self.AMPLITUDE * (1 if self._pulse % 2 == 0 else -1)
        return self._command

    def update(self, *, command: float | None, observed: float | None,
               speed: float | None, engaged: bool,
               lane_change: bool, dt: float, manual_override: bool = False,
               user_axis: float | None = None) -> bool:
        """Guard a verified binding. True requests disengagement."""
        if (not self.ready or not engaged or lane_change or manual_override
                or command is None or observed is None
                or speed is None
                or not all(math.isfinite(x) for x in (command, observed, speed))
                or abs(speed) < 6.0):
            self._commands.clear()
            self._bad_s = 0.0
            return False
        self._commands.append(command)
        if len(self._commands) < self._commands.maxlen:
            return False
        # All plausible delayed commands must have the same nontrivial sign.
        lo, hi = min(self._commands), max(self._commands)
        if (lo < self.MIN_AXIS and hi > -self.MIN_AXIS
                or abs(observed) < self.MIN_AXIS):
            self._bad_s = 0.0
            return False
        expected = sum(self._commands) / len(self._commands)
        # A large opposing response is a driver correction, not evidence
        # of a mirrored binding. Raw input, when available, has no game lag.
        magnitude = abs(expected)
        player_opposing = (user_axis is not None and math.isfinite(user_axis)
                           and user_axis * expected * self.sign < 0
                           and abs(user_axis) > 2 * magnitude + self.PLAYER_THRESHOLD)
        if player_opposing or not 0.5 * magnitude <= abs(observed) <= 2 * magnitude:
            self._bad_s = 0.0
            return False
        if expected * observed * self.sign < 0:
            self._bad_s += min(max(dt, 0.0), 0.25)
        else:
            self._bad_s = 0.0
        return self._bad_s >= self.CONTRADICTION_S

    def to_dict(self) -> dict:
        return {"method": self.METHOD, "sign": self.sign,
                "pairs": list(self.evidence.pairs) if self.ready else []}

    def load_dict(self, d: dict) -> None:
        if d.get("method") != self.METHOD:
            return
        self.evidence.load(d.get("pairs", []))
        sign = d.get("sign")
        if sign in (-1, 1) and self.evidence.sign == sign:
            self.sign = int(sign)
