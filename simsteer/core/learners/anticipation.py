"""Learn how far ahead steering must act from the measured response delay.

The lateral controller reads the plan's curvature `lookahead_s +
curvature_anticipation_s` into the future, so that by the time the
command has gone through the model, the output device, the game's input
filter and the vehicle's yaw response, the truck turns where the plan
turns. A fixed lead turns in late wherever that chain is slower, e.g. a
heavy truck at highway speed.

While engaged, this records the commanded curvature and the curvature
the vehicle actually drives (yaw rate / speed). Every few seconds it finds
the delay that best aligns the two, and moves the anticipation so the
total lead matches that delay. The delay belongs to the vehicle and the
game, not to the lead, so the estimate does not feed on itself.

Only well-excited, steady stretches count: engaged, above
`MIN_SPEED_MPS`, no lane change, enough curvature variation, and a strong
correlation at the best delay. Anything else leaves the value alone.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

MIN_SPEED_MPS = 8.0
WINDOW_S = 20.0
ESTIMATE_EVERY_S = 2.0
MAX_DELAY_S = 1.2
DELAY_STEP_S = 0.05
MIN_CURVATURE_STD = 0.0005    # 1/m; below this the stretch is too straight
MIN_CORRELATION = 0.8
SMOOTHING = 0.25              # share of the gap closed per accepted estimate
ANTICIPATION_MIN_S = -0.3
ANTICIPATION_MAX_S = 0.5      # the tuner knob's range


class AnticipationLearner:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._samples: deque[tuple[float, float, float]] = deque()
        self._t = 0.0
        self._next_estimate = ESTIMATE_EVERY_S
        self.last_delay_s: float | None = None
        self.last_correlation: float | None = None
        self.estimates = 0

    def reset(self) -> None:
        """Drop the window, e.g. on disengage: a gap would fake a delay."""
        self._samples.clear()
        self._next_estimate = self._t + ESTIMATE_EVERY_S

    def update(self, *, dt: float, commanded_k: float, yaw_rate: float | None,
               speed: float | None, engaged: bool, lane_change: bool) -> bool:
        """Feed one control tick. Returns True when the anticipation moved."""
        if not (math.isfinite(dt) and dt > 0):
            return False
        self._t += dt
        usable = (getattr(self.cfg, "auto_anticipation", False) and engaged and not lane_change
                  and speed is not None and yaw_rate is not None
                  and math.isfinite(speed) and math.isfinite(yaw_rate)
                  and math.isfinite(commanded_k) and speed >= MIN_SPEED_MPS)
        if not usable:
            self.reset()
            return False
        self._samples.append((self._t, commanded_k, yaw_rate / speed))
        while self._samples and self._samples[0][0] < self._t - WINDOW_S:
            self._samples.popleft()
        if self._t < self._next_estimate:
            return False
        self._next_estimate = self._t + ESTIMATE_EVERY_S
        return self._estimate()

    def _estimate(self) -> bool:
        if len(self._samples) < 2 or self._samples[-1][0] - self._samples[0][0] < WINDOW_S / 2:
            return False
        t, cmd, act = (np.asarray(c) for c in zip(*self._samples))
        if float(np.std(cmd)) < MIN_CURVATURE_STD:
            return False
        best_delay, best_corr = None, 0.0
        for k in range(int(round(MAX_DELAY_S / DELAY_STEP_S)) + 1):
            delay = k * DELAY_STEP_S
            usable = t - delay >= t[0]
            if usable.sum() < 20:
                break
            shifted = np.interp(t[usable] - delay, t, cmd)
            response = act[usable]
            if np.std(shifted) <= 0 or np.std(response) <= 0:
                continue
            # The yaw sign convention is the game's; only alignment matters.
            corr = abs(float(np.corrcoef(shifted, response)[0, 1]))
            if corr > best_corr:
                best_delay, best_corr = delay, corr
        self.last_correlation = best_corr
        if best_delay is None or best_corr < MIN_CORRELATION:
            return False
        self.last_delay_s = best_delay
        target = min(ANTICIPATION_MAX_S, max(ANTICIPATION_MIN_S, best_delay - self.cfg.lookahead_s))
        current = self.cfg.curvature_anticipation_s
        new = round(current + SMOOTHING * (target - current), 3)
        self.estimates += 1
        if abs(new - current) < 1e-3:
            return False
        self.cfg.curvature_anticipation_s = new
        return True
