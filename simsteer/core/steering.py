"""Clock-driven steering execution; no threads or device access."""
from __future__ import annotations

import math
from dataclasses import dataclass

from simsteer.core.finite import NonFiniteError, require_finite

STEERING_PERIOD_S = .01
STEERING_TIMEOUT_S = .25


def limit_curvature(desired: float, previous: float, speed: float, dt: float,
                    max_jerk: float = 5.) -> float:
    """Bound the steering contribution to lateral jerk: v² * dκ/dt.

    This is a constant-speed bound; acceleration also contributes to the
    vehicle's total lateral jerk. It is not a vehicle certification.
    """
    require_finite("curvature_limit", desired=desired, previous=previous,
                   speed=speed, dt=dt, jerk=max_jerk)
    delta = max(0., max_jerk) * max(0., dt) / max(abs(speed), 1.) ** 2
    return max(previous - delta, min(previous + delta, desired))


@dataclass(frozen=True)
class SteeringTarget:
    axis: float
    speed: float
    # Snapshot of axis = rack_gain * wheel_angle + rack_bias, in device units.
    rack_gain: float
    rack_bias: float
    wheelbase: float
    max_jerk: float | None = 5.


class SteeringExecutor:
    """Smooth complete axis targets, then enforce the curvature jerk bound.

    Targets include trim and probe, which are calculated once per model
    frame. Rack snapshots keep the executor independent of mutable learners.
    A missing plan is tolerated briefly; stale plans or a stalled executor
    clear state and return None so the caller can hand control back.
    """

    def __init__(self, smoothing_s: float = .05) -> None:
        self.smoothing_s = smoothing_s
        self.reset()

    def reset(self) -> None:
        self.target: SteeringTarget | None = None
        self.axis = 0.
        self.submitted_at: float | None = None
        self.last_tick: float | None = None

    def submit(self, target: SteeringTarget, now: float) -> None:
        require_finite("steering_target", **vars(target), now=now)
        if abs(target.rack_gain) < 1e-6 or target.wheelbase <= 0:
            raise NonFiniteError("steering_target.rack")
        self.target = target
        self.submitted_at = now
        if self.last_tick is None:
            self.last_tick = now - STEERING_PERIOD_S

    def tick(self, now: float) -> float | None:
        require_finite("steering_tick", now=now, smoothing=self.smoothing_s)
        target = self.target
        if target is None:
            return None
        dt = now - self.last_tick
        age = now - self.submitted_at
        if dt < 0 or age < 0 or max(dt, age) > STEERING_TIMEOUT_S:
            self.reset()
            return None
        self.last_tick = now
        alpha = 1. if self.smoothing_s <= 0 else -math.expm1(-dt / self.smoothing_s)
        desired_axis = self.axis + alpha * (max(-1., min(1., target.axis)) - self.axis)
        if target.max_jerk is not None:
            # Bound the final command, including trim/probe and authority.
            # Limiting in axis space alone would depend on the rack ratio.
            def curvature(axis):
                wheel = (axis - target.rack_bias) / target.rack_gain
                if abs(wheel) >= 1.5:
                    raise NonFiniteError("steering_tick.rack_range")
                return math.tan(wheel) / target.wheelbase

            previous = curvature(self.axis)
            desired = curvature(desired_axis)
            limited = limit_curvature(desired, previous, target.speed, dt, target.max_jerk)
            desired_axis = (target.rack_gain * math.atan(limited * target.wheelbase)
                            + target.rack_bias)
        require_finite("steering_tick", axis=desired_axis)
        self.axis = max(-1., min(1., desired_axis))
        return self.axis
