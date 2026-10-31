"""Pure wheel ownership and handover. All times are monotonic seconds."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

ENGAGE_BLEND_S = 0.25
TAKEOVER_ARM_S = 0.30
TAKEOVER_ANGLE_DEG = 12.0
TAKEOVER_HOLD_S = 0.08
TAKEOVER_RATE_DPS = 180.0
TAKEOVER_RATE_HOLD_S = 0.04
TAKEOVER_RATE_MIN_DEG = 4.0
TAKEOVER_HARD_DEG = 30.0
ANCHOR_TAU_S = 2.0
RATE_FILTER_TAU_S = 0.010
TAKEOVER_FOLLOW_ERROR_DEG = 15.0  # Reserved for rim follow.
BRAKE_TAKEOVER = 0.10
BRAKE_TAKEOVER_HOLD_S = 0.05
THROTTLE_OVERLAY_MARGIN = 0.05
HANDBACK_MIN_S = 0.30
HANDBACK_RATE = 0.20
HANDBACK_MAX_S = 2.0
AI_STALE_S = 0.25
OVERRIDE_HOLD_S = 0.20


def clamp(value, lo=-1.0, hi=1.0):
    if not math.isfinite(value):
        raise ValueError("Non-finite wheel command")
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class Physical:
    steer: float = 0.0
    throttle: float | None = None
    brake: float | None = None
    clutch: float | None = None
    ok: bool = False
    sampled_at: float = 0.0


@dataclass(frozen=True)
class Commands:
    request: str = ""
    serial: int = 0
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    ai_at: float = 0.0
    forced: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Output:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    clutch: float = 0.0
    source: str = "LOST"
    engaged: bool = False
    reason: str = ""


class Override:
    def __init__(self, rim_range_deg=900):
        self.rim_range_deg = rim_range_deg
        self.state, self.reason = "PLAYER", ""
        self.output = Output()
        self.last_at, self.serial, self.lost = None, -1, False
        self.anchor = self.last_rim = self.rate = 0.0
        self.since = {}
        self.started = self.offset = self.duration = 0.0
        self.forced_axes = set()

    @property
    def engaged(self):
        return self.state in ("ENGAGING", "AI")

    def _held(self, name, condition, now, seconds):
        if not condition:
            self.since.pop(name, None)
            return False
        return now - self.since.setdefault(name, now) + 1e-9 >= seconds

    def _handback(self, now, rim, reason):
        self.state, self.reason, self.started = "HANDBACK", reason, now
        self.offset = self.output.steer - rim
        rate = max(HANDBACK_RATE, abs(self.offset) / HANDBACK_MAX_S)
        self.duration = max(HANDBACK_MIN_S, abs(self.offset) / rate)
        self.since.clear()

    def step(self, now: float, phys: Physical, cmd: Commands) -> Output:
        dt = max(0.0, now - self.last_at) if self.last_at is not None else 0.0
        self.last_at = now
        values = (phys.steer, phys.throttle or 0.0, phys.brake or 0.0, phys.clutch or 0.0)
        if not phys.ok or not all(math.isfinite(v) for v in values):
            # Disconnect/error safety takes precedence over ongoing ownership.
            self.state, self.reason, self.lost, self.serial = "PLAYER", "lost", True, cmd.serial
            self.since.clear()
            self.forced_axes.clear()
            self.output = Output(reason="lost")
            return self.output
        rim, throttle, brake, clutch = values
        if self.lost:
            self._handback(now, rim, "recovered")
            self.lost = False
        if cmd.serial != self.serial:
            self.serial = cmd.serial
            if cmd.request == "engage" and brake < BRAKE_TAKEOVER:
                start = self.output.steer if self.state in ("HANDBACK", "OVERRIDE") else rim
                self.state, self.reason, self.started, self.offset = "ENGAGING", "", now, start
                self.anchor = self.last_rim = rim * self.rim_range_deg / 2
                self.rate, self.since = 0.0, {}
            elif cmd.request in ("insert", "loop") and self.engaged:
                self._handback(now, rim, cmd.request)
        if self.engaged:
            degrees = rim * self.rim_range_deg / 2
            speed = (degrees - self.last_rim) / dt if dt > 0 else 0.0
            self.rate = (self.rate + (speed - self.rate) * (1 - math.exp(-dt / RATE_FILTER_TAU_S))
                         if self.rate else speed)
            self.last_rim = degrees
            travel = abs(degrees - self.anchor)
            armed = now - self.started + 1e-9 >= TAKEOVER_ARM_S
            if not armed:
                self.anchor = degrees
            elif (travel < TAKEOVER_ANGLE_DEG / 2
                  or (0 < abs(self.rate) < TAKEOVER_ANGLE_DEG / ANCHOR_TAU_S and travel < TAKEOVER_ANGLE_DEG)):
                # Continue absorbing sustained slow drift: stopping at half
                # the travel threshold would freeze a 5°/s drift into a grab.
                self.anchor += (degrees - self.anchor) * (1 - math.exp(-dt / ANCHOR_TAU_S))
            steering = armed and (
                travel + 1e-9 >= TAKEOVER_HARD_DEG
                or self._held("travel", travel + 1e-9 >= TAKEOVER_ANGLE_DEG, now, TAKEOVER_HOLD_S)
                or (self._held("flick", abs(self.rate) >= TAKEOVER_RATE_DPS,
                               now, TAKEOVER_RATE_HOLD_S) and travel > TAKEOVER_RATE_MIN_DEG))
            reason = ("brake" if self._held("brake", brake >= BRAKE_TAKEOVER, now, BRAKE_TAKEOVER_HOLD_S)
                      else "steering" if steering else "stale" if now - cmd.ai_at >= AI_STALE_S else "")
            if reason:
                self._handback(now, rim, reason)
        forced = {k: v for k, (v, until) in cmd.forced.items() if now < until}
        if self.engaged:
            blend = min(1.0, max(0.0, (now - self.started) / ENGAGE_BLEND_S))
            steer = self.offset * (1 - blend) + cmd.steer * blend
            if blend == 1:
                self.state = "AI"
            throttle = max(cmd.throttle, throttle) if throttle > cmd.throttle + THROTTLE_OVERLAY_MARGIN else cmd.throttle
            brake = cmd.brake
        else:
            if ((self.state == "OVERRIDE" and self.forced_axes and not forced)
                    or ("steer" in self.forced_axes and "steer" not in forced)):
                self._handback(now, rim, "override")
            elif forced and self.state != "HANDBACK":
                self.state = "OVERRIDE"
            steer = rim
            if self.state == "HANDBACK":
                blend = max(0.0, 1 - (now - self.started) / self.duration)
                steer += self.offset * blend
                if blend <= 0:
                    self.state = "OVERRIDE" if forced else "PLAYER"
            steer, throttle, brake = (forced.get(k, v) for k, v in
                                      (("steer", steer), ("throttle", throttle), ("brake", brake)))
        self.output = Output(clamp(steer), clamp(throttle, 0, 1), clamp(brake, 0, 1),
                             clamp(clutch, 0, 1), self.state, self.engaged, self.reason)
        self.forced_axes = set(forced) if not self.engaged else set()
        return self.output
