"""Navigation-on-openpilot port.

openpilot's NOOP has three layers:
  1. Route source (Mapbox API in production, OSM-routing alternative).
  2. Maneuver queue — step-by-step list of (direction, distance) entries
     that count down with v_ego as the truck drives.
  3. Execution — at a trigger distance before each maneuver, fire the
     corresponding lane-change desire so the model commits to the
     correct lane for the turn / exit.

The current driving_policy.onnx public model does NOT accept nav
features as an input — comma 's recent split removed `nav_features`
and `nav_instructions` from the policy head. So "executing" a
maneuver in our port means injecting the same lane-change one-hot
desire that the ← / → keys use today. That's enough for the model
to commit to the inside lane before a turn.

Route source for games (none have a real Mapbox-equivalent):
  - **ETS2**: SCS exposes `navigation_distance` / `navigation_time` to
    DESTINATION only — no per-maneuver breakdown. We display it but
    can't auto-queue maneuvers from it.
  - **AC / Forza**: no nav data exposed at all.

So practically the queue is populated by the user via global hotkeys
(F1 = queue LEFT, F2 = queue RIGHT, F3 = clear queue). The queue
auto-decrements each frame and fires lane-changes at the trigger
distance. Future work: read ETS2's HUD minimap to auto-detect turns,
or hook into a separate Mapbox client.

Hotkeys (global — work while game is focused):
    F1  queue LEFT  at NAV_DEFAULT_DISTANCE_M (default 250 m)
    F2  queue RIGHT at NAV_DEFAULT_DISTANCE_M
    F3  clear queue
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# Default planning distance when the user queues a maneuver via the
# global hotkey (PageUp/PageDown). Set just above the trigger so the
# lane change fires within a couple seconds at normal driving speed —
# previously 250 m (= 5 s @ 50 km/h, easy to perceive as "doesn't
# work"). Bumping the queue further out is still possible by calling
# `nav.queue(direction, distance_m=...)` explicitly.
NAV_DEFAULT_DISTANCE_M = 180.0

# When the next maneuver's remaining distance drops below this, fire
# the lane-change desire. Below the openpilot DesireHelper trigger of
# `LANE_CHANGE_TIME_S * v_ego` (~50 m at highway). We use a fixed
# distance so it works at any speed.
NAV_TRIGGER_DISTANCE_M = 150.0

# After firing, the desire is sustained for this long (matches
# `ControllerConfig.lane_change_hold_s`). Without this the single-frame
# pulse washes out of the model's 5 s temporal buffer.
NAV_HOLD_S = 2.5

# How long after the trigger distance to keep the maneuver in the
# queue. Beyond this we assume the maneuver has been executed and pop
# it off, even if v_ego didn't quite drive past the planned point
# (game pause, going backward, etc).
NAV_EXPIRY_S = 30.0


class ManeuverDir(IntEnum):
    LEFT = 0
    RIGHT = 1


# Matches openpilot's desire indices (selfdrive/controls/lib/desire_helper.py).
_DESIRE_LANE_CHANGE_LEFT = 3
_DESIRE_LANE_CHANGE_RIGHT = 4


@dataclass
class Maneuver:
    direction: ManeuverDir
    distance_m: float          # remaining distance until execution
    fired: bool = False        # True once the lane-change pulse started
    fire_until_t: float = 0.0  # wall time the pulse is held until
    age_s: float = 0.0         # seconds since added to queue


class NavManager:
    """Holds a queue of pending maneuvers, counts them down with
    v_ego, and decides what desire one-hot (if any) to feed the model
    this frame."""

    def __init__(self,
                 trigger_distance_m: float = NAV_TRIGGER_DISTANCE_M,
                 hold_s: float = NAV_HOLD_S,
                 default_distance_m: float = NAV_DEFAULT_DISTANCE_M) -> None:
        self.trigger_distance_m = trigger_distance_m
        self.hold_s = hold_s
        self.default_distance_m = default_distance_m
        self._queue: list[Maneuver] = []
        self._dest_distance_m: float | None = None   # SCS nav distance, if any
        self._dest_time_s: float | None = None
        # Diagnostics
        self.last_fired_direction: ManeuverDir | None = None
        self.executed_count = 0

    # ----- queue management -----

    def queue(self, direction: ManeuverDir,
              distance_m: float | None = None) -> None:
        d = float(distance_m if distance_m is not None
                  else self.default_distance_m)
        self._queue.append(Maneuver(direction=direction, distance_m=d))

    def clear(self) -> None:
        self._queue.clear()
        self.last_fired_direction = None

    @property
    def queue_len(self) -> int:
        return len(self._queue)

    @property
    def next_maneuver(self) -> Maneuver | None:
        return self._queue[0] if self._queue else None

    # ----- destination info from telemetry (display only) -----

    def update_destination(self, distance_m: float | None,
                           time_s: float | None) -> None:
        self._dest_distance_m = distance_m
        self._dest_time_s = time_s

    @property
    def dest_distance_m(self) -> float | None:
        return self._dest_distance_m

    @property
    def dest_time_s(self) -> float | None:
        return self._dest_time_s

    # ----- per-frame tick -----

    def tick(self, v_ego: float, dt: float, now_t: float
             ) -> int | None:
        """Decrement the head-of-queue maneuver by v_ego * dt and
        determine whether to fire the lane-change desire this frame.

        Returns the desire one-hot index to inject this frame, or None
        for no nav-driven desire. The caller can OR this with manually
        triggered lane-changes (← / →) — last-write-wins on the model
        input.
        """
        if not self._queue:
            return None
        m = self._queue[0]
        m.age_s += dt
        m.distance_m -= max(0.0, v_ego) * dt

        # Has the maneuver triggered?
        if not m.fired and m.distance_m <= self.trigger_distance_m:
            m.fired = True
            m.fire_until_t = now_t + self.hold_s
            self.last_fired_direction = m.direction

        # Hold the desire active for the configured window. After the
        # window expires OR the maneuver ages out, pop it.
        if m.fired and now_t >= m.fire_until_t:
            self._queue.pop(0)
            self.executed_count += 1
            return None
        if m.age_s > NAV_EXPIRY_S:
            self._queue.pop(0)
            return None

        if m.fired:
            return (_DESIRE_LANE_CHANGE_LEFT if m.direction == ManeuverDir.LEFT
                    else _DESIRE_LANE_CHANGE_RIGHT)
        return None

    # ----- HUD helper -----

    def hud_line(self) -> str:
        parts: list[str] = []
        if self._queue:
            m = self._queue[0]
            dir_arrow = "←" if m.direction == ManeuverDir.LEFT else "→"
            state = " FIRING" if m.fired else ""
            parts.append(f"NAV next: {dir_arrow} {m.distance_m:5.0f} m"
                         f"  (queue {len(self._queue)}){state}")
        else:
            parts.append("NAV queue empty")
        if self._dest_distance_m is not None:
            parts.append(f"dest: {self._dest_distance_m/1000:.1f} km")
        if self._dest_time_s is not None:
            mins = int(self._dest_time_s // 60)
            parts.append(f"ETA {mins} min")
        if self.executed_count > 0:
            parts.append(f"done={self.executed_count}")
        return "  ".join(parts)
