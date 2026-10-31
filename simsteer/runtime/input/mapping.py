"""Axis conversion and explicit pedal calibration, without device I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass

MAP_PEDAL_WINDOW_S = 3.0
PEDALS = ("throttle", "brake", "clutch")


def validate_axes(value) -> dict:
    if not isinstance(value, dict) or set(value) - {"steer", *PEDALS}:
        return {}
    used = {value.get("steer", 0)} if type(value.get("steer", 0)) is int else set()
    if not used or not 0 <= next(iter(used)) < 8:
        return {}
    for name in PEDALS:
        if name not in value:
            continue
        p = value[name]
        if (not isinstance(p, dict) or set(p) != {"slot", "rest", "full"}
                or type(p["slot"]) is not int or not 0 <= p["slot"] < 8 or p["slot"] in used
                or any(type(p[k]) not in (int, float) or not math.isfinite(p[k]) for k in ("rest", "full"))
                or p["rest"] == p["full"]):
            return {}
        used.add(p["slot"])
    return {k: dict(v) if isinstance(v, dict) else v for k, v in value.items()}


def resolve_slots(names: dict[int, str]) -> dict[str, int]:
    wanted = {"wheel axis": "steer", "accelerator": "throttle", "brake": "brake", "clutch": "clutch"}
    slots = {wanted[n.strip().casefold()]: i for i, n in names.items() if n.strip().casefold() in wanted}
    return {"steer": 0, **slots}


@dataclass(frozen=True)
class AxisMap:
    slot: int = 0
    rest: float = 0
    full: float = 65535

    def pedal(self, axes) -> float:
        return max(0.0, min(1.0, (axes[self.slot] - self.rest) / (self.full - self.rest)))

    def steering(self, axes) -> float:
        return self.pedal(axes) * 2.0 - 1.0

    @staticmethod
    def mapped_pedal(axes, mapping) -> float | None:
        return AxisMap(**mapping).pedal(axes) if mapping else None


class PedalMapper:
    """Three timed presses; failed required pedals never publish a partial map."""
    def __init__(self, now, axes, names, ranges, steer):
        self.started, self.rest, self.ranges = now, tuple(axes), ranges
        self.names, self.result, self.index = names, {"steer": steer}, 0
        self.peak = tuple(axes)
        self.done = False
        self.status = "Press throttle fully"

    def step(self, now, axes):
        if self.done:
            return
        self.peak = tuple(v if abs(v - r) > abs(p - r) else p for v, r, p in zip(axes, self.rest, self.peak))
        if now - self.started < MAP_PEDAL_WINDOW_S:
            return
        used = {self.result["steer"], *(p["slot"] for p in self.result.values() if isinstance(p, dict))}
        candidates = [i for i in self.names if i not in used and "combined" not in self.names[i].casefold()]
        travel = lambda i: abs(self.peak[i] - self.rest[i]) / max(1, self.ranges[i][1] - self.ranges[i][0])
        slot = max(candidates, key=travel, default=None)
        if slot is not None and travel(slot) >= 0.1:
            self.result[PEDALS[self.index]] = {"slot": slot, "rest": self.rest[slot], "full": self.peak[slot]}
        elif self.index < 2:
            self.done, self.result, self.status = True, {}, "No pedal movement; retry"
            return
        self.index += 1
        self.done = self.index == len(PEDALS)
        self.status = "Pedals mapped" if self.done else f"Release pedals, press {PEDALS[self.index]} fully"
        self.started, self.rest, self.peak = now, tuple(axes), tuple(axes)
