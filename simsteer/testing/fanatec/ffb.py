"""What a simulated base received, and the force it would have produced.

The base appends events as SimSteer's DirectInput calls arrive:

- `call`: every force-related call (exclusive Acquire, SetProperty,
  CreateEffect, SetParameters, Start, Stop, Unload, effect Release,
  SendForceFeedbackCommand) with its HRESULT;
- `gain`: the device gain (0..10000) changed;
- `play` / `end`: an effect started producing force (from `start`, until
  `until` or for ever) or stopped;
- `gate`: the motor as a whole turned on or off (unplugged, actuators
  off, paused, safety switch).

The force model is an upper bound of the commanded torque as a fraction of
the base's peak: per effect |constant|, max(|ramp start|, |ramp end|),
periodic magnitude + |offset|, condition saturation, the envelope's
attack/fade levels when higher, times the effect gain; summed, clipped to
1, times the device gain, and 0 while the gate is off. It is not a physics
model, but it is exact for "never above X" and "stopped by T".

An effect with a duration keeps playing until its duration ends even if
the process that started it dies: the worst case the bridge's renewal
windows are built for. `FfbLog.load` rebuilds a log from a JSONL file, so
a killed process can still be judged.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from simsteer.core import ffb_model as fm

NOMINAL = fm.DI_FFNOMINALMAX
FORCE_OPS = frozenset({"Acquire", "SetProperty", "CreateEffect", "SetParameters", "Start", "Stop", "Unload",
                       "Release", "SendForceFeedbackCommand", "Download"})
EPS = 1e-9


def level_of(kind: str, params: fm.DiParams) -> float:
    """Upper bound of an effect's force, 0..1 of the base's peak."""
    ts = params.type_specific
    if isinstance(ts, fm.DiConstant):
        level = abs(ts.magnitude)
    elif isinstance(ts, fm.DiRamp):
        level = max(abs(ts.start), abs(ts.end))
    elif isinstance(ts, fm.DiPeriodic):
        level = ts.magnitude + abs(ts.offset)
    elif isinstance(ts, fm.DiCondition):
        level = max(ts.positive_saturation, ts.negative_saturation)
        if not level and (ts.positive_coefficient or ts.negative_coefficient):
            level = NOMINAL  # no saturation given: assume the worst
    else:
        level = NOMINAL if kind == "CustomForce" else 0
    if params.envelope is not None:
        level = max(level, params.envelope.attack_level, params.envelope.fade_level)
    return min(1.0, level / NOMINAL) * min(1.0, params.gain / NOMINAL)


@dataclass
class Interval:
    effect: int
    kind: str
    start: float
    end: float  # math.inf while unbounded
    level: float


@dataclass
class FfbLog:
    on_event: Callable[[dict], None] | None = None
    events: list[dict] = field(default_factory=list)
    intervals: list[Interval] = field(default_factory=list)
    _open: dict[int, Interval] = field(default_factory=dict)
    _gain: list[tuple[float, int]] = field(default_factory=lambda: [(-math.inf, NOMINAL)])
    _gate: list[tuple[float, bool]] = field(default_factory=lambda: [(-math.inf, True)])
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ---- recording

    def add(self, event: dict) -> None:
        with self._lock:
            self._apply(event)
            self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def _apply(self, e: dict) -> None:
        ev, t = e["ev"], e["t"]
        if ev == "gain":
            self._gain.append((t, e["value"]))
        elif ev == "gate":
            self._gate.append((t, e["on"]))
        elif ev == "play":
            self._close(e["effect"], t)
            until = math.inf if e.get("until") is None else e["until"]
            iv = Interval(e["effect"], e.get("kind", ""), e.get("start", t), until, e["level"])
            self._open[iv.effect] = iv
            self.intervals.append(iv)
        elif ev == "end":
            self._close(e["effect"], t)

    def _close(self, effect: int, t: float) -> None:
        iv = self._open.pop(effect, None)
        if iv is not None:
            iv.end = min(iv.end, t)

    @classmethod
    def from_events(cls, events: Iterable[dict]) -> FfbLog:
        log = cls()
        for e in events:
            log.add(e)
        return log

    @classmethod
    def load(cls, path: str | Path) -> FfbLog:
        """A log written with `on_event=FfbLog.writer(path)`; a torn last line
        (the writer was killed) is ignored."""
        events = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break
        return cls.from_events(events)

    @staticmethod
    def writer(path: str | Path) -> Callable[[dict], None]:
        f = open(path, "w", encoding="utf-8")  # noqa: SIM115 - closed with the process
        lock = threading.Lock()

        def write(event: dict) -> None:
            with lock:
                f.write(json.dumps(event) + "\n")
                f.flush()
        write.close = f.close  # type: ignore[attr-defined]
        return write

    # ---- queries

    @staticmethod
    def _at(timeline, t):
        value = timeline[0][1]
        for when, v in timeline:
            if when > t:
                break
            value = v
        return value

    def gain_at(self, t: float) -> int:
        return self._at(self._gain, t)

    def gate_at(self, t: float) -> bool:
        return self._at(self._gate, t)

    def gain_timeline(self) -> list[tuple[float, int]]:
        return self._gain[1:]

    def playing(self, at: float) -> list[Interval]:
        return [iv for iv in self.intervals if iv.start <= at < iv.end and iv.level > 0]

    def force_at(self, t: float) -> float:
        if not self.gate_at(t):
            return 0.0
        total = min(1.0, sum(iv.level for iv in self.playing(t)))
        return total * self.gain_at(t) / NOMINAL

    def _breakpoints(self) -> list[float]:
        points = {iv.start for iv in self.intervals}
        points |= {iv.end for iv in self.intervals if math.isfinite(iv.end)}
        points |= {t for t, _ in self._gain[1:]} | {t for t, _ in self._gate[1:]}
        return sorted(points)

    def peak_force(self) -> float:
        return max((self.force_at(t) for t in self._breakpoints()), default=0.0)

    def last_force_time(self) -> float | None:
        """When the force last ended (inf: still on; None: never any)."""
        points = self._breakpoints()
        on = [t for t in points if self.force_at(t) > 0]
        if not on:
            return None
        return next((p for p in points if p > on[-1]), math.inf)

    def calls(self, op: str | None = None) -> list[dict]:
        return [e for e in self.events if e["ev"] == "call" and (op is None or e["op"] == op)]

    def commands(self) -> list[str]:
        return [e["detail"] for e in self.calls("SendForceFeedbackCommand")]

    def plays(self) -> list[dict]:
        return [e for e in self.events if e["ev"] == "play" and e["level"] > 0]

    # ---- assertions

    def assert_force_never_above(self, fraction: float) -> None:
        peak = self.peak_force()
        assert peak <= fraction + EPS, f"force reached {peak:.4f} of peak, limit {fraction:.4f}"
        for e in self.plays():
            gain = self.gain_at(e["start"])
            assert gain <= fraction * NOMINAL + EPS, (
                f"effect {e['effect']} ({e['kind']}) started at t={e['start']:.4f} with device gain {gain}, "
                f"above {fraction:.4f}")

    def assert_gain_set_before_first_start(self, cap: float | None = None) -> None:
        plays = self.plays()
        if not plays:
            return
        first = plays[0]["t"]
        gains = [t for t, _ in self.gain_timeline() if t <= first]
        assert gains, f"first effect started at t={first:.4f} before any device gain was set"
        if cap is not None:
            gain = self.gain_at(first)
            assert gain <= cap * NOMINAL + EPS, f"device gain {gain} at the first effect, cap {cap:.4f}"

    def assert_all_stopped(self, by: float | None = None, *, after: float | None = None,
                           within_ms: float | None = None) -> None:
        """No force at or after `by` (or `after` + `within_ms`)."""
        if by is None:
            if after is None or within_ms is None:
                raise TypeError("give `by`, or `after` and `within_ms`")
            by = after + within_ms / 1000
        for t in [by, *(p for p in self._breakpoints() if p >= by)]:
            force = self.force_at(t)
            assert force == 0, (
                f"force {force:.4f} still on at t={t:.4f} (deadline {by:.4f}): "
                + ", ".join(f"{iv.kind}#{iv.effect} until {iv.end}" for iv in self.playing(t)))

    def assert_quiet_after(self, t: float) -> None:
        """Nothing starts and the device gain never rises after `t`."""
        late = [e for e in self.plays() if e["t"] > t]
        assert not late, f"{len(late)} effect start(s) after t={t:.4f}, first {late[0]}"
        before = self.gain_at(t)
        for when, gain in self.gain_timeline():
            if when > t:
                assert gain <= before, f"device gain rose to {gain} at t={when:.4f}, after t={t:.4f}"
                before = gain

    def assert_no_force_writes(self) -> None:
        """Read-only use: no exclusive acquire, property write, effect or command."""
        calls = self.calls()
        assert not calls, "force-related calls on a read-only path: " + ", ".join(
            f"{c['op']}({c.get('detail', '')})" for c in calls[:10])
