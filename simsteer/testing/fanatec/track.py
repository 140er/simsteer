"""Scripted and recorded wheel input as a function of time.

A `Track` is a recipe: builder steps that run when the wheel starts
playing it, from the wheel's state at that moment (so `flick(steer_deg=6)`
means "6 degrees from wherever the rim is"). Times are seconds from the
start of the track. Steering and pedals interpolate linearly between
keyframes; buttons, the POV hat and the plug channels step.

    Track().hold(1).ramp(steer_deg=12.5, over=0.08).hold(0.5)
    Track().noise(seconds=60, steer_deg=2, seed=7)
    Track().pedal("brake", 0.12, hold=0.06)
    Track().unplug(for_s=1.5)
    Track.load("drive.json")
    Track.from_simsteer_jsonl("run.jsonl")   # rim_axis / phys_throttle / phys_brake
"""

from __future__ import annotations

import bisect
import json
import math
import random
from pathlib import Path

LINEAR = ("steer_deg", "throttle", "brake", "clutch")
STEP = ("plugged", "pedals_plugged", "pov")
PEDALS = ("throttle", "brake", "clutch")


def _channel_ok(name: str) -> bool:
    return name in LINEAR or name in STEP or name.startswith("button:")


class Compiled:
    """Concrete keyframes: channel -> sorted [(t, value)]."""

    def __init__(self, keys: dict[str, list[tuple[float, object]]]) -> None:
        self.keys = {ch: sorted(ks, key=lambda k: k[0]) for ch, ks in keys.items() if ks}
        self._times = {ch: [k[0] for k in ks] for ch, ks in self.keys.items()}
        self.duration = max((ks[-1][0] for ks in self.keys.values()), default=0.0)

    def value(self, channel: str, t: float):
        keys = self.keys.get(channel)
        if not keys:
            return None
        i = bisect.bisect_right(self._times[channel], t)
        if i == 0:
            return None  # the track has not touched this channel yet
        t0, v0 = keys[i - 1]
        if channel not in LINEAR or i == len(keys):
            return v0
        t1, v1 = keys[i]
        if t1 <= t0:
            return v1
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    def values(self, t: float) -> dict:
        out = {}
        for ch in self.keys:
            v = self.value(ch, t)
            if v is not None:
                out[ch] = v
        return out

    def transitions(self, channel: str, t0: float, t1: float) -> list[tuple[float, object]]:
        """Step keyframes in (t0, t1]."""
        return [(t, v) for t, v in self.keys.get(channel, ()) if t0 < t <= t1]


class Track:
    def __init__(self) -> None:
        self._steps: list[tuple[str, tuple, dict]] = []

    # ---- builders (each returns self)

    def _add(self, op: str, *args, **kwargs) -> Track:
        for name in kwargs:
            if op in ("set", "ramp", "noise") and not _channel_ok(name):
                raise ValueError(f"unknown track channel {name!r}")
        self._steps.append((op, args, kwargs))
        return self

    def set(self, **values) -> Track:
        """Jump to these values now."""
        return self._add("set", **values)

    def hold(self, seconds: float) -> Track:
        return self._add("hold", seconds)

    def ramp(self, over: float, **values) -> Track:
        """Move linearly to these values over `over` seconds."""
        return self._add("ramp", over, **values)

    def flick(self, steer_deg: float, rate_dps: float) -> Track:
        """Turn the rim by `steer_deg` (signed) at `rate_dps` degrees/second."""
        return self._add("flick", steer_deg, rate_dps)

    def noise(self, seconds: float, seed: int = 0, step: float = 0.01, **amplitudes) -> Track:
        """Random wander of +/- amplitude around the current values (hands
        resting on the rim); ends back on the current values."""
        return self._add("noise", seconds, seed, step, **amplitudes)

    def pedal(self, name: str, value: float, hold: float | None = None, over: float = 0.0) -> Track:
        """Press pedal `name` to `value`; with `hold`, release it after."""
        if name not in PEDALS:
            raise ValueError(f"unknown pedal {name!r}")
        self.ramp(over, **{name: value})
        if hold is not None:
            self.hold(hold).ramp(over, **{name: 0.0})
        return self

    def press(self, button: int, hold: float = 0.1) -> Track:
        return self.set(**{f"button:{button}": True}).hold(hold).set(**{f"button:{button}": False})

    def unplug(self, for_s: float | None = None) -> Track:
        """Pull the base's USB; with `for_s`, plug it back in after."""
        self.set(plugged=False)
        return self.hold(for_s).set(plugged=True) if for_s is not None else self

    def unplug_pedals(self, for_s: float | None = None) -> Track:
        self.set(pedals_plugged=False)
        return self.hold(for_s).set(pedals_plugged=True) if for_s is not None else self

    # ---- compiling

    def compile(self, start: dict | None = None) -> Compiled:
        """Run the recipe from the wheel state `start`."""
        cur = {"steer_deg": 0.0, "throttle": 0.0, "brake": 0.0, "clutch": 0.0, **(start or {})}
        keys: dict[str, list[tuple[float, object]]] = {}
        t = 0.0

        def key(ch, at, v):
            keys.setdefault(ch, []).append((at, v))

        def jump(values):
            for ch, v in values.items():
                if ch in LINEAR and ch in cur:
                    key(ch, t, cur[ch])
                key(ch, t, v)
                cur[ch] = v

        for op, args, kw in self._steps:
            if op == "keys":  # loaded keyframes, absolute
                for at, values in args[0]:
                    for ch, v in values.items():
                        key(ch, at, v)
                        cur[ch] = v
                    t = max(t, at)
            elif op == "set":
                jump(kw)
            elif op == "hold":
                for ch in LINEAR:
                    if ch in keys:
                        key(ch, t + args[0], cur[ch])
                t += args[0]
            elif op == "ramp":
                over = args[0]
                if over <= 0:
                    jump(kw)
                    continue
                for ch, v in kw.items():
                    if ch not in LINEAR:
                        raise ValueError(f"cannot ramp step channel {ch!r}")
                    key(ch, t, cur[ch])
                    key(ch, t + over, v)
                    cur[ch] = v
                t += over
            elif op == "flick":
                deg, rate = args
                over = abs(deg) / rate
                key("steer_deg", t, cur["steer_deg"])
                cur["steer_deg"] += deg
                key("steer_deg", t + over, cur["steer_deg"])
                t += over
            elif op == "noise":
                seconds, seed, step = args
                rng = random.Random(seed)
                base = {ch: cur[ch] for ch in kw}
                n = max(1, math.ceil(seconds / step))
                for ch in kw:
                    key(ch, t, base[ch])
                for i in range(1, n):
                    for ch, amp in kw.items():
                        key(ch, t + i * step, base[ch] + amp * rng.uniform(-1, 1))
                for ch in kw:
                    key(ch, t + seconds, base[ch])
                t += seconds
        return Compiled(keys)

    # ---- files

    @classmethod
    def from_keyframes(cls, frames: list[tuple[float, dict]]) -> Track:
        for _, values in frames:
            for ch in values:
                if not _channel_ok(ch):
                    raise ValueError(f"unknown track channel {ch!r}")
        track = cls()
        track._steps.append(("keys", (sorted(frames, key=lambda f: f[0]),), {}))
        return track

    @classmethod
    def load(cls, path: str | Path) -> Track:
        """JSON `{"keyframes": [{"t": 0, "steer_deg": 0, ...}, ...]}` or JSONL,
        one keyframe per line."""
        text = Path(path).read_text(encoding="utf-8")
        if text.lstrip().startswith("{") and '"keyframes"' in text[:200]:
            rows = json.loads(text)["keyframes"]
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return cls.from_keyframes([(float(r.pop("t")), r) for r in rows])

    @classmethod
    def from_simsteer_jsonl(cls, path: str | Path, rim_range_deg: float = 900) -> Track:
        """Replay a SimSteer `--log-jsonl` run that logged the physical wheel
        (`rim_axis`, `phys_throttle`, `phys_brake`)."""
        frames = []
        t0 = None
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "t" not in row:
                continue
            values = {}
            if row.get("rim_axis") is not None:
                values["steer_deg"] = float(row["rim_axis"]) * rim_range_deg / 2
            for src, ch in (("phys_throttle", "throttle"), ("phys_brake", "brake")):
                if row.get(src) is not None:
                    values[ch] = float(row[src])
            if values:
                t0 = row["t"] if t0 is None else t0
                frames.append((float(row["t"]) - t0, values))
        if not frames:
            raise ValueError(f"{path}: no rim_axis / phys_throttle / phys_brake frames")
        return cls.from_keyframes(frames)

    def save(self, path: str | Path, start: dict | None = None) -> None:
        # One row per keyframe: a jump is two keyframes at the same time.
        rows = sorted(({"t": t, ch: v} for ch, ks in self.compile(start).keys.items() for t, v in ks),
                      key=lambda r: r["t"])
        Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
