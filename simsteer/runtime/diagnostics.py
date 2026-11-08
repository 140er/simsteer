"""Bounded numeric flight recorder and observational loop timings.

No images or free-form fault text are retained.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from simsteer.core.finite import NonFiniteError

STAGES = ("capture_wait", "warp", "model", "control", "draw")
SLOW_MODEL_BANNER = "Model below 20 Hz: temporal state slowed — lower graphics or model size"
# Stable numeric event codes keep exception/device text out of incident files.
USER_TAKEOVER, OUTPUT_DIRECTION, INVALID_DATA, AUTO_DISENGAGE, EXECUTOR_FAULT, FAULT, LANE_EXCURSION = range(1, 8)
STOP_REASONS = {
    "Capture unavailable": 8,
    "Telemetry unavailable": 9,
    "Output device changed": 10,
    "Steering calibration required": 11,
    "Restarting": 12,
}


def number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def contained(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        if not self.disabled:
            try:
                return method(self, *args, **kwargs)
            except Exception:
                self._failed()
    return call


class FlightRing:
    def __init__(self, seconds=15., max_samples=600):
        if seconds <= 0 or max_samples < 1:
            raise ValueError("positive recorder bounds required")
        self.seconds = seconds
        self.samples = deque(maxlen=max_samples)

    def append(self, sample):
        t = sample["t"]
        if self.samples and t < self.samples[-1]["t"]:
            self.samples.clear()
        self.samples.append(sample)
        while self.samples and t - self.samples[0]["t"] > self.seconds:
            self.samples.popleft()


class TimingStats:
    """Five-second rolling means and model cadence, with rate hysteresis."""

    def __init__(self, window_s=5., max_samples=600, sustain_s=3.):
        self.window_s, self.max_samples, self.sustain_s = window_s, max_samples, sustain_s
        self.samples = deque()
        self.totals = [0.] * len(STAGES)
        self.inferences = 0
        self.started = None
        self.low_since = self.high_since = None
        self.hz = 20.
        self.slow = False

    def update(self, t, values, inferred):
        if self.started is None:
            self.started = t
        self.samples.append((t, values, int(inferred)))
        self.inferences += int(inferred)
        for i, value in enumerate(values):
            self.totals[i] += value
        while (len(self.samples) > self.max_samples
               or t - self.samples[0][0] > self.window_s):
            _, old, count = self.samples.popleft()
            self.inferences -= count
            for i, value in enumerate(old):
                self.totals[i] -= value
        elapsed = t - self.samples[0][0]
        if elapsed > 0:
            self.hz = (self.inferences - self.samples[0][2]) / elapsed
        if t - self.started < self.window_s:
            return
        if not self.slow:
            self.high_since = None
            if self.hz < 18.:
                if self.low_since is None:
                    self.low_since = t
            else:
                self.low_since = None
            if self.low_since is not None and t - self.low_since >= self.sustain_s:
                self.slow = True
        else:
            self.low_since = None
            if self.hz >= 19.:
                if self.high_since is None:
                    self.high_since = t
            else:
                self.high_since = None
            if self.high_since is not None and t - self.high_since >= self.sustain_s:
                self.slow = False

    def summary(self):
        n = len(self.samples)
        return {stage + "_mean_ms": round(max(0., total / max(n, 1)), 3)
                for stage, total in zip(STAGES, self.totals)}


class RunDiagnostics:
    def __init__(self, root: Path, *, timing_log=False, dump_interval_s=10.):
        self.disabled = False
        self.fields = {}
        self.hud_text = ""
        self.slow = False
        self._thread = None
        self._reported = False
        try:
            self.root = root / "logs" / "incidents"
            self.timing_log = timing_log
            self.dump_interval_s = dump_interval_s
            self.ring = FlightRing()
            self.timing = TimingStats()
            self._queue = queue.Queue(maxsize=1)
            self._pending = set()
            self._last_dump = float("-inf")
            self._manual = self._excursion = False
            self._last_intervention = 0.
            self._control = {}
            self._sample = None
        except Exception:
            self._failed()

    def _failed(self):
        self.disabled = True
        self.fields = {}
        self.hud_text = ""
        self.slow = False
        if not self._reported:
            self._reported = True
            try:
                logging.getLogger(__name__).warning("Session diagnostics unavailable")
            except Exception:
                pass

    @contained
    def begin_tick(self):
        self._control = {}
        self._sample = None
        self.fields = {}

    @contained
    def incident(self, reason="", *, user_stop=False):
        if user_stop:
            code = USER_TAKEOVER
        elif isinstance(reason, NonFiniteError):
            code = INVALID_DATA
        elif reason == "Output direction changed":
            code = OUTPUT_DIRECTION
        elif reason in ("Steering executor failed", "Steering plan expired"):
            code = EXECUTOR_FAULT
        elif isinstance(reason, str) and reason.startswith("frame rate dropped"):
            code = AUTO_DISENGAGE
        elif reason:
            code = STOP_REASONS.get(reason, FAULT)
        else:
            return
        self._pending.add(code)

    @contained
    def control(self, lateral, probe_axis, speed_scale=1.0):
        # Called before learning or a handback can reset the controller.
        self._control = {
            "raw_k": number(lateral.last_desired_k_raw),
            "steer_authority": number(lateral.last_authority),
            "understeer_ff": number(getattr(lateral, "last_understeer_ff", 1.0)),
            "speed_scale": number(speed_scale),
            "feedforward_axis": number(lateral.last_ff),
            "trim_axis": number(lateral.axis_trim_state),
            "probe_axis": number(probe_axis),
        }

    @contained
    def observe(self, *, t, decoded, speed, yaw, wheel, observed_axis, user_axis,
                final_axis, executor_axis, engaged, manual, intervention, fresh,
                calibration, camera_pose, frame_fov, output_sign, counters, fault):
        def vector(name, count):
            values = getattr(decoded, name, None)
            return ([number(x) for x in values[:count]] if values is not None else None)

        probs = vector("lane_lines_prob", 4)
        lanes = getattr(decoded, "lane_lines", None)
        offset = width = None
        if (lanes is not None and probs and len(probs) >= 3
                and probs[1] is not None and probs[2] is not None
                and min(probs[1:3]) >= .5):
            left, right = float(lanes[1, 0, 0]), float(lanes[2, 0, 0])
            offset, width = number(-(left + right) / 2), number(abs(right - left))
        plan = decoded.plan
        self.fields = {
            "desire_state": vector("desire_state", 8),
            "lane_line_probs": probs,
            # The pinned vision model has no road-edge probability head.
            "road_edge_probs": vector("road_edges_prob", 2),
            "plan_speed_mps": number(plan[0, 3]),
            "plan_end_speed_mps": number(plan[-1, 3]),
            "pose_speed_mps": number(decoded.pose[0]) if decoded.pose is not None else None,
            "pose_vertical_mps": number(decoded.pose[2]) if decoded.pose is not None and len(decoded.pose) > 2 else None,
            "model_height_m": number(decoded.road_transform[2]) if decoded.road_transform is not None else None,
            "lane_offset_m": offset, "lane_width_m": width,
            "plan_curvature": number(plan[0, 14] / max(float(plan[0, 3]), 1.)),
            "raw_k": None, "steer_authority": None, "feedforward_axis": None,
            "trim_axis": None, "probe_axis": None,
            **self._control,
            "final_axis": number(final_axis), "executor_axis": number(executor_axis),
            "output_sign": number(output_sign),
            "model_fresh": bool(fresh),
            "calibration_pitch_deg": number(camera_pose["pitch_deg"]),
            "calibration_height_m": number(camera_pose["height_m"]),
            "camera_fy_px": number(calibration.image_w / (2 * math.tan(math.radians(frame_fov) / 2))),
            "camera_cy_px": number(calibration.cy),
        }
        self._sample = {
            "t": number(t), "speed_mps": number(speed), "yaw_rad_s": number(yaw),
            "wheel_rad": number(wheel), "observed_axis": number(observed_axis),
            "user_axis": number(user_axis), "engaged": bool(engaged), "fault": bool(fault),
            **self.fields,
        }
        self._counters = counters
        self._fresh = fresh
        if (manual and not self._manual) or intervention > self._last_intervention:
            self._pending.add(USER_TAKEOVER)
        self._manual = manual
        self._last_intervention = intervention
        excursion = bool(fresh and offset is not None and abs(offset) > 1.5)
        if excursion and not self._excursion:
            self._pending.add(LANE_EXCURSION)
        self._excursion = excursion

    def finish(self, end, row=None, log_f=None):
        # Logging still works after diagnostics has disabled itself.
        if not self.disabled:
            try:
                self._finish(end)
            except Exception:
                self._failed()
        if log_f is not None and row is not None:
            try:
                row.update(self.fields)
                log_f.write(json.dumps(row) + "\n")
            except Exception:
                self._failed()

    def _finish(self, end):
        if self._sample is None:
            return
        grab, captured, warp, model, control, draw, extra_model = self._counters
        values = tuple(max(0., x * 1000.) for x in (
            captured - grab, model - warp, control - model + extra_model,
            draw - control - extra_model, end - draw))
        t = self._sample["t"]
        self.timing.update(t, values, self._fresh)
        self.slow = self.timing.slow
        stats = self.timing.summary()
        self.hud_text = (f"model {self.timing.hz:.1f} Hz | " + "  ".join(
            f"{name} {stats[name + '_mean_ms']:.1f}" for name in STAGES) + " ms")
        timing = {name + "_ms": round(value, 3) for name, value in zip(STAGES, values)}
        timing.update(effective_model_hz=round(self.timing.hz, 3),
                      model_under_rate=self.slow, timing_stats=stats)
        self._sample.update(timing)
        if self.timing_log:
            # Preserve the legacy cap/warp/model/ctrl/frame timing columns.
            self.fields.update(timing)
        self.ring.append(self._sample)
        self._submit(t)

    def _submit(self, t):
        codes = sorted(self._pending)
        self._pending.clear()
        if not codes or t - self._last_dump < self.dump_interval_s:
            return
        payload = {"schema": 1, "events": [{"t": t, "codes": codes}],
                   "samples": list(self.ring.samples)}
        if self._thread is None:
            self._thread = threading.Thread(target=self._write, name="session-diagnostics", daemon=True)
            self._thread.start()
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            return
        self._last_dump = t

    def _write(self):
        while True:
            payload = self._queue.get()
            try:
                if payload is None:
                    return
                if not self.disabled:
                    self.root.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    with (self.root / (stamp + ".json")).open("x", encoding="utf-8") as f:
                        json.dump(payload, f, allow_nan=False, separators=(",", ":"))
            except Exception:
                self._failed()
            finally:
                self._queue.task_done()

    def close(self):
        try:
            if not self.disabled and self.ring.samples:
                self._submit(self.ring.samples[-1]["t"])
            if self._thread is not None:
                self._queue.put(None)
                self._thread.join()
        except Exception:
            self._failed()
