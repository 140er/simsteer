"""Residual vision-speed calibration, downstream of FOV and camera learning.

Ideas ported from SpysyWeeb/Beamng-onnx (MIT), commits 24f4c23, 02c57b7.
The decoded pose and plan stay in model units; only longitudinal targets
use this conversion. A measured FOV may already remove the entire bias.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from simsteer.paths import state_path
from simsteer.runtime.telemetry import vehicle_context
from simsteer.state import (
    atomic_save_json, bounded_number, read_json_object, snapshot_exists, state_warning,
)

if TYPE_CHECKING:
    from simsteer.calib.flow import CalibFlow
    from simsteer.core.control import ControllerConfig
    from simsteer.runtime.telemetry import Telemetry

SPEED_SCALE_LIMIT = 1.4
SMOOTHING_S = 2.0
CONVERGED_S = 10.0


class SpeedScale:
    """Warm-start per game/vehicle; save only a stable, bounded estimate."""

    def __init__(self, game: str, cfg: ControllerConfig, *,
                 path: Path | None = None) -> None:
        self.cfg = cfg
        self.path = path if path is not None else state_path("longitudinal", game)
        self.value = 1.0
        self._key = None
        self._context = None
        self._stable_s = 0.0
        self._converged: float | None = None
        self._entries = {}
        if snapshot_exists(self.path):
            try:
                data = read_json_object(self.path)
                if data.get("version") != 1 or not isinstance(data.get("vehicles"), dict):
                    raise ValueError("invalid longitudinal calibration")
                self._entries = data["vehicles"]
            except (OSError, ValueError, TypeError):
                state_warning(self.path, "unreadable; using defaults")

    def save(self) -> None:
        if self._key is None or self._converged is None:
            return
        ceiling = min(SPEED_SCALE_LIMIT, max(1.0, self.cfg.speed_scale_max))
        if ceiling <= 1.0:
            return
        self._entries[self._key] = {
            "context": self._context, "scale": min(ceiling, max(1.0, self._converged)),
        }
        atomic_save_json(self.path, {"version": 1, "vehicles": self._entries})

    def update(self, flow: CalibFlow, telemetry: Telemetry,
               pose_v: float | None, dt: float, *, healthy: bool = True) -> float:
        """Return the active scale. Never use fallback speed as telemetry.

        Freeze the estimate and suspend its application during calibration.
        The context prevents a pre-FOV estimate being reused after the warp
        changes. Steering settings do not participate in this calibration.
        """
        ceiling = min(SPEED_SCALE_LIMIT, max(1.0, self.cfg.speed_scale_max))
        if not healthy or not flow.calibrated or flow.camera_learning or ceiling <= 1.0:
            self._stable_s = 0.0
            return 1.0
        key = vehicle_context(telemetry).get("vehicle") or ""
        camera = asdict(flow.calib)
        camera.pop("lateral_sign")
        context = hashlib.sha256(json.dumps(
            {"camera": camera, "model": flow.model_identity},
            sort_keys=True).encode()).hexdigest()
        if (key, context) != (self._key, self._context):
            self.save()
            self._key, self._context = key, context
            self.value, self._stable_s, self._converged = 1.0, 0.0, None
            entry = self._entries.get(key, {})
            try:
                if entry.get("context") == context:
                    self.value = min(ceiling, bounded_number(entry["scale"], 1.0,
                                                            SPEED_SCALE_LIMIT))
            except (KeyError, ValueError, TypeError, AttributeError):
                state_warning(self.path, "invalid speed scale; using defaults")
        self.value = min(ceiling, max(1.0, self.value))
        v = telemetry.speed_mps() if telemetry.available else None
        if (v is None or pose_v is None or not all(math.isfinite(x) for x in (v, pose_v, dt))
                or v <= 5.0 or pose_v <= 3.0 or not 0 < dt <= 0.25):
            self._stable_s = 0.0
            return self.value
        raw = max(1.0, v / pose_v)
        # Reject rather than saturate a bad pose sample into a learned ceiling.
        if raw > ceiling:
            self._stable_s = 0.0
            return self.value
        self.value += (1.0 - math.exp(-dt / SMOOTHING_S)) * (raw - self.value)
        self._stable_s = (self._stable_s + dt if abs(raw - self.value) <= 0.01
                          else 0.0)
        if self._stable_s >= CONVERGED_S:
            # Retain the last stable estimate through parking and dropouts.
            self._converged = self.value
        return self.value
