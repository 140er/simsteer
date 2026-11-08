"""Deterministic longitudinal plant; run with python -m tools.speed_scale_bench."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from simsteer.core.calibration import Calibration
from simsteer.core.control import ControllerConfig, LongitudinalController
from simsteer.core.postprocess import decode
from simsteer.runtime.speedscale import SpeedScale


RATIO = 0.88
TARGET_MPS = 25.0
DT = 0.05


def decoded_plan(v, target, *, fraction=1.0, accel=0.0):
    decoded = decode(np.zeros(1576, np.float32), np.zeros(1000, np.float32))
    decoded.pose[0] = RATIO * v
    decoded.plan[:, 3] = RATIO * target * fraction
    decoded.plan[:, 6] = RATIO * accel
    decoded.lead_prob[:] = 0.0
    return decoded


def benchmark(*, corrected=True):
    with TemporaryDirectory() as root:
        return _benchmark(corrected=corrected, path=Path(root) / "longitudinal.json")


def _benchmark(*, corrected, path):
    cfg = ControllerConfig(accel_deadband_mps2=0.0,
                           speed_scale_max=1.3 if corrected else 1.0)
    controller = LongitudinalController(cfg)
    learner = SpeedScale("speed-benchmark", cfg, path=path)
    flow = SimpleNamespace(calibrated=True, camera_learning=False,
                           calib=Calibration(), model_identity={})
    v = 20.0
    telemetry = SimpleNamespace(available=True, speed_mps=lambda: v)
    for _ in range(3600):
        # A plan to close half the remaining gap to the road's target speed.
        # Both this desired velocity and current pose share the 0.88 bias.
        decoded = decoded_plan(v, v + 0.5 * (TARGET_MPS - v))
        controller.speed_scale = learner.update(flow, telemetry, float(decoded.pose[0]), DT)
        throttle, brake = controller.compute(decoded, v)
        v = max(0.0, v + DT * (throttle * cfg.max_accel_mps2
                                - brake * cfg.max_decel_mps2))
    cruise = v
    # Freeze the calibrated scale; evaluate actual model slowdowns and leads.
    decoded = decoded_plan(v, v, fraction=0.5, accel=-1.0)
    controller.compute(decoded, v)
    slowdown_target = controller.last_v_target
    slowdown_accel = controller.last_a_target
    slowdown_brake = controller.last_brake
    decoded = decoded_plan(v, v)
    decoded.lead_prob[0] = 0.99
    decoded.leads[0, 0, :3] = [30.0, 0.0, 15.0]
    controller.compute(decoded, v)
    lead_brake = controller.last_brake
    decoded.leads[0, 0, 0] = 5.0
    controller.compute(decoded, v)
    return {
        "target_mps": TARGET_MPS, "cruise_mps": round(cruise, 6),
        "scale": round(controller.speed_scale, 6),
        "half_plan_target_mps": round(slowdown_target, 6),
        "half_plan_fraction": round(slowdown_target / v, 6),
        "slowdown_accel_mps2": round(slowdown_accel, 6),
        "slowdown_brake": round(slowdown_brake, 6),
        "lead_brake": round(lead_brake, 6),
        "aeb_brake": controller.last_brake,
    }


if __name__ == "__main__":
    print(json.dumps({"before": benchmark(corrected=False),
                      "after": benchmark(corrected=True)}, indent=2))
