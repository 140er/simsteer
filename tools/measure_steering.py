"""Deterministic 20 Hz plan step/ramp, with optional historical controllers."""
from __future__ import annotations

import argparse
import importlib
import io
import json
import subprocess
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

from simsteer.core.constants import IDX_N, PLAN_WIDTH, T_IDXS
from simsteer.core.learners.liveparams import LiveParams


def measure(ref: str | None, speed: float) -> dict:
    # Load the actual historical implementation without a second checkout.
    for name in ("postprocess", "control"):
        module = importlib.import_module(f"simsteer.core.{name}")
        if ref:
            source = subprocess.check_output(
                ["git", "show", f"{ref}:simsteer/core/{name}.py"], text=True)
            exec(compile(source, name, "exec"), module.__dict__)
    from simsteer.core.control import ControllerConfig, LateralController

    cfg = ControllerConfig(wheel_trim_enabled=False)
    with redirect_stdout(io.StringIO()):
        lp = LiveParams(clock=lambda: 0.)
    lp.x[:] = [4., 0., 0.]
    controller = LateralController(cfg, lp)
    executor = None
    if not ref:
        from simsteer.core.steering import SteeringExecutor, SteeringTarget
        executor = SteeringExecutor()
    outputs = []
    for tick in range(400):
        now = tick * .01
        if tick % 5 == 0:
            k = 0. if now < .5 else .006 if now < 1.5 else max(-.006, .006 - .006 * (now - 1.5))
            plan = np.zeros((IDX_N, PLAN_WIDTH), dtype=np.float32)
            plan[:, 3] = speed
            plan[:, 11] = np.asarray(T_IDXS) * speed * k
            plan[:, 14] = speed * k
            axis = controller.compute(SimpleNamespace(plan=plan, desire_state=np.zeros(8)),
                                      speed, dt=.05)
            if executor:
                executor.submit(SteeringTarget(axis, speed, 4., 0., cfg.wheelbase_m), now)
            else:
                outputs.append((now, axis))
        if executor:
            axis = executor.tick(now)
            if axis is not None:
                outputs.append((now, axis))
    steps, jerks = [], []
    for (t0, a0), (t1, a1) in zip(outputs, outputs[1:]):
        steps.append(abs(a1 - a0))
        k0, k1 = np.tan(np.array([a0, a1]) / 4.) / cfg.wheelbase_m
        jerks.append(speed * speed * abs(k1 - k0) / (t1 - t0))
    return dict(speed_mps=speed, output_hz=len(outputs) / 4.,
                max_axis_step=max(steps), max_lateral_jerk_mps3=max(jerks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Read control and postprocess from this git revision")
    args = parser.parse_args()
    print(json.dumps([measure(args.ref, v) for v in (10., 20., 30.)], indent=2))
