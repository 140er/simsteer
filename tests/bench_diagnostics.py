"""Run with the dev environment's Python; use the same file on both revisions."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics_harness import drive


def deep_size(value, seen=None):
    from collections import deque

    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, deque)):
        size += sum(deep_size(x, seen) for x in value)
    return size


def component_bench(frames):
    from simsteer.core.calibration import Calibration
    from simsteer.runtime.diagnostics import RunDiagnostics
    from simsteer.runtime.faults import neutral_decoded

    before, after = [], []
    decoded = neutral_decoded()
    decoded.pose[0] = 22.
    lateral = SimpleNamespace(last_desired_k_raw=.001, last_authority=1.,
                              last_ff=.05, axis_trim_state=0.)
    inputs = dict(decoded=decoded, speed=22., yaw=0., wheel=0., observed_axis=0.,
                  user_axis=0., final_axis=.05, executor_axis=.05, engaged=False,
                  manual=False, intervention=0., fresh=True, calibration=Calibration(),
                  camera_pose={"pitch_deg": 0., "height_m": 1.2}, frame_fov=90.,
                  output_sign=-1, counters=(0., .001, .001, .002, .003, .004, 0.), fault=False)

    def tick(diagnostics):
        if diagnostics is not None:
            diagnostics.begin_tick()
            diagnostics.control(lateral, 0.)
            diagnostics.observe(**inputs)
            diagnostics.finish(.005)

    with tempfile.TemporaryDirectory() as directory:
        diagnostics = RunDiagnostics(Path(directory))
        for i in range(frames):
            inputs["t"] = i * .01
            order = ((None, before), (diagnostics, after))
            for instance, samples in order if i % 2 else reversed(order):
                started = time.perf_counter_ns()
                tick(instance)
                samples.append((time.perf_counter_ns() - started) / 1000.)
        diagnostics.close()
    before, after = np.array(before[100:]), np.array(after[100:])
    return dict(before_mean_us=float(np.mean(before)), before_p99_us=float(np.percentile(before, 99)),
                after_mean_us=float(np.mean(after)), after_p99_us=float(np.percentile(after, 99)),
                added_mean_us=float(np.mean(after - before)),
                added_p99_us=float(np.percentile(after - before, 99)),
                ring_samples=len(diagnostics.ring.samples),
                ring_max_samples=diagnostics.ring.samples.maxlen,
                ring_bytes=deep_size(diagnostics.ring.samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--period", type=float, default=.05)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--component", action="store_true",
                        help="paired diagnostics calls versus the base's absence of those calls")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        patch.setenv("SIMSTEER_DATA_DIR", directory)
        if args.component:
            print(json.dumps(component_bench(args.frames)))
            return
        state = drive(patch, Path(directory), frames=args.frames, log=args.log,
                      period=args.period, instrument=args.instrument)
    samples = np.array(state.durations[100:]) / 1000.
    cpu = np.array(state.cpu_durations[100:]) / 1000.
    result = dict(ticks=len(samples), mean_us=float(np.mean(samples)),
                  p99_us=float(np.percentile(samples, 99)),
                  cpu_mean_us=float(np.mean(cpu)), cpu_p99_us=float(np.percentile(cpu, 99)))
    if hasattr(state, "diagnostics"):
        ring = state.diagnostics.ring
        result.update(ring_samples=len(ring.samples), ring_max_samples=ring.samples.maxlen,
                      ring_bytes=deep_size(ring.samples))
    if hasattr(state, "diagnostic_costs"):
        costs = np.array(state.diagnostic_costs[100:]) / 1000.
        result.update(diagnostics_mean_us=float(np.mean(costs)),
                      diagnostics_p99_us=float(np.percentile(costs, 99)))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
