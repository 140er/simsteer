"""Measure timestamp queues on a jittered 20 Hz capture-reader grid.

The consumer grabs the latest sample at 8–20 Hz, with ±5% loop jitter.
The reader wakes 0–1.5 ms late. No camera, game or model is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simsteer.core.calibration import Calibration
from simsteer.core.preprocess import FrameQueue

RATES = (20, 18, 15, 12, 10, 9, 8)


def capture_timestamps(hz: int, rng: random.Random):
    """Reproduce a phase-locked reader and latest-sample consumer for 60 s."""
    ticks = [k * .05 + rng.uniform(0, .0015) for k in range(2000)]
    t, last_index = 0., -1
    while t < 60:
        t += 1 / hz * rng.uniform(.95, 1.05)
        index = int(t / .05)
        if index <= last_index:
            index = last_index + 1
            t = ticks[index]
        last_index = index
        yield ticks[index]


def measure():
    cv2.setNumThreads(1)
    rng = random.Random(0)
    frame = np.zeros((540, 960, 3), np.uint8)
    results = []
    for hz in RATES:
        queue = FrameQueue()
        ready = pair_ok = frames = 0
        for timestamp in capture_timestamps(hz, rng):
            queue.push(frame, Calibration(), timestamp=timestamp)
            frames += 1
            ready += queue.ready
            # The reviewed adapter has no flag. Derive its baseline pair
            # eligibility using the same approved 200 ± 60 ms contract.
            pair_ok += getattr(queue, "pair_ok", queue.ready and abs(queue.span_s - .2) <= .060001)
        row = {"hz": hz, "frames": frames, "resets": queue.resets,
               "ready_pct": 100 * ready / frames,
               "pair_ok_pct": 100 * pair_ok / frames}
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "pair_ok_baseline_derived": not hasattr(FrameQueue(), "pair_ok"),
              "measurements": measure()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
