"""Run the full pipeline on a synthetic frame to verify everything wires up.

Generates a fake 1280x720 BGR frame with a road-ish gradient, pushes it
through preprocess + model + postprocess a few times, and prints timings
and decoded shapes. Doesn't require ETS2 or any capture device.

    python tools/smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pilot.model import DrivingModel
from pilot.postprocess import decode
from pilot.preprocess import FrameQueue


def fake_frame(w: int = 1280, h: int = 720, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sky = np.linspace(180, 80, h // 2, dtype=np.uint8)[:, None]
    road = np.linspace(60, 110, h - h // 2, dtype=np.uint8)[:, None]
    col = np.concatenate([sky, road], axis=0)         # (h, 1)
    bgr = np.repeat(col[:, :, None], 3, axis=2)       # (h, 1, 3) -> broadcast
    bgr = np.repeat(bgr, w, axis=1)
    bgr = np.clip(bgr.astype(np.int32) + rng.integers(-10, 10, bgr.shape), 0, 255).astype(np.uint8)
    # crude lane stripes
    cx = w // 2
    for offset, width in [(-220, 4), (220, 4), (-440, 6), (440, 6)]:
        x = cx + offset
        bgr[h // 2:, x:x + width] = 240
    return bgr


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="force CPU EP")
    args = ap.parse_args()

    print("loading model...")
    providers = ["CPUExecutionProvider"] if args.cpu else None
    model = DrivingModel(providers=providers)
    print(f"  vision provider: {model.active_provider}")
    print(f"  policy provider: {model.policy_provider}")

    fq = FrameQueue()
    n_steps = 10
    print(f"running {n_steps} steps on synthetic frame...")
    times = []
    for i in range(n_steps):
        bgr = fake_frame(seed=i)
        img_narrow, img_wide = fq.push(bgr)
        t0 = time.perf_counter()
        vision_out, policy_out = model.step(img_narrow, img_wide)
        times.append((time.perf_counter() - t0) * 1000)
    decoded = decode(vision_out, policy_out)

    print(f"  per-step ms: min={min(times):.1f} avg={sum(times)/len(times):.1f} max={max(times):.1f}")
    print()
    print("decoded shapes:")
    print(f"  plan            {decoded.plan.shape}")
    print(f"  plan[0]         {decoded.plan[0]}")
    print(f"  lane_lines      {decoded.lane_lines.shape}")
    print(f"  lane_lines_prob {decoded.lane_lines_prob}")
    print(f"  road_edges      {decoded.road_edges.shape}")
    print(f"  pose            {decoded.pose}")
    print(f"  desire_state    {decoded.desire_state}")
    print(f"  lead_prob       {decoded.lead_prob}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
