"""One-shot smoke test for the desire pulse pipeline.

1. Single-frame pulse (old behavior) — model's argmax response.
2. Sustained pulse for N frames (new behavior) — verify the response
   stays high longer.
"""
from __future__ import annotations

import numpy as np

from pilot.constants import DESIRE_LEN
from pilot.model import DrivingModel
from pilot.postprocess import decode


def main() -> None:
    img = np.zeros((1, 12, 128, 256), dtype=np.uint8)
    big = np.zeros((1, 12, 128, 256), dtype=np.uint8)
    one_hot_left = np.zeros(DESIRE_LEN, dtype=np.float32)
    one_hot_left[3] = 1.0

    print("=== single-frame pulse ===")
    m = DrivingModel()
    v, p = m.step(img, big, desire=one_hot_left)
    print(f"  pulse: argmax={decode(v, p).desire_state.argmax()} "
          f"top={decode(v, p).desire_state.max():.3f}")
    for i in range(1, 12):
        v, p = m.step(img, big, desire=None)
        dec = decode(v, p)
        print(f"  +{i:2d}f:  argmax={dec.desire_state.argmax()} "
              f"top={dec.desire_state.max():.3f} "
              f"lc_prob={float(dec.desire_state[3])+float(dec.desire_state[4]):.3f}")

    print("\n=== sustained pulse (50 frames = 2.5 s @ 20 Hz) ===")
    m = DrivingModel()  # fresh model state
    for i in range(50):
        v, p = m.step(img, big, desire=one_hot_left)
        if i % 10 == 0 or i in (1, 5, 49):
            dec = decode(v, p)
            print(f"  hold f{i:2d}: argmax={dec.desire_state.argmax()} "
                  f"top={dec.desire_state.max():.3f} "
                  f"lc_prob={float(dec.desire_state[3])+float(dec.desire_state[4]):.3f}")

    print("\n  -- desire released --")
    for i in range(1, 20):
        v, p = m.step(img, big, desire=None)
        if i in (1, 5, 10, 19):
            dec = decode(v, p)
            print(f"  +{i:2d}f after release: argmax={dec.desire_state.argmax()} "
                  f"top={dec.desire_state.max():.3f} "
                  f"lc_prob={float(dec.desire_state[3])+float(dec.desire_state[4]):.3f}")


if __name__ == "__main__":
    main()
