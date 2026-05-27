"""Verify the fake stereo shear actually displaces road-plane content
horizontally in the expected direction + magnitude, and leaves
above-horizon content roughly mirrored (sky shifts opposite, which we
accept as the trade-off).

Synthesizes a frame with a vertical line at column 200, runs the same
content through with stereo_baseline_m = 0 and 0.3, and reports the
horizontal displacement at three depths.
"""
from __future__ import annotations

import numpy as np

from pilot.preprocess import _apply_stereo_shift


def find_white_col(row: np.ndarray) -> int:
    """Index of the brightest pixel in `row`. -1 if none above threshold."""
    if row.max() < 200:
        return -1
    return int(np.argmax(row))


def main() -> None:
    h, w = 480, 800
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Vertical white line at column 200 — represents a road-plane
    # feature (lane line) at a fixed lateral position.
    frame[:, 200] = (255, 255, 255)
    frame[:, 600] = (255, 255, 255)  # second line on the other side

    # Camera height matches `height_m` in calibration.
    h_cam = 1.4

    for baseline in (0.0, 0.1, 0.3, -0.3):
        sheared = _apply_stereo_shift(frame, baseline_m=baseline,
                                       height_m=h_cam)
        # Compare displacement at three rows.
        v_horizon = h // 2  # 240
        rows = [v_horizon, v_horizon + 60, h - 1]
        labels = [f"horizon (v={v_horizon})",
                  f"near horizon (v={v_horizon + 60})",
                  f"bottom (v={h - 1})"]
        results = []
        for v, lbl in zip(rows, labels):
            c_orig = find_white_col(frame[v, :, 0])
            c_new = find_white_col(sheared[v, :, 0])
            results.append((lbl, c_orig, c_new, c_new - c_orig))
        bl_str = f"baseline={baseline:+.2f} m"
        print(f"\n{bl_str}")
        for lbl, co, cn, d in results:
            expected = int(round((rows[results.index((lbl, co, cn, d))]
                                   - v_horizon) * baseline / h_cam))
            print(f"  {lbl:24s}  orig col={co:3d}  new col={cn:3d}  "
                  f"shift={d:+4d}  (expected {expected:+4d})")


if __name__ == "__main__":
    main()
