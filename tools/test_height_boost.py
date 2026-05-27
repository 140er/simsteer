"""Render a synthetic road through the full preprocess pipeline at
several camera_height_boost values, and dump them to PNGs so the
effect is visually inspectable.

Synthetic scene: blue sky, green ground, with horizontal lines at
specific world distances drawn on the ground plane. With boost > 1,
the same world lines should appear FURTHER DOWN in the boosted image
(taller camera = more pixels per meter at distance).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pilot.calibration import Calibration
from pilot.preprocess import _apply_height_boost


def synth_frame(w: int = 1280, h: int = 720) -> np.ndarray:
    """Sky + ground + dashboard-like band, plus a series of horizontal
    grid lines on the ground that we can track through the warp."""
    img = np.empty((h, w, 3), dtype=np.uint8)
    horizon = h // 2
    # Sky gradient
    for y in range(horizon):
        shade = int(200 - 60 * y / horizon)
        img[y, :] = (255, shade, shade // 2)
    # Ground gradient
    for y in range(horizon, h):
        t = (y - horizon) / (h - horizon)
        img[y, :] = (60 + int(60 * t), 80 + int(40 * t), 50)
    # Horizontal "distance markers" on the ground at known image rows
    # below horizon. These are what we'll track.
    for marker_row in (horizon + 40, horizon + 100, horizon + 200,
                       horizon + 320):
        cv2.line(img, (0, marker_row), (w, marker_row),
                 (255, 255, 255), 2)
        cv2.putText(img, f"row +{marker_row - horizon}",
                    (20, marker_row - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # "Dashboard" band at the very bottom
    img[h - 60:h, :] = (40, 40, 60)
    return img


def main() -> None:
    frame = synth_frame()
    out_dir = Path(".")

    for boost in (1.0, 1.5, 2.0, 2.5):
        boosted = _apply_height_boost(frame, boost)
        out_path = out_dir / f"debug_height_boost_{boost:.1f}.png"
        cv2.imwrite(str(out_path), boosted)
        print(f"wrote {out_path}  boost={boost}")

    # Side-by-side comparison
    panels = [_apply_height_boost(frame, b) for b in (1.0, 1.5, 2.0)]
    labels = ["1.0x (none)", "1.5x boost", "2.0x boost"]
    for img, lbl in zip(panels, labels):
        cv2.putText(img, lbl, (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3, cv2.LINE_AA)
    side = np.hstack(panels)
    out_path = out_dir / "debug_height_boost_comparison.png"
    cv2.imwrite(str(out_path), side)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
