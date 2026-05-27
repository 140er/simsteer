"""One-shot sanity check: feed `crop_for_model` a frame with a single
white pixel column at the exact horizontal center, then look at where
that column ends up in BOTH the narrow and wide outputs. Any non-zero
offset means we're not feeding the model the geometric middle of the
captured frame — which would systematically push the model's lane-
keeping bias.

    python -m tools.check_crop_centering
"""

from __future__ import annotations

import numpy as np

from pilot.calibration import Calibration
from pilot.constants import SRC_H, SRC_W
from pilot.preprocess import crop_for_model


def _white_offset(view: np.ndarray) -> tuple[int, int, int]:
    h, w = view.shape[:2]
    col = view.sum(axis=(0, 2))
    if col.max() == 0:
        return -1, w // 2, 0
    white_at = int(np.argmax(col))
    crop_center = w // 2
    return white_at, crop_center, white_at - crop_center


def check(w: int, h: int) -> None:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cx = w // 2
    frame[:, cx, :] = 255

    cal = Calibration(image_w=w, image_h=h)
    narrow, wide = crop_for_model(frame, cal)

    print(f"  capture {w}x{h:>4}")
    for name, view in (("wide  ", wide), ("narrow", narrow)):
        ch, cw = view.shape[:2]
        white_at, crop_center, asym = _white_offset(view)
        if white_at < 0:
            print(f"    {name}: {cw}x{ch:>4}  WHITE LINE LOST")
            continue
        flag = "ok" if asym == 0 else f"BIAS {asym:+d}px"
        print(f"    {name}: {cw}x{ch:>4} (aspect {cw / ch:.3f})  "
              f"center px {asym:+d} from middle  [{flag}]")


def main() -> None:
    print(f"target model input: {SRC_W}x{SRC_H}  (aspect {SRC_W / SRC_H:.3f})")
    print()
    for w, h in [(3440, 1440), (2560, 1440), (3840, 2160),
                 (2560, 1080), (1920, 1080), (1920, 1200), (1366, 768)]:
        check(w, h)


if __name__ == "__main__":
    main()
