"""v1 (pilot) vs v2 (simsteer) end-to-end parity check.

Phase 1 verification: feed one identical frame through both pipelines
and assert the output is bit-identical at every stage — preprocess
(warp/YUV), the two-stage model, the decoded tensors, and the rendered
overlay. Both models run on the CPU execution provider so the result is
deterministic and isolates the question "is the ported code identical?"
from any GPU floating-point variance.

This complements the structural Compare-Object check done during the
port (which proved only import lines changed): this proves the *wiring*
is right too — that the loop's modules resolve to the identical math.

    python tools/test_v2_parity.py [image.png]

Exits non-zero on any mismatch. Defaults to a repo sample frame.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CPU = ["CPUExecutionProvider"]


def _load_image(arg: str | None) -> np.ndarray:
    candidates = [arg] if arg else [
        "debug_overlay_test.png",
        "debug/hill_path_smoke.png",
        "hud_smoke.png",
    ]
    for c in candidates:
        if c is None:
            continue
        p = (REPO / c) if not Path(c).is_absolute() else Path(c)
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                print(f"frame: {p.name}  {img.shape[1]}x{img.shape[0]}")
                return img
    # Fall back to a deterministic synthetic frame.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    print("frame: synthetic 1920x1080 (no sample image found)")
    return img


def _assert_eq(name: str, a: np.ndarray, b: np.ndarray) -> None:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise SystemExit(f"FAIL {name}: shape {a.shape} != {b.shape}")
    if not np.array_equal(a, b):
        # Report the worst element so a near-miss is debuggable.
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        raise SystemExit(
            f"FAIL {name}: not identical (max|Δ|={diff.max():.3e}, "
            f"n_diff={int((diff > 0).sum())}/{a.size})")
    print(f"  ok  {name}  shape={a.shape}")


def main() -> int:
    img = _load_image(sys.argv[1] if len(sys.argv) > 1 else None)

    # --- v1 (pilot) ---
    from pilot.calibration import Calibration as Cal1
    from pilot.model import DrivingModel as DM1
    from pilot.postprocess import decode as decode1
    from pilot.preprocess import FrameQueue as FQ1
    from debug.overlay import draw_overlay as draw1

    # --- v2 (simsteer) ---
    from simsteer.core.calibration import Calibration as Cal2
    from simsteer.core.model import DrivingModel as DM2
    from simsteer.core.postprocess import decode as decode2
    from simsteer.core.preprocess import FrameQueue as FQ2
    from simsteer.ui.overlay import draw_overlay as draw2

    cal1, cal2 = Cal1.load("ets2"), Cal2.load("ets2")

    print("preprocess:")
    n1, w1 = FQ1().push(img, cal1)
    n2, w2 = FQ2().push(img, cal2)
    _assert_eq("img_narrow", n1, n2)
    _assert_eq("img_wide", w1, w2)

    print("model (CPU):")
    m1 = DM1(providers=CPU, policy_providers=CPU)
    m2 = DM2(providers=CPU, policy_providers=CPU)
    v1o, p1o = m1.step(n1, w1)
    v2o, p2o = m2.step(n2, w2)
    _assert_eq("vision_out", v1o, v2o)
    _assert_eq("policy_out", p1o, p2o)

    print("decode:")
    d1, d2 = decode1(v1o, p1o), decode2(v2o, p2o)
    for f in fields(d1):
        _assert_eq(f"decoded.{f.name}", getattr(d1, f.name), getattr(d2, f.name))

    print("overlay render:")
    o1 = draw1(img.copy(), d1, cal1)
    o2 = draw2(img.copy(), d2, cal2)
    _assert_eq("overlay", o1, o2)

    print("\nPARITY OK — v1 and v2 produce bit-identical output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
