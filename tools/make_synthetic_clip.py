"""Render a synthetic highway clip for headless replay.

Draws a three-lane road in perspective with dashed lane lines that move
at the clip's speed and a gentle S-bend, plus matching telemetry. The
output is fully deterministic, so it is a stable input for replay tests
and perf runs when no recorded drive is around.

    python tools/make_synthetic_clip.py
    python tools/make_synthetic_clip.py --out /tmp/gap --telemetry-gap 10:13
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simsteer.runtime.replay import ClipWriter  # noqa: E402

DEFAULT_OUT = Path.home() / ".cache" / "simsteer" / "clips" / "synthetic"
W, H = 960, 540
FOV_H_DEG = 75.0
CAM_HEIGHT_M = 1.4
HORIZON_Y = int(H * 0.47)
LANE_W = 3.6
SPEED = 22.0
WHEELBASE = 3.8
DASH, GAP = 3.0, 9.0


def curvature_at(s: float) -> float:
    """Road curvature (1/m) at distance s along the road: straight, then a
    gentle S-bend, so the model sees both."""
    return 0.0015 * math.sin(2 * math.pi * s / 500.0) if s > 120.0 else 0.0


def _background() -> np.ndarray:
    img = np.empty((H, W, 3), np.uint8)
    sky = np.linspace(0, 1, HORIZON_Y)[:, None]
    img[:HORIZON_Y] = (np.array([235, 205, 170]) * (1 - sky) +
                       np.array([250, 230, 210]) * sky)[:, None, :].astype(np.uint8)
    ground = np.linspace(0, 1, H - HORIZON_Y)[:, None]
    img[HORIZON_Y:] = (np.array([70, 120, 80]) * (1 - ground) +
                       np.array([50, 100, 60]) * ground)[:, None, :].astype(np.uint8)
    return img


def render(s0: float) -> np.ndarray:
    f = (W / 2) / math.tan(math.radians(FOV_H_DEG) / 2)
    img = _background()
    zs = np.linspace(2.0, 160.0, 240)
    # Lateral road offset at distance z ahead, integrating curvature.
    ds = zs[1] - zs[0]
    k = np.array([curvature_at(s0 + z) for z in zs])
    heading = np.cumsum(k) * ds
    xoff = np.cumsum(np.sin(heading)) * ds

    def project(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        u = W / 2 + f * x / z
        v = HORIZON_Y + f * CAM_HEIGHT_M / z
        return np.stack([u, v], axis=1)

    edge_l = project(xoff - 1.5 * LANE_W - 1.0, zs)
    edge_r = project(xoff + 1.5 * LANE_W + 1.0, zs)
    road = np.concatenate([edge_l, edge_r[::-1]]).astype(np.int32)
    cv2.fillPoly(img, [road], (95, 95, 100), lineType=cv2.LINE_AA)

    for lane_x, dashed in ((-1.5, False), (-0.5, True), (0.5, True), (1.5, False)):
        pts = project(xoff + lane_x * LANE_W, zs)
        for i in range(len(zs) - 1):
            if dashed and (s0 + zs[i]) % (DASH + GAP) > DASH:
                continue
            width = max(1, int(round(f * 0.15 / zs[i])))
            p0 = tuple(int(round(c)) for c in pts[i])
            p1 = tuple(int(round(c)) for c in pts[i + 1])
            cv2.line(img, p0, p1, (235, 235, 235), width, cv2.LINE_AA)

    # Roadside posts give the model motion parallax off the road surface.
    for side in (-1, 1):
        for n in range(int(s0 // 25) + 1, int((s0 + 160) // 25) + 1):
            z = n * 25.0 - s0
            if z < 3.0:
                continue
            x = float(np.interp(z, zs, xoff)) + side * (1.5 * LANE_W + 2.5)
            u, v = project(np.array([x]), np.array([z]))[0]
            hgt = f * 1.2 / z
            cv2.rectangle(img, (int(u - hgt / 12), int(v - hgt)),
                          (int(u + hgt / 12), int(v)), (40, 40, 40), -1)
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--telemetry-gap", metavar="START:END",
                    help="leave out telemetry between these seconds")
    args = ap.parse_args(argv)

    gap = None
    if args.telemetry_gap:
        a, b = args.telemetry_gap.split(":")
        gap = (float(a), float(b))

    writer = ClipWriter(args.out, game="ets2", overwrite=True)
    for i in range(args.frames):
        t = i / args.fps
        s = SPEED * t
        writer.write_frame(t, render(s))
        if gap and gap[0] <= t < gap[1]:
            continue
        k = curvature_at(s)
        writer.write_telemetry(t, {
            "speed_mps": SPEED,
            "yaw_rate_rad_s": SPEED * k,
            "wheel_angle_rad": math.atan(WHEELBASE * k),
            "game_steer": 0.0,
            "user_steer": 0.0,
        })
    meta = writer.close("synthetic")
    print(f"{args.out}: {meta['frames']} frames, {meta['duration_s']} s"
          + (f", telemetry gap {gap[0]}-{gap[1]} s" if gap else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
