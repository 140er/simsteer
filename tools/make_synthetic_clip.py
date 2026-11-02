"""Render a synthetic highway clip for headless replay.

Draws a three-lane road in perspective with dashed lane lines that move
at the clip's speed and a gentle S-bend, plus matching telemetry. The
output is fully deterministic, so it is a stable input for replay tests
and perf runs when no recorded drive is around.

`--calib` renders a long calibration drive instead: straights and
sweeps, a changing speed, lane changes and a small weave, from a camera
with a known FOV, pitch and yaw, and a steering rack with known
LiveParams coefficients. The calibration flow should recover them.

    python tools/make_synthetic_clip.py
    python tools/make_synthetic_clip.py --out /tmp/gap --telemetry-gap 10:13
    python tools/make_synthetic_clip.py --calib --minutes 6 \
        --out ~/.cache/simsteer/clips/synthetic-calib
"""

from __future__ import annotations

import argparse
import json
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


# ----- calibration drive (--calib) -----

LAG_FRAMES = 5  # LiveParams pairs the axis with the wheel this many frames later


def calib_road_curvature(s: float) -> float:
    """Repeat every 2 km: 600 m straight, a 900 m sweep each way, 500 m
    straight. Peak 0.0012 1/m keeps the yaw rate under LiveCalib's gate."""
    s = s % 2000.0
    if 600.0 <= s < 1500.0:
        return 0.0012 * math.sin(2 * math.pi * (s - 600.0) / 900.0)
    return 0.0


def calib_speed(t: float) -> float:
    return 22.0 + 5.0 * math.sin(2 * math.pi * t / 120.0)


def lane_offset(s: float) -> float:
    """Lateral offset from the middle lane's centre (m, +right): one lane
    change to the right and back per 3 km, plus a small weave."""
    def ramp(x: float) -> float:
        return 0.0 if x <= 0 else 1.0 if x >= 1 else 0.5 - 0.5 * math.cos(math.pi * x)
    c = s % 3000.0
    lc = LANE_W * (ramp((c - 2000.0) / 150.0) - ramp((c - 2400.0) / 150.0))
    return lc + 0.25 * math.sin(2 * math.pi * s / 350.0)


def _second_derivative(fn, s: float, h: float = 1.0) -> float:
    return (fn(s + h) - 2 * fn(s) + fn(s - h)) / (h * h)


class CameraModel:
    def __init__(self, fov_deg: float, pitch_deg: float, yaw_deg: float,
                 height_m: float) -> None:
        self.f = (W / 2) / math.tan(math.radians(fov_deg) / 2)
        self.p = math.radians(pitch_deg)
        self.y = math.radians(yaw_deg)
        self.h = height_m

    def project(self, x: np.ndarray, z: np.ndarray, up: float = 0.0) -> np.ndarray:
        """Car frame (x right, z forward, `up` above the road) to pixels.
        Yaw turns the camera left, pitch tips it down."""
        down = self.h - up
        x1 = x * math.cos(self.y) + z * math.sin(self.y)
        z1 = -x * math.sin(self.y) + z * math.cos(self.y)
        y2 = down * math.cos(self.p) - z1 * math.sin(self.p)
        z2 = down * math.sin(self.p) + z1 * math.cos(self.p)
        return np.stack([W / 2 + self.f * x1 / z2, H / 2 + self.f * y2 / z2], axis=1)


def dash_intervals(s0: float, near: float, far: float):
    """Paint boundaries in fixed world coordinates, clipped to the view."""
    period = DASH + GAP
    for n in range(math.floor((s0 + near) / period), math.ceil((s0 + far) / period) + 1):
        start = max(near, n * period - s0)
        end = min(far, n * period + DASH - s0)
        if end > start:
            yield start, end


def render_calib(s0: float, cam: CameraModel, *, continuous_dashes: bool = False) -> np.ndarray:
    img = np.empty((H, W, 3), np.uint8)
    horizon = int(round(H / 2 - cam.f * math.tan(cam.p)))
    horizon = max(1, min(H - 1, horizon))
    sky = np.linspace(0, 1, horizon)[:, None]
    img[:horizon] = (np.array([235, 205, 170]) * (1 - sky) +
                     np.array([250, 230, 210]) * sky)[:, None, :].astype(np.uint8)
    ground = np.linspace(0, 1, H - horizon)[:, None]
    img[horizon:] = (np.array([70, 120, 80]) * (1 - ground) +
                     np.array([50, 100, 60]) * ground)[:, None, :].astype(np.uint8)

    zs = np.linspace(2.0, 160.0, 240)
    ds = zs[1] - zs[0]
    k = np.array([calib_road_curvature(s0 + z) for z in zs])
    xoff = np.cumsum(np.sin(np.cumsum(k) * ds)) * ds
    # Car heading relative to the road is the slope of its lane offset.
    e0 = lane_offset(s0)
    psi = (lane_offset(s0 + 1.0) - lane_offset(s0 - 1.0)) / 2.0
    centre = xoff - e0 - psi * zs

    edge_l = cam.project(centre - 1.5 * LANE_W - 1.0, zs)
    edge_r = cam.project(centre + 1.5 * LANE_W + 1.0, zs)
    road = np.concatenate([edge_l, edge_r[::-1]]).astype(np.int32)
    cv2.fillPoly(img, [road], (95, 95, 100), lineType=cv2.LINE_AA)
    for lane_x, dashed in ((-1.5, False), (-0.5, True), (0.5, True), (1.5, False)):
        if dashed and continuous_dashes:
            for start, end in dash_intervals(s0, float(zs[0]), float(zs[-1])):
                z_paint = np.concatenate([[start], zs[(zs > start) & (zs < end)], [end]])
                x_paint = np.interp(z_paint, zs, centre) + lane_x * LANE_W
                points = cam.project(x_paint, z_paint)
                for i in range(len(points) - 1):
                    width = max(1, int(round(cam.f * .15 / z_paint[i])))
                    a, b = (tuple(int(round(c)) for c in point) for point in points[i:i + 2])
                    cv2.line(img, a, b, (235, 235, 235), width, cv2.LINE_AA)
            continue
        pts = cam.project(centre + lane_x * LANE_W, zs)
        for i in range(len(zs) - 1):
            if dashed and (s0 + zs[i]) % (DASH + GAP) > DASH:
                continue
            width = max(1, int(round(cam.f * 0.15 / zs[i])))
            p0 = tuple(int(round(c)) for c in pts[i])
            p1 = tuple(int(round(c)) for c in pts[i + 1])
            cv2.line(img, p0, p1, (235, 235, 235), width, cv2.LINE_AA)
    for side in (-1, 1):
        for n in range(int(s0 // 25) + 1, int((s0 + 160) // 25) + 1):
            z = n * 25.0 - s0
            if z < 3.0:
                continue
            x = float(np.interp(z, zs, centre)) + side * (1.5 * LANE_W + 2.5)
            (u, v), = cam.project(np.array([x]), np.array([z]))
            (_, vt), = cam.project(np.array([x]), np.array([z]), up=1.2)
            half = max(1, int((v - vt) / 12))
            cv2.rectangle(img, (int(u) - half, int(vt)), (int(u) + half, int(v)),
                          (40, 40, 40), -1)
    return img


def write_calib_clip(args) -> dict:
    cam = CameraModel(args.fov, args.pitch_deg, args.yaw_deg, CAM_HEIGHT_M)
    a, b, c = args.rack
    frames = int(round(args.minutes * 60 * args.fps))
    dt = 1.0 / args.fps
    ts = [i * dt for i in range(frames + LAG_FRAMES)]
    vs, ss = [], []
    s = 0.0
    for t in ts:
        v = calib_speed(t)
        vs.append(v)
        ss.append(s)
        s += v * dt
    wheels = [math.atan(WHEELBASE * (calib_road_curvature(x)
                                     + _second_derivative(lane_offset, x)))
              for x in ss]
    writer = ClipWriter(args.out, game="ets2", overwrite=True)
    for i in range(frames):
        writer.write_frame(ts[i], render_calib(ss[i], cam))
        k_car = calib_road_curvature(ss[i]) + _second_derivative(lane_offset, ss[i])
        w_next = wheels[i + LAG_FRAMES]
        axis = a * w_next + b * w_next * vs[i] ** 2 + c
        writer.write_telemetry(ts[i], {
            "speed_mps": vs[i],
            "yaw_rate_rad_s": vs[i] * k_car,
            "wheel_angle_rad": wheels[i],
            "game_steer": axis,
            "user_steer": axis,
        })
    meta = writer.close("synthetic-calib")
    meta["truth"] = {"fov_h_deg": args.fov, "pitch_deg": args.pitch_deg,
                     "yaw_deg": args.yaw_deg, "height_m": CAM_HEIGHT_M,
                     "rack": [a, b, c], "lag_frames": LAG_FRAMES}
    (Path(args.out) / "clip.json").write_text(json.dumps(meta, indent=2) + "\n",
                                              encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--telemetry-gap", metavar="START:END",
                    help="leave out telemetry between these seconds")
    ap.add_argument("--calib", action="store_true",
                    help="render the long calibration drive")
    ap.add_argument("--minutes", type=float, default=6.0,
                    help="--calib drive length")
    ap.add_argument("--fov", type=float, default=FOV_H_DEG,
                    help="--calib horizontal FOV (deg)")
    ap.add_argument("--pitch-deg", type=float, default=2.0,
                    help="--calib camera pitch, + looks down")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="--calib camera yaw, + looks left")
    ap.add_argument("--rack", type=lambda v: tuple(float(x) for x in v.split(",")),
                    default=(4.2, 0.0010, 0.01), metavar="A,B,C",
                    help="--calib steering rack: axis = a*w + b*w*v^2 + c")
    args = ap.parse_args(argv)

    if args.calib:
        meta = write_calib_clip(args)
        print(f"{args.out}: {meta['frames']} frames, {meta['duration_s']} s, "
              f"truth {meta['truth']}")
        return 0

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
