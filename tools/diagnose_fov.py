"""Measure a real-model FOV response on identical timestamped frame pairs.

    python tools/diagnose_fov.py --clip /tmp/clip --out /tmp/fov-response

Each candidate sees the same 20 Hz timestamps, covering the whole clip,
so candidate scheduling and interpolation can be checked independently.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import median
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simsteer.core.calibration import Calibration
from simsteer.core.model import DrivingModel
from simsteer.core.preprocess import bgr_to_yuv6, model_views
from simsteer.runtime.fov import FovSolver, model_vision, solve_from_ratio
from tools.measure_time_base import generate, measure, sha256


def cpu_model():
    import simsteer.core.model as adapter
    cv2.setNumThreads(1)

    def session(path, providers):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    adapter._make_session = session
    model = DrivingModel(providers=["CPUExecutionProvider"])
    return model, model_vision(model)


def pairs(clip: Path, samples: int):
    meta = json.loads((clip / "clip.json").read_text())
    times = [json.loads(line)["t"] for line in (clip / "frames.jsonl").read_text().splitlines()]
    step = round(meta["fps"] / 20)
    indices = np.linspace(40, (len(times) - 1) // step, samples).astype(int) * step
    frames = {}
    for i in sorted(set(indices) | {i - 4 * step for i in indices}):
        frame = cv2.imread(str(clip / "frames" / f"{i:06d}.jpg"))
        if frame is None:
            raise ValueError(f"unreadable frame {i}")
        frames[i] = frame
    return meta, [(times[i], times[i] - times[i - 4 * step],
                   frames[i - 4 * step], frames[i]) for i in indices]


def response(vision, frame_pairs, calib, views=model_views, speed=22.):
    records = []
    for t, span, previous, current in frame_pairs:
        a, b = views(previous, calib), views(current, calib)
        img, big = (np.concatenate([a[j], b[j]], axis=0)[None] for j in (0, 1))
        vx, std = vision(img, big)
        ratio = vx / speed * .2 / span
        records.append({"t": t, "span_s": span, "vx_ratio": ratio, "vx_std": std})
    ratios = [r["vx_ratio"] for r in records]
    ratio = median(ratios)
    return {"fov": calib.fov_h_deg, "median_ratio": ratio,
            "mean_ratio": float(np.mean(ratios)), "ratio_std": float(np.std(ratios)),
            "p10_ratio": float(np.percentile(ratios, 10)),
            "p90_ratio": float(np.percentile(ratios, 90)),
            "median_vx_std": median(r["vx_std"] for r in records),
            "coarse_fov": solve_from_ratio(calib.fov_h_deg, ratio), "records": records}


def crossings(rows):
    return [a["fov"] + (b["fov"] - a["fov"]) * (a["median_ratio"] - 1)
            / (a["median_ratio"] - b["median_ratio"])
            for a, b in zip(rows, rows[1:])
            if a["median_ratio"] >= 1 >= b["median_ratio"]
            and a["median_ratio"] > b["median_ratio"]]


def geometry_views(kind, frame, calib, shift=0.):
    import simsteer.core.warp as warp
    if shift:
        frame = cv2.warpAffine(frame, np.float32([[1, 0, 0], [0, 1, shift]]),
                               (frame.shape[1], frame.shape[0]))
    if kind == "pinned_nearest":
        h, w = frame.shape[:2]
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        planes = (yuv[:h], yuv[h:h + h // 4].reshape(h // 2, w // 2),
                  yuv[h + h // 4:].reshape(h // 2, w // 2))
        results = []
        for big in (False, True):
            matrix = warp.get_warp_matrix(calib.rpy_calib_rad(),
                                          warp.camera_intrinsics(w, h, calib.fov_h_deg), big)
            warped = []
            for j, plane in enumerate(planes):
                scale = 2 if j else 1
                uv_scale = np.diag([1 / scale, 1 / scale, 1]).astype(np.float32)
                transform = uv_scale @ matrix @ np.linalg.inv(uv_scale)
                yy, xx = np.indices((256 // scale, 512 // scale), dtype=np.float32)
                rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ transform.T
                sx = np.rint(rays[..., 0] / rays[..., 2]).astype(int).clip(0, plane.shape[1] - 1)
                sy = np.rint(rays[..., 1] / rays[..., 2]).astype(int).clip(0, plane.shape[0] - 1)
                warped.append(plane[sy, sx])
            y, u, v = warped
            results.append(np.stack([y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2], u, v]))
        return tuple(results)
    if kind != "subpixel":
        return model_views(frame, replace(calib, simple_warp=kind == "simple"))
    h, w = frame.shape[:2]
    focal = (w / 2) / math.tan(math.radians(calib.fov_h_deg / 2))
    horizon = h / 2 - focal * math.tan(math.radians(calib.pitch_deg))
    result = []
    for fl, cy in ((warp.MEDMODEL_FL, warp.MEDMODEL_CY),
                   (warp.SBIGMODEL_FL, warp.SBIGMODEL_CY)):
        scale = fl / focal
        affine = np.float32([[scale, 0, 256 - scale * w / 2],
                             [0, scale, cy - scale * horizon]])
        view = cv2.warpAffine(frame, affine, (512, 256), flags=cv2.INTER_LINEAR)
        result.append(bgr_to_yuv6(view))
    return tuple(result)


def solve20(vision, frames, times, calib, views):
    solver = FovSolver(vision, views)
    previous = []
    for i, (t, frame) in enumerate(zip(times, frames)):
        current = views(frame, calib)
        previous.append((t, current))
        past_time, past = min(previous, key=lambda value: abs(value[0] - (t - .2)))
        span = t - past_time
        img, big = (np.concatenate([past[j], current[j]], axis=0)[None] for j in (0, 1))
        vx, _ = vision(img, big)
        if i >= 5:
            solver.update(frame, calib, vx, 22., 0., timestamp=t, frame_span_s=span)
        if solver.done:
            break
        previous = previous[-6:]
    return {"result": solver.result, "processed_frames": i + 1, "solver": solver.to_dict()}


def geometry_probe(clip, frame_pairs, calib, vision, selected=None):
    import simsteer.core.warp as warp
    meta = json.loads((clip / "clip.json").read_text())
    step = round(meta["fps"] / 20)
    all_times = [json.loads(line)["t"] for line in (clip / "frames.jsonl").read_text().splitlines()]
    times = all_times[::step]
    frames = [cv2.imread(str(clip / "frames" / f"{i:06d}.jpg"))
              for i in range(0, len(all_times), step)]
    variants = [("simple", "simple", 0., 152., 0.),
                ("perspective", "perspective", 0., 152., 0.),
                ("simple_pinned_cy", "simple", 0., 151.8, 0.),
                ("perspective_pinned_cy", "perspective", 0., 151.8, 0.),
                ("subpixel_simple_pinned_cy", "subpixel", 0., 151.8, 0.),
                ("pinned_nearest_yuv", "pinned_nearest", 0., 151.8, 0.),
                ("pitch_minus_025", "simple", -.25, 152., 0.),
                ("pitch_plus_025", "simple", .25, 152., 0.),
                ("principal_minus_1px", "simple", 0., 152., -1.),
                ("principal_plus_1px", "simple", 0., 152., 1.)]
    results = []
    for name, kind, pitch_delta, cy, shift in variants:
        if selected and name not in selected:
            continue
        old_cy = warp.SBIGMODEL_CY
        old_intrinsics = warp.SBIGMODEL_INTRINSICS.copy()
        old_inverse = warp._CALIB_FROM_SBIGMODEL
        warp.SBIGMODEL_CY = cy
        warp.SBIGMODEL_INTRINSICS[1, 2] = cy
        warp._CALIB_FROM_SBIGMODEL = np.linalg.inv(
            warp.SBIGMODEL_INTRINSICS @ warp.VIEW_FROM_DEVICE)
        try:
            def views(frame, config):
                return geometry_views(kind, frame, config, shift)
            config = replace(calib, pitch_deg=calib.pitch_deg + pitch_delta)
            row = response(vision, frame_pairs, config, views)
            solved = solve20(vision, frames, times, config, views)
            result = {"variant": name, "pitch_delta": pitch_delta, "wide_cy": cy,
                      "source_y_shift": shift, "response": row, **solved}
            results.append(result)
            print(json.dumps({"variant": name, "ratio_at_true_fov": row["median_ratio"],
                              "coarse_fov": row["coarse_fov"], "solved_fov": solved["result"]}), flush=True)
        finally:
            warp.SBIGMODEL_CY = old_cy
            warp.SBIGMODEL_INTRINSICS[:] = old_intrinsics
            warp._CALIB_FROM_SBIGMODEL = old_inverse
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--fov", type=float, default=75.)
    parser.add_argument("--baseline-out", type=Path)
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--variants", help="Comma-separated geometry variants")
    args = parser.parse_args()
    if args.generate:
        generate(args.clip, 24., args.fov)
    if args.baseline_out:
        measure(args.clip, args.baseline_out)
    args.out.mkdir(parents=True, exist_ok=False)
    model, vision = cpu_model()
    meta, frame_pairs = pairs(args.clip, args.samples)
    truth = meta["truth"]
    calib = Calibration(image_w=meta["width"], image_h=meta["height"],
                        fov_h_deg=truth["fov_h_deg"], pitch_deg=truth["pitch_deg"],
                        height_m=truth["height_m"])
    if args.geometry:
        result = {"clip_meta": meta, "model_identity": model.certification_identity,
                  "variants": geometry_probe(args.clip, frame_pairs, calib, vision,
                                             args.variants.split(",") if args.variants else None)}
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    rows = []
    for offset in (-12, -10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10, 12):
        row = response(vision, frame_pairs, replace(calib, fov_h_deg=calib.fov_h_deg + offset))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "records"}), flush=True)
    result = {"clip_meta": meta, "model_identity": model.certification_identity,
              "harness_sha256": sha256(Path(__file__)), "samples": args.samples,
              "candidates": rows, "crossings": crossings(rows)}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("crossings:", result["crossings"], flush=True)


if __name__ == "__main__":
    main()
