"""Audit synthetic projection and probe model speed without fitted factors."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simsteer.core.calibration import Calibration
from tools import make_synthetic_clip as scene
from tools.diagnose_fov import cpu_model, pairs, response, solve20
from simsteer.core.preprocess import model_views


def projection_audit(fov):
    camera = scene.CameraModel(fov, 2., 0., 1.4)
    x, z = np.meshgrid(np.linspace(-6., 6., 13), [3., 5., 10., 25., 60., 150.])
    x, z = x.ravel(), z.ravel()
    rendered = camera.project(x, z)
    points = np.column_stack([x, np.full(len(x), 1.4), z])
    # Independent OpenCV projection: identity intrinsics first, then fit
    # focal lengths and principal points from the renderer's coordinates.
    normalised, _ = cv2.projectPoints(points, np.array([math.radians(2.), 0., 0.]),
                                     np.zeros(3), np.eye(3), None)
    normalised = normalised.reshape(-1, 2)
    fx, cx = np.linalg.lstsq(np.column_stack([normalised[:, 0], np.ones(len(x))]),
                            rendered[:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.column_stack([normalised[:, 1], np.ones(len(x))]),
                            rendered[:, 1], rcond=None)[0]
    reconstructed = normalised * [fx, fy] + [cx, cy]
    measured_fov = math.degrees(2 * math.atan(scene.W / (2 * fx)))
    # Recover the speed of a fixed ground point from two projected pixels.
    points_z = np.array([30., 30. - 22. * .2])
    projected = camera.project(np.array([3., 3.]), points_z)
    rays = np.column_stack([(projected[:, 0] - cx) / fx,
                            (projected[:, 1] - cy) / fy, np.ones(2)])
    rotation, _ = cv2.Rodrigues(np.array([math.radians(2.), 0., 0.]))
    world_rays = rays @ rotation
    depths = world_rays[:, 2] * 1.4 / world_rays[:, 1]
    speed = (depths[0] - depths[1]) / .2
    assert abs(measured_fov - fov) < 1e-9
    assert abs(fx - fy) < 1e-9 and abs(cx - scene.W / 2) < 1e-9
    assert abs(cy - scene.H / 2) < 1e-9 and abs(speed - 22.) < 1e-9
    return {"true_fov": fov, "recovered_hfov": measured_fov,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": scene.W, "height": scene.H, "pixel_aspect": fx / fy,
            "reprojection_max_px": float(abs(reconstructed - rendered).max()),
            "recovered_speed_mps": speed,
            "source_horizon_px": scene.H / 2 - camera.f * math.tan(camera.p)}


def render_pairs(times, fov, width, speed, height, span=.2, continuous_dashes=False):
    old_w, old_h = scene.W, scene.H
    scene.W, scene.H = width, width * 9 // 16
    try:
        camera = scene.CameraModel(fov, 2., 0., height)
        for t in times:
            # Hold the current scene fixed when probing speed. Only the
            # previous position changes, so bends/texture phase cannot
            # confound the comparison with velocity.
            position = 22. * t
            yield t, span, scene.render_calib(position - speed * span, camera, continuous_dashes=continuous_dashes), scene.render_calib(position, camera, continuous_dashes=continuous_dashes)
    finally:
        scene.W, scene.H = old_w, old_h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--dash-audit", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model, vision = cpu_model()
    results = []
    for clip in args.clip:
        meta, saved = pairs(clip, args.samples)
        fov = float(meta["truth"]["fov_h_deg"])
        audit = projection_audit(fov)
        print(json.dumps({"projection_audit": audit}), flush=True)
        config = Calibration(image_w=960, image_h=540, fov_h_deg=fov,
                             pitch_deg=2., height_m=1.4)
        rows = [{"variant": "saved_jpeg_960", "speed": 22., "height": 1.4,
                 "response": response(vision, saved, config)}]
        times = [pair[0] for pair in saved]
        if args.dash_audit:
            for continuous in (False, True):
                row = response(vision, render_pairs(times, fov, 960, 22., 1.4,
                                                    continuous_dashes=continuous), config)
                camera = scene.CameraModel(fov, 2., 0., 1.4)
                ts = np.arange(480) / 20
                frames = (scene.render_calib(22 * t, camera, continuous_dashes=continuous) for t in ts)
                solved = solve20(vision, frames, ts, config, model_views)
                rows.append({"variant": "continuous_dashes" if continuous else "quantised_dashes",
                             "response": row, **solved})
                print(json.dumps({"fov": fov, "continuous_dashes": continuous,
                                  "ratio": row["median_ratio"], "solved_fov": solved["result"]}), flush=True)
            results.append({"meta": meta, "variants": rows})
            continue
        for name, width, speed, height, span in [
            ("lossless_960", 960, 22., 1.4, .2),
            ("lossless_2560", 2560, 22., 1.4, .2),
            ("speed_16", 960, 16., 1.4, .2), ("speed_28", 960, 28., 1.4, .2),
            ("height_12", 960, 22., 1.2, .2), ("height_16", 960, 22., 1.6, .2),
            ("span_400ms", 960, 22., 1.4, .4),
        ]:
            frame_pairs = render_pairs(times, fov, width, speed, height, span)
            row = response(vision, frame_pairs,
                           replace(config, image_w=width, image_h=width * 9 // 16, height_m=height),
                           speed=speed)
            rows.append({"variant": name, "speed": speed, "height": height,
                         "span_s": span, "response": row})
            print(json.dumps({"true_fov": fov, "variant": name,
                              "ratio": row["median_ratio"], "coarse_fov": row["coarse_fov"]}), flush=True)
        results.append({"meta": meta, "projection_audit": audit, "variants": rows})
    result = {"model_identity": model.certification_identity, "results": results}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
