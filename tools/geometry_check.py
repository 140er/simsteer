"""Audit numeric model geometry from a run log or incident recording.

Replay a clip with --log-jsonl run.jsonl first; this tool needs the model
outputs, not its images. Works with any game's telemetry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median


def read_recording(path):
    path = Path(path)
    if path.is_dir():
        path = next((path / name for name in ("run.jsonl", "replay.jsonl")
                     if (path / name).is_file()), None)
        if path is None:
            raise ValueError("Replay the clip with --log-jsonl run.jsonl first")
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".json":
            return json.load(f)["samples"]
        return [json.loads(line) for line in f if line.strip()]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def summary(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
    values = sorted(values)
    return {"n": len(values), "mean": mean(values), "median": median(values),
            "p05": values[int((len(values) - 1) * .05)],
            "p95": values[int((len(values) - 1) * .95)]}


def audit(rows, *, lane_width_m=3.7):
    metrics = {name: [] for name in ("lane_width_m", "speed_ratio", "pitch_residual_deg",
                                    "calibration_horizon_px", "model_horizon_px",
                                    "horizon_delta_px", "implied_height_m", "height_delta_m")}
    for row in rows:
        if not isinstance(row, dict) or row.get("model_fresh") is False:
            continue
        sample = row.get("camera_sample") or {}
        pose = sample.get("pose") or []
        calibration = sample.get("calibration") or {}
        speed = row.get("actual_speed_mps", row.get("speed_mps"))
        # Stationary/slow poses cannot constrain metric scale or pitch.
        if not finite(speed) or speed < 8.:
            continue
        width = row.get("lane_width_m")
        if finite(width):
            metrics["lane_width_m"].append(width)
        vx = row.get("pose_speed_mps", row.get("pose_vx", pose[0] if pose else None))
        vz = row.get("pose_vertical_mps", pose[2] if len(pose) > 2 else None)
        if finite(vx):
            metrics["speed_ratio"].append(vx / speed)
        if finite(vx) and vx > 1. and finite(vz):
            residual = -math.atan2(vz, vx)
            metrics["pitch_residual_deg"].append(math.degrees(residual))
            pitch = row.get("calibration_pitch_deg", calibration.get("pitch_deg"))
            fy, cy = row.get("camera_fy_px"), row.get("camera_cy_px")
            if finite(pitch) and finite(fy) and finite(cy):
                horizon = cy - fy * math.tan(math.radians(pitch))
                implied = cy - fy * math.tan(math.radians(pitch) + residual)
                metrics["calibration_horizon_px"].append(horizon)
                metrics["model_horizon_px"].append(implied)
                metrics["horizon_delta_px"].append(implied - horizon)
        height = row.get("model_height_m", row.get("road_height_m"))
        calibrated = row.get("calibration_height_m", calibration.get("height_m"))
        if finite(height):
            metrics["implied_height_m"].append(height)
            if finite(calibrated):
                metrics["height_delta_m"].append(height - calibrated)
    checks = {}
    for name, target, tolerance in (("lane_width_m", lane_width_m, .8),
                                    ("speed_ratio", 1., .1),
                                    ("pitch_residual_deg", 0., 1.),
                                    ("height_delta_m", 0., .5)):
        values = metrics[name]
        checks[name] = ("insufficient" if not values else
                        "ok" if abs(median(values) - target) <= tolerance else "warning")
    return {"metrics": {k: summary(v) for k, v in metrics.items()}, "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--lane-width-m", type=float, default=3.7,
                        help="reference lane width; roads vary (default: 3.7)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.lane_width_m) or args.lane_width_m <= 0:
        parser.error("lane width must be positive and finite")
    try:
        result = audit(read_recording(args.recording), lane_width_m=args.lane_width_m)
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(2, "A numeric run log or incident recording is required; replay clips with --log-jsonl run.jsonl first.\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if "warning" in result["checks"].values() else 0


if __name__ == "__main__":
    raise SystemExit(main())
