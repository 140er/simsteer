"""Measure model speed and FOV on one clip resampled at 10/15/18/20 Hz.

Generate a dense, deterministic 90-degree calibration clip once, then use
the same files for both revisions. Runs the real pinned vision and policy
models on CPU with two inference threads; timings are not performance gates.
Supports the frame-counted adapter as well as the timestamped adapter.

    python tools/measure_time_base.py --clip /tmp/time-base-clip --generate \
        --out /tmp/time-base-before
    python tools/measure_time_base.py --clip /tmp/time-base-clip \
        --out /tmp/time-base-after
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simsteer.core.calibration import Calibration
from simsteer.core.model import DrivingModel
from simsteer.core.model_identity import actual_model_identity
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue, model_views
from simsteer.runtime.fov import FovSolver, model_vision
from simsteer.runtime.replay import ClipWriter
from tools.make_synthetic_clip import CAM_HEIGHT_M, CameraModel, render_calib

RATES = (10, 15, 18, 20)
SOURCE_HZ = math.lcm(*RATES)
SPEED_MPS = 22.0
TRUE_FOV = 90.0
PITCH_DEG = 2.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fov_target_met(measurements: list[dict], true_fov: float, tolerance: float = 1.5) -> bool:
    """Only full real-model solves count toward the FOV accuracy target."""
    if (len(measurements) != len(RATES)
            or {row.get("hz") for row in measurements} != set(RATES)):
        return False
    return all(isinstance(row.get("real_model_fov_deg"), (int, float))
               and math.isfinite(row["real_model_fov_deg"])
               and abs(row["real_model_fov_deg"] - true_fov) <= tolerance
               for row in measurements)


def generate(clip: Path, seconds: float, fov: float = TRUE_FOV) -> None:
    if clip.exists():
        raise FileExistsError(f"refusing to overwrite {clip}")
    camera = CameraModel(fov, PITCH_DEG, 0.0, CAM_HEIGHT_M)
    writer = ClipWriter(clip, game="ets2")
    for i in range(round(seconds * SOURCE_HZ)):
        t = i / SOURCE_HZ
        writer.write_frame(t, render_calib(SPEED_MPS * t, camera))
        writer.write_telemetry(t, {"speed_mps": SPEED_MPS,
                                  "yaw_rate_rad_s": 0.0,
                                  "wheel_angle_rad": 0.0,
                                  "game_steer": 0.0, "user_steer": 0.0})
    meta = writer.close("synthetic-time-base")
    meta["truth"] = {"fov_h_deg": fov, "pitch_deg": PITCH_DEG,
                     "height_m": CAM_HEIGHT_M, "speed_mps": SPEED_MPS}
    (clip / "clip.json").write_text(json.dumps(meta, indent=2) + "\n")


def synthetic_fov(rate: int, true_fov: float = TRUE_FOV) -> float | None:
    """Exact optical flow isolates the review's time-span FOV regression.

    Keep this distinct from the real-model solve: a network can have a
    scene-dependent speed bias even at its trained frame interval.
    """
    def views(frame, calib):
        view = np.zeros((6, 1, 2))
        view[:, 0, 0] = calib.fov_h_deg
        view[:, 0, 1] = frame.item()
        return view, view

    def vision(img, big):
        span = float(img[0, 6, 0, 1] - img[0, 0, 0, 1])
        fov = float(img[0, 6, 0, 0])
        ratio = math.tan(math.radians(true_fov / 2)) / math.tan(math.radians(fov / 2))
        return SPEED_MPS * ratio * span / 0.2, 0.1

    stamped = "timestamp" in inspect.signature(FovSolver.update).parameters
    solver = FovSolver(vision, views)
    calib = Calibration(fov_h_deg=true_fov)
    for i in range(1000):
        t = i / rate
        solver.update(np.array([t]), calib, SPEED_MPS * 20 / rate, SPEED_MPS, 0.0,
                      **({"timestamp": t, "frame_span_s": 4 / rate} if stamped else {}))
        if solver.done:
            break
    return solver.result


def measure(clip: Path, out: Path) -> dict:
    # Bound CPU inference independently of the collector host's core count.
    import simsteer.core.model as adapter

    def session(path, providers):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), options,
                                   providers=["CPUExecutionProvider"])

    adapter._make_session = session
    stamped_frames = "timestamp" in inspect.signature(FrameQueue.push).parameters
    stamped_model = "timestamp" in inspect.signature(DrivingModel.step).parameters
    stamped_solver = "timestamp" in inspect.signature(FovSolver.update).parameters
    timed_decode = "frame_span_s" in inspect.signature(decode).parameters
    files = sorted((clip / "frames").glob("*.jpg"))
    times = [json.loads(line)["t"] for line in
             (clip / "frames.jsonl").read_text().splitlines()]
    meta = json.loads((clip / "clip.json").read_text())
    true_fov = float(meta["truth"]["fov_h_deg"])
    if abs(float(meta["fps"]) - SOURCE_HZ) > 1e-3:
        raise ValueError(f"clip must be {SOURCE_HZ} Hz")
    out.mkdir(parents=True, exist_ok=False)
    measurements = []
    for rate in RATES:
        model = DrivingModel(providers=["CPUExecutionProvider"])
        queue = FrameQueue()
        solver = FovSolver(model_vision(model), model_views)
        calib = Calibration(image_w=meta["width"], image_h=meta["height"], fov_h_deg=true_fov,
                            pitch_deg=meta["truth"]["pitch_deg"],
                            height_m=meta["truth"]["height_m"])
        ratios = []
        with (out / f"{rate}hz.jsonl").open("w") as log:
            for n, i in enumerate(range(0, len(files), SOURCE_HZ // rate)):
                frame = cv2.imread(str(files[i]))
                if frame is None:
                    raise ValueError(f"unreadable frame {files[i]}")
                t = float(times[i])
                images = queue.push(frame, calib, **({"timestamp": t}
                                                     if stamped_frames else {}))
                vision, policy = model.step(*images, **({"timestamp": t}
                                                        if stamped_model else {}))
                # The fallback is a measurement of the old four-frame pair,
                # not a correction applied to the baseline adapter.
                span = getattr(queue, "span_s", min(n, 4) / rate)
                decoded = decode(vision, policy, **({"frame_span_s": span}
                                                    if timed_decode else {}))
                raw_vx = float(decode(vision, policy).pose[0])
                if n >= 5:
                    solver.update(frame, calib, raw_vx, SPEED_MPS, 0.0,
                                  **({"timestamp": t, "frame_span_s": span}
                                     if stamped_solver else {}),
                                  **({"pair_ok": queue.pair_ok} if hasattr(queue, "pair_ok") else {}))
                ratio = float(decoded.pose[0]) / SPEED_MPS if decoded.pose is not None else None
                if t >= 2.0 and ratio is not None:
                    ratios.append(ratio)
                log.write(json.dumps({"t": t, "source_frame": i,
                                      "span_s": span, "raw_vx": raw_vx,
                                      "vx_ratio": ratio,
                                      "fov_progress": solver.progress,
                                      "fov_result": solver.result}) + "\n")
        row = {"hz": rate, "samples": len(ratios),
               "median_vx_ratio": median(ratios),
               "real_model_fov_deg": solver.result,
               "synthetic_fov_deg": synthetic_fov(rate, true_fov), "solver": solver.to_dict()}
        measurements.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "solver"}), flush=True)
    reference = measurements[-1]["median_vx_ratio"]
    for row in measurements:
        row["speed_delta_pct"] = 100 * (row["median_vx_ratio"] / reference - 1)
    result = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        text=True).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                         text=True).splitlines(),
        "harness_sha256": sha256(Path(__file__)),
        "model_identity": actual_model_identity(adapter.VISION_PATH, adapter.POLICY_PATH),
        "onnxruntime": ort.__version__, "inference_threads": 2,
        "clip_meta": meta, "frames_manifest_sha256": sha256(clip / "frames.jsonl"),
        "frame_bytes_sha256": hashlib.sha256(b"".join(
            bytes.fromhex(sha256(p)) for p in files)).hexdigest(),
        "measurements": measurements,
        "real_model_fov_target_met": fov_target_met(measurements, true_fov),
        "real_model_fov_tolerance_deg": 1.5,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--seconds", type=float, default=24.0)
    parser.add_argument("--fov", type=float, default=TRUE_FOV,
                        help="Known horizontal FOV when generating a clip")
    args = parser.parse_args()
    if args.generate:
        generate(args.clip, args.seconds, args.fov)
    measure(args.clip, args.out)


if __name__ == "__main__":
    main()
