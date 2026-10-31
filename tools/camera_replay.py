"""Replay Camera inputs without capture, output devices, or saved calibration.

Use --kind gates for full learner-input logs, frames for older telemetry-only
logs, or samples for the JSONL produced by --collect-clip. --tree selects the
implementation under test so the same inputs can measure parent and head.
Telemetry-only results explicitly leave acceptance unknown when model fields
were not recorded. They never substitute an accepted synthetic observation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np


def read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                yield line, json.loads(text)


def replay(paths: list[Path], kind: str, *, game: str = "ac") -> tuple[dict, list[dict]]:
    # Completing blocks can call Calibration.save even with LiveCalib(game=None).
    # Keep those writes away from user state, including when imported as a tool.
    with tempfile.TemporaryDirectory(prefix="camera-replay-") as data:
        with patch.dict(os.environ, {"SIMSTEER_DATA_DIR": data}):
            return _replay(paths, kind, game=game)


def _replay(paths: list[Path], kind: str, *, game: str) -> tuple[dict, list[dict]]:
    from simsteer.core.calibration import Calibration
    from simsteer.core.learners.livecalib import LiveCalib

    reasons = Counter()
    details = []
    accepted = unknown = total = 0
    calibrated_at = []
    discarded_blocks = 0
    for path in paths:
        now = 0.0
        learner = LiveCalib(clock=lambda: now)
        calib = Calibration()
        start = None
        calibrated = False
        for line, row in read_rows(path):
            if kind == "frames" and row.get("calib") != "camera":
                continue
            if "camera_sample" in row and row["camera_sample"] is None:
                continue
            now = float(row.get("t", row.get("wall", line / 20)))
            if start is None:
                start = now
            total += 1
            if kind == "frames":
                speed = row.get("v_ego") if row.get("telemetry") == "ok" else None
                yaw = None
                if game == "ac" and speed is not None and row.get("wheel_angle") is not None:
                    # AC's logged synthetic wheel angle uses a 4 m wheelbase.
                    yaw = math.tan(row["wheel_angle"]) * speed / 4.0
                # Deliberately fail the first model gate. Earlier failures are
                # measurable; reaching model_speed means the real pose is unknown.
                inputs = dict(pose=np.zeros(3), road_transform=None,
                              actual_yaw_rate=yaw, v_ego=speed,
                              game_steer=row.get("observed_axis"))
            else:
                sample = row.get("camera_sample") or row
                inputs = {name: (None if sample.get(name) is None else np.asarray(sample[name]))
                          for name in ("pose", "road_transform", "pose_std",
                                       "road_transform_std", "wide_from_device_euler")}
                inputs.update(actual_yaw_rate=sample.get("actual_yaw_rate", sample.get("actual_yaw")),
                              v_ego=sample.get("v_ego", sample.get("speed")),
                              game_steer=sample.get("game_steer"))
                # New logs can preserve the exact warp baseline per observation.
                for name, value in sample.get("calibration", dict(pitch_deg=0., yaw_deg=0.)).items():
                    setattr(calib, name, value)
            previous = learner.samples
            learner.update(calib, **inputs)
            delta = int(learner.samples > previous or (
                learner.samples < previous and learner.last_rejection_reason is None))
            reason = learner.last_rejection_reason
            if kind == "frames" and reason == "model_speed":
                reason = "model_inputs_unlogged"
                unknown += 1
            elif kind == "frames" and game != "ac" and reason not in ("missing_speed", "speed"):
                reason = "yaw_and_model_inputs_unlogged"
                unknown += 1
            accepted += delta
            if reason and not delta:
                reasons[str(getattr(reason, "value", reason))] += 1
            details.append(dict(source=path.name, line=line, t=now, accepted=delta,
                                reason=getattr(reason, "value", reason),
                                block_discarded=reason == "block_jump"))
            if learner.status_label == "CALIBRATED" and not calibrated:
                calibrated = True
                calibrated_at.append(dict(source=path.name, line=line,
                                          elapsed_s=round(now - start, 3)))
        discarded_blocks += learner.rej_block_jump
    assert total == accepted + sum(reasons.values()), "each frame needs an outcome"
    return dict(frames=total, accepted=None if unknown else accepted,
                measured_accepted=accepted, unresolved=unknown,
                reasons=dict(reasons.most_common()),
                coverage="telemetry gates only" if kind == "frames" else "full learner inputs",
                first_calibrated=calibrated_at, discarded_blocks=discarded_blocks,
                inputs=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in paths]), details


def collect_clip(clip: Path, output: Path, limit: int, models: Path | None = None) -> None:
    """Cache real model outputs once, with a frozen zero-pose warp for both runs."""
    import onnxruntime as ort
    from simsteer.core.calibration import Calibration
    from simsteer.core import model as model_module
    from simsteer.core.postprocess import decode
    from simsteer.core.preprocess import FrameQueue
    from simsteer.runtime.replay import ReplayClock, ReplaySource, ReplayTelemetry, clip_meta

    def session(path, providers):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])

    model_module._make_session = session
    if models is not None:
        model_module.VISION_PATH = models / "driving_vision.onnx"
        model_module.POLICY_PATH = models / "driving_policy.onnx"
    model = model_module.DrivingModel()
    clock = ReplayClock()
    source = ReplaySource(clip, clock)
    telemetry = ReplayTelemetry(clip, clock)
    meta = clip_meta(clip)
    calib = Calibration(image_w=meta["width"], image_h=meta["height"], fov_h_deg=75.)
    queue = FrameQueue()
    output.parent.mkdir(parents=True, exist_ok=True)
    with source, output.open("w") as stream:
        for _ in range(limit):
            frame = source.grab()
            if frame is None:
                break
            decoded = decode(*model.step(*queue.push(frame, calib)))
            sample = {name: (value.tolist() if value is not None else None)
                      for name in ("pose", "road_transform", "pose_std",
                                   "road_transform_std", "wide_from_device_euler")
                      for value in [getattr(decoded, name)]}
            sample.update(t=clock.now(), speed=telemetry.speed_mps(),
                          actual_yaw=telemetry.yaw_rate_rad_s(),
                          game_steer=telemetry.game_steer(),
                          calibration={name: getattr(calib, name)
                                       for name in ("pitch_deg", "yaw_deg", "height_m")})
            stream.write(json.dumps(sample, allow_nan=False) + "\n")
    output.with_suffix(".meta.json").write_text(json.dumps(dict(
        clip=meta, models=model.certification_identity,
        scope="fixed 75 degree FOV and zero pose; model outputs cached for identical learner replay"),
        indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--kind", choices=("gates", "frames", "samples"))
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--game", default="ac", choices=("ac", "ets2"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collect-clip", type=Path)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--revision", help="revision label for a source archive without git metadata")
    args = parser.parse_args()
    sys.path.insert(0, str(args.tree.resolve()))
    if args.collect_clip:
        collect_clip(args.collect_clip, args.out, args.limit, args.models)
        return
    if not args.kind or not args.input:
        parser.error("replay needs --kind and --input")
    summary, details = replay(args.input, args.kind, game=args.game)
    summary["revision"] = args.revision or subprocess.check_output(
        ["git", "-C", str(args.tree), "rev-parse", "HEAD"], text=True).strip()
    summary["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    with args.out.with_suffix(".frames.jsonl").open("w") as stream:
        for row in details:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
