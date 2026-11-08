import importlib.util
import json
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("geometry_check", Path(__file__).resolve().parents[1] / "tools" / "geometry_check.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def recording(**changes):
    return {"model_fresh": True, "actual_speed_mps": 20., "pose_speed_mps": 20.,
            "pose_vertical_mps": 0., "lane_width_m": 3.7, "model_height_m": 1.4,
            "calibration_height_m": 1.4, "calibration_pitch_deg": 2.,
            "camera_fy_px": 500., "camera_cy_px": 270., **changes}


def test_audit_synthetic_recording_and_incident(tmp_path, capsys):
    rows = [recording() for _ in range(100)]
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows))
    assert geometry.main([str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result["checks"].values()) == {"ok"}
    metrics = result["metrics"]
    assert metrics["speed_ratio"]["median"] == 1.
    assert metrics["calibration_horizon_px"]["mean"] == pytest.approx(270 - 500 * math.tan(math.radians(2)))
    assert metrics["implied_height_m"]["median"] == 1.4
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({"schema": 1, "samples": rows}))
    assert geometry.audit(geometry.read_recording(incident)) == result


def test_audit_flags_scale_pitch_height_and_lane_width():
    result = geometry.audit([recording(pose_speed_mps=10., pose_vertical_mps=1.,
                                       model_height_m=3., lane_width_m=6.)])
    assert set(result["checks"].values()) == {"warning"}
    assert result["metrics"]["horizon_delta_px"]["mean"] > 0.


def test_audit_old_logs_and_missing_data():
    old = {"actual_speed_mps": 20., "pose_vx": 20., "road_height_m": 1.4,
           "camera_sample": {"pose": [20., 0., 0.],
                             "calibration": {"pitch_deg": 2., "height_m": 1.4}}}
    result = geometry.audit([old, recording(model_fresh=False), recording(actual_speed_mps=0.),
                             recording(actual_speed_mps=None), {}])
    assert result["checks"]["speed_ratio"] == "ok"
    assert result["checks"]["lane_width_m"] == "insufficient"
    assert result["metrics"]["speed_ratio"]["n"] == 1
