"""Live-loop regressions that can also run against the unmodified base."""

import json
import re
from types import SimpleNamespace

from diagnostics_harness import drive


def test_auto_disengage_writes_numeric_incident(monkeypatch, tmp_path):
    state = drive(monkeypatch, tmp_path, gap_at=30, frames=40)
    assert state.rc == 0
    assert any(row["output_fault"] == "Low frame rate" for row in state.rows)
    files = list((tmp_path / "logs" / "incidents").glob("*.json"))
    assert files, "auto-disengagement must leave a flight recording"
    assert len(files) == 1
    assert re.fullmatch(r"\d{8}T\d{12}Z.json", files[0].name)
    payload = json.loads(files[0].read_text())
    assert payload["schema"] == 1
    assert 4 in payload["events"][0]["codes"]
    assert len(payload["samples"]) == 30
    assert payload["samples"][-1]["fault"]
    assert payload["samples"][-2]["engaged"]

    def numeric(value):
        if isinstance(value, dict):
            return all(numeric(x) for x in value.values())
        if isinstance(value, list):
            return all(numeric(x) for x in value)
        return value is None or isinstance(value, (int, float, bool))

    assert numeric(payload)


def test_new_columns_and_original_values(monkeypatch, tmp_path):
    state = drive(monkeypatch, tmp_path, frames=4, log_timing=True)
    row = state.rows[0]
    assert {"desire_state", "lane_line_probs", "road_edge_probs", "plan_speed_mps",
            "pose_speed_mps", "lane_offset_m", "plan_curvature", "raw_k",
            "steer_authority", "understeer_ff", "speed_scale",
            "feedforward_axis", "trim_axis", "probe_axis",
            "final_axis", "executor_axis", "draw_ms", "timing_stats"} <= row.keys()
    assert row["desire_state"] == [0.] * 8
    assert row["lane_line_probs"] == [.95] * 4
    assert row["road_edge_probs"] is None
    assert row["lane_width_m"] == 3.6 and row["lane_offset_m"] == 0.
    assert row["plan_speed_mps"] == row["pose_speed_mps"] == 22.
    assert row["plan_curvature"] == state.decoded.plan[0, 14] / 22.
    assert row["probe_axis"] == 0.
    assert row["understeer_ff"] == 1. and row["speed_scale"] == 1.
    assert row["final_axis"] == state.writes[0]
    assert {"cap_ms", "warp_ms", "model_ms", "ctrl_ms", "frame_ms"} <= row.keys()
    assert row["t"] == .05 and row["frame"] == 1
    assert row["v_ego"] == 22. and row["telemetry"] == "ok"
    assert row["output_fault"] is None and row["engaged"]
    assert row["curvature"] == round(row["raw_k"] * row["steer_authority"], 7)
    assert row["steer_axis"] == round(row["final_axis"], 6)
    assert row["camera_sample"] is None


def test_under_rate_banner_and_debug_timings(monkeypatch, tmp_path):
    from simsteer.runtime.diagnostics import SLOW_MODEL_BANNER

    state = drive(monkeypatch, tmp_path, frames=140, period=.1, render=True)
    assert state.rc == 0
    assert any(hud.banner_text == SLOW_MODEL_BANNER for hud in state.hud)
    assert any("capture_wait" in text for text, _ in state.hud[-1].dev_lines)


def test_diagnostics_failure_does_not_change_device_writes(monkeypatch, tmp_path, caplog):
    from simsteer.runtime.diagnostics import RunDiagnostics

    healthy = tmp_path / "healthy"
    healthy.mkdir()
    with monkeypatch.context() as patch:
        expected = drive(patch, healthy, frames=50)

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostics failure")

    monkeypatch.setattr(RunDiagnostics, "_finish", fail)
    actual = drive(monkeypatch, tmp_path, frames=50)
    assert actual.rc == 0 and actual.writes == expected.writes
    # Every old field retains its value on every tick, even on failure.
    for before, after in zip(expected.rows, actual.rows):
        assert all(after[key] == value for key, value in before.items() if key in after)
    assert caplog.text.count("Session diagnostics unavailable") == 1


def test_executor_fault_is_an_incident(monkeypatch, tmp_path):
    from simsteer.runtime import loop

    class Executor:
        last_axis = .125
        ticks = 0

        def __init__(self, *args):
            pass

        def take_fault(self):
            self.ticks += 1
            return "Steering executor failed" if self.ticks == 4 else ""

        def reset(self):
            self.last_axis = None

        def submit(self, *args):
            self.last_axis = .125

        def close(self):
            pass

    monkeypatch.setattr(loop, "LiveClock", SimpleNamespace)
    monkeypatch.setattr(loop, "SteeringOutput", Executor)
    state = drive(monkeypatch, tmp_path, frames=10)
    assert state.rc == 0
    assert state.rows[2]["executor_axis"] == .125
    files = list((tmp_path / "logs" / "incidents").glob("*.json"))
    payload = json.loads(files[0].read_text())
    assert payload["events"][0]["codes"] == [5]
    assert payload["samples"][-1]["fault"]


def test_output_direction_change_is_an_incident(monkeypatch, tmp_path):
    def flip(state):
        if state.frame == 4:
            state.flow.observe_output = lambda **kwargs: True

    state = drive(monkeypatch, tmp_path, frames=10, on_frame=flip)
    assert state.rc == 0
    assert state.rows[3]["output_fault"] == "Output direction changed"
    files = list((tmp_path / "logs" / "incidents").glob("*.json"))
    payload = json.loads(files[0].read_text())
    assert payload["events"][0]["codes"] == [2]


def test_legacy_fields_match_when_diagnostics_disabled(monkeypatch, tmp_path):
    from simsteer.runtime.diagnostics import RunDiagnostics

    normal = tmp_path / "normal"
    normal.mkdir()
    with monkeypatch.context() as patch:
        expected = drive(patch, normal, frames=30)
    monkeypatch.setattr(RunDiagnostics, "begin_tick", lambda self: setattr(self, "disabled", True))
    actual = drive(monkeypatch, tmp_path, frames=30)
    assert actual.writes == expected.writes
    for before, after in zip(expected.rows, actual.rows):
        assert {key: before[key] for key in after} == after
