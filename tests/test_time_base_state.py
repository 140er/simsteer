"""Timed FOV evidence follows the shared state persistence contract."""

import json

import pytest

from simsteer.calib.flow import flow_path
from simsteer.core.calibration import Calibration
from simsteer.state import background_saves, read_json_object, take_state_warnings
from simsteer.state_import import import_legacy_state
from test_calib_flow import Driver, new_flow


def test_timed_fov_evidence_round_trips_via_the_atomic_writer(data_dir, monkeypatch):
    import simsteer.state as state

    replaced = []
    replace = state.os.replace

    def record(source, target):
        replaced.append(target)
        return replace(source, target)

    monkeypatch.setattr(state.os, "replace", record)
    with background_saves():
        flow = new_flow(cal=Calibration(fov_h_deg=95.))
        Driver(flow).frames(60)
        assert flow.fov_solver.stage == "bracket"
        flow.save()
        expected = json.loads(json.dumps(flow.to_dict()))["fov"]
        resumed = new_flow(cal=flow.calib)
        assert json.loads(json.dumps(resumed.to_dict()))["fov"] == expected
    path = flow_path("ets2", "gamepad")
    assert path.parent == data_dir and path in replaced
    assert read_json_object(path)["fov"] == expected
    assert expected["solver"]["time_base_version"] == 1
    assert not take_state_warnings(data_dir)


@pytest.mark.parametrize("field,value", [
    ("time_base_version", True), ("tick", -1), ("tick", 1e20), ("tick", True),
    ("shifts", 3), ("ratios", [True]), ("ratios", [1e20]),
    ("ratios", "bad"), ("candidates", ["75"] * 4),
    ("cand_ratios", [[1e20]] * 4), ("stage", "bad"),
    ("geometry", [None] * 9),
    ("geometry", [[2, 2], 1920, 1080, 90., .5, .5, 0., 0., "yes"]),
])
@pytest.mark.parametrize("through_import", [False, True], ids=["load", "import"])
def test_invalid_timed_evidence_is_rejected_by_load_and_import(
        field, value, through_import, data_dir):
    raw = new_flow(persist=False).to_dict()
    raw["fov"]["solver"][field] = value
    path = flow_path("ets2", "gamepad")
    if through_import:
        source = data_dir / "legacy"
        source.mkdir()
        (source / path.name).write_text(json.dumps(raw))
        assert import_legacy_state(source, data_dir)
        assert not path.exists()
        assert "using defaults" in take_state_warnings(data_dir)
    else:
        path.write_text(json.dumps(raw))
        assert new_flow().fov_solver.samples == 0
        assert "using defaults" in take_state_warnings(data_dir)
