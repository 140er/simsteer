"""Corrupt nested state, camera provenance and upgrade recovery."""

import json
import subprocess
import sys

import pytest

from simsteer.calib.flow import Obs, Step, flow_path
from simsteer.core.calibration import Calibration
from simsteer.core.learners.livecalib import LiveCalib
from simsteer.core.learners.liveparams import LiveParams, _seed_for
from simsteer.state import read_json_object, take_state_warnings
from simsteer.state_import import IMPORT_MARKER, import_legacy_state
from test_calib_flow import FakeLiveCalib, new_flow


BAD_VALUES = [1e20, -1, "x", float("nan"), None, True]
BAD_SHAPES = [None, [], "x", 5]


def flow_snapshot(step="camera"):
    value = new_flow(persist=False).to_dict()
    value["step"] = step
    return value


def load_flow(value, data_dir):
    flow_path("ets2", "gamepad").write_text(json.dumps(value))
    # Uncalibrated Camera must invalidate Steering without dereferencing
    # malformed saved controls. Use the constructor so the real loader runs.
    flow = new_flow(lc=FakeLiveCalib())
    if value["step"] == "steering":
        flow.tick(Obs(now=.05, dt=.05, v=20.))
    flow.engage_block()
    if value["step"] != "steering":
        flow.tick(Obs(now=.05, dt=.05, v=20.))
    assert flow.step is Step.FOV
    assert "using defaults" in take_state_warnings(data_dir)


@pytest.mark.parametrize("step", ["camera", "fov", "steering"])
@pytest.mark.parametrize("member", ["controls", "camera"])
@pytest.mark.parametrize("value", BAD_SHAPES)
def test_corrupt_check_context_resets_before_engage_or_tick(step, member, value, data_dir):
    raw = flow_snapshot(step)
    raw["check_context"] = {member: value}
    load_flow(raw, data_dir)


def livecalib_snapshot():
    value = LiveCalib().state_dict()
    value.update(certainty_state_version=1, bootstrap_samples=30,
                 bootstrap_rpy_sum=[-.5, .25, -.1], bootstrap_height_sum=36.,
                 bootstrap_height_n=30, reset_generation=3,
                 blocked_counts={"speed": 2}, last_rejection_reason="speed",
                 samples=7, rejected=2)
    return value


def load_livecalib(raw, data_dir):
    path = data_dir / "livecalib_state_ets2.json"
    path.write_text(json.dumps(raw))
    learner = LiveCalib(game="ets2")
    assert learner.samples == learner.rejected == learner.blocks == 0
    assert "using defaults" in take_state_warnings(data_dir)
    learner.save_state("ets2")
    assert read_json_object(path)["samples"] == 0


@pytest.mark.parametrize("field", ["bootstrap_samples", "bootstrap_height_sum",
                                   "bootstrap_height_n", "reset_generation"])
@pytest.mark.parametrize("value", BAD_VALUES)
def test_livecalib_camera_numbers_are_bounded(field, value, data_dir):
    raw = livecalib_snapshot()
    raw[field] = value
    load_livecalib(raw, data_dir)


@pytest.mark.parametrize("field", ["blocked_counts", "last_rejection_reason"])
@pytest.mark.parametrize("value", BAD_VALUES[:4])
def test_livecalib_blocked_diagnostics_are_guarded(field, value, data_dir):
    raw = livecalib_snapshot()
    raw[field] = {"speed": value} if field == "blocked_counts" else value
    load_livecalib(raw, data_dir)


@pytest.mark.parametrize("value", BAD_VALUES[:4])
def test_livecalib_blocked_count_keys_are_guarded(value, data_dir):
    raw = livecalib_snapshot()
    raw["blocked_counts"] = {value: 1}
    load_livecalib(raw, data_dir)


@pytest.mark.parametrize("value", BAD_SHAPES)
def test_livecalib_blocked_counts_requires_object(value, data_dir):
    raw = livecalib_snapshot()
    raw["blocked_counts"] = value
    load_livecalib(raw, data_dir)


@pytest.mark.parametrize("value", BAD_VALUES[:5] + [[0, 0], [[0, 0, 0]],
                                                  [0, 0, 0, 0], [0, True, 0], [0, 1e20, 0]])
def test_livecalib_bootstrap_pose_is_a_bounded_vector(value, data_dir):
    raw = livecalib_snapshot()
    raw["bootstrap_rpy_sum"] = value
    load_livecalib(raw, data_dir)


@pytest.mark.parametrize("version", [None, 1e20, -1, "x", float("nan"), True, 0, 2])
def test_old_livecalib_provenance_is_dropped_silently(version, data_dir, tmp_path):
    raw = livecalib_snapshot()
    if version is None:
        raw.pop("certainty_state_version")
    else:
        raw["certainty_state_version"] = version
    path = data_dir / "livecalib_state_ets2.json"
    path.write_text(json.dumps(raw))
    learner = LiveCalib(game="ets2")
    assert learner.samples == learner.blocks == 0
    assert not take_state_warnings(data_dir)
    target = tmp_path / "imported"
    assert import_legacy_state(data_dir, target)
    assert not (target / path.name).exists()
    assert not take_state_warnings(target)
    learner.save_state("ets2")


@pytest.mark.parametrize("field", ["no_progress_s", "discard_no_progress_s", "progress_mark",
                                   "samples_seen", "reset_seen", "blocks_seen", "jumps_seen",
                                   "discard_jumps"])
@pytest.mark.parametrize("value", BAD_VALUES)
def test_flow_camera_numbers_are_bounded(field, value, data_dir):
    raw = flow_snapshot()
    raw.setdefault("camera", {})[field] = value
    load_flow(raw, data_dir)


@pytest.mark.parametrize("value", BAD_SHAPES)
def test_flow_camera_requires_object(value, data_dir):
    raw = flow_snapshot()
    raw["camera"] = value
    load_flow(raw, data_dir)


@pytest.mark.parametrize("value", BAD_VALUES)
def test_flow_camera_count_values_are_bounded(value, data_dir):
    raw = flow_snapshot()
    raw.setdefault("camera", {})["no_progress_counts"] = {"speed": value}
    load_flow(raw, data_dir)


@pytest.mark.parametrize("value", BAD_VALUES[:4])
def test_flow_camera_count_keys_are_guarded(value, data_dir):
    raw = flow_snapshot()
    raw.setdefault("camera", {})["no_progress_counts"] = {value: 1}
    load_flow(raw, data_dir)


def test_current_camera_fields_round_trip_through_import(data_dir, tmp_path):
    raw = livecalib_snapshot()
    path = data_dir / "livecalib_state_ets2.json"
    path.write_text(json.dumps(raw))
    flow = new_flow()
    flow._camera_no_progress_s = 2.5
    flow.save()
    original = {p.name: p.read_bytes() for p in data_dir.glob("*.json")}
    target = tmp_path / "imported"
    assert import_legacy_state(data_dir, target)
    for name, content in original.items():
        assert (data_dir / name).read_bytes() == content
        assert read_json_object(target / name) == json.loads(content)
    assert read_json_object(target / path.name)["bootstrap_rpy_sum"] == [-.5, .25, -.1]


def test_rejection_saves_remain_throttled(data_dir, monkeypatch):
    import simsteer.core.learners.livecalib as module

    now, saved = [0.], []
    learner = LiveCalib(game="ets2", clock=lambda: now[0])
    monkeypatch.setattr(module, "atomic_save_json", lambda *args: saved.append(args) or True)
    now[0] = 5.
    learner.record_blocked(module.CameraBlockedReason.SPEED)
    for _ in range(10):
        now[0] += .05
        learner.record_blocked(module.CameraBlockedReason.SPEED)
    assert len(saved) == 1
    now[0] = 10.
    learner.record_blocked(module.CameraBlockedReason.SPEED)
    assert len(saved) == 2 and learner._last_diagnostic_save == 10.


@pytest.mark.parametrize("name", ["calibration_ets2.json", "liveparams_ets2_gamepad.json",
                                  "livecalib_state_ets2.json", "calibflow_ets2_gamepad.json"])
def test_corrupt_import_leaves_profile_seeds_available(name, data_dir, tmp_path, monkeypatch):
    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    content = '{"unfinished":'
    (source / name).write_text(content)
    assert import_legacy_state(source, target)
    assert not (target / name).exists()
    assert (source / name).read_text() == content
    assert "using defaults" in take_state_warnings(target)
    monkeypatch.setenv("SIMSTEER_DATA_DIR", str(target))
    seeds = {"gamepad": (3.5, .0015, 0.)}
    learner = LiveParams(game="ets2", device_kind="gamepad", seeds=seeds)
    assert (learner.a_linear, learner.b_quad, learner.c_bias) == _seed_for(seeds, "gamepad")
    assert Calibration.load(game="ets2", fov_h_deg=75.).fov_h_deg == 75.


def test_settings_import_keeps_valid_fields_when_another_is_bad(tmp_path):
    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    (source / "settings.json").write_text('{"game":"ats","max_width":"bad"}')
    assert import_legacy_state(source, target)
    assert read_json_object(target / "settings.json")["game"] == "ats"
    assert "using defaults" in take_state_warnings(target)


def test_first_drive_flags_import_once_and_keep_originals(tmp_path):
    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    name, content = "first_drive_done_ets2.flag", b"done\xff\n"
    (source / name).write_bytes(content)
    assert import_legacy_state(source, target)
    assert (target / name).read_bytes() == (source / name).read_bytes() == content
    assert name in read_json_object(target / IMPORT_MARKER)["files"]
    (target / name).write_bytes(b"player edit")
    assert not import_legacy_state(source, target)
    assert (target / name).read_bytes() == b"player edit"


@pytest.mark.parametrize("boundary", ["fsync", "replace"])
def test_first_drive_flag_interruption_preserves_previous_file(boundary, data_dir):
    path = data_dir / "first_drive_done_ets2.flag"
    path.write_bytes(b"previous")
    script = """
import os, sys
from pathlib import Path
from simsteer.state import atomic_save_bytes
def interrupt(*args): os._exit(9)
setattr(os, sys.argv[2], interrupt)
atomic_save_bytes(Path(sys.argv[1]), b'next')
"""
    run = subprocess.run([sys.executable, "-c", script, str(path), boundary], timeout=5)
    assert run.returncode == 9 and path.read_bytes() == b"previous"
