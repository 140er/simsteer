"""State recovery regressions for pending telemetry, imports and queued saves."""

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from simsteer.calib.flow import Step
from simsteer.calib.score import evaluate
from simsteer.core.learners.livecalib import LiveCalib
from simsteer.core.learners.liveparams import LiveParams
from simsteer.state import atomic_save_json, read_json_object
from test_calib_flow import Driver, FakeLiveCalib, FakeLiveParams, _to_check, new_flow
from test_calib_score import _drive
from test_state_context import STATE_FILES, _save


@pytest.mark.parametrize("next_vehicle", ["test-truck", "other-truck"])
def test_launch_waits_for_vehicle_without_discarding_certificate(next_vehicle):
    vehicle = ["test-truck"]
    controls = lambda: {"vehicle": vehicle[0], "device_id": 1}
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    original = new_flow(lc, lp, certification_context=controls)
    _to_check(original, lc, lp)
    original._finish_check(evaluate(_drive()))
    result = original.result.copy()
    vehicle[0] = None
    resumed = new_flow(lc, lp, cal=original.calib, certification_context=controls)
    assert resumed.step is Step.DONE and resumed.result == result
    assert not resumed.calibrated and not resumed.ai_steering_ready
    assert resumed.engage_block() == "Waiting for vehicle identity"
    resumed.confirm_check()
    Driver(resumed).frames(3)
    assert resumed.step is Step.DONE and resumed.result == result
    vehicle[0] = next_vehicle
    Driver(resumed).frames(1)
    if next_vehicle == "test-truck":
        assert resumed.calibrated and resumed.ai_steering_ready
        assert resumed.result == result and not resumed.engage_block()
    else:
        assert not resumed.calibrated and not resumed.ai_steering_ready
        assert resumed.engage_block() == "Check again: press INSERT"


@pytest.mark.parametrize("controls", [{}, {"vehicle": None}])
def test_unavailable_vehicle_reader_keeps_matching_certificate(controls):
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    original = new_flow(lc, lp, certification_context=lambda: controls.copy())
    _to_check(original, lc, lp)
    original._finish_check(evaluate(_drive()))
    for _ in range(2):
        resumed = new_flow(lc, lp, cal=original.calib,
                           certification_context=lambda: controls.copy())
        assert resumed.calibrated and resumed.ai_steering_ready
        assert not resumed.engage_block()


def test_only_capable_readers_put_vehicle_in_context():
    from simsteer.runtime.telemetry import vehicle_context
    from simsteer.runtime.telemetry.scs_shm import ScsTelemetry, SHARED_MEM_SIZE, OFF_PLUGIN_REV
    import struct

    assert vehicle_context(SimpleNamespace()) == {}
    assert vehicle_context(SimpleNamespace(vehicle_id=lambda: None)) == {"vehicle": None}
    tel = ScsTelemetry.__new__(ScsTelemetry)
    tel._mm = bytearray(SHARED_MEM_SIZE)
    struct.pack_into("<I", tel._mm, OFF_PLUGIN_REV, 99)
    assert vehicle_context(tel) == {}
    struct.pack_into("<I", tel._mm, OFF_PLUGIN_REV, 12)
    assert vehicle_context(tel) == {"vehicle": None}


@pytest.mark.parametrize("kind,field", [("liveparams", "samples"), ("livecalib", "rejected"),
                                      ("livecalib", "block_n")])
@pytest.mark.parametrize("value", [1e20, True, 1.5, -1, [], {}, "4", None])
def test_invalid_learner_counts_reset_with_banner(kind, field, value, data_dir, caplog):
    _save(kind)
    path = data_dir / STATE_FILES[kind]
    raw = json.loads(path.read_text())
    raw[field] = value
    path.write_text(json.dumps(raw))
    if kind == "liveparams":
        learner = LiveParams(game="ets2", device_kind="gamepad")
        assert learner.samples == 0
        learner.save()
    else:
        learner = LiveCalib()
        learner.load_state("ets2")
        assert learner.rejected == learner._block_n == 0
        learner.save_state("ets2")
    assert "using defaults" in caplog.text.lower()


@pytest.mark.parametrize("value", ["1", True, None, [], {}])
def test_wrong_learner_coefficient_types_report_defaults(value, data_dir, caplog):
    path = data_dir / STATE_FILES["liveparams"]
    path.write_text(json.dumps({"a": value, "samples": 100}))
    assert LiveParams(game="ets2", device_kind="gamepad").samples == 0
    assert "using defaults" in caplog.text.lower()


@pytest.mark.parametrize("kind,field,value", [
    ("liveparams", "coverage_speed_ranges", {"7": [True, 22]}),
    ("liveparams", "coverage_speed_min", 1001),
    ("livecalib", "rpys", [[0, 100, 0]]),
    ("livecalib", "heights", [-1]),
])
def test_learner_history_ranges_are_guarded(kind, field, value, data_dir, caplog):
    _save(kind)
    path = data_dir / STATE_FILES[kind]
    raw = read_json_object(path)
    raw[field] = value
    path.write_text(json.dumps(raw))
    if kind == "liveparams":
        assert LiveParams(game="ets2", device_kind="gamepad").samples == 0
    else:
        learner = LiveCalib()
        learner.load_state("ets2")
        assert learner.blocks == 0
    assert "using defaults" in caplog.text.lower()


def test_imports_legacy_state_once_and_keeps_originals(tmp_path, monkeypatch):
    from simsteer.state_import import import_legacy_state, IMPORT_MARKER

    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    with monkeypatch.context() as env:
        env.setenv("SIMSTEER_DATA_DIR", str(source))
        for kind in STATE_FILES:
            _save(kind)
    originals = {p.name: p.read_bytes() for p in source.glob("*.json")}
    assert import_legacy_state(source, target)
    assert all((target / name).exists() for name in originals)
    assert all((source / name).read_bytes() == raw for name, raw in originals.items())
    assert read_json_object(target / IMPORT_MARKER)["complete"]
    (source / "controller_ats.json").write_text('{"steer_max": 0.4}')
    assert not import_legacy_state(source, target)
    assert not (target / "controller_ats.json").exists()


def test_import_preserves_existing_user_state(tmp_path):
    from simsteer.state_import import import_legacy_state

    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    target.mkdir()
    (source / "calibration_ets2.json").write_text('{"fov_h_deg": 80}')
    (target / "settings.json").write_text('{"game": "ats"}')
    assert not import_legacy_state(source, target)
    assert not (target / "calibration_ets2.json").exists()
    assert json.loads((target / "settings.json").read_text()) == {"game": "ats"}


def test_import_validates_corrupt_legacy_state(tmp_path, caplog):
    from simsteer.state_import import import_legacy_state

    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    (source / "liveparams_ets2_gamepad.json").write_text('{"a": 1, "samples": 1e20}')
    (source / "calibration_ets2.json").write_text('{"unfinished":')
    assert import_legacy_state(source, target)
    assert not (target / "liveparams_ets2_gamepad.json").exists()
    assert not (target / "calibration_ets2.json").exists()
    assert "using defaults" in caplog.text.lower()


def test_live_entrypoint_imports_before_loading_settings(tmp_path, monkeypatch):
    import simsteer.state_import as importer
    import simsteer.runtime.loop as loop
    from simsteer.app.main import main

    source = tmp_path / "old"
    source.mkdir()
    original = '{"game": "ats", "max_width": 1234}'
    (source / "settings.json").write_text(original)
    monkeypatch.delenv("SIMSTEER_DATA_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user"))
    monkeypatch.setattr(importer, "bundle_dir", lambda: source)
    captured = []
    monkeypatch.setattr(loop, "run", lambda settings, game, **kwargs:
                        captured.append((settings.game, settings.max_width)) or 0)
    assert main(["--no-gamepad", "--no-tuner", "--headless"]) == 0
    assert captured == [("ats", 1234)]
    assert (source / "settings.json").read_text() == original


def test_import_retry_keeps_completed_files(tmp_path, monkeypatch):
    import simsteer.state_import as importer

    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    (source / "settings.json").write_text('{"game": "ets2"}')
    (source / "controller_ets2.json").write_text('{"steer_max": 0.4}')
    save = importer.atomic_save_json
    with monkeypatch.context() as patch:
        patch.setattr(importer, "atomic_save_json", lambda p, d: False if p.name == "settings.json" else save(p, d))
        assert not importer.import_legacy_state(source, target)
    assert not read_json_object(target / importer.IMPORT_MARKER)["complete"]
    (target / "controller_ets2.json").write_text('{"steer_max": 0.6}')
    assert importer.import_legacy_state(source, target)
    assert read_json_object(target / "controller_ets2.json")["steer_max"] == 0.6


@pytest.mark.parametrize("field,value", [("files", {}), ("complete", "yes"),
                                       ("pending", ["../settings.json"])])
def test_corrupt_import_manifest_is_guarded(field, value, data_dir, caplog):
    from simsteer.state_import import import_legacy_state, IMPORT_MARKER

    source = data_dir / "old"
    source.mkdir()
    raw = {"version": 1, "complete": False, "pending": ["settings.json"], "files": []}
    raw[field] = value
    (data_dir / IMPORT_MARKER).write_text(json.dumps(raw))
    assert not import_legacy_state(source, data_dir)
    assert "import paused" in caplog.text.lower()


def test_session_saves_use_one_worker_and_keep_latest_snapshot(data_dir, monkeypatch):
    from simsteer.state import background_saves

    entered, release = threading.Event(), threading.Event()
    writers = []
    fsync = os.fsync

    def slow_sync(fd):
        writers.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        fsync(fd)

    monkeypatch.setattr(os, "fsync", slow_sync)
    path = data_dir / "queued.json"
    try:
        with background_saves():
            assert atomic_save_json(path, {"value": 0})
            assert entered.wait(2)
            for value in range(1, 11):
                assert atomic_save_json(path, {"value": value})
            # The caller reached this point while fsync is still blocked.
            assert not release.is_set()
            release.set()
    finally:
        release.set()
    assert read_json_object(path) == {"value": 10}
    assert len(writers) == 2 and len(set(writers)) == 1
    assert writers[0] != threading.get_ident()


def test_queued_payload_is_immutable_and_reads_flush(data_dir):
    from simsteer.state import background_saves

    path = data_dir / "queued.json"
    value = {"samples": [1]}
    with background_saves():
        atomic_save_json(path, value)
        value["samples"].append(2)
        assert read_json_object(path) == {"samples": [1]}


def test_queued_deletion_cannot_resurrect_state(data_dir):
    from simsteer.state import background_saves, delete_state

    path = data_dir / "queued.json"
    with background_saves():
        atomic_save_json(path, {"samples": 1})
        delete_state(path)
    assert not path.exists()


def test_real_loop_persists_on_worker_and_flushes_before_return(data_dir, monkeypatch):
    from test_output_safety import drive

    writers = []
    fsync = os.fsync

    def record(fd):
        writers.append(threading.get_ident())
        fsync(fd)

    monkeypatch.setattr(os, "fsync", record)
    drive(monkeypatch, frames=2, force=False)
    assert writers and all(writer != threading.get_ident() for writer in writers)
    for family in ("calibration", "liveparams", "livecalib", "flow"):
        assert read_json_object(data_dir / STATE_FILES[family])


def test_failed_background_save_preserves_prior_snapshot_and_banner(data_dir, monkeypatch):
    from simsteer.state import background_saves, take_state_warnings

    path = data_dir / "settings.json"
    assert atomic_save_json(path, {"value": 1})

    def reject(*args):
        raise PermissionError("replacement refused")

    monkeypatch.setattr(os, "replace", reject)
    with background_saves():
        atomic_save_json(path, {"value": 2})
    assert read_json_object(path) == {"value": 1}
    assert "not saved" in take_state_warnings(data_dir)


def test_load_cleans_stale_temps_and_keeps_recent_siblings(data_dir):
    path = data_dir / "settings.json"
    path.write_text('{}')
    stale = data_dir / ".settings.json.old.tmp"
    recent = data_dir / ".settings.json.recent.tmp"
    unrelated = data_dir / ".controller.json.old.tmp"
    for p in (stale, recent, unrelated):
        p.write_text("partial")
    then = time.time() - 2 * 86400
    os.utime(stale, (then, then))
    assert read_json_object(path) == {}
    assert not stale.exists() and recent.exists() and unrelated.exists()


def test_state_ignore_patterns_only_apply_at_root():
    for name in ("tests/fixtures/calibration_ets2.json", "examples/controller_ac.json",
                 "editor/settings.json", "examples/calibflow_game.json"):
        assert subprocess.run(["git", "check-ignore", "-q", name]).returncode == 1


def test_camera_defaults_are_generic():
    from simsteer.core.calibration import Calibration
    assert (Calibration().image_w, Calibration().image_h) == (1920, 1080)
