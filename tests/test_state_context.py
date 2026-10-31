"""Crash recovery, context binding and clean source-run state."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simsteer.app.settings import Settings
from simsteer.calib.flow import Step
from simsteer.calib.score import evaluate
from simsteer.core.calibration import Calibration
from simsteer.core.control import ControllerConfig
from simsteer.paths import data_dir as user_data_dir
from test_calib_flow import FakeLiveCalib, FakeLiveParams, _to_check, new_flow
from test_calib_score import _drive


STATE_FILES = {
    "calibration": "calibration_ets2.json",
    "controller": "controller_ets2.json",
    "liveparams": "liveparams_ets2_gamepad.json",
    "livecalib": "livecalib_state_ets2.json",
    "flow": "calibflow_ets2_gamepad.json",
    "settings": "settings.json",
}


def _save(kind):
    from simsteer.core.learners.livecalib import LiveCalib
    from simsteer.core.learners.liveparams import LiveParams

    if kind == "calibration":
        Calibration().save(game="ets2")
    elif kind == "controller":
        ControllerConfig().save(game="ets2")
    elif kind == "liveparams":
        LiveParams(game="ets2", device_kind="gamepad").save()
    elif kind == "livecalib":
        LiveCalib().save_state("ets2")
    elif kind == "flow":
        new_flow().save()
    else:
        Settings().save()


def _interrupted_writer(kind, root, connection):
    os.environ["SIMSTEER_DATA_DIR"] = root

    def pause():
        connection.send("writing")
        connection.recv()  # The parent kills this process at the write boundary.

    def direct_write(path, text, *args, **kwargs):
        # Reproduce a partial direct write on the old implementation. The
        # atomic implementation reaches the replace hook with a complete temp.
        with path.open("w", encoding="utf-8") as stream:
            stream.write(text[:max(1, len(text) // 2)])
            stream.flush()
            os.fsync(stream.fileno())
            pause()

    Path.write_text = direct_write
    os.replace = lambda *args, **kwargs: pause()
    _save(kind)


@pytest.mark.parametrize("kind", STATE_FILES)
def test_kill_mid_write_preserves_previous_state(kind, data_dir):
    path = data_dir / STATE_FILES[kind]
    _save(kind)
    previous = json.loads(path.read_text(encoding="utf-8"))
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe()
    process = context.Process(target=_interrupted_writer, args=(kind, str(data_dir), writer))
    process.start()
    try:
        assert reader.poll(10), "writer did not reach the interruption point"
        assert reader.recv() == "writing"
        process.kill()
        process.join(timeout=5)
        assert not process.is_alive()
        assert json.loads(path.read_text(encoding="utf-8")) == previous
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        reader.close()
        writer.close()


@pytest.mark.parametrize("kind", STATE_FILES)
def test_state_write_fsyncs_temp_before_replace(kind, data_dir, monkeypatch):
    events = []
    fsync, replace = os.fsync, os.replace

    def sync(fd):
        events.append("fsync")
        fsync(fd)

    def swap(source, target):
        source, target = Path(source), Path(target)
        assert source != target and source.parent == target.parent == data_dir
        json.loads(source.read_text(encoding="utf-8"))
        events.append("replace")
        replace(source, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", swap)
    _save(kind)
    assert "replace" in events and events.index("fsync") < events.index("replace")
    assert not list(data_dir.glob("*.tmp"))


@pytest.mark.parametrize("kind", STATE_FILES)
def test_failed_replace_keeps_snapshot_and_reports_warning(kind, data_dir, monkeypatch, caplog):
    _save(kind)
    path = data_dir / STATE_FILES[kind]
    previous = path.read_bytes()

    def reject(*args):
        raise OSError("replacement refused")

    monkeypatch.setattr(os, "replace", reject)
    _save(kind)
    assert path.read_bytes() == previous
    assert "not saved" in caplog.text.lower()
    assert not list(data_dir.glob("*.tmp"))


@pytest.mark.parametrize("config,field", [(ControllerConfig, "lookahead_s"),
                                        (Calibration, "fov_h_deg")])
@pytest.mark.parametrize("raw", ['{"unfinished":', "[]", "null", '"invalid"',
                                 '{"FIELD": "bad"}', '{"FIELD": true}',
                                 '{"FIELD": NaN}', '{"FIELD": 1e999}'])
def test_corrupt_config_starts_with_defaults_and_warning(config, field, raw, data_dir, caplog):
    path = data_dir / "broken.json"
    path.write_text(raw.replace("FIELD", field), encoding="utf-8")
    assert config.load(path=path) == config()
    assert "using defaults" in caplog.text.lower()


@pytest.mark.parametrize("filename", ["controller_ets2.json", "calibration_ets2.json"])
def test_corrupt_config_banner_reaches_live_hud(filename, data_dir, monkeypatch):
    from test_output_safety import drive

    (data_dir / filename).write_text('{"unfinished":', encoding="utf-8")
    states = []
    drive(monkeypatch, frames=2, force=False, tuner_factory=lambda st, pad: SimpleNamespace(
        take_restart_request=lambda: None, update_state=states.append))
    assert any("using defaults" in s.banner_text.lower() for s in states)
    assert any(filename.split("_")[0] in s.banner_text.lower() for s in states)


@pytest.mark.parametrize("config,field,value", [
    (Calibration, "image_w", 0), (Calibration, "image_h", -1),
    (Calibration, "fov_h_deg", 180), (Calibration, "height_m", 0),
    (Calibration, "crop_top_pct", 0.5), (Calibration, "lateral_sign", 0),
    (ControllerConfig, "wheelbase_m", 0), (ControllerConfig, "steer_max", 2),
    (ControllerConfig, "default_speed", -1), (ControllerConfig, "max_speed_mps", 0),
])
def test_config_rejects_invalid_ranges(config, field, value, data_dir, caplog):
    path = data_dir / "invalid.json"
    path.write_text(json.dumps({field: value}), encoding="utf-8")
    assert config.load(path=path) == config()
    assert "using defaults" in caplog.text.lower()


@pytest.mark.parametrize("config", [Calibration, ControllerConfig])
def test_unreadable_config_recovers_with_game_defaults(config, data_dir, monkeypatch, caplog):
    path = data_dir / "unreadable.json"
    path.touch()

    def reject(*args, **kwargs):
        raise PermissionError("unreadable file")

    monkeypatch.setattr(Path, "read_text", reject)
    if config is Calibration:
        loaded = config.load(path=path, fov_h_deg=75.0)
        assert loaded.fov_h_deg == 75.0
    else:
        loaded = config.load(path=path, seed={"steer_max": 0.4})
        assert loaded.steer_max == 0.4
    assert "using defaults" in caplog.text.lower()


CONTEXT_FIELDS = ["game", "device", "device_id", "binding", "vehicle", "model",
                  "capture", "fov", "warp", "schema"]


def _change_context(flow, controls, field):
    if field == "game":
        flow.game = "ats"
    elif field == "device":
        flow.device = "wheel"
    elif field in ("device_id", "binding", "vehicle"):
        controls[field] = "changed"
    elif field == "model":
        flow.model_identity = {**flow.model_identity, "vision_sha256": "changed"}
    elif field == "capture":
        flow.calib.image_w += 1
    elif field == "fov":
        flow.calib.fov_h_deg += 1
    elif field == "warp":
        flow.calib.simple_warp = not flow.calib.simple_warp
    else:
        flow._check_context["schema"] -= 1


@pytest.mark.parametrize("field", CONTEXT_FIELDS)
def test_changed_context_blocks_engagement_before_next_tick(field):
    controls = dict(device_id=1, binding="steering-axis", vehicle="test-car")
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp, certification_context=lambda: controls.copy())
    _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    assert flow.calibrated and flow.ai_steering_ready and not flow.engage_block()
    _change_context(flow, controls, field)
    assert flow.engage_block()
    assert not flow.calibrated and flow.device_axis(0.5) == 0.0
    # Explicit confirmation only starts a new Check; it never restores PASS.
    flow.confirm_check()
    assert not flow.calibrated
    if field == "device_id":
        assert not flow.output_direction.ready and not flow.ai_steering_ready
    else:
        assert flow.step is Step.CHECK and flow.ai_steering_ready


@pytest.mark.parametrize("field", CONTEXT_FIELDS)
def test_saved_context_mismatch_is_untrusted_on_reload(field, data_dir):
    controls = dict(device_id=1, binding="steering-axis", vehicle="test-car")
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    flow = new_flow(lc, lp, certification_context=lambda: controls.copy())
    _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    _change_context(flow, controls, field)
    # Persist a stale certificate under the current game/device, so all
    # fingerprint fields are tested independently of filename isolation.
    state = flow.to_dict()
    state.update(game="ets2", device="gamepad")
    if field in ("game", "device"):
        state["check_context"][field] = "different"
    (data_dir / STATE_FILES["flow"]).write_text(json.dumps(state), encoding="utf-8")
    resumed = new_flow(lc, lp, cal=flow.calib, model_identity=flow.model_identity,
                       certification_context=lambda: controls.copy())
    assert resumed.step in (Step.CHECK, Step.STEERING)
    assert not resumed.calibrated and not resumed.ai_steering_ready
    assert resumed.engage_block()


def test_saved_pass_survives_matching_context():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    context = lambda: {"vehicle": "test-car", "binding": "steering-axis", "device_id": 1}
    flow = new_flow(lc, lp, certification_context=context)
    _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    resumed = new_flow(lc, lp, cal=flow.calib, certification_context=context)
    assert resumed.calibrated and resumed.ai_steering_ready and not resumed.engage_block()


def test_reader_without_vehicle_keeps_matching_certificate_after_restart():
    lc, lp = FakeLiveCalib(), FakeLiveParams()
    context = lambda: {"vehicle": None}
    flow = new_flow(lc, lp, certification_context=context)
    _to_check(flow, lc, lp)
    flow._finish_check(evaluate(_drive()))
    assert flow.calibrated
    resumed = new_flow(lc, lp, cal=flow.calib, certification_context=context)
    assert resumed.calibrated and resumed.ai_steering_ready
    assert not resumed.engage_block()


@pytest.mark.parametrize("field", ["vehicle", "binding_index"])
def test_live_context_change_stops_device_output(field, monkeypatch):
    from test_output_safety import drive

    vehicle = ["test-car"]
    telemetry = SimpleNamespace(
        available=True, speed_mps=lambda: 20., yaw_rate_rad_s=lambda: 0.,
        wheel_angle_rad=lambda wb: .01, game_steer=lambda: .2,
        user_steer=lambda: 0., vehicle_id=lambda: vehicle[0], close=lambda: None)

    def change(st, manager):
        if st.i == 1:
            if field == "vehicle":
                vehicle[0] = "another-car"
            else:
                manager.device._xinput_index = 2

    st, pad, _ = drive(monkeypatch, frames=4, on_grab=change, telemetry_source=telemetry)
    assert st.checkpoints[1][0] and any(st.checkpoints[1][1])
    assert all(not engaged and report == (0., 0., 0.) for engaged, report in st.checkpoints[2:])
    assert not st.flow.ai_steering_ready and st.flow.engage_block()


def test_virtual_binding_records_slot_and_axes():
    from simsteer.runtime.output.base import DeviceManager
    manager = DeviceManager.__new__(DeviceManager)
    manager._kind, manager._vjoy_id = "gamepad", 3
    manager._device = SimpleNamespace(_xinput_index=2)
    binding = manager.binding_identity
    assert binding["index"] == 2 and binding["steering"] == "left-stick-x"
    manager._kind = "wheel"
    binding = manager.binding_identity
    assert binding["index"] == 3 and binding["steering"] == "x"


def test_scs_vehicle_identity_tracks_truck_and_ignores_pause():
    import struct
    from simsteer.runtime.telemetry import scs_shm as scs

    tel = scs.ScsTelemetry.__new__(scs.ScsTelemetry)
    tel._mm = bytearray(scs.SHARED_MEM_SIZE)
    struct.pack_into("<I", tel._mm, scs.OFF_PLUGIN_REV, 12)
    tel._mm[2300:2305] = b"brand"
    tel._mm[2428:2433] = b"truck"
    assert tel.vehicle_id() == "brand:truck"
    tel._mm[scs.OFF_PAUSED] = 1
    assert tel.vehicle_id() == "brand:truck"
    tel._mm[2428:2433] = b"other"
    assert tel.vehicle_id() == "brand:other"
    struct.pack_into("<I", tel._mm, scs.OFF_PLUGIN_REV, 99)
    assert tel.vehicle_id() is None


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
@pytest.mark.parametrize("frozen", [False, True])
def test_source_and_bundle_use_per_user_data_dir(platform, frozen, tmp_path, monkeypatch):
    monkeypatch.delenv("SIMSTEER_DATA_DIR")
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    expected = tmp_path / ("local" if platform == "win32" else "xdg") / "SimSteer"
    assert user_data_dir() == expected
    assert expected.is_dir()


def test_fresh_source_run_keeps_git_status_clean(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    clone = tmp_path / "clone"
    clone.mkdir()
    # Export current source, including the proposed change, to an independent
    # fresh checkout without requiring a network connection or a commit.
    import shutil
    shutil.copytree(repo / "simsteer", clone / "simsteer", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(repo / "tests", clone / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(repo / ".gitignore", clone)
    subprocess.run(["git", "init", "-q", str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "add", "."], check=True)
    subprocess.run(["git", "-C", str(clone), "-c", "user.name=State test", "-c",
                    "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "Source fixture"], check=True)
    before = subprocess.check_output(["git", "-C", str(clone), "status", "--porcelain"])
    assert before == b""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "XDG_DATA_HOME": str(tmp_path / "xdg"),
           "LOCALAPPDATA": str(tmp_path / "local"), "PYTHONPATH": str(clone / "tests")}
    env.pop("SIMSTEER_DATA_DIR", None)
    code = """
from simsteer.paths import data_dir
from test_output_safety import drive
from pytest import MonkeyPatch
from test_state_context import _save, STATE_FILES
with MonkeyPatch.context() as patch:
    drive(patch, frames=2, force=False)
for kind in STATE_FILES:
    _save(kind)
assert all((data_dir() / p).exists() for p in STATE_FILES.values())
"""
    subprocess.run([sys.executable, "-c", code], cwd=clone, env=env,
                   check=True, timeout=20, capture_output=True, text=True)
    after = subprocess.check_output(["git", "-C", str(clone), "status", "--porcelain"])
    assert after == before
    expected = tmp_path / ("local" if sys.platform == "win32" else "xdg") / "SimSteer"
    assert all((expected / p).exists() for p in STATE_FILES.values())


def test_runtime_calibration_and_controller_files_are_untracked_and_ignored():
    repo = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "calibration_*.json", "controller_*.json"],
                                      cwd=repo, text=True)
    assert not tracked
    for name in ("calibration_new.json", "controller_new.json", "calibflow_new.json", "settings.json"):
        assert subprocess.run(["git", "check-ignore", "-q", name], cwd=repo).returncode == 0
