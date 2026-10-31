"""Regressions for game writes, interrupted recovery and capture signals."""

import os
import base64
from pathlib import Path

import pytest

from test_testrig import FakeWindows, snapshot, seed_profile
from tools.testrig import RigError
from tools.testrig.cli import main
from tools.testrig.display_api import NativeDisplay
from tools.testrig.display import previous_boot, recover_leftover_displays, stop_worker
from tools.testrig.guard import blocking_prompt, require_free
from tools.testrig.session import Session
from tools.testrig.state import Journal, atomic_json, read_json, monitor_covers
from tools.testrig.windows import Windows, output_log_active
from tools.testrig.capture import growing_output, configured_outputs


@pytest.fixture
def rig(tmp_path):
    backend = FakeWindows(tmp_path)
    return Session("ac", backend.directory, backend), backend


def mark_owner_input(session):
    receipt = session.journal.directory / "display-state.json"
    row = read_json(receipt)
    row["owner_input_seen"] = True
    atomic_json(receipt, row)


def test_game_exit_rewrites_are_restored_byte_identically(rig):
    session, backend = rig
    target = backend.gamepath / "cfg/controls.ini"
    original = target.read_bytes()
    session.start({}, 30)
    stop = backend.stop_process
    def resave(process):
        target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
        stop(process)
    backend.stop_process = resave
    assert session.stop()["phase"] == "restored"
    assert target.read_bytes() == original


def test_owner_input_makes_rewrites_loud_conflicts_and_preserves_original(rig):
    session, backend = rig
    target = backend.gamepath / "cfg/controls.ini"
    original = target.read_bytes()
    session.start({}, 30)
    mark_owner_input(session)
    target.write_bytes(b"owner wheel setup")
    with pytest.raises(RigError, match="conflict.*controls.ini"):
        session.stop()
    assert session.data["phase"] == "restored-with-conflicts"
    assert target.read_bytes() == b"owner wheel setup"
    row = next(row for row in session.data["backups"] if row["path"] == str(target))
    assert Path(row["archive"]).read_bytes() == original
    with pytest.raises(RigError, match="conflict"):
        Session("ac", backend.directory, backend).restore()


def test_conflicts_give_nonzero_cli_exit(rig, capsys):
    session, backend = rig
    session.start({}, 30)
    mark_owner_input(session)
    (backend.gamepath / "cfg/controls.ini").write_bytes(b"owner wheel setup")
    root = backend.directory.parents[1]
    code = main(["ac", "stop", "--state-dir", str(root), "--session", backend.directory.name], backend=backend)
    assert code == 2
    assert "conflict" in capsys.readouterr().err


def test_displace_intent_survives_kill_after_move(rig, monkeypatch):
    session, backend = rig
    target = backend.gamepath / "cfg/controls.ini"
    session.start({}, 30)
    mark_owner_input(session)
    target.write_bytes(b"owner edit")
    original = Journal._owner_changed
    def crash(journal, row):
        original(journal, row)
        raise KeyboardInterrupt("interrupted after displace")
    monkeypatch.setattr(Journal, "_owner_changed", crash)
    with pytest.raises(KeyboardInterrupt):
        session.stop()
    monkeypatch.setattr(Journal, "_owner_changed", original)
    resumed = Session("ac", backend.directory, backend)
    with pytest.raises(RigError, match="conflict"):
        resumed.restore()
    assert resumed.data["phase"] == "restored-with-conflicts"
    assert target.read_bytes() == b"owner edit"
    assert all(row.get("result") in ("restored", "owner_changed") for row in resumed.data["backups"])


@pytest.mark.parametrize("field,folder", [("stored", "backup"), ("archive", "displaced")])
def test_manifest_storage_paths_cannot_name_owner_files(rig, field, folder):
    session, backend = rig
    session.prepare({})
    victim = backend.document_game / "owner.bin"
    before = victim.read_bytes()
    row = next(row for row in session.data["backups"] if row["baseline"]["kind"] == "file")
    row[field] = str(victim)
    session.save()
    with pytest.raises(RigError, match=folder):
        session.journal.purge_backups()
    assert victim.read_bytes() == before


def test_changed_session_home_files_are_not_owner_conflicts(tmp_path):
    backend = FakeWindows(tmp_path, "ets2")
    session = Session("ets2", backend.directory, backend)
    profile = seed_profile(tmp_path / "offline-seed")
    session.start({"profile_source": str(profile)}, 30)
    target = next(Path(row["path"]) for row in session.data["backups"] if row["path"].endswith("profile.sii"))
    target.write_bytes(target.read_bytes() + b"\n// game exit write\n")
    result = session.stop()
    row = next(row for row in result["restoration"] if row["path"] == str(target))
    assert row["result"] == "left_session_file"
    assert "owner" not in row.get("message", "")


@pytest.mark.parametrize("kind", ["replay buffer", "virtual camera"])
def test_obs_replay_and_virtual_camera_outputs(kind):
    assert output_log_active(f"==== {kind.title()} Start ====", kind)
    assert not output_log_active(f"==== {kind.title()} Start ====\n==== {kind.title()} Stop ====", kind)


@pytest.mark.parametrize("signal", ["encoder", "high_cpu", "growing_output", "recording_indicator"])
def test_unread_capture_state_with_active_signals_refuses(signal):
    with pytest.raises(RigError, match="recording|capture"):
        require_free(snapshot([{"pid": 5, "name": "obs64.exe"}], activity=[
            {"pid": 5, "active": False, "observable": False, "signals": [signal], "kind": "capture"}]))


def test_unobservable_idle_obs_warns_and_continues():
    warning = "OBS output activity is not observable; verify recording/streaming is stopped"
    assert warning in require_free(snapshot([{"pid": 5, "name": "obs64.exe"}], warnings=[warning], activity=[
        {"pid": 5, "active": False, "observable": False, "signals": [], "kind": "capture"}]))


def test_maximized_browser_is_not_unknown_fullscreen_game():
    require_free(snapshot([{"pid": 7, "name": "msedge.exe"}], foreground={
        "pid": 7, "fullscreen": True, "maximized": True, "caption": True, "exclusive": False}))
    with pytest.raises(RigError, match="fullscreen"):
        require_free(snapshot([{"pid": 7, "name": "unknown.exe"}], foreground={
            "pid": 7, "fullscreen": True, "maximized": True, "exclusive": True}))


def test_vdd_source_waits_for_async_arrival_and_times_out():
    api = NativeDisplay.__new__(NativeDisplay)
    clock = [0.0]
    def pause(seconds):
        clock[0] += seconds
    source = {"name": "DISPLAY3", "adapter_path": r"\\?\ROOT#DISPLAY#0000#{00000000-0000-0000-0000-000000000000}"}
    api.sources = lambda: [source] if clock[0] >= 0.6 else []
    assert api.wait_virtual(r"ROOT\DISPLAY\0000", clock=lambda: clock[0], pause=pause) == "DISPLAY3"
    api.sources = lambda: []
    start = clock[0]
    with pytest.raises(RigError, match="timed out|unavailable"):
        api.wait_virtual(r"ROOT\DISPLAY\0000", clock=lambda: clock[0], pause=pause)
    assert 10 <= clock[0] - start < 10.3


def test_manifest_tracks_temporary_before_target_replace(rig, monkeypatch):
    session, backend = rig
    replace = os.replace
    def kill(source, destination):
        if Path(destination).name == "controls.ini":
            durable = Journal(backend.directory)
            row = next(row for row in durable.data["backups"] if row["path"] == str(destination))
            assert row["temporary"] == str(source)
            assert Path(source).name.startswith(".testrig-")
            raise KeyboardInterrupt("killed before target replace")
        return replace(source, destination)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "replace", kill)
        # A killed process does not execute its finally block.
        interrupted.setattr(Journal, "cleanup_temporary", lambda *_: None)
        with pytest.raises(KeyboardInterrupt):
            session.prepare({})
    assert list(backend.document_game.glob("cfg/.testrig-*"))
    Session("ac", backend.directory, backend).restore()
    assert not list(backend.gamepath.glob("cfg/.testrig-*"))
    assert not list(backend.document_game.glob("cfg/.testrig-*"))


@pytest.mark.parametrize("title", ["Allow game launch?", "Launch Options", "Steam - Launch Confirmation"])
def test_modern_steam_launch_dialogs(title):
    assert "Steam launch confirmation" in blocking_prompt([{"title": title, "class": "SDL_app", "owner": 10}])


def test_dropped_hook_heartbeat_aborts_but_injected_input_does_not():
    from tools.testrig.activity import InputHeartbeat
    clock = [0.0]
    heartbeat = InputHeartbeat(100, clock=lambda: clock[0])
    heartbeat.observe(200)
    assert not heartbeat.missed(200)
    clock[0] = 0.1
    assert not heartbeat.missed(300)
    clock[0] = 1.2
    assert heartbeat.missed(300)


@pytest.mark.parametrize("action", ["status", "guard", "start"])
def test_previous_boot_display_recovers_before_command(rig, action):
    session, backend = rig
    session.start({"display": "vdd", "vdd_device": "virtual"}, 30)
    session.data["boot_id"] = "counter:10"
    session.save()
    backend.rows = backend.rows[:1]
    original = backend.snapshot
    backend.snapshot = lambda: {**original(), "boot_id": "counter:11"}
    command = ["guard"] if action == "guard" else ["ac", action, "--session", backend.directory.name]
    main(command + ["--state-dir", str(backend.directory.parents[1])], backend=backend)
    assert not backend.display.enabled
    assert backend.display.topology == backend.display.original


def test_owner_edit_after_a_sealed_session_stays_a_conflict(rig):
    session, backend = rig
    session.start({}, 30)
    target = backend.gamepath / "cfg/controls.ini"
    session._stop()
    target.write_bytes(b"later owner edit")
    with pytest.raises(RigError, match="conflict"):
        Session("ac", backend.directory, backend).restore()
    assert target.read_bytes() == b"later owner edit"


def test_crash_recovery_does_not_extend_the_game_lifetime(rig):
    session, backend = rig
    session.start({}, 30)
    backend.rows = backend.rows[:1]
    target = backend.gamepath / "cfg/controls.ini"
    target.write_bytes(b"owner after game exit")
    later = session.data["last_owned_at"] + 10
    os.utime(target, (later, later))
    with pytest.raises(RigError, match="conflict"):
        Session("ac", backend.directory, backend).restore()
    assert target.read_bytes() == b"owner after game exit"


def test_displace_intent_is_durable_before_the_move(rig, monkeypatch):
    session, backend = rig
    session.start({}, 30)
    mark_owner_input(session)
    target = backend.gamepath / "cfg/controls.ini"
    target.write_bytes(b"owner edit")
    replace = os.replace
    def before_move(source, destination):
        if Path(destination).parent == backend.directory / "displaced":
            durable = Journal(backend.directory)
            row = next(row for row in durable.data["backups"] if row["path"] == str(target))
            assert row["archive_pending"] and row["archive"] == str(destination)
            raise KeyboardInterrupt("intent committed, move not started")
        return replace(source, destination)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "replace", before_move)
        with pytest.raises(KeyboardInterrupt):
            session.stop()
    with pytest.raises(RigError, match="conflict"):
        Session("ac", backend.directory, backend).restore()
    assert target.read_bytes() == b"owner edit"


def test_one_bad_backup_does_not_block_other_files(rig):
    session, backend = rig
    session.start({}, 30)
    row = next(row for row in session.data["backups"] if row["baseline"]["kind"] == "file")
    Path(row["stored"]).write_bytes(b"damaged backup")
    with pytest.raises(RigError, match="Backup hash mismatch"):
        session.stop()
    assert any(row.get("result") == "restored" for row in session.data["backups"])
    assert session.data["phase"] == "cleanup-required"


def test_temporary_cleanup_preserves_a_changed_or_untracked_file(rig):
    session, backend = rig
    session.prepare({})
    row = session.data["backups"][0]
    temporary = Path(row["temporary"])
    temporary.write_bytes(b"changed temporary")
    untracked = temporary.parent / ".testrig-other-session-0.tmp"
    untracked.write_bytes(b"untracked")
    with pytest.raises(RigError, match="temporary file changed"):
        session.restore()
    assert temporary.read_bytes() == b"changed temporary"
    assert untracked.read_bytes() == b"untracked"


@pytest.mark.parametrize("name", ["obs64.exe", "Streamlabs Desktop.exe", "NVIDIA Share.exe", "bcastdvr.exe", "Discord.exe"])
@pytest.mark.parametrize("signal", ["cpu_percent", "encoder_active"])
def test_native_capture_signals_are_used_for_unread_output(name, signal, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    process = {"pid": 5, "name": name, signal: 50 if signal == "cpu_percent" else True}
    backend = Windows.__new__(Windows)
    records, warnings = backend.activity([process], [])
    assert warnings
    with pytest.raises(RigError, match="recording|capture"):
        require_free(snapshot([process], activity=records, warnings=warnings))


def test_obs_idle_and_active_portable_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    root = tmp_path / "portable"
    log = root / "config/obs-studio/logs/current.txt"
    log.parent.mkdir(parents=True)
    process = {"pid": 5, "name": "obs64.exe", "path": str(root / "bin/64bit/obs64.exe"), "command": "obs64.exe --portable"}
    backend = Windows.__new__(Windows)
    log.write_text("OBS started\n")
    records, warnings = backend.activity([process], [])
    assert records[0]["observable"] and not records[0]["active"]
    require_free(snapshot([process], activity=records, warnings=warnings))
    log.write_text("==== Replay Buffer Start ====\n")
    records, warnings = backend.activity([process], [])
    with pytest.raises(RigError, match="recording"):
        require_free(snapshot([process], activity=records, warnings=warnings))


def test_streamlabs_log_and_configured_growing_output(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    root = tmp_path / "slobs-client"
    log = root / "node-obs/logs/current.log"
    log.parent.mkdir(parents=True)
    log.write_text("==== Recording Start ====\n")
    backend = Windows.__new__(Windows)
    records, _ = backend.activity([{"pid": 5, "name": "Streamlabs Desktop.exe"}], [])
    assert records[0]["active"]
    output = tmp_path / "recordings"
    output.mkdir()
    media = output / "video.mkv"
    media.write_bytes(b"one")
    config = root / "basic/profiles/test/basic.ini"
    config.parent.mkdir(parents=True)
    config.write_text(f"[AdvOut]\nRecFilePath={output}\n")
    folders = configured_outputs([root])
    assert folders == {output}
    assert growing_output(folders, 0, pause=lambda _: media.write_bytes(b"one two"))
    assert not growing_output(folders, 0, pause=lambda _: None)


@pytest.mark.parametrize("name,control", [("bcastdvr.exe", "Recording in progress"), ("NVIDIA Share.exe", "Stop recording"),
                                         ("Discord.exe", "Go Live — Streaming"), ("Streamlabs Desktop.exe", "Stop streaming")])
def test_visible_capture_indicators(name, control):
    backend = Windows.__new__(Windows)
    process = {"pid": 5, "name": name}
    records, _ = backend.activity([process], [{"pid": 5, "title": name, "text": control}])
    assert records[0]["active"]


def test_unobservable_capture_warnings_are_saved_in_the_session(rig):
    session, backend = rig
    snap = backend.snapshot
    backend.snapshot = lambda: {**snap(), "warnings": ["Capture state is not observable"]}
    session.prepare({})
    assert "Capture state is not observable" in Journal(backend.directory).data["warnings"]


def test_current_boot_does_not_recover_an_active_display(rig):
    session, backend = rig
    session.start({"display": "vdd", "vdd_device": "virtual"}, 30)
    session.data["boot_id"] = "counter:10"
    session.save()
    snap = backend.snapshot
    backend.snapshot = lambda: {**snap(), "boot_id": "counter:10"}
    assert main(["ac", "status", "--session", backend.directory.name,
                 "--state-dir", str(backend.directory.parents[1])], backend=backend) == 0
    assert backend.display.enabled


def test_vdd_wait_retries_a_changing_ccd_inventory():
    api = NativeDisplay.__new__(NativeDisplay)
    clock = [0.0]
    def sources():
        if clock[0] < 0.5:
            raise RigError("Display topology kept changing")
        return [{"name": "DISPLAY3", "adapter_path": r"\\?\ROOT#DISPLAY#0000#{00000000-0000-0000-0000-000000000000}"}]
    def pause(seconds):
        clock[0] += seconds
    api.sources = sources
    assert api.wait_virtual(r"ROOT\DISPLAY\0000", clock=lambda: clock[0], pause=pause) == "DISPLAY3"


def test_generic_streamlabs_log_does_not_hide_active_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    log = tmp_path / "slobs-client/logs/electron.log"
    log.parent.mkdir(parents=True)
    log.write_text("Electron UI update\n")
    process = {"pid": 5, "name": "Streamlabs Desktop.exe", "cpu_percent": 60}
    backend = Windows.__new__(Windows)
    records, warnings = backend.activity([process], [])
    assert not records[0]["observable"]
    with pytest.raises(RigError, match="recording"):
        require_free(snapshot([process], activity=records, warnings=warnings))


def test_previous_boot_recovery_failure_refuses_start(rig):
    session, backend = rig
    session.start({"display": "vdd", "vdd_device": "virtual"}, 30)
    session.data["boot_id"] = "counter:10"
    session.save()
    backend.rows = backend.rows[:1]
    snap = backend.snapshot
    backend.snapshot = lambda: {**snap(), "boot_id": "counter:11"}
    def failed(journal):
        raise RigError("Physical topology recovery failed")
    backend.stop_display = failed
    uris = len(backend.uris)
    assert main(["ac", "start", "--session", backend.directory.name,
                 "--state-dir", str(backend.directory.parents[1])], backend=backend) == 2
    assert len(backend.uris) == uris


def test_stalled_hook_pump_aborts_without_a_new_input_tick(monkeypatch):
    import ctypes
    import threading
    from tools.testrig.activity import OwnerInput, InputHeartbeat
    class Info(ctypes.Structure):
        _fields_ = [("time", ctypes.c_uint32)]
    monitor = OwnerInput.__new__(OwnerInput)
    monitor.info, monitor.error, monitor.failure = Info(100), None, None
    monitor.get_input = lambda _: True
    monitor.heartbeat = InputHeartbeat(100)
    monitor.triggered = threading.Event()
    monitor.last_pump = 0
    monkeypatch.setattr("tools.testrig.activity.time.monotonic", lambda: 10)
    assert monitor.active()
    assert "heartbeat lost" in monitor.failure


def test_input_during_autonomous_game_close_prevents_rewrite_attribution(rig):
    session, backend = rig
    session.start({}, 30)
    target = backend.gamepath / "cfg/controls.ini"
    class Monitor:
        present = False
        def active(self):
            return self.present
    backend.owner_monitor = Monitor()
    stop = backend.stop_process
    def close_with_owner_input(process):
        backend.owner_monitor.present = True
        target.write_bytes(b"owner returned during close")
        stop(process)
    backend.stop_process = close_with_owner_input
    with pytest.raises(RigError, match="conflict"):
        session.stop()
    assert target.read_bytes() == b"owner returned during close"


def test_game_bar_service_residency_and_active_counters():
    backend = Windows.__new__(Windows)
    process = {"pid": 5, "name": "svchost.exe", "capture_tool": "Game Bar capture service", "cpu_percent": 0}
    records, warnings = backend.activity([process], [])
    require_free(snapshot([process], activity=records, warnings=warnings))
    process["encoder_active"] = True
    records, warnings = backend.activity([process], [])
    with pytest.raises(RigError, match="Game Bar capture service"):
        require_free(snapshot([process], activity=records, warnings=warnings))


def test_dead_native_worker_never_authorizes_an_owner_rewrite(rig, monkeypatch):
    session, backend = rig
    session.start({}, 30)
    target = backend.gamepath / "cfg/controls.ini"
    baseline = Path(next(row["stored"] for row in session.data["backups"] if row["path"] == str(target))).read_bytes()
    session.data["display_worker"] = {"pid": 999, "created": "dead", "path": "worker.exe"}
    session.save()
    target.write_bytes(b"owner changed after worker death")
    backend.stop_display = lambda journal: stop_worker(journal, backend)
    monkeypatch.setattr("tools.testrig.display.NativeDisplay", lambda: backend.display)
    monkeypatch.setattr("tools.testrig.display.BlackOverlay", lambda: backend.display)
    with pytest.raises(RigError, match="conflict"):
        session.stop()
    assert target.read_bytes() == b"owner changed after worker death"
    assert read_json(backend.directory / "display-state.json")["owner_input_seen"]
    row = next(row for row in session.data["backups"] if row["path"] == str(target))
    assert Path(row["archive"]).read_bytes() == baseline


def test_prepare_start_gap_is_not_blessed_by_a_later_game_rewrite(rig):
    session, backend = rig
    session.prepare({})
    target = backend.gamepath / "cfg/controls.ini"
    target.write_bytes(b"owner edited while only prepared")
    session.start({}, 30)
    stop = backend.stop_process
    def resave(process):
        target.write_bytes(target.read_bytes() + b"\n; game resaved\n")
        stop(process)
    backend.stop_process = resave
    with pytest.raises(RigError, match="conflict"):
        session.stop()
    assert target.read_bytes().startswith(b"owner edited while only prepared")


@pytest.mark.parametrize("failure", ["missing", "no-start", "no-stop", "errors", "stale-coverage"])
def test_monitor_receipt_must_prove_coverage(rig, failure):
    session, backend = rig
    session.start({}, 30)
    path = backend.directory / "display-state.json"
    receipt = read_json(path)
    now = backend.wall_time()
    receipt.update(stop_reason="explicit-stop", monitor_stopped_at=now + 100, monitor_covered_until=now + 100)
    if failure == "no-start":
        receipt.pop("monitor_started_at")
    elif failure == "no-stop":
        receipt.pop("stop_reason")
    elif failure == "errors":
        receipt["errors"] = ["hook failed"]
    elif failure == "stale-coverage":
        receipt["monitor_covered_until"] = now - 100
    if failure == "missing":
        path.unlink()
    else:
        atomic_json(path, receipt)
    backend.stop_display = lambda _: None  # receipt left by a terminated monitor
    target = backend.gamepath / "cfg/controls.ini"
    target.write_bytes(b"unknown writer")
    with pytest.raises(RigError, match="conflict"):
        session.stop()
    assert target.read_bytes() == b"unknown writer"


def test_monitor_coverage_excludes_edits_before_monitor_start(rig):
    session, backend = rig
    session.start({}, 30)
    target = backend.gamepath / "cfg/controls.ini"
    target.write_bytes(b"unmonitored edit")
    earlier = session.data["monitor_started_at"] - 1
    os.utime(target, (earlier, earlier))
    with pytest.raises(RigError, match="conflict"):
        session.stop()
    assert target.read_bytes() == b"unmonitored edit"


def test_monitor_can_prove_autonomous_shutdown_while_still_alive():
    row = {"monitor_started_at": 10, "monitor_covered_until": 30, "monitor_live": True,
           "stop_reason": "deadline", "errors": []}
    assert monitor_covers(row, 20)
    row.pop("monitor_live")
    assert not monitor_covers(row, 20)
    row["monitor_stopped_at"] = 30
    assert monitor_covers(row, 20)


@pytest.mark.parametrize("damage", ["prefix", "zeros"])
def test_torn_temporary_recovers_automatically(rig, damage):
    session, backend = rig
    session.prepare({})
    row = session.data["backups"][0]
    content = base64.b64decode(row["temporary_content"])
    temporary = Path(row["temporary"])
    temporary.write_bytes(content[:3] if damage == "prefix" else b"\0" * 3)
    assert session.restore()["phase"] == "restored"
    assert not temporary.exists()
    assert all("temporary_content" not in row for row in session.data["backups"])


@pytest.mark.parametrize("action", ["guard", "status", "prepare", "start"])
def test_boot_identity_change_never_tears_down_live_owned_processes(rig, action):
    session, backend = rig
    session.start({"display": "vdd", "vdd_device": "virtual"}, 30)
    snap = backend.snapshot
    backend.snapshot = lambda: {**snap(), "boot_id": "counter:11"}
    command = ["guard"] if action == "guard" else ["ac", action, "--session", backend.directory.name]
    main(command + ["--state-dir", str(backend.directory.parents[1])], backend=backend)
    assert backend.display.enabled
    assert not (backend.directory / "display-stop").exists()
    assert not backend.killed


def test_boot_identity_change_never_stops_a_live_monitor(rig):
    session, backend = rig
    session.start({"display": "vdd", "vdd_device": "virtual"}, 30)
    worker = {"pid": 50, "created": "worker-now", "path": "python.exe", "name": "python.exe"}
    session.data["display_worker"] = worker
    session.save()
    backend.rows = backend.rows[:1] + [worker]
    snap = {**backend.snapshot(), "boot_id": "counter:11"}
    assert recover_leftover_displays(backend.directory.parents[1], backend, snap) == []
    assert backend.display.enabled


def test_previous_boot_recovery_rechecks_liveness_under_the_lock(rig):
    session, backend = rig
    session.start({"display": "vdd", "vdd_device": "virtual"}, 30)
    initial = {**backend.snapshot(), "processes": backend.rows[:1], "boot_id": "counter:11"}
    current = {**backend.snapshot(), "boot_id": "counter:11"}
    backend.snapshot = lambda: current
    assert recover_leftover_displays(backend.directory.parents[1], backend, initial) == []
    assert backend.display.enabled


def test_reused_worker_pid_does_not_prevent_previous_boot_recovery(rig):
    session, backend = rig
    session.start({"display": "vdd", "vdd_device": "virtual"}, 30)
    session.data["display_worker"] = {"pid": 50, "created": "old-worker", "path": "python.exe"}
    session.save()
    backend.rows = backend.rows[:1] + [{"pid": 50, "created": "new-worker", "path": "python.exe", "name": "python.exe"}]
    snap = {**backend.snapshot(), "boot_id": "counter:11"}
    backend.snapshot = lambda: snap
    assert recover_leftover_displays(backend.directory.parents[1], backend, snap)
    assert not backend.display.enabled
    assert not backend.killed


def test_boot_time_fallback_requires_uptime_discontinuity():
    saved = {"boot_id": "133000000000000000", "boot_uptime": 1000, "boot_observed_at": 10000}
    drift = {"boot_id": "133000000012340000", "boot_uptime": 1002, "boot_observed_at": 10003.234}
    assert not previous_boot(saved, drift)
    continuous = {"boot_id": "133000005000000000", "boot_uptime": 1600, "boot_observed_at": 10600}
    assert not previous_boot(saved, continuous)
    rebooted = {**continuous, "boot_uptime": 5}
    assert previous_boot(saved, rebooted)
    assert not previous_boot({"boot_id": "counter:10"}, {**drift, "boot_id": "counter:10", "boot_time": drift["boot_id"]})


def test_fake_file_clock_is_independent_of_polling_time(rig, monkeypatch):
    _, backend = rig
    backend.t += 1000
    monkeypatch.setattr("test_testrig.time.time", lambda: 12345.5)
    assert backend.wall_time() == 12345.5


@pytest.mark.parametrize("name", ["nvcontainer.exe", "other-encoder.exe"])
def test_unknown_hardware_encoder_refuses_by_name_unless_owned(name):
    backend = Windows.__new__(Windows)
    process = {"pid": 5, "name": name, "encoder_active": True, "created": "encoder-now", "path": name}
    records, warnings = backend.activity([process], [])
    with pytest.raises(RigError, match=name):
        require_free(snapshot([process], activity=records, warnings=warnings))
    require_free(snapshot([process], activity=records, warnings=warnings), owned=[process])


@pytest.mark.parametrize("kind", ["SDL_app", "Chrome_WidgetWin_1", "vguiPopupWindow"])
@pytest.mark.parametrize("title", ["Launching...", "Preparing to launch", "Steam"])
def test_normal_owned_steam_windows_do_not_request_confirmation(kind, title):
    assert blocking_prompt([{"title": title, "text": "Preparing to launch your game", "class": kind,
                             "steam": True, "owner": 10}]) is None
