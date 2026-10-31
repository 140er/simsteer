"""Safety regressions for interrupted sessions and concurrent owner activity."""

import os
from pathlib import Path
import stat

import pytest

from test_testrig import FakeWindows, snapshot, restart
from tools.testrig import RigError
from tools.testrig.guard import blocking_prompt, require_free
from tools.testrig.session import Session
from tools.testrig.state import atomic_json, read_json
from tools.testrig.state import Journal, lock
from tools.testrig.activity import real_input
from tools.testrig.display import watch_session, recover_session, DisplayLifecycle
from tools.testrig.display_api import NativeDisplay, device_instance, encoded
from tools.testrig.windows import output_log_active, connection_log_active, Windows
from tools.testrig.cli import main
from test_testrig import FakeDisplay
import ctypes
import struct
import base64
from datetime import datetime, timezone, timedelta
from copy import deepcopy


@pytest.fixture
def rig(tmp_path):
    backend = FakeWindows(tmp_path)
    return Session("ac", backend.directory, backend), backend


def failed_cleanup(session, backend):
    backend.busy = {"pid": 99, "parent": 1, "name": "acs.exe", "path": str(backend.gamepath / "acs.exe"),
                    "created": "unowned", "born": 0}
    with pytest.raises(RigError, match="still running"):
        session.stop()
    backend.busy = None


def test_delayed_restore_keeps_owner_progress_and_new_files(rig):
    session, backend = rig
    session.start({}, 30)
    failed_cleanup(session, backend)
    new_setup = backend.document_game / "owner-new-setup.ini"
    new_setup.write_bytes(b"new setup")
    (backend.document_game / "owner.bin").write_bytes(b"later progress")
    Session("ac", backend.directory, backend).restore()
    assert new_setup.read_bytes() == b"new setup"
    assert (backend.document_game / "owner.bin").read_bytes() == b"later progress"


def test_restore_never_deletes_owner_directories_or_replays(rig, monkeypatch):
    session, backend = rig
    for i in range(20):
        (backend.document_game / f"replay{i:02}.acreplay").write_bytes(bytes([i]))
    locked = backend.document_game / "locked"
    locked.mkdir()
    (locked / "file").write_bytes(b"owner")
    session.start({}, 30)
    def no_tree_deletion(*args, **kwargs):
        raise AssertionError("Owner folder deletion is forbidden")
    monkeypatch.setattr("shutil.rmtree", no_tree_deletion)
    session.stop()
    Session("ac", backend.directory, backend).restore()
    assert len(list(backend.document_game.glob("replay*.acreplay"))) == 20
    assert (locked / "file").read_bytes() == b"owner"


def test_owner_changed_edited_file_is_left_and_baseline_retained(rig):
    session, backend = rig
    target = backend.gamepath / "cfg/controls.ini"
    baseline = target.read_bytes()
    session.start({}, 30)
    failed_cleanup(session, backend)
    target.write_bytes(b"owner changed controls")
    recovered = Session("ac", backend.directory, backend)
    result = recovered.restore()
    assert target.read_bytes() == b"owner changed controls"
    assert any("owner changed" in entry.get("message", "") for entry in result["restoration"])
    archives = list((backend.directory / "displaced").glob("*"))
    assert any(path.is_file() and path.read_bytes() == baseline for path in archives)
    recovered.restore()
    assert target.read_bytes() == b"owner changed controls"


def test_backups_contain_only_edited_files_and_are_purged_after_restore(rig):
    session, backend = rig
    session.start({}, 30)
    assert session.data["backups"]
    for row in session.data["backups"]:
        assert row["baseline"]["kind"] in ("file", "absent")
        assert Path(row["path"]).suffix == ".ini"
        assert "userdata" not in Path(row["path"]).parts
    session.stop()
    assert not (backend.directory / "backup").exists()
    assert (backend.directory / "manifest.json").is_file()


def test_readonly_backed_up_file_restores_bytes_and_mode(rig):
    session, backend = rig
    target = backend.gamepath / "cfg/controls.ini"
    baseline = target.read_bytes()
    target.chmod(stat.S_IRUSR)
    try:
        session.start({}, 30)
        session.stop()
        assert target.read_bytes() == baseline
        assert not target.stat().st_mode & stat.S_IWUSR
    finally:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_locked_restore_is_partial_restartable_and_keeps_owner_changes(rig, monkeypatch):
    session, backend = rig
    target = backend.gamepath / "cfg/controls.ini"
    baseline = target.read_bytes()
    session.start({}, 30)
    replace = os.replace
    locked = True
    def replace_with_lock(source, destination):
        if Path(destination) == target and locked:
            error = PermissionError("file is shared")
            error.winerror = 32
            raise error
        return replace(source, destination)
    monkeypatch.setattr("tools.testrig.state.os.replace", replace_with_lock)
    monkeypatch.setattr("tools.testrig.state.time.sleep", lambda _: None, raising=False)
    with pytest.raises(RigError, match="controls.ini"):
        session.stop()
    assert session.data["phase"] == "cleanup-required"
    assert (backend.document_game / "owner.bin").exists()
    # A successful row from the first attempt must not be rewritten on retry.
    other = backend.gamepath / "cfg/race.ini"
    other.write_bytes(b"owner edit after partial restore")
    locked = False
    Session("ac", backend.directory, backend).restore()
    assert target.read_bytes() == baseline
    assert other.read_bytes() == b"owner edit after partial restore"


def test_json_replace_and_read_retry_transient_sharing_errors(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    atomic_json(target, {"old": True})
    replace = os.replace
    attempts = 0
    def transient(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise PermissionError("sharing violation")
        return replace(source, destination)
    monkeypatch.setattr("tools.testrig.state.os.replace", transient)
    atomic_json(target, {"new": True})
    assert attempts == 4
    original = Path.read_text
    reads = 0
    def transient_read(path, *args, **kwargs):
        nonlocal reads
        reads += 1
        if reads < 4:
            raise PermissionError("delete pending")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", transient_read)
    assert read_json(target) == {"new": True}
    assert reads == 4


def test_corrupt_state_recovers_from_manifest(rig):
    session, backend = rig
    session.start({}, 30)
    session.journal.path.write_text("{truncated")
    recovered = Session("ac", backend.directory, backend)
    assert recovered.status()["recovery"]["source"] == "manifest.json"
    recovered.restore()
    assert recovered.data["phase"] == "restored"


def test_pending_rename_is_only_a_warning():
    warnings = require_free(snapshot(restart=restart(signals=["PendingFileRenameOperations"])))
    assert any("PendingFileRenameOperations" in warning for warning in warnings)


def test_future_reboot_warns_but_imminent_reboot_refuses():
    task = {"name": "Reboot", "state": "Ready", "next_run": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()}
    assert require_free(snapshot(restart=restart(tasks=[task])))
    task["next_run"] = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    with pytest.raises(RigError, match="restart pending"):
        require_free(snapshot(restart=restart(tasks=[task])))


@pytest.mark.parametrize("name", ["NVIDIA Share.exe", "NVIDIA Overlay.exe", "GameBar.exe",
                                  "OVRServer_x64.exe", "sunshine.exe", "parsecd.exe", "wallpaper64.exe"])
def test_idle_resident_helpers_do_not_block(name):
    require_free(snapshot([{"pid": 5, "name": name, "path": rf"D:\Steam\steamapps\common\Tool\{name}"}]))


def test_fullscreen_unknown_game_and_active_obs_block():
    row = {"pid": 5, "name": "new-game.exe", "path": r"D:\Games\new-game.exe"}
    with pytest.raises(RigError, match="gaming|fullscreen"):
        require_free(snapshot([row], foreground={"pid": 5, "fullscreen": True, "title": "Game"}))
    obs = {"pid": 7, "name": "obs64.exe"}
    require_free(snapshot([obs]))
    with pytest.raises(RigError, match="recording|streaming"):
        require_free(snapshot([obs], activity=[{"pid": 7, "active": True, "kind": "OBS recording"}]))


def test_owned_helper_ancestry_ignores_game_renderer():
    owned = {"pid": 10, "created": "parent", "path": r"D:\Steam\steamapps\common\Game\Game.exe", "name": "Game.exe", "born": 1}
    child = {"pid": 11, "parent": 10, "created": "child", "path": r"D:\Steam\steamapps\common\Game\renderer.exe", "name": "renderer.exe", "born": 2}
    require_free(snapshot([owned, child], foreground={"pid": 11, "fullscreen": True}), [owned])


def test_steam_launch_confirmation_fails_fast():
    assert "Steam launch confirmation" in blocking_prompt([{"title": "Allow game launch?", "class": "vguiPopupWindow"}])


def test_prepare_finished_session_refuses(rig):
    session, _ = rig
    session.start({}, 30)
    session.stop()
    with pytest.raises(RigError, match="finished|new --session"):
        session.prepare({})


def test_manifest_wins_if_crash_prevents_state_publication(rig, monkeypatch):
    session, backend = rig
    session.prepare({})
    original = os.replace
    def blocked(source, destination):
        if Path(destination) == session.journal.path:
            raise PermissionError("state still open")
        return original(source, destination)
    monkeypatch.setattr("tools.testrig.state.os.replace", blocked)
    monkeypatch.setattr("tools.testrig.state.time.sleep", lambda _: None)
    session.data["error"] = "latest recovery information"
    with pytest.raises(PermissionError):
        session.save()
    recovered = Journal(backend.directory)
    assert recovered.data["error"] == "latest recovery information"
    assert recovered.recovery["source"] == "manifest.json"


def test_json_retry_exhaustion_keeps_previous_valid_file(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    atomic_json(target, {"old": True})
    monkeypatch.setattr("tools.testrig.state.time.sleep", lambda _: None)
    def locked(*args):
        raise PermissionError("busy")
    monkeypatch.setattr("tools.testrig.state.os.replace", locked)
    with pytest.raises(PermissionError):
        atomic_json(target, {"new": True})
    assert read_json(target) == {"old": True}
    assert not list(tmp_path.glob(".state-*"))


def test_created_file_changed_by_owner_is_not_removed(rig):
    session, backend = rig
    session.start({}, 30)
    file = backend.document_game / "cfg/video.ini"
    file.write_bytes(b"owner replacement")
    session.stop()
    assert file.read_bytes() == b"owner replacement"


@pytest.mark.parametrize("prefix", ["\\\\?\\", "\\??\\"])
def test_ccd_adapter_interface_path_is_matched_to_pnp_instance(prefix):
    interface = prefix + "ROOT#DISPLAY#0000#{5b45201d-f2f2-4f3b-85bb-30ff1f953599}"
    assert device_instance(interface) == r"root\display\0000"
    assert device_instance(r"Root\SomeVirtualHardware") != r"root\display\0000"


def test_ccd_ambiguous_virtual_outputs_fail_closed():
    api = NativeDisplay.__new__(NativeDisplay)
    path = r"\\?\ROOT#DISPLAY#0000#{5b45201d-f2f2-4f3b-85bb-30ff1f953599}"
    api.sources = lambda: [{"name": name, "adapter_path": path} for name in ("DISPLAY2", "DISPLAY3")]
    with pytest.raises(RigError, match="multiple|one CCD"):
        api.extend_primary(r"ROOT\DISPLAY\0000", {}, 1920, 1080)


def test_ccd_source_adapter_and_monitor_requests_use_documented_layouts():
    luid = bytes(range(8))
    target_luid = bytes(range(8, 16))
    interface = r"\\?\ROOT#DISPLAY#0000#{5b45201d-f2f2-4f3b-85bb-30ff1f953599}"
    requests = []
    class User:
        def DisplayConfigGetDeviceInfo(self, data):
            kind, size = struct.unpack_from("<II", data)
            identifier = struct.unpack_from("<I", data, 16)[0]
            requests.append((kind, size, bytes(data)[8:16], identifier))
            offset, value = {1: (20, "DISPLAY3"), 4: (20, interface), 2: (164, "monitor-interface")}[kind]
            content = (value + "\0").encode("utf-16-le")
            data[offset:offset + len(content)] = content
            return 0
    path = bytearray(72)
    path[:8], path[20:28] = luid, target_luid
    struct.pack_into("<I", path, 8, 7)
    struct.pack_into("<I", path, 28, 3)
    api = NativeDisplay.__new__(NativeDisplay)
    api.user, api.query = User(), lambda _: {"paths": encoded(path), "path_count": 1}
    source = api.sources()[0]
    assert requests == [(1, 84, luid, 7), (4, 276, luid, 0), (2, 420, target_luid, 3)]
    assert source["name"] == "DISPLAY3" and device_instance(source["adapter_path"]) == r"root\display\0000"
    assert source["target_paths"] == ["monitor-interface"]


def test_physical_fallback_remaps_names_and_restores_saved_modes():
    mode = ctypes.create_string_buffer(220)
    struct.pack_into("<H", mode, 68, 220)
    struct.pack_into("<III", mode, 168, 32, 2560, 1440)
    struct.pack_into("<I", mode, 184, 144)
    saved = {"name": "DISPLAY1", "adapter_path": r"\\?\PCI#VEN_0000#INSTANCE#{00000000-0000-0000-0000-000000000000}",
             "target_paths": ["stable-monitor-interface"], "source_id": 1,
             "x": 0, "y": 0, "width": 2560, "height": 1440, "primary": True, "mode": encoded(mode)}
    api = NativeDisplay.__new__(NativeDisplay)
    api.sources = lambda **kwargs: [{**saved, "name": "DISPLAY9", "source_id": 7}]
    applied = []
    def change(name, data, flags):
        if name:
            applied.append({**saved, "name": name, "mode": encoded(data), "primary": bool(flags & 0x10)})
    api.change, api.active = change, lambda: applied
    api.restore_physical({"displays": [saved]})
    assert applied[0]["name"] == "DISPLAY9" and applied[0]["primary"]
    assert struct.unpack_from("<I", base64.b64decode(applied[0]["mode"]), 184)[0] == 144


def test_failed_blob_restore_accepts_verified_physical_fallback():
    api = FakeDisplay()
    lifecycle = DisplayLifecycle(api, api, lambda _: None,
        {"display": "vdd", "overlay": True, "vdd_device": "virtual", "width": 1920, "height": 1080}, "session")
    lifecycle.start()
    api.failure = "restore-topology"
    lifecycle.close()
    assert lifecycle.row["topology_restored"] and lifecycle.row["topology_fallback"]
    assert lifecycle.row["vdd_disabled"] and lifecycle.row["overlay_closed"]
    assert api.topology == api.original and lifecycle.row["warnings"]


@pytest.mark.parametrize("flags,keyboard,expected", [(0, True, True), (0x10, True, False), (0x02, True, False),
                                                    (0, False, True), (1, False, False), (2, False, False)])
def test_owner_activity_filters_injected_input(flags, keyboard, expected):
    assert real_input(flags, keyboard=keyboard) == expected


@pytest.mark.parametrize("reason", ["owner-input", "explicit-stop", "game-exited", "deadline"])
def test_worker_abort_is_responsive_and_recovers_session(rig, reason):
    session, backend = rig
    session.start({"display": "vdd", "vdd_device": "virtual"}, 30)
    state = deepcopy(session.data)
    state["options"]["max_seconds"] = 0.5
    class Parent:
        def alive(self):
            return False
    class Activity:
        def active(self):
            return reason == "owner-input"
    class Handle:
        def alive(self):
            return reason != "game-exited"
        def close(self):
            pass
    if reason == "explicit-stop":
        (backend.directory / "display-stop").touch()
    # There is no inventory callback in this pump; process handles suffice.
    actual, recover = watch_session(backend.directory, state, Parent(), backend.display_lifecycle, Activity(),
        open_process=lambda _: Handle(), clock=backend.monotonic, pause=backend.sleep)
    assert (actual, recover) == (reason, True)
    assert "overlay-pump" in backend.display.events
    backend.display_lifecycle.close()
    recover_session(backend.directory, backend, backend.display_lifecycle)
    assert backend.killed == [21, 20]
    assert not backend.display.enabled and not backend.display.covered
    assert Session("ac", backend.directory, backend).data["phase"] == "restored"


def test_activity_log_start_stop_and_modal_confirmation():
    assert output_log_active("==== Recording Start ====", "recording")
    assert not output_log_active("==== Recording Start ====\n==== Recording Stop ====", "recording")
    assert output_log_active("Starting streaming session", "streaming")
    assert "Steam launch confirmation" in blocking_prompt([{"title": "Steam", "class": "vguiPopupWindow", "owner": 10}])


@pytest.mark.parametrize("game", ["ac", "ets2"])
def test_game_guard_alias(rig, game, capsys):
    _, backend = rig
    assert main([game, "guard"], backend=backend) == 0
    assert '"guard": "clear"' in capsys.readouterr().out


def test_lock_file_does_not_grow(tmp_path):
    for _ in range(10):
        with lock(tmp_path):
            pass
    assert (tmp_path / ".lock").stat().st_size == 1


def test_streaming_service_activity_uses_connections_not_presence(tmp_path):
    executable = tmp_path / "sunshine.exe"
    log = tmp_path / "config/sunshine.log"
    log.parent.mkdir()
    log.write_text("Sunshine version: test\nCLIENT CONNECTED\nCLIENT DISCONNECTED\n")
    backend = Windows.__new__(Windows)
    rows = [{"pid": 1, "name": "sunshine.exe", "path": str(executable), "born": 0}]
    assert not backend.activity(rows, [])[0][0]["active"]
    log.write_text("Sunshine version: test\nCLIENT CONNECTED\n")
    assert backend.activity(rows, [])[0][0]["active"]
    assert connection_log_active("CLIENT CONNECTED\nCLIENT CONNECTED\nCLIENT DISCONNECTED")


def test_owner_replaced_managed_file_with_directory_is_untouched(rig):
    session, backend = rig
    session.start({}, 30)
    target = backend.gamepath / "cfg/controls.ini"
    target.unlink()
    target.mkdir()
    (target / "owner-data").write_bytes(b"keep")
    session.stop()
    assert (target / "owner-data").read_bytes() == b"keep"
