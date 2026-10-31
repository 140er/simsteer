from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib
import os
from pathlib import Path
import pkgutil
import time
from urllib.parse import unquote

import pytest

from tools import testrig
from tools.testrig import RigError
from tools.testrig.cli import main
from tools.testrig.discovery import discover, libraries, read_vdf
from tools.testrig.display import DisplayLifecycle, recover_session, stop_worker
from tools.testrig.display_api import NativeDisplay, encoded
from tools.testrig.games import ADAPTERS
from tools.testrig.guard import blocking_prompt, require_free, restart_reasons, same_process
from tools.testrig.session import DEFAULTS, Session
from tools.testrig.state import Journal, atomic_json, fingerprint, lock, read_json


def restart(**changes):
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "signals": [], "tasks": [],
            "errors": [], **changes}


def snapshot(processes=(), **changes):
    return {"processes": list(processes), "idle_seconds": 600, "restart": restart(), "errors": [], **changes}


class FakeDisplay:
    def __init__(self, failure=None):
        self.events, self.failure = [], failure
        self.original = {"primary": "physical", "displays": [{"name": "physical", "x": 0, "y": 0,
                            "width": 2560, "height": 1440, "primary": True}]}
        self.topology = deepcopy(self.original)
        self.enabled = False
        self.covered = False

    def step(self, event):
        self.events.append(event)
        if event == self.failure:
            raise RigError("injected " + event)

    def capture(self):
        self.step("capture")
        return deepcopy(self.topology)

    def device_state(self, instance):
        self.step("device-state")
        return 0 if self.enabled else 22

    def keep_awake(self):
        self.step("keep-awake")

    def enable(self, instance):
        self.enabled = True
        self.step("enable")  # failure can happen AFTER a side effect

    def extend_primary(self, instance, original, width, height):
        self.topology = {"primary": "virtual", "displays": [
            {"name": "physical", "x": -2560, "y": 0, "width": 2560, "height": 1440, "primary": False},
            {"name": "virtual", "x": 0, "y": 0, "width": width, "height": height, "primary": True}]}
        self.step("primary")
        return {"name": "virtual", "width": width, "height": height}

    def physical_rectangles(self, original):
        self.step("rectangles")
        return [(-2560, 0, 2560, 1440)]

    def restore(self, original):
        self.step("restore-topology")
        self.topology = deepcopy(original)

    def disable(self, instance):
        self.step("disable")
        self.enabled = False

    def physical_primary(self, original):
        self.step("physical-primary-fallback")
        self.topology["primary"] = original["primary"]

    def restore_physical(self, original):
        self.step("restore-physical")
        self.topology = deepcopy(original)

    def pump(self):
        self.step("overlay-pump")

    def start(self, rectangles):
        assert rectangles == [(-2560, 0, 2560, 1440)]
        self.covered = True
        self.step("overlay-start")

    def close(self):
        self.step("overlay-close")
        self.covered = False


class FakeWindows:
    def __init__(self, root, game="ac"):
        self.root, self.game = root, game
        self.steam, self.library = root / "Steam", root / "Other library"
        self.docs = root / "Redirected Documents"
        for folder in (self.steam / "steamapps", self.library / "steamapps", self.docs):
            folder.mkdir(parents=True)
        (self.steam / "steam.exe").write_bytes(b"steam")
        escaped = str(self.library).replace("\\", "\\\\")
        (self.steam / "steamapps/libraryfolders.vdf").write_text(
            f'"libraryfolders" {{ "0" {{ "path" "{escaped}" }} }}')
        self.appid = "244210" if game == "ac" else "227300"
        (self.library / f"steamapps/appmanifest_{self.appid}.acf").write_text(
            f'"AppState" {{ "appid" "{self.appid}" "installdir" "Game" "StateFlags" "4" }}')
        self.gamepath = self.library / "steamapps/common/Game"
        self.gamepath.mkdir(parents=True)
        self.document_game = self.docs / ("Assetto Corsa" if game == "ac" else "Euro Truck Simulator 2")
        self.document_game.mkdir()
        (self.document_game / "owner.bin").write_bytes(b"\x00original\xff\r\n")
        self.userdata = self.steam / "userdata/1" / self.appid
        self.userdata.mkdir(parents=True)
        (self.userdata / "remotecache.vdf").write_bytes(b"original cloud metadata\r\n")
        if game == "ac":
            for name in ("AssettoCorsa.exe", "acs.exe"):
                (self.gamepath / name).write_bytes(b"game")
            (self.gamepath / "cfg").mkdir()
            (self.gamepath / "cfg/controls.ini").write_bytes(b"[HEADER]\r\nINPUT_METHOD=WHEEL\r\n")
            (self.gamepath / "cfg/race.ini").write_text("[SESSION_1]\nTYPE=3\n")
            (self.gamepath / "content/tracks/magione").mkdir(parents=True)
            (self.gamepath / "content/cars/lotus_elise_sc/skins/default").mkdir(parents=True)
        else:
            (self.gamepath / "bin/win_x64/plugins").mkdir(parents=True)
            (self.gamepath / "bin/win_x64/eurotrucks2.exe").write_bytes(b"game")
            (self.gamepath / "bin/win_x64/plugins/scs-telemetry.dll").write_bytes(b"existing plugin")
        self.rows = [{"pid": 10, "parent": 1, "name": "steam.exe", "path": str(self.steam / "steam.exe"),
                      "created": "steam-identity", "born": 0}]
        self.t, self.base = 0.0, time.time()
        self.uris, self.killed = [], []
        self.token, self.stale = 0, False
        self.busy, self.extra_restart, self.bad_home = None, {}, False
        self.display = FakeDisplay()
        self.display_lifecycle = None
        self.monitor_alive = True
        self.directory = root / "state/ac/run"

    def steam_paths(self):
        return [self.steam]

    def steam_user(self):
        return 1

    def documents(self):
        return self.docs

    def snapshot(self):
        rows = deepcopy(self.rows)
        if self.busy:
            rows.append(self.busy)
        return snapshot(rows, restart=restart(**self.extra_restart), boot_id="counter:10")

    def add_game(self, name, pid, parent):
        path = self.gamepath / ("bin/win_x64/eurotrucks2.exe" if self.game == "ets2" else name)
        self.rows.append({"pid": pid, "parent": parent, "name": name, "path": str(path),
                          "created": f"identity-{pid}", "born": self.wall_time()})

    def launch_uri(self, uri):
        self.uris.append(uri)
        if self.game == "ac":
            self.add_game("AssettoCorsa.exe", 20, 10)
        else:
            self.add_game("eurotrucks2.exe", 20, 10)
            home = self.directory / "ets2-home"
            (home / "game.log.txt").write_text(
                "New profile selected: 'SimSteer Test'\n[zipfs] comma_wide_cam.scs: Mounted, 3 entries\n")

    def launch_ac(self, executable):
        self.add_game("acs.exe", 21, os.getpid())

    def windows(self):
        return [{"pid": row["pid"], "title": "Game", "text": "", "hwnd": row["pid"]}
                for row in self.rows if row["pid"] >= 20]

    def argument(self, process, name):
        return "wrong home" if self.bad_home else str(self.directory / "ets2-home")

    def telemetry(self, game):
        if not self.stale:
            self.token += 1
        return [self.token, self.token] if game == "ac" else [self.token]

    def stop_process(self, process):
        self.killed.append(process["pid"])
        self.rows = [row for row in self.rows if not same_process(row, process)]

    def place_window(self, window, process, width, height):
        assert window["pid"] == process["pid"]
        assert self.display.topology["primary"] == "virtual"
        self.display.events.append("place-game")

    def start_display(self, journal, options):
        path = journal.directory / "display-state.json"
        self.display_lifecycle = DisplayLifecycle(self.display, self.display, lambda row: atomic_json(path, row),
                                                 options, journal.data["id"])
        self.display_lifecycle.row["monitor_started_at"] = self.wall_time()
        self.display_lifecycle.start()

    def stop_display(self, journal):
        if self.display_lifecycle:
            path = journal.directory / "display-state.json"
            self.display_lifecycle.row.update(read_json(path))
            if self.monitor_alive:
                self.display_lifecycle.row.update(stop_reason="explicit-stop", monitor_covered_until=self.wall_time(),
                                                  monitor_stopped_at=self.wall_time())
            self.display_lifecycle.close()

    def sleep(self, seconds):
        self.t += seconds

    def monotonic(self):
        return self.t

    def wall_time(self):
        # Game files use the filesystem's real clock, not virtual polling time.
        return time.time()


@pytest.fixture
def rig(tmp_path):
    backend = FakeWindows(tmp_path)
    return Session("ac", backend.directory, backend), backend


def seed_profile(path):
    path.mkdir()
    (path / "profile.sii").write_text('SiiNunit\n{\nprofile : _nameless.test\n{\n'
        'profile_name: "Offline Test"\nis_local: true\nactive_mods: 0\nonline_user_name: ""\n}\n}\n')
    entries = ["device joy `old`", "input j_steer `old`", "input j_throttle `old`", "input j_brake `old`",
               "constant c_steer_dz 0.1", "constant c_relatsteer 1.0", "constant c_ff_enabled 1.0"]
    entries += [f"device joy{i} `old`" for i in range(2, 7)]
    (path / "controls.sii").write_text("SiiNunit\n{\n" + "\n".join(
        f' config_lines[{i}]: "{entry}"' for i, entry in enumerate(entries)) + "\n}\n")
    save = path / "save/autosave"
    save.mkdir(parents=True)
    (save / "game.sii").write_bytes(b"offline scenario")
    return path


def test_all_testrig_modules_import_without_windows():
    for module in pkgutil.walk_packages(testrig.__path__, "tools.testrig."):
        importlib.import_module(module.name)


def test_vdf_escapes_comments_and_unicode():
    data = read_vdf('// comment\n"LibraryFolders" { "0" { "path" "D:\\\\Games é" } }')
    assert data["libraryfolders"]["0"]["path"] == "D:\\Games é"


@pytest.mark.parametrize("text", ['"x" {', '"x"', '"x" }', '{ "x" "y" }', '"x" "y" }', 'bare value'])
def test_malformed_vdf_refused(text):
    with pytest.raises(RigError):
        read_vdf(text)


def test_discovery_libraries_redirected_documents_and_user(rig):
    session, backend = rig
    installed = discover("ac", backend)
    assert installed.game == backend.gamepath
    assert installed.documents == backend.document_game
    assert installed.userdata == backend.userdata
    assert libraries(backend.steam) == [backend.steam, backend.library]
    metadata = backend.steam / "steamapps/libraryfolders.vdf"
    metadata.write_text(f'"libraryfolders" {{ "1" "{backend.library}" }}')
    assert libraries(backend.steam)[1] == backend.library


@pytest.mark.parametrize("name", ["..", "../Other", "C:\\Other"])
def test_discovery_rejects_path_escape(rig, name):
    _, backend = rig
    (backend.library / "steamapps/appmanifest_244210.acf").write_text(
        '"AppState" { "appid" "244210" "installdir" "' + name + '" "StateFlags" "4" }')
    with pytest.raises(RigError, match="Invalid install"):
        discover("ac", backend)


def test_discovery_missing_game_and_signed_out_steam(rig):
    _, backend = rig
    backend.steam_user = lambda: 0
    with pytest.raises(RigError, match="signed-in"):
        discover("ac", backend)
    (backend.library / "steamapps/appmanifest_244210.acf").unlink()
    with pytest.raises(RigError, match="not installed or not owned"):
        discover("ac", backend)


def test_backup_restore_files_absence_and_crash_retry(tmp_path):
    original, absent, file = tmp_path / "original", tmp_path / "created", tmp_path / "single"
    original.write_bytes(b"\x00\xfforiginal\r\n")
    file.write_bytes(b"single\xff")
    journal = Journal(tmp_path / "session")
    journal.data = {"version": Journal.VERSION, "backups": []}
    journal.backup([original, absent, file])
    expected = [fingerprint(path) for path in (original, absent, file)]
    for path in (original, absent, file):
        journal.write(path, b"session change")
    # First file already restored, as if a crash occurred before progress saved.
    original.write_bytes(b"\x00\xfforiginal\r\n")
    recovered = Journal(journal.directory)
    recovered.restore()
    recovered.restore()
    assert [fingerprint(path) for path in (original, absent, file)] == expected


def test_corrupt_backup_never_deletes_live_data(tmp_path):
    original = tmp_path / "owner"
    original.write_bytes(b"baseline")
    journal = Journal(tmp_path / "session")
    journal.data = {"version": Journal.VERSION}
    journal.backup([original])
    original.write_bytes(b"still here")
    Path(journal.data["backups"][0]["stored"]).write_bytes(b"corrupt")
    with pytest.raises(RigError, match="Backup hash mismatch"):
        journal.restore()
    assert original.read_bytes() == b"still here"


def test_backup_rejects_links_and_session_overlap(tmp_path):
    original = tmp_path / "owner"
    original.mkdir()
    link = tmp_path / "link"
    link.symlink_to(original, target_is_directory=True)
    with pytest.raises(RigError, match="symlink"):
        fingerprint(link)
    journal = Journal(tmp_path / "state")
    journal.data = {"version": Journal.VERSION}
    with pytest.raises(RigError, match="outside"):
        journal.backup([tmp_path])


def test_lock_releases_and_refuses_concurrent_mutations(tmp_path):
    with lock(tmp_path):
        with pytest.raises(RigError, match="Another"):
            with lock(tmp_path):
                pytest.fail("lock should refuse")
    with lock(tmp_path):
        pass


@pytest.mark.parametrize("name", ["vrserver.exe", "ForzaHorizon5.exe", "Le Mans Ultimate.exe"])
def test_guard_busy_software(name):
    with pytest.raises(RigError, match="gaming"):
        require_free(snapshot([{"pid": 1, "name": name}]))


def test_guard_unknown_steam_game_and_reused_pid():
    row = {"pid": 1, "created": "a", "name": "unknown.exe", "path": r"D:\SteamLibrary\steamapps\common\NewGame\unknown.exe"}
    require_free(snapshot([row], foreground={"pid": 1, "fullscreen": True}), owned=[row])
    with pytest.raises(RigError, match="gaming"):
        require_free(snapshot([row], foreground={"pid": 1, "fullscreen": True}), owned=[{**row, "created": "old"}])


@pytest.mark.parametrize("idle", [None, -1, 0, 119, float("nan"), float("inf")])
def test_guard_active_or_unknown_user(idle):
    with pytest.raises(RigError, match="activity|active"):
        require_free(snapshot(idle_seconds=idle))


@pytest.mark.parametrize("signal", ["RebootRequired", "RebootPending", "UpdateOrchestrator RebootDeadline"])
def test_restart_signals_refuse(signal):
    with pytest.raises(RigError, match=signal):
        require_free(snapshot(restart=restart(signals=[signal])))


def test_restart_tasks_staleness_missing_and_denied():
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    task = {"name": "Reboot", "state": "Ready", "next_run": future}
    assert restart_reasons(restart(tasks=[task])) == ["UpdateOrchestrator/Reboot"]
    assert restart_reasons(restart(tasks=[{**task, "state": "Disabled"}])) == []
    assert restart_reasons(restart(tasks=[{**task, "next_run": None}])) == []
    for invalid in ({}, restart(errors=["access denied"]), restart(tasks=[{}]),
                    restart(checked_at="2000-01-01T00:00:00Z")):
        with pytest.raises(RigError, match="incomplete"):
            restart_reasons(invalid)


@pytest.mark.parametrize("title, expected", [("Steam Sign in", "Sign-in"), ("EULA", "EULA"),
    ("Windows Defender Firewall", "Firewall"), ("Game purchase", "ownership")])
def test_blocking_prompts(title, expected):
    assert expected in blocking_prompt([{"title": title}])


def test_ac_complete_idempotent_session_and_byte_restore(rig):
    session, backend = rig
    baseline = [fingerprint(path) for path in (backend.document_game, backend.gamepath / "cfg", backend.userdata)]
    session.prepare({})
    first_state = deepcopy(session.data)
    session.prepare({})
    assert session.data == first_state
    cfg = backend.document_game / "cfg"
    assert "INPUT_METHOD=X360" in (cfg / "controls.ini").read_text()
    assert "FULLSCREEN=0" in (cfg / "video.ini").read_text()
    race = (cfg / "race.ini").read_text()
    assert "Practice" in race and "TRACK=magione" in race and "SESSION_1" not in race
    session.start({}, 5)
    session.start({}, 5)
    assert backend.uris == ["steam://rungameid/244210"]
    assert session.data["phase"] == "running"
    session.stop()
    assert session.data["phase"] == "restored"
    session.stop()
    assert backend.killed == [21, 20]
    session.restore()
    assert [fingerprint(path).get("files") for path in (backend.document_game, backend.gamepath / "cfg", backend.userdata)] == [item.get("files") for item in baseline]
    (backend.document_game / "owner.bin").write_bytes(b"later owner edit")
    session.restore()  # a completed session must never roll back later owner edits
    assert (backend.document_game / "owner.bin").read_bytes() == b"later owner edit"


def test_crash_after_steam_launch_recovers_ownership(rig):
    session, backend = rig
    session.prepare({})
    original = backend.telemetry
    backend.telemetry = lambda _: (_ for _ in ()).throw(RigError("crash"))
    with pytest.raises(RigError, match="crash"):
        session.start({}, 5)
    assert backend.uris
    recovered = Session("ac", backend.directory, backend)
    backend.telemetry = original
    recovered.restore()
    assert backend.killed == [21, 20]
    assert recovered.data["phase"] == "restored"


def test_crash_before_first_process_observation_adopts_only_intended_launch(rig):
    session, backend = rig
    launch = backend.launch_uri
    def launch_then_crash(uri):
        launch(uri)
        raise RigError("lost startup caller")
    backend.launch_uri = launch_then_crash
    with pytest.raises(RigError, match="lost startup"):
        session.start({}, 5)
    assert session.data["launch"] and not session.data["owned"]
    recovered = Session("ac", backend.directory, backend)
    recovered.restore()
    assert backend.killed == [20]
    assert recovered.data["phase"] == "restored"


def test_orphaned_session_files_never_overwritten(rig):
    session, backend = rig
    (backend.directory / "backup").mkdir(parents=True)
    sentinel = backend.directory / "backup/original"
    sentinel.write_bytes(b"retain")
    with pytest.raises(RigError, match="without a journal"):
        session.prepare({})
    assert sentinel.read_bytes() == b"retain"


@pytest.mark.parametrize("device", [None, r"ROOT\*", r"ROOT\DISPLAY\????"])
def test_display_requires_one_explicit_device_before_any_edits(rig, device):
    session, backend = rig
    with pytest.raises(RigError, match="exact|wildcards"):
        session.prepare({"display": "vdd", "vdd_device": device})
    assert not backend.display.events and session.data is None


def test_status_observes_identity_without_rewriting_state(rig):
    session, backend = rig
    session.start({}, 5)
    before = session.journal.path.read_bytes()
    status = session.status(live=True)
    assert status["observed_owned_pids"] == [20, 21]
    assert session.journal.path.read_bytes() == before


def test_reused_pid_never_killed_and_live_writer_blocks_restore(rig):
    session, backend = rig
    session.start({}, 5)
    backend.rows[-1].update(created="reused", born=backend.base + 1000)
    with pytest.raises(RigError, match="still running"):
        session.restore()
    assert 21 not in backend.killed
    assert session.data["phase"] != "restored"


def test_stale_telemetry_is_bounded_and_cli_rolls_back(rig, capsys):
    session, backend = rig
    backend.stale, backend.token = True, 7
    assert main(["ac", "start", "--state-dir", str(backend.root / "state"), "--session", "run", "--timeout", "3"], backend=backend) == 2
    assert "live changing telemetry" in capsys.readouterr().err
    assert backend.killed == [21, 20]
    assert Session("ac", backend.directory, backend).data["phase"] == "restored"


def test_guard_rechecked_before_launch_and_display_failure_restores_config(rig):
    session, backend = rig
    baseline = fingerprint(backend.document_game)
    backend.display.failure = "primary"
    with pytest.raises(RigError, match="injected primary") as caught:
        session.start({"display": "vdd", "vdd_device": r"ROOT\VIRTUAL\0001"}, 5)
    session.failure_cleanup(caught.value)
    assert not backend.display.enabled and not backend.display.covered
    assert backend.display.topology == backend.display.original
    assert fingerprint(backend.document_game)["files"] == baseline["files"]
    assert backend.uris == []


def test_owner_activity_rechecked_after_display_setup(rig):
    session, backend = rig
    real_start = backend.start_display
    def display_then_busy(journal, options):
        real_start(journal, options)
        backend.busy = {"pid": 500, "name": "ForzaHorizon5.exe"}
    backend.start_display = display_then_busy
    with pytest.raises(RigError, match="gaming") as caught:
        session.start({"display": "vdd", "vdd_device": "virtual"}, 5)
    session.failure_cleanup(caught.value)
    assert backend.uris == []
    assert not backend.display.enabled and backend.display.topology == backend.display.original


def test_vdd_game_window_placement_then_stop_restores_everything(rig):
    session, backend = rig
    baseline = fingerprint(backend.gamepath / "cfg")
    session.start({"display": "vdd", "vdd_device": "virtual", "width": 1920, "height": 1080}, 5)
    assert "place-game" in backend.display.events
    assert session.data["placed_windows"]
    session.stop()
    assert backend.display.topology == backend.display.original
    assert not backend.display.enabled and not backend.display.covered
    assert fingerprint(backend.gamepath / "cfg") == baseline


def test_display_failure_after_start_recovers_game_and_config(rig):
    session, backend = rig
    baseline = fingerprint(backend.document_game)
    session.start({"display": "vdd", "vdd_device": "virtual"}, 5)
    backend.display_lifecycle.close()
    recover_session(backend.directory, backend, backend.display_lifecycle)
    assert backend.killed == [21, 20]
    assert fingerprint(backend.document_game)["files"] == baseline["files"]
    assert Session("ac", backend.directory, backend).data["phase"] == "restored"


def test_prepare_failure_after_config_write_rolls_back(rig, monkeypatch):
    session, backend = rig
    baseline = fingerprint(backend.document_game)
    def failed_prepare(installed, directory, options, journal):
        journal.write(installed.documents / "cfg/controls.ini", b"changed")
        raise RigError("config failure")
    monkeypatch.setattr(session.adapter, "prepare", failed_prepare)
    with pytest.raises(RigError, match="config failure") as caught:
        session.prepare({})
    session.failure_cleanup(caught.value)
    assert fingerprint(backend.document_game)["files"] == baseline["files"]


def test_ets2_local_profile_only_camera_controls_and_cleanup(tmp_path):
    backend = FakeWindows(tmp_path, "ets2")
    backend.directory = tmp_path / "state/ets2/run"
    seed = seed_profile(tmp_path / "offline-seed")
    baseline = fingerprint(seed)
    session = Session("ets2", backend.directory, backend)
    session.prepare({"profile_source": str(seed)})
    home = backend.directory / "ets2-home"
    profile = next((home / "profiles").iterdir())
    text = (profile / "controls.sii").read_text()
    assert "c_relatsteer 0.000000" in text and "c_steer_dz 0.000000" in text
    assert "xinput.xinput_gamepad_1" in text
    assert fingerprint(home / "mod/comma_wide_cam.scs") == fingerprint(ADAPTERS["ets2"].camera)
    assert "active_mods: 1" in (profile / "profile.sii").read_text()
    assert not (backend.document_game / "mod").exists()
    session.start({}, 5)
    assert unquote(backend.uris[0]).startswith('steam://rungameid/227300//-nointro -homedir ')
    assert '-profile "SimSteer Test"' in unquote(backend.uris[0])
    session.restore()
    assert home.exists()  # directories and untracked game logs are retained
    assert not (profile / "profile.sii").exists()
    assert not (home / "mod/comma_wide_cam.scs").exists()
    assert fingerprint(seed) == baseline


@pytest.mark.parametrize("kind", ["encrypted", "cloud", "account"])
def test_ets2_rejects_unsafe_or_uneditable_seeds(tmp_path, kind):
    backend = FakeWindows(tmp_path, "ets2")
    seed = seed_profile(tmp_path / "seed")
    if kind == "encrypted":
        (seed / "profile.sii").write_bytes(b"ScsC\xffencrypted")
    elif kind == "cloud":
        (seed / "profile.sii").write_text((seed / "profile.sii").read_text().replace("is_local: true", "steam_cloud: true"))
    else:
        (seed / "profile.sii").write_text((seed / "profile.sii").read_text().replace('online_user_name: ""', 'online_user_name: "account"'))
    session = Session("ets2", tmp_path / "state/ets2/run", backend)
    with pytest.raises(RigError, match="plaintext|Cloud|account"):
        session.prepare({"profile_source": str(seed)})
    assert session.data is None


def test_ets2_rejects_ignored_homedir_and_stops_owned_launch(tmp_path):
    backend = FakeWindows(tmp_path, "ets2")
    backend.directory = tmp_path / "state/ets2/run"
    backend.bad_home = True
    seed = seed_profile(tmp_path / "seed")
    session = Session("ets2", backend.directory, backend)
    with pytest.raises(RigError, match="did not forward") as caught:
        session.start({"profile_source": str(seed)}, 5)
    session.failure_cleanup(caught.value)
    assert backend.killed == [20]
    assert not list((backend.directory / "ets2-home").rglob("profile.sii"))


def test_display_success_order_saved_original_and_idempotence():
    api, receipts = FakeDisplay(), []
    lifecycle = DisplayLifecycle(api, api, lambda row: receipts.append(deepcopy(row)),
                                 DEFAULTS | {"display": "vdd", "vdd_device": "virtual", "overlay": True}, "test")
    lifecycle.start()
    assert receipts[0]["original"] == api.original and not receipts[0]["enable_intent"]
    assert api.events == ["capture", "keep-awake", "device-state", "enable", "primary", "rectangles", "overlay-start"]
    assert api.topology["primary"] == "virtual" and len(api.topology["displays"]) == 2
    lifecycle.close()
    assert api.events[-3:] == ["restore-topology", "disable", "overlay-close"]
    assert api.topology == api.original and not api.enabled and not api.covered
    before = list(api.events)
    lifecycle.close()
    assert api.events == before


@pytest.mark.parametrize("failure", ["capture", "keep-awake", "device-state", "enable", "primary", "rectangles", "overlay-start"])
def test_display_each_start_failure_cleans_partial_effects(failure):
    api, receipts = FakeDisplay(failure), []
    lifecycle = DisplayLifecycle(api, api, lambda row: receipts.append(deepcopy(row)),
                                 DEFAULTS | {"display": "vdd", "vdd_device": "virtual", "overlay": True}, "test")
    with pytest.raises(RigError, match="injected"):
        lifecycle.start()
    assert not api.enabled and not api.covered
    assert api.topology == api.original
    assert not lifecycle.row["ready"] and lifecycle.row["overlay_closed"]
    if failure in ("enable", "primary", "rectangles", "overlay-start"):
        assert "disable" in api.events


@pytest.mark.parametrize("failure", ["restore-topology", "disable", "overlay-close"])
def test_display_cleanup_failure_does_not_skip_other_steps(failure):
    api = FakeDisplay()
    lifecycle = DisplayLifecycle(api, api, lambda _: None,
                                 DEFAULTS | {"display": "vdd", "vdd_device": "virtual", "overlay": True}, "test")
    lifecycle.start()
    api.failure = failure
    lifecycle.close()
    assert {"restore-topology", "disable", "overlay-close"}.issubset(api.events)
    assert lifecycle.row["errors"] or lifecycle.row["warnings"]
    assert not lifecycle.row["ready"]
    api.failure = None
    lifecycle.close()
    assert api.topology == api.original and not api.enabled and not api.covered
    assert all(lifecycle.row[key] for key in ("topology_restored", "vdd_disabled", "overlay_closed"))


def test_dead_display_worker_recovery_uses_saved_topology(rig, monkeypatch):
    session, backend = rig
    session.prepare({"display": "vdd", "vdd_device": "virtual"})
    api = FakeDisplay()
    path = backend.directory / "display-state.json"
    lifecycle = DisplayLifecycle(api, api, lambda row: atomic_json(path, row), session.data["options"], session.data["id"])
    lifecycle.start()
    session.data["display_worker"] = {"pid": 999, "created": "gone", "path": "worker.exe"}
    monkeypatch.setattr("tools.testrig.display.NativeDisplay", lambda: api)
    monkeypatch.setattr("tools.testrig.display.BlackOverlay", lambda: api)
    stop_worker(session.journal, backend)
    assert api.topology == api.original and not api.enabled and not api.covered
    assert read_json(path)["topology_restored"]


def test_native_display_extension_preserves_physical_mode_and_uses_primary_flag():
    import ctypes
    import struct
    def mode(width, height):
        data = ctypes.create_string_buffer(220)
        struct.pack_into("<H", data, 68, 220)
        struct.pack_into("<III", data, 168, 32, width, height)
        struct.pack_into("<I", data, 184, 60)
        return data
    original = {"primary": "physical", "displays": [{"name": "physical", "x": 0, "y": 0,
        "width": 2560, "height": 1440, "primary": True, "mode": encoded(mode(2560, 1440))}]}
    api = NativeDisplay.__new__(NativeDisplay)
    api.sources = lambda: [{"name": "virtual", "adapter_path": r"\\?\ROOT#DISPLAY#0000#{5b45201d-f2f2-4f3b-85bb-30ff1f953599}"}]
    api.physical_displays = lambda layout: layout["displays"]
    api.mode = lambda name, index: mode(1920, 1080)
    queued, active = [], []
    def change(name, data, flags):
        queued.append((name, bytes(data) if data is not None else None, flags))
        if name:
            active.append({"name": name, "primary": bool(flags & 0x10),
                "x": struct.unpack_from("<i", data, 76)[0], "y": struct.unpack_from("<i", data, 80)[0],
                "width": struct.unpack_from("<I", data, 172)[0], "height": struct.unpack_from("<I", data, 176)[0]})
    api.change, api.active = change, lambda: active
    assert api.extend_primary(r"ROOT\DISPLAY\0000", original, 1920, 1080)["name"] == "virtual"
    assert [item[0] for item in queued] == ["physical", "virtual", None]
    assert queued[0][2] == 0x10000001 and queued[1][2] == 0x10000011
    assert struct.unpack_from("<I", queued[0][1], 72)[0] == 0x20  # position only
    assert active[0] == {"name": "physical", "primary": False, "x": -2560, "y": 0, "width": 2560, "height": 1440}
    assert active[1]["primary"] and active[1]["x"] == 0


def test_native_display_restore_supplies_saved_paths_modes_and_verifies_primary():
    class User:
        def SetDisplayConfig(self, paths, path_data, modes, mode_data, flags):
            assert paths == 1 and modes == 2
            assert bytes(path_data) == bytes(72) and bytes(mode_data) == bytes(128)
            assert flags == 0x2a0  # USE_SUPPLIED_DISPLAY_CONFIG | APPLY | SAVE_TO_DATABASE
            return 0
    original = {"path_count": 1, "mode_count": 2, "paths": encoded(bytes(72)), "modes": encoded(bytes(128)),
                "displays": [{"name": "physical", "x": 0, "y": 0, "width": 2560, "height": 1440, "primary": True}]}
    api = NativeDisplay.__new__(NativeDisplay)
    api.user, api.active = User(), lambda: original["displays"]
    api.restore(original)
    api.active = lambda: [{**original["displays"][0], "primary": False}]
    with pytest.raises(RigError, match="physical primary"):
        api.restore(original)
