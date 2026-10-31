"""Hook timing evidence and tightly owned first-launch Documents cleanup."""

from contextlib import nullcontext
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from test_testrig import rig  # noqa: F401
from tools.testrig import RigError
from tools.testrig import activity
from tools.testrig.activity import InputHeartbeat
from tools.testrig.state import Journal, atomic_json, fingerprint, read_json


def heartbeat():
    clock = [0.0]
    beat = InputHeartbeat(100, clock=lambda: clock[0], wall=lambda: 1000 + clock[0],
                          ticks=lambda: int(clock[0] * 1000),
                          timeout={"registry_ms": 300, "effective_ms": 300, "classification_ms": 300, "source": "registry"})
    return beat, clock


def test_unmatched_tick_with_400ms_gap_records_both_checks():
    beat, clock = heartbeat()
    clock[0] = 0.4
    beat.pump()
    assert not beat.check(200)
    for tick in (0.6, 0.8, 1.0, 1.2, 1.4):
        clock[0] = tick
        beat.pump()
    clock[0] = 1.5
    assert beat.check(200)
    row = beat.diagnostics()
    assert row["failure"]["cause"] == "hook-pump-stall"
    assert row["failure"]["checks"] == {"unobserved_input_tick": True, "pump_stall": False}
    assert row["effective_timeout_ms"] == 300
    assert row["max_pump_gap"]["milliseconds"] == pytest.approx(400)
    assert row["max_pump_gap"]["start"]["wall_time"] == 1000
    assert row["max_pump_gap"]["end"]["tick"] == 400
    assert row["unmatched_ticks"][0]["first_poll"] == 0.4
    assert row["last_input_tick"] == 200 and row["last_hook_tick"] == 100
    assert row["tick_gap_ms"] == 100


def test_unobserved_tick_label_without_a_timeout_sized_pump_gap():
    beat, clock = heartbeat()
    assert not beat.check(200)
    for tick in range(1, 12):
        clock[0] = tick / 10
        beat.pump()
    assert beat.check(200)
    assert beat.diagnostics()["failure"]["cause"] == "unobserved-input-tick"


def test_pump_stall_without_input_still_stops():
    beat, clock = heartbeat()
    clock[0] = 1.1
    assert beat.check(100)
    row = beat.diagnostics()
    assert row["failure"]["checks"] == {"unobserved_input_tick": False, "pump_stall": True}
    assert row["pump_age_ms"] == pytest.approx(1100)
    assert row["failure"]["cause"] == "hook-pump-stall"


def test_400ms_gap_alone_does_not_change_the_stop_policy():
    beat, clock = heartbeat()
    clock[0] = 0.4
    beat.pump()
    assert not beat.check(100)
    assert beat.diagnostics()["failure"] is None


def test_unmatched_ticks_keep_individual_first_polls_and_original_deadline():
    beat, clock = heartbeat()
    assert not beat.check(200)
    clock[0] = 0.5
    beat.pump()
    assert not beat.check(300)
    clock[0] = 1.1
    beat.pump()
    assert beat.check(300)
    rows = {row["tick"]: row for row in beat.diagnostics()["unmatched_ticks"]}
    assert rows[200]["first_poll"] == 0 and rows[300]["first_poll"] == 0.5


def test_late_hook_match_is_preserved_after_the_final_poll():
    beat, clock = heartbeat()
    assert not beat.check(200)
    clock[0] = 1.1
    beat.pump()
    assert beat.check(200)
    beat.observe(200, keyboard=True, flags=0x10, delivered=550)
    beat.check(200)  # the final active() call must not erase the earlier failure
    row = beat.diagnostics()
    assert row["unmatched_ticks"][0]["status"] == "late"
    assert row["unmatched_ticks"][0]["seen_at_last_poll"] is True
    assert row["failure"] is not None
    assert row["max_delivery_lag_ms"] == 350 and row["delivery_lag_at_timeout_count"] == 1


def test_event_counts_and_wrapped_delivery_lag():
    beat, _ = heartbeat()
    beat.observe(0xfffffff0, keyboard=False, flags=1, delivered=0x10)
    beat.observe(200, keyboard=True, flags=0, delivered=210)
    beat.observe(300, keyboard=False, flags=0, delivered=305)
    beat.observe(400, keyboard=True, flags=0x10, delivered=405)
    row = beat.diagnostics()
    assert row["event_counts"] == {"keyboard": {"real": 1, "injected": 1}, "mouse": {"real": 1, "injected": 1}}
    assert row["max_delivery_lag_ms"] == 32


def test_owner_input_stays_triggered_when_the_missing_hook_arrives_late():
    import ctypes
    beat, clock = heartbeat()
    class Info(ctypes.Structure):
        _fields_ = [("time", ctypes.c_uint32)]
    monitor = activity.OwnerInput.__new__(activity.OwnerInput)
    monitor.heartbeat, monitor.info = beat, Info(200)
    monitor.error = monitor.failure = None
    monitor.triggered = threading.Event()
    monitor.get_input = lambda _: True
    assert not monitor.active()
    clock[0] = 1.1
    beat.pump()
    assert monitor.active()
    assert "hook-pump-stall" in monitor.failure and "last_input_tick=200" in monitor.failure
    beat.observe(200, keyboard=True, flags=0x10, delivered=1400)
    assert monitor.active()  # late delivery is evidence, never a bypass
    assert monitor.diagnostics()["unmatched_ticks"][0]["status"] == "late"


@pytest.mark.parametrize("value,expected", [(300, 300), ("400", 400), (5000, 1000), (0, 0)])
def test_registry_timeout_is_reported_with_the_windows_cap(value, expected):
    registry = SimpleNamespace(HKEY_CURRENT_USER=1, OpenKey=lambda *a: nullcontext(1),
                               QueryValueEx=lambda *a: (value, 4))
    row = activity.hook_timeout(registry=registry)
    assert row["registry_ms"] == int(value) and row["effective_ms"] == expected


def test_unreadable_timeout_is_reported_without_inventing_a_value():
    def denied(*args):
        raise PermissionError("denied")
    row = activity.hook_timeout(registry=SimpleNamespace(HKEY_CURRENT_USER=1, OpenKey=denied))
    assert row["effective_ms"] is None and row["classification_ms"] == 1000
    assert "denied" in row["error"]


@pytest.fixture
def new_documents(rig, monkeypatch):
    session, backend = rig
    root = backend.document_game
    (root / "owner.bin").unlink()
    os.rmdir(root)
    clock, births = [100.0], {}
    prepared = {root, root / "cfg"} | {root / "cfg" / name for name in session.adapter.files}
    monkeypatch.setattr(backend, "wall_time", lambda: clock[0])
    monkeypatch.setattr("tools.testrig.state.creation_time", lambda path, info=None:
                        births.get(Path(path), 100.0 if Path(path) in prepared else 102.0), raising=False)
    return SimpleNamespace(session=session, backend=backend, root=root, clock=clock, births=births)


def start_clean(rig):
    rig.session.prepare({})
    rig.clock[0] = 101.0
    rig.session.start({}, 30)
    path = rig.backend.directory / "display-state.json"
    row = read_json(path)
    row["owner_input_seen"] = False
    atomic_json(path, row)
    (rig.root / "logs").mkdir()
    (rig.root / "logs/session.log").write_bytes(b"session log")
    (rig.root / "cfg/generated.ini").write_bytes(b"game generated")
    (rig.root / "cfg/controls.ini").write_bytes(b"game rewrote config")
    rig.clock[0] = 110.0


def test_new_whole_documents_root_is_archived_verified_and_removed(new_documents):
    rig = new_documents
    start_clean(rig)
    original = fingerprint(rig.root)
    assert rig.session.stop()["phase"] == "restored"
    row = rig.session.data["created_roots"][0]
    assert row["path"] == str(rig.root) and row["created_by_session"] is True
    assert row["result"] == "removed" and fingerprint(Path(row["archive"])) == original
    assert not rig.root.exists() and rig.root.parent.is_dir()
    assert Journal(rig.backend.directory).data["created_roots"][0]["result"] == "removed"
    rig.session.restore()
    assert not rig.root.exists()


@pytest.mark.parametrize("attack", ["owner", "heartbeat", "errors", "coverage", "reason", "gap", "old", "late", "failure"])
def test_unproven_root_keeps_the_whole_tree_and_reports_conflict(new_documents, attack):
    rig = new_documents
    if attack == "gap":
        rig.session.prepare({})
        (rig.root / "gap.txt").write_bytes(b"unmonitored owner file")
        rig.births[rig.root / "gap.txt"] = 100.5
    start_clean(rig) if attack != "gap" else rig.session.start({}, 30)
    if attack == "gap":
        rig.clock[0] = 110
    receipt = rig.backend.directory / "display-state.json"
    row = read_json(receipt)
    if attack == "owner":
        row["owner_input_seen"] = True
    elif attack == "heartbeat":
        row["warnings"] = ["Owner input hook heartbeat lost: hook-pump-stall"]
    elif attack == "errors":
        row["errors"] = ["monitor failed"]
    elif attack in ("coverage", "reason"):
        row.update(monitor_covered_until=109, monitor_stopped_at=109,
                   stop_reason="owner-input" if attack == "reason" else "explicit-stop")
        rig.backend.stop_display = lambda _: None
    elif attack in ("old", "late"):
        rig.births[rig.root / "cfg/generated.ini"] = 100.5 if attack == "old" else 111.0
    elif attack == "failure":
        rig.session.data["error"] = "aborted startup"
    atomic_json(receipt, row)
    original = fingerprint(rig.root)
    with pytest.raises(RigError, match="conflict"):
        rig.session.stop()
    assert fingerprint(rig.root) == original
    assert rig.session.data["phase"] == "restored-with-conflicts"
    assert rig.session.data["created_roots"][0]["result"] == "owner_changed"


def test_prepared_but_unmonitored_tree_is_an_explicit_conflict(new_documents):
    rig = new_documents
    rig.session.prepare({})
    original = fingerprint(rig.root)
    with pytest.raises(RigError, match="conflict"):
        rig.session.restore()
    assert fingerprint(rig.root) == original


def test_aborted_start_with_empty_error_text_keeps_created_tree(new_documents):
    rig = new_documents
    start_clean(rig)
    original = fingerprint(rig.root)
    rig.session.failure_cleanup(RigError(""))
    assert rig.session.data["failure_seen"]
    assert rig.session.data["phase"] == "restored-with-conflicts"
    assert fingerprint(rig.root) == original


def test_junction_attribute_is_refused_without_unlink(new_documents, monkeypatch):
    rig = new_documents
    start_clean(rig)
    junction = rig.root / "logs"
    lstat = Path.lstat
    def attributes(path):
        info = lstat(path)
        if path == junction:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info
    with monkeypatch.context() as fake:
        fake.setattr(Path, "lstat", attributes)
        with pytest.raises(RigError, match="conflict"):
            rig.session.stop()
    assert (rig.root / "logs/session.log").read_bytes() == b"session log"
    assert (rig.root / "cfg/controls.ini").read_bytes() == b"game rewrote config"


def test_bad_archive_hash_prevents_every_unlink(new_documents, monkeypatch):
    import shutil
    rig = new_documents
    start_clean(rig)
    original = fingerprint(rig.root)
    copy = shutil.copyfile
    def corrupt(source, destination):
        result = copy(source, destination)
        Path(destination).write_bytes(b"corrupt archive")
        return result
    monkeypatch.setattr("tools.testrig.state.shutil.copyfile", corrupt)
    with pytest.raises(RigError, match="conflict"):
        rig.session.stop()
    assert fingerprint(rig.root) == original


def test_file_created_after_listing_prevents_every_unlink(new_documents, monkeypatch):
    import shutil
    rig = new_documents
    start_clean(rig)
    copy = shutil.copyfile
    def late(source, destination):
        result = copy(source, destination)
        (rig.root / "late.txt").write_bytes(b"new owner file")
        rig.births[rig.root / "late.txt"] = 111
        return result
    monkeypatch.setattr("tools.testrig.state.shutil.copyfile", late)
    with pytest.raises(RigError, match="conflict"):
        rig.session.stop()
    assert (rig.root / "late.txt").read_bytes() == b"new owner file"
    assert (rig.root / "cfg/controls.ini").read_bytes() == b"game rewrote config"


def test_existing_owner_root_is_never_eligible_for_tree_cleanup(rig):
    session, backend = rig
    session.start({}, 30)
    generated = backend.document_game / "generated.txt"
    generated.write_bytes(b"owner directory output")
    session.stop()
    assert generated.read_bytes() == b"owner directory output"
    assert (backend.document_game / "owner.bin").exists()
    assert not session.data.get("created_roots")


def test_empty_directory_in_the_prepare_start_gap_prevents_tree_removal(new_documents):
    rig = new_documents
    rig.session.prepare({})
    (rig.root / "owner-directory").mkdir()
    rig.clock[0] = 101
    rig.session.start({}, 30)
    with pytest.raises(RigError, match="conflict"):
        rig.session.stop()
    assert (rig.root / "owner-directory").is_dir()


@pytest.mark.parametrize("change", ["source", "archive"])
def test_entry_change_during_remove_intent_save_is_rechecked(new_documents, monkeypatch, change):
    rig = new_documents
    start_clean(rig)
    save = rig.session.journal.save
    changed = False
    def edit():
        nonlocal changed
        save()
        row = rig.session.data.get("created_roots", [{}])[0]
        name = row.get("remove_intent")
        if name and not changed:
            changed = True
            base = rig.root if change == "source" else Path(row["archive"])
            (base / name).write_bytes(b"changed after intent save")
    monkeypatch.setattr(rig.session.journal, "save", edit)
    with pytest.raises(RigError, match="conflict"):
        rig.session.stop()
    assert changed and len(fingerprint(rig.root)["files"]) == 8  # all entries retained


def test_late_file_between_unlinks_is_kept_with_verified_archive(new_documents, monkeypatch):
    from tools.testrig import state
    rig = new_documents
    start_clean(rig)
    original = fingerprint(rig.root)
    remove = state.remove
    def late(path):
        remove(path)
        (rig.root / "late-owner.txt").write_bytes(b"owner file")
        rig.births[rig.root / "late-owner.txt"] = 111
    monkeypatch.setattr(state, "remove", late)
    with pytest.raises(RigError, match="conflict"):
        rig.session.stop()
    row = rig.session.data["created_roots"][0]
    assert (rig.root / "late-owner.txt").read_bytes() == b"owner file"
    assert fingerprint(Path(row["archive"])) == original
    assert row["result"] == "owner_changed" and row["removed_entries"]


def test_owned_actions_have_wall_and_input_clock_in_state_and_display_receipt(rig):
    session, backend = rig
    session.start({"display": "vdd", "vdd_device": "virtual"}, 30)
    session.stop()
    receipt = session.owner_receipt()
    actions = receipt["owned_actions"]
    assert {row["action"] for row in actions} >= {
        "display-apply-request", "display-applied", "overlay-created", "steam-uri-launch", "owned-process-observed"}
    assert all(isinstance(row["wall_time"], float) and isinstance(row["tick"], int) for row in actions)


def test_final_worker_poll_persists_late_match_diagnostics(tmp_path, monkeypatch):
    from tools.testrig import display
    beat, clock = heartbeat()
    beat.check(200)
    clock[0] = 1.1
    beat.pump()
    beat.check(200)
    (tmp_path / "display-go").touch()
    state = {"display_parent": {}, "options": {"display": None, "overlay": False, "max_seconds": 30}, "id": "test"}
    monkeypatch.setattr(display, "Journal", lambda _: SimpleNamespace(data=state))
    monkeypatch.setattr(display, "Windows", lambda: SimpleNamespace(wall_time=lambda: 20,
                       action_time=lambda: {"wall_time": 20, "tick": 20000}))
    monkeypatch.setattr(display, "ProcessHandle", lambda _: SimpleNamespace(alive=lambda: True, close=lambda: None))
    monkeypatch.setattr(display, "NativeDisplay", lambda: SimpleNamespace())
    monkeypatch.setattr(display, "BlackOverlay", lambda: SimpleNamespace(close=lambda: None))
    calls = 0
    def active():
        nonlocal calls
        calls += 1
        if calls == 3:  # watch poll, cleanup poll, final poll
            beat.observe(200, keyboard=True, flags=0x10, delivered=1400)
        beat.check(200)
        return True
    monitor = SimpleNamespace(active=active, diagnostics=beat.diagnostics, close=lambda: None, failure="heartbeat lost")
    monkeypatch.setattr(activity, "OwnerInput", lambda: monitor)
    display.worker(tmp_path)
    row = read_json(tmp_path / "display-state.json")
    assert calls == 3, row
    assert row["input_diagnostics"]["unmatched_ticks"][0]["status"] == "late"
    assert row["input_diagnostics"]["failure"]["checks"]["unobserved_input_tick"] is True
