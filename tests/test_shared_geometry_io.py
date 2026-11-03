"""Deterministic writer liveness, deadline propagation and lock ownership."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from simsteer import state, state_import
from simsteer.calib.flow import Step
from simsteer.core.learners import livecalib as lc_module
from simsteer.paths import state_path
from test_shared_camera_regression import make_flow, measure
from test_shared_geometry import trusted, consume, assert_released


class Clock:
    def __init__(self):
        self.now = 100.
    def __call__(self):
        return self.now


def writer(*, alive=True):
    result = object.__new__(state._SnapshotWriter)
    result.condition = threading.Condition()
    result.pending = {}
    result.active = None
    result.closing = False
    result.thread = SimpleNamespace(is_alive=lambda: alive)
    return result


@pytest.mark.parametrize("operation", ["submit", "flush"])
@pytest.mark.parametrize("pending", [False, True])
def test_dead_worker_never_acknowledges_or_hangs(operation, pending, tmp_path):
    worker = writer(alive=False)
    path = tmp_path / "file.json"
    if pending:
        worker.pending[path] = "{}"
    with pytest.raises(state.StateIOUnavailable):
        if operation == "submit":
            worker.submit(path, "{}", deadline=time.monotonic() + 2.)
        else:
            worker.flush(path, deadline=time.monotonic() + 2.)
    assert bool(worker.pending) == pending


@pytest.mark.parametrize("death", [False, True])
def test_no_progress_wake_checks_same_deadline_and_liveness(death, tmp_path):
    clock = Clock()
    worker = writer()
    path = tmp_path / "file.json"
    worker.pending[path] = "{}"
    waits = []
    def wake(timeout):
        waits.append(timeout)
        if death:
            worker.thread.is_alive = lambda: False
        else:
            clock.now += 1.
        return True
    worker.condition.wait = wake
    with pytest.raises(state.StateIOUnavailable):
        worker.flush(path, deadline=state.StateDeadline(102., clock))
    assert waits == ([2.] if death else [2., 1.])


@pytest.mark.parametrize("mutex", ["global", "condition"])
@pytest.mark.parametrize("action", ["save", "load", "reset", "flow"])
def test_blocked_mutex_fails_closed_and_releases_owner(mutex, action, monkeypatch):
    current, owner = trusted()
    held = threading.Event()
    release = threading.Event()
    lock = threading.RLock()
    worker = writer()
    if mutex == "global":
        monkeypatch.setattr(state, "_writer_lock", lock)
    else:
        worker.condition = threading.Condition(lock)
        monkeypatch.setattr(state, "_writer", worker)
    def hold():
        with lock:
            held.set()
            assert release.wait(3)
    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(2)
    deadline = time.monotonic() + .02
    operations = {
        "save": lambda: current.calib.save(game="ets2", deadline=deadline),
        "load": lambda: current.lc.load_state("ets2", deadline=deadline),
        "reset": lambda: current.lc.reset(deadline=deadline),
        "flow": lambda: current._load(deadline=deadline),
    }
    try:
        with pytest.raises(state.StateIOUnavailable):
            operations[action]()
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert owner.epoch is None and not owner.trusted
    assert_released(owner)


@pytest.mark.parametrize("stage", range(1, 10))
def test_timeout_at_each_publication_barrier_revokes_epoch(stage, monkeypatch):
    current = make_flow()
    current.lc.reset()
    measure(current, 299)
    owner = current._geometry_owner
    clock = Clock()
    owner.io_clock = clock
    count = [0]
    saved_deadlines = []
    def boundary(function):
        def call(*args, **kw):
            count[0] += 1
            saved_deadlines.append(kw["deadline"])
            if count[0] == stage:
                clock.now = 103.
            return function(*args, **kw)
        return call
    monkeypatch.setattr(state, "flush_state", boundary(state.flush_state))
    monkeypatch.setattr(state, "atomic_save_json", boundary(state.atomic_save_json))
    with pytest.raises(state.StateIOUnavailable):
        measure(current, 1)
    assert count[0] == stage
    assert all(value is saved_deadlines[0] for value in saved_deadlines)
    assert owner.epoch is None and not owner.trusted
    assert_released(owner)


def test_late_physical_read_cannot_publish_after_deadline(monkeypatch):
    current = make_flow()
    current.lc.reset()
    measure(current, 299)
    owner = current._geometry_owner
    clock = Clock()
    owner.io_clock = clock
    original = Path.read_bytes
    reads = []
    def read(path):
        result = original(path)
        reads.append(path)
        if len(reads) == 5:  # last payload read in final C barrier
            clock.now = 103.
        return result
    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(state.StateIOUnavailable):
        measure(current, 1)
    assert len(reads) == 5
    assert not owner.trusted and owner.epoch is None
    assert_released(owner)


@pytest.mark.parametrize("phase", ["publish", "consume"])
def test_owner_excludes_full_derived_setter_and_other_producer(phase, monkeypatch):
    current, owner = trusted()
    if phase == "publish":
        current.lc.reset()
        measure(current, 299)
    paused = threading.Event()
    proceed = threading.Event()
    original = state.flush_state
    errors = []
    def flush(path=None, **kw):
        if path == owner.paths[2] and not paused.is_set():
            paused.set()
            assert proceed.wait(3)
        return original(path, **kw)
    monkeypatch.setattr(state, "flush_state", flush)
    def run(action):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)
    consumer = threading.Thread(target=lambda: run(
        (lambda: measure(current, 1)) if phase == "publish" else (lambda: make_flow("wheel", cal=current.calib, lc=current.lc))))
    # For consume, create a legacy device snapshot before the thread starts.
    if phase == "consume":
        snapshot = current.to_dict()
        snapshot.update(device="wheel", simple_warp_version=1, step="done", result={"passed": True})
        from simsteer.calib.flow import flow_path
        flow_path("ets2", "wheel").write_text(json.dumps(snapshot))
    consumer.start()
    assert paused.wait(2)
    attempted = [threading.Event(), threading.Event()]
    finished = [threading.Event(), threading.Event()]
    def edit(index, action):
        attempted[index].set()
        run(action)
        finished[index].set()
    setter = threading.Thread(target=lambda: edit(0, lambda: setattr(current.calib, "fov_v_deg", 50.)))
    producer = threading.Thread(target=lambda: edit(1, lambda: measure(current, 1)))
    setter.start()
    producer.start()
    assert all(event.wait(2) for event in attempted)
    assert not any(event.is_set() for event in finished)
    proceed.set()
    for thread in (consumer, setter, producer):
        thread.join(3)
        assert not thread.is_alive()
    assert not errors
    assert all(event.is_set() for event in finished)
    assert not owner.trusted
    assert_released(owner)


@pytest.mark.parametrize("boundary", ["flag", "normalize", "commit", "manifest"])
def test_all_import_error_guards_propagate_dedicated_failure(boundary, monkeypatch, tmp_path):
    error = state.StateIOUnavailable(boundary)
    def fail(*args, **kw):
        raise error
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    if boundary == "flag":
        path = source / "first_drive_done_ets2.flag"
        path.touch()
        monkeypatch.setattr(Path, "read_bytes", fail)
        action = lambda: state_import._normalized(path)
    elif boundary == "normalize":
        path = source / "calibflow_ets2_gamepad.json"
        monkeypatch.setattr(state_import, "read_json_object", fail)
        action = lambda: state_import._normalized(path)
    elif boundary == "commit":
        monkeypatch.setattr(state_import, "atomic_save_json", lambda *args: True)
        monkeypatch.setattr(state_import, "read_json_object", fail)
        action = lambda: state_import._commit(destination / "file.json", {})
    else:
        (destination / state_import.IMPORT_MARKER).touch()
        monkeypatch.setattr(state_import, "read_json_object", fail)
        action = lambda: state_import.import_legacy_state(source, destination)
    with pytest.raises(state.StateIOUnavailable) as caught:
        action()
    assert caught.value is error


@pytest.mark.parametrize("where", ["write", "cleanup", "stale-temp", "worker"])
def test_low_level_dedicated_errors_are_not_reduced_to_disk_fallback(where, monkeypatch, tmp_path):
    error = state.StateIOUnavailable(where)
    def fail(*args, **kw):
        raise error
    path = tmp_path / "state.json"
    if where == "write":
        monkeypatch.setattr(state.tempfile, "NamedTemporaryFile", fail)
        action = lambda: state._write_snapshot(path, "{}")
    elif where == "cleanup":
        monkeypatch.setattr(Path, "unlink", fail)
        action = lambda: state._write_snapshot(path, "{}")
    elif where == "stale-temp":
        temp = tmp_path / ".state.json.orphan.tmp"
        temp.touch()
        monkeypatch.setattr(Path, "stat", fail)
        action = lambda: state._clean_stale_temps(path)
    else:
        worker = writer()
        worker.pending[path] = "{}"
        monkeypatch.setattr(state, "_write_snapshot", fail)
        action = worker._run
    with pytest.raises(state.StateIOUnavailable) as caught:
        action()
    assert caught.value is error


def test_submission_true_is_only_queue_ack(monkeypatch, tmp_path):
    worker = writer()
    monkeypatch.setattr(state, "_writer", worker)
    path = tmp_path / "state.json"
    assert state.atomic_save_json(path, {"x": 1}, deadline=time.monotonic() + 2.) is True
    assert path in worker.pending and not path.exists()
    monkeypatch.setattr(state, "_writer", None)
    monkeypatch.setattr(state, "_write_snapshot", lambda *args: False)
    assert state.atomic_save_json(path, {"x": 1}) is False


def test_pending_old_receipt_coalescing_has_an_explicit_drain_barrier(monkeypatch):
    current, owner = trusted()
    old_receipt = owner._receipt
    current.lc.reset()
    measure(current, 299)
    blocker = state_path("writer_barrier", "ets2")
    blocked = threading.Event()
    release = threading.Event()
    draining = threading.Event()
    trace = []
    errors = []
    original_write, original_flush = state._write_snapshot, state.flush_state
    def write(path, payload):
        if path == blocker:
            blocked.set()
            assert release.wait(3)
        trace.append(path)
        return original_write(path, payload)
    def flush(path=None, **kw):
        if path == owner.paths[2]:
            draining.set()
        return original_flush(path, **kw)
    monkeypatch.setattr(state, "_write_snapshot", write)
    monkeypatch.setattr(state, "flush_state", flush)
    def produce():
        try:
            measure(current, 1)
        except BaseException as exc:
            errors.append(exc)
    with state.background_saves():
        state.atomic_save_json(blocker, {})
        assert blocked.wait(2)
        state.atomic_save_json(owner.paths[2], {"old": "coalesced"})
        state.atomic_save_json(owner.paths[2], old_receipt)
        with state._writer.condition:
            assert owner.paths[2] in state._writer.pending
        producer = threading.Thread(target=produce)
        producer.start()
        try:
            assert draining.wait(2)
        finally:
            release.set()
        producer.join(3)
        assert not producer.is_alive() and not errors
        assert owner.trusted
    positions = [i for i, path in enumerate(trace) if path == owner.paths[2]]
    assert len(positions) == 2
    for path in owner.paths[:2]:
        # Generic saves may precede A; the publication's final payload write
        # must follow the drained receipt and precede the new receipt.
        assert positions[0] < max(i for i, value in enumerate(trace) if value == path) < positions[1]


def test_actual_stopped_worker_with_pending_work_fails_bounded_flush(tmp_path):
    worker = state._SnapshotWriter()
    worker.close()
    path = tmp_path / "never-written.json"
    worker.pending[path] = "{}"
    with pytest.raises(state.StateIOUnavailable):
        worker.flush(path, deadline=time.monotonic() + 2.)
    assert path in worker.pending and not path.exists()
