"""Atomic JSON snapshots, strict JSON types and visible recovery notices."""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

_notices: dict[Path, str] = {}
_notice_lock = threading.Lock()


class StateIOUnavailable(OSError):
    """The snapshot worker cannot make progress within the caller's deadline."""


class StateDeadline(float):
    """An absolute monotonic deadline, with an injectable clock for tests."""

    def __new__(cls, value, clock=time.monotonic):
        result = super().__new__(cls, value)
        result.clock = clock
        return result


def _remaining(deadline):
    remaining = deadline - getattr(deadline, "clock", time.monotonic)()
    if remaining <= 0:
        raise StateIOUnavailable("snapshot deadline expired")
    return remaining


@contextmanager
def _state_lock(lock, deadline):
    if deadline is None:
        lock.acquire()
    elif not lock.acquire(timeout=_remaining(deadline)):
        raise StateIOUnavailable("snapshot lock unavailable")
    try:
        if deadline is not None:
            _remaining(deadline)
        yield
    finally:
        lock.release()


def state_warning(path: Path, reason: str) -> None:
    message = f"{path.stem.split('_')[0].capitalize()} {reason}"
    logging.getLogger(__name__).warning(message)
    with _notice_lock:
        _notices[path] = message


def take_state_warnings(directory: Path) -> str:
    """The loop displays notices from its own state directory as a banner."""
    with _notice_lock:
        paths = [p for p in _notices if p.parent == directory]
        return "; ".join(_notices.pop(p) for p in paths)


def _write_snapshot(path: Path, payload: str | bytes | None) -> bool:
    temporary = None
    try:
        if payload is None:
            path.unlink(missing_ok=True)
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload.encode("utf-8") if isinstance(payload, str) else payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    except StateIOUnavailable:
        raise
    except OSError:
        state_warning(path, "not saved; check storage and retry")
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except StateIOUnavailable:
                raise
            except OSError:
                pass


class _SnapshotWriter:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending: dict[Path, str | bytes | None] = {}
        self.active: Path | None = None
        self.closing = False
        self.thread = threading.Thread(target=self._run, name="state-writer", daemon=True)
        self.thread.start()

    def _available(self, deadline):
        if deadline is not None:
            _remaining(deadline)
            if not self.thread.is_alive() or self.closing:
                raise StateIOUnavailable("snapshot worker unavailable")

    def submit(self, path, payload, *, deadline=None):
        with _state_lock(self.condition, deadline):
            self._available(deadline)
            self.pending[path] = payload
            self.condition.notify_all()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.closing)
                if not self.pending:
                    return
                path = next(iter(self.pending))
                payload = self.pending.pop(path)
                self.active = path
            try:
                _write_snapshot(path, payload)
            except StateIOUnavailable:
                # Disk writes receive no caller deadline, so this is defensive
                # against injected/future failures. Let the worker terminate;
                # finally wakes waiters and deadline-bound flush detects it.
                raise
            except Exception:
                state_warning(path, "not saved; check storage and retry")
            finally:
                with self.condition:
                    self.active = None
                    self.condition.notify_all()

    def flush(self, path=None, *, deadline=None):
        with _state_lock(self.condition, deadline):
            self._available(deadline)
            def finished():
                return (
                not self.pending and self.active is None if path is None
                else path not in self.pending and self.active != path)
            while not finished():
                self._available(deadline)
                self.condition.wait(None if deadline is None else _remaining(deadline))
                self._available(deadline)

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join()


_writer_lock = threading.RLock()
_writer: _SnapshotWriter | None = None
_writer_users = 0


@contextmanager
def background_saves():
    """A session uses one writer; the last session flushes after output stops."""
    global _writer, _writer_users
    with _writer_lock:
        if _writer is None:
            _writer = _SnapshotWriter()
        _writer_users += 1
    try:
        yield
    finally:
        with _writer_lock:
            _writer_users -= 1
            if _writer_users == 0:
                _writer.close()
                _writer = None


def with_background_saves(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with background_saves():
            return function(*args, **kwargs)
    return wrapped


def _submit_snapshot(path, payload, *, deadline=None):
    with _state_lock(_writer_lock, deadline):
        if _writer is not None:
            if deadline is None:
                _writer.submit(path, payload)
            else:
                _writer.submit(path, payload, deadline=deadline)
            return True
        return _write_snapshot(path, payload)


def atomic_save_json(path: Path, value, *, deadline=None) -> bool:
    """Snapshot on the caller; fsync and replace on the session's writer.

    Pending snapshots of the same file coalesce. Outside a driving session,
    saves remain synchronous for setup tools and explicit one-shot writes.
    """
    try:
        payload = json.dumps(value, indent=2, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        state_warning(path, "not saved; check storage and retry")
        return False
    return _submit_snapshot(path, payload, deadline=deadline)


def delete_state(path: Path, *, deadline=None) -> None:
    # A deletion follows any in-flight save and supersedes queued snapshots.
    _submit_snapshot(path, None, deadline=deadline)


def atomic_save_bytes(path: Path, value: bytes) -> bool:
    """Commit a flag through the same ordered, atomic writer as JSON state."""
    return _submit_snapshot(path, value)


def flush_state(path: Path | None = None, *, deadline=None) -> None:
    with _state_lock(_writer_lock, deadline):
        writer = _writer
    if writer is not None:
        if deadline is None:
            writer.flush(path)
        else:
            writer.flush(path, deadline=deadline)


def snapshot_exists(path: Path, *, deadline=None) -> bool:
    flush_state(path, deadline=deadline)
    return path.exists()


def _clean_stale_temps(path: Path) -> None:
    # A recent sibling may belong to another running process. Only remove
    # old orphans; a live writer never holds a snapshot open for a day.
    cutoff = time.time() - 86400
    for sibling in path.parent.glob(f".{path.name}.*.tmp"):
        try:
            if sibling.stat().st_mtime < cutoff:
                sibling.unlink(missing_ok=True)
        except StateIOUnavailable:
            raise
        except OSError:
            pass


def _reject_constant(value):
    raise ValueError("non-finite JSON number")


def read_json_object(path: Path, *, allow_non_finite: bool = False, deadline=None) -> dict:
    """Read state; provenance-aware loaders may defer finite validation."""
    flush_state(path, deadline=deadline)
    _clean_stale_temps(path)
    raw = json.loads(path.read_text(encoding="utf-8"),
                     parse_constant=float if allow_non_finite else _reject_constant)
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object")
    if not allow_non_finite:
        _finite_json(raw)
    return raw


def _finite_json(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")


def loaded_value(value, expected):
    """Validate JSON types without parsing strings or truncating fractions."""
    if type(value) is float and not math.isfinite(value):
        raise ValueError("non-finite number")
    if type(value) is expected:
        return value
    origin = get_origin(expected)
    args = get_args(expected)
    if origin in (Union, UnionType):
        for member in args:
            try:
                return loaded_value(value, member)
            except (TypeError, ValueError, OverflowError):
                continue
    elif expected is float and type(value) is int:
        return float(value)
    elif expected is int and type(value) is float and value.is_integer():
        return int(value)
    elif origin is list and isinstance(value, list):
        return [loaded_value(item, args[0]) for item in value]
    elif origin is dict and isinstance(value, dict):
        return {loaded_value(k, args[0]): loaded_value(v, args[1])
                for k, v in value.items()}
    raise ValueError("incompatible setting type")


def dataclass_values(cls, raw: dict) -> dict:
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    return {k: loaded_value(v, hints[k]) for k, v in raw.items() if k in names}


MAX_STATE_COUNT = (1 << 31) - 1


def bounded_number(value, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError("invalid saved number")
    return float(value)


def bounded_int(value, minimum: int = 0, maximum: int = MAX_STATE_COUNT) -> int:
    bounded_number(value, minimum, maximum)
    if type(value) is float and not value.is_integer():
        raise ValueError("invalid saved count")
    return int(value)
