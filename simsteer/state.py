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


def _write_snapshot(path: Path, payload: str | None) -> bool:
    temporary = None
    try:
        if payload is None:
            path.unlink(missing_ok=True)
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    except OSError:
        state_warning(path, "not saved; check storage and retry")
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class _SnapshotWriter:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending: dict[Path, str | None] = {}
        self.active: Path | None = None
        self.closing = False
        self.thread = threading.Thread(target=self._run, name="state-writer", daemon=True)
        self.thread.start()

    def submit(self, path, payload):
        with self.condition:
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
            except Exception:
                state_warning(path, "not saved; check storage and retry")
            finally:
                with self.condition:
                    self.active = None
                    self.condition.notify_all()

    def flush(self, path=None):
        with self.condition:
            self.condition.wait_for(lambda: (
                not self.pending and self.active is None if path is None
                else path not in self.pending and self.active != path))

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


def _submit_snapshot(path, payload):
    with _writer_lock:
        if _writer is not None:
            _writer.submit(path, payload)
            return True
        return _write_snapshot(path, payload)


def atomic_save_json(path: Path, value) -> bool:
    """Snapshot on the caller; fsync and replace on the session's writer.

    Pending snapshots of the same file coalesce. Outside a driving session,
    saves remain synchronous for setup tools and explicit one-shot writes.
    """
    try:
        payload = json.dumps(value, indent=2, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        state_warning(path, "not saved; check storage and retry")
        return False
    return _submit_snapshot(path, payload)


def delete_state(path: Path) -> None:
    # A deletion follows any in-flight save and supersedes queued snapshots.
    _submit_snapshot(path, None)


def flush_state(path: Path | None = None) -> None:
    with _writer_lock:
        writer = _writer
    if writer is not None:
        writer.flush(path)


def snapshot_exists(path: Path) -> bool:
    flush_state(path)
    return path.exists()


def _clean_stale_temps(path: Path) -> None:
    # A recent sibling may belong to another running process. Only remove
    # old orphans; a live writer never holds a snapshot open for a day.
    cutoff = time.time() - 86400
    for sibling in path.parent.glob(f".{path.name}.*.tmp"):
        try:
            if sibling.stat().st_mtime < cutoff:
                sibling.unlink(missing_ok=True)
        except OSError:
            pass


def _reject_constant(value):
    raise ValueError("non-finite JSON number")


def read_json_object(path: Path) -> dict:
    flush_state(path)
    _clean_stale_temps(path)
    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object")
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
