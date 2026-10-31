"""Atomic JSON snapshots, strict JSON types and visible recovery notices."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
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


def atomic_save_json(path: Path, value) -> bool:
    """Commit a complete snapshot; interruption leaves the old file intact.

    Each writer owns a unique temporary file in the destination directory.
    Close it before replacement, including on Windows. A failed save leaves
    the previous snapshot in place and reports to both the log and HUD.
    """
    temporary = None
    try:
        payload = json.dumps(value, indent=2, allow_nan=False)
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
    except (OSError, ValueError, TypeError, OverflowError):
        state_warning(path, "not saved; check storage and retry")
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _reject_constant(value):
    raise ValueError("non-finite JSON number")


def read_json_object(path: Path) -> dict:
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
