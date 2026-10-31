"""One-time import of user-owned state from an older source/app directory."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

from simsteer.paths import bundle_dir, data_dir, is_frozen
from simsteer.state import (
    atomic_save_bytes, atomic_save_json, flush_state, read_json_object, snapshot_exists,
    state_warning, take_state_warnings,
)

IMPORT_MARKER = "state-import.json"
_STATE_NAME = re.compile(
    r"(?:(?:settings|(?:calibration|controller|liveparams|livecalib_state|calibflow)"
    r"(?:_[a-zA-Z0-9_-]+)?)\.json|first_drive_done_[a-zA-Z0-9_-]+\.flag)\Z")


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*") if p.is_file() and _STATE_NAME.fullmatch(p.name))


def _normalized(source: Path) -> dict | bytes | None:
    if source.suffix == ".flag":
        try:
            return source.read_bytes()
        except OSError:
            state_warning(source, "unreadable; keeping original")
            return None
    # Use the same typed loaders as startup. Never copy an unchecked blob.
    from simsteer.app.settings import Settings
    from simsteer.calib.flow import CalibFlow
    from simsteer.core.calibration import Calibration
    from simsteer.core.control import ControllerConfig
    from simsteer.core.learners.livecalib import LiveCalib
    from simsteer.core.learners.liveparams import LiveParams, TRUST_COVERAGE_VERSION

    name = source.stem
    if name == "settings":
        return asdict(Settings.load(path=source))
    if name.startswith("calibration"):
        return asdict(Calibration.load(path=source))
    if name.startswith("controller"):
        return asdict(ControllerConfig.load(path=source))
    if name.startswith("liveparams"):
        return {**asdict(LiveParams._load(path=source)), "coverage_version": TRUST_COVERAGE_VERSION}
    if name.startswith("livecalib_state"):
        learner = LiveCalib()
        if not learner.load_state("default", path=source):
            return None
        return learner.state_dict()
    try:
        value = read_json_object(source)
        CalibFlow.validate_snapshot(value)
        return value
    except (OSError, ValueError, TypeError, OverflowError):
        state_warning(source, "unreadable; using defaults")
        return None


def _commit(path: Path, value: dict | bytes) -> bool:
    save = atomic_save_bytes if isinstance(value, bytes) else atomic_save_json
    if not save(path, value):
        return False
    flush_state(path)
    try:
        if isinstance(value, bytes):
            return path.read_bytes() == value
        return read_json_object(path) == value
    except (OSError, ValueError):
        return False


def import_legacy_state(source: Path | None = None, destination: Path | None = None) -> bool:
    """Import once into empty user storage; preserve originals and new edits.

    A durable manifest resumes a failed import without overwriting files
    already copied or changed by the player. Replay/portable overrides
    opt out. Bundled assets are never used as calibration seeds.
    """
    if source is None and os.environ.get("SIMSTEER_DATA_DIR"):
        return False
    if source is None:
        source = Path(sys.executable).resolve().parent if is_frozen() else bundle_dir()
    destination = destination if destination is not None else data_dir()
    if source.resolve() == destination.resolve():
        return False
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / IMPORT_MARKER
    if snapshot_exists(marker):
        try:
            manifest = read_json_object(marker)
            if (type(manifest.get("version")) is not int or manifest["version"] != 1
                    or type(manifest.get("complete")) is not bool):
                raise ValueError("invalid import manifest")
            for field in ("pending", "files"):
                entries = manifest.get(field)
                if not isinstance(entries, list) or not all(
                        isinstance(name, str) and len(name) <= 255
                        and _STATE_NAME.fullmatch(name) for name in entries):
                    raise ValueError("invalid import manifest")
            if manifest["complete"]:
                return False
            names = manifest["pending"]
        except (OSError, ValueError, KeyError, TypeError):
            state_warning(marker, "unreadable; import paused")
            return False
    else:
        names = [p.name for p in _files(source)] if not _files(destination) else []
        manifest = {"version": 1, "complete": not bool(names), "pending": names, "files": []}
        if not _commit(marker, manifest) or not names:
            return False
    for name in list(names):
        target = destination / name
        if not snapshot_exists(target):
            value = _normalized(source / name)
            if take_state_warnings(source):
                state_warning(target, "unreadable; using defaults")
                if name.startswith(("calibration", "liveparams", "livecalib_state", "calibflow")):
                    # A missing snapshot lets startup choose the profile and
                    # device seeds. Persisting universal defaults would hide them.
                    value = None
            if value is not None and not _commit(target, value):
                return False
        manifest["files"].append(name)
        manifest["pending"].remove(name)
        if not _commit(marker, manifest):
            return False
    manifest["complete"] = True
    return _commit(marker, manifest)
