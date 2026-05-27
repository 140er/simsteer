"""Filesystem layout — bundle dir (read-only assets) vs data dir (writable state).

Under PyInstaller (frozen), code and bundled assets live in `sys._MEIPASS` —
an extraction directory that is wiped on exit. Writes there are lost. So:

  - Read-only assets (model .onnx files, bundled DLLs, docs) live in
    `bundle_dir()`. PyInstaller's `datas=[...]` ships things here.
  - Writable state (per-game calibration, liveparams, livecalib_state, wizard
    flags) lives in `data_dir()`, which is `%LOCALAPPDATA%\\SimSteer\\` when
    frozen and the repo root when not.

Per project policy: NO seeding from bundle into data_dir on first launch.
Each user calibrates fresh, every game.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_APP_NAME = "SimSteer"


def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def bundle_dir() -> Path:
    """Read-only assets: model .onnx files, bundled DLLs, docs/screenshots.

    Frozen: `sys._MEIPASS` (the PyInstaller extraction dir).
    Dev:    the repo root (parent of `pilot/`).
    """
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Writable state. Created on first call.

    Frozen: `%LOCALAPPDATA%\\SimSteer\\`.
    Dev:    the repo root (so existing JSONs at the project root still load).
    """
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = str(Path.home() / "AppData" / "Local")
        d = Path(base) / _APP_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(__file__).resolve().parent.parent


def state_path(base_name: str, game: str | None = None) -> Path:
    """Per-game writable state file.

    Returns `<data_dir>/<base>_<game>.json` when `game` is given and not
    "default"; otherwise `<data_dir>/<base>.json` (the legacy single-game
    location). Callers prefer the per-game path on save; on load they should
    use `load_with_fallback` to try per-game first, then legacy.
    """
    root = data_dir()
    if game and game != "default":
        return root / f"{base_name}_{game}.json"
    return root / f"{base_name}.json"


def load_with_fallback(base_name: str, game: str | None) -> Path | None:
    """Resolve the path to load FROM: per-game first, then legacy.
    Returns None if neither file exists (caller uses defaults)."""
    game_path = state_path(base_name, game)
    if game_path.exists():
        return game_path
    legacy = state_path(base_name, None)
    if legacy.exists():
        return legacy
    return None


def model_path(name: str) -> Path:
    """Path to a model file under `<bundle>/models/`. Read-only."""
    return bundle_dir() / "models" / name


def asset_path(name: str) -> Path:
    """Path to a bundled asset under `<bundle>/assets/`. Read-only.
    May not exist — callers (e.g. `pilot.audio`) should handle missing files."""
    return bundle_dir() / "assets" / name


def prereq_path(name: str) -> Path:
    """Path to a bundled prerequisite (e.g. scs-telemetry.dll) under
    `<bundle>/prereqs/`. May not exist outside the shipped bundle."""
    return bundle_dir() / "prereqs" / name
