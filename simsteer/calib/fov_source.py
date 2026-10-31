"""Read a game's FOV from its own config, for the FOV step.

A profile names the file in `[camera.fov_source]` (`file`, `key`, `axis`,
see docs/PROFILES.md). `profile_reader` turns that into a reader that
takes the capture's aspect ratio and returns the horizontal FOV in
degrees, or None when the file, the key or a sane value is missing. The
FOV step then measures instead.

Values come from SCS `uset <key> "<value>"` lines or plain `key = value`
lines. ETS2's built-in profile names no source: its `config.cfg` holds
the multi-monitor FOV, and trucks keep their own `camera_fov`, so the
key for the camera the player drives with is not confirmed.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from pathlib import Path

FOV_MIN_DEG, FOV_MAX_DEG = 40.0, 150.0

FovReader = Callable[[float], float | None]


def read_scs_cfg(path: Path, key: str) -> float | None:
    """Value of `uset <key> "<number>"` in an SCS config file, or None."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(rf'^\s*uset\s+{re.escape(key)}\s+"([-+0-9.eE]+)"', text,
                  flags=re.MULTILINE)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def read_cfg_value(path: Path, key: str) -> float | None:
    """`uset key "1.5"`, `key = 1.5` or `key "1.5"` in a text config."""
    value = read_scs_cfg(path, key)
    if value is not None:
        return value
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(rf'^\s*{re.escape(key)}\s*[=:]?\s*"?([-+0-9.eE]+)"?\s*$', text,
                  flags=re.MULTILINE)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def horizontal_from_vertical(vfov_deg: float, aspect: float) -> float:
    return math.degrees(2.0 * math.atan(
        math.tan(math.radians(vfov_deg) / 2.0) * aspect))


def config_reader(file: str, key: str, axis: str = "horizontal") -> FovReader:
    """`file` may use `~` and environment variables (`%USERPROFILE%` on
    Windows)."""
    path = Path(os.path.expandvars(os.path.expanduser(file)))

    def read(aspect: float) -> float | None:
        value = read_cfg_value(path, key)
        if value is None:
            return None
        if axis == "vertical":
            value = horizontal_from_vertical(value, aspect)
        return value if FOV_MIN_DEG <= value <= FOV_MAX_DEG else None
    return read


def profile_reader(source) -> FovReader | None:
    """Reader for a profile's `camera.fov_source`, or None without one."""
    if source is None:
        return None
    return config_reader(source.file, source.key, source.axis)
