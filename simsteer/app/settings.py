"""Persistent user-facing settings — the bits a non-technical user
should be able to change without a terminal.

CLI flags still win for the current run (so a dev can `--device wheel`
to override their saved setting), but anything the user picks in the
tuner's Setup tab lands here and applies on the next launch.

Lives at `%LOCALAPPDATA%\\SimSteer\\settings.json` when bundled, or at
the repo root in dev mode (via `simsteer.paths.data_dir()`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, field
import uuid
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

from simsteer.paths import data_dir
from simsteer.runtime.input.mapping import validate_axes


_SETTINGS_FILE = "settings.json"


def _loaded_value(value, expected):
    """Validate JSON types without parsing strings or truncating fractions."""
    if type(value) is expected:
        return value
    origin = get_origin(expected)
    args = get_args(expected)
    if origin in (Union, UnionType):
        for member in args:
            try:
                return _loaded_value(value, member)
            except (TypeError, ValueError, OverflowError):
                continue
    elif expected is float and type(value) is int:
        return float(value)
    elif expected is int and type(value) is float and value.is_integer():
        return int(value)
    elif origin is list and isinstance(value, list):
        return [_loaded_value(item, args[0]) for item in value]
    elif origin is dict and isinstance(value, dict):
        return {_loaded_value(k, args[0]): _loaded_value(v, args[1])
                for k, v in value.items()}
    raise ValueError("incompatible setting type")


@dataclass
class Settings:
    # A game profile id, or "auto" for the first profile whose process
    # is running. An id with no profile falls back to "auto".
    game: str = "auto"
    # Output device kind. "gamepad" = ViGEm Xbox 360. "wheel" = vJoy.
    device: str = "gamepad"
    # Let LiveParams fit while disengaged in games whose profile does
    # not set `passive_fit_safe` (off by default: in ETS2 the gamepad
    # rack assist makes wheel/keyboard-sourced samples bias the fit).
    passive_fit: bool = False
    # Disable active steering probing during onboarding calibration.
    # Default ON because probes dramatically accelerate LP convergence;
    # only turn it OFF if the small steering wobble during setup is
    # unwanted.
    no_probe: bool = False
    # Bypass the engagement gate (camera-calibrated + telemetry + FPS).
    # DEV ONLY — without calibration the truck won't track lanes.
    force_engage: bool = False
    # Forza Data Out UDP port — must match the in-game setting.
    forza_port: int = 7777
    # vJoy device index when device=wheel.
    vjoy_device: int = 1
    wheel_passthrough: bool = True
    rim_range_deg: int = 900
    wheel_instance: str = ""
    wheel_axes: dict = field(default_factory=dict)
    # cv2 overlay max width (px). Larger = nicer view, more GPU/CPU.
    max_width: int = 1600
    # Suppress the tuner window entirely (advanced; you lose the
    # Setup tab too, so leave this off unless you're scripting).
    no_tuner: bool = False
    # Run loop with no virtual-pad output (overlay-only mode for
    # capturing screenshots / debugging the model without driving).
    no_gamepad: bool = False
    # HUD mode: "user" (clean panels) or "dev" (stack-of-text dump).
    # Toggle live with the H hotkey; this value is the persisted default.
    hud_mode: str = "user"
    # List of simsteer.app.hotkeys ids that are disabled. Managed via
    # the Hotkeys tab. Stored as a list for JSON friendliness; lookup
    # treats it as a set.
    disabled_hotkeys: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Dataclass defaults for mutable types (list/dict) can't be
        # literal — share-state hazard. Resolve to a fresh empty list.
        if self.disabled_hotkeys is None:
            self.disabled_hotkeys = []
        # Runtime overrides are deliberately outside the persisted dataclass.
        self._runtime_overrides: dict = {}
        # load() validates types; these are wheel-specific value constraints.
        self.rim_range_deg = max(180, min(2520, self.rim_range_deg))
        try:
            self.wheel_instance = str(uuid.UUID(self.wheel_instance)).upper() if self.wheel_instance else ""
        except (TypeError, ValueError, AttributeError):
            self.wheel_instance = ""
        self.wheel_axes = validate_axes(self.wheel_axes)

    def override(self, **values) -> None:
        """Apply values for this run, retaining their persisted baseline."""
        for name, value in values.items():
            self._runtime_overrides.setdefault(name, getattr(self, name))
            setattr(self, name, value)

    def set_persistent(self, **values) -> None:
        """An explicit user edit replaces any runtime override of that field."""
        for name, value in values.items():
            self._runtime_overrides.pop(name, None)
            setattr(self, name, value)

    @classmethod
    def load(cls) -> "Settings":
        """Read settings from disk. Missing file -> defaults, no
        warning. Unknown fields ignored (forward compat). Bad fields
        fall back to defaults (don't crash launch on a corrupted file).
        """
        p = data_dir() / _SETTINGS_FILE
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        if "passive_fit_ets2" in raw:
            raw.setdefault("passive_fit", raw.pop("passive_fit_ets2"))
        # Resolve postponed annotations before validating known fields.
        valid = {f.name for f in fields(cls)}
        hints = get_type_hints(cls)
        kwargs: dict = {}
        for k, v in raw.items():
            if k not in valid:
                continue
            try:
                kwargs[k] = _loaded_value(v, hints[k])
            except (TypeError, ValueError, OverflowError):
                print(f"warning: invalid setting {k!r}; using default")
        return cls(**kwargs)

    def save(self) -> None:
        """Write to disk. Failures are swallowed — settings are
        nice-to-have, not critical to the loop."""
        try:
            p = data_dir() / _SETTINGS_FILE
            saved = asdict(self)
            saved.update(self._runtime_overrides)
            p.write_text(json.dumps(saved, indent=2),
                         encoding="utf-8")
        except OSError:
            pass

    def path(self) -> str:
        """Filesystem path for display in the UI."""
        return str(data_dir() / _SETTINGS_FILE)
