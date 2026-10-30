"""A game profile: one TOML file that tells SimSteer how to drive one game.

Built-in profiles live in `simsteer/games/profiles/`, user profiles in
`data_dir()/games/`. `docs/PROFILES.md` documents every key. Parsing is
strict: an unknown key, a wrong type or a bad value raises `ProfileError`
naming the file and the dotted key, so a typo never silently drops a
setting.

This module stays light (stdlib only plus the telemetry kind table) because
`--list-games` and argparse load every profile before anything heavy.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from simsteer.runtime.telemetry import KINDS

# `runtime.output.base.DeviceKind`, not imported because it pulls in the
# device drivers.
DEVICE_KINDS = ("gamepad", "wheel")
GATE_PROFILES = ("strict", "loose")
SPEED_SOURCES = ("telemetry", "model")
FOV_AXES = ("horizontal", "vertical")
# Seeds for `ControllerConfig` fields when the game has no saved
# controller file yet.
CONTROLLER_KEYS = ("wheelbase_m", "steer_authority", "steer_max",
                   "lookahead_s", "curvature_anticipation_s")
# Ids name state files (`calibration_<id>.json`). "default" maps to the
# legacy unsuffixed file in `paths.state_path`, "auto" is the detector.
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_RESERVED_IDS = ("auto", "default")


class ProfileError(ValueError):
    def __init__(self, path: Path | str, key: str | None, message: str) -> None:
        self.path = Path(path)
        self.key = key
        self.message = message
        where = f"{self.path}: {key}" if key else str(self.path)
        super().__init__(f"{where}: {message}")


@dataclass(frozen=True)
class TelemetrySpec:
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    longitudinal: bool = True
    speed_source: str = "telemetry"
    synth_wheel_angle: bool = False
    passive_fit_safe: bool = False


@dataclass(frozen=True)
class FovSource:
    file: str
    key: str
    axis: str = "horizontal"


@dataclass(frozen=True)
class Camera:
    fov_h_deg: float | None = None
    hint: str = ""
    fov_source: FovSource | None = None


@dataclass(frozen=True)
class LiveParamsSpec:
    gates: str = "strict"
    seeds: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class SetupCheck:
    check: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GameProfile:
    id: str
    display_name: str
    telemetry: TelemetrySpec
    process_names: tuple[str, ...] = ()
    tested: bool = False
    steam_app_id: int | None = None
    controller: Mapping[str, float] = field(default_factory=dict)
    camera: Camera = Camera()
    capabilities: Capabilities = Capabilities()
    liveparams: LiveParamsSpec = LiveParamsSpec()
    setup: tuple[SetupCheck, ...] = ()
    path: Path | None = None
    builtin: bool = False

    def make_telemetry(self):
        from simsteer.runtime.telemetry import open_telemetry
        return open_telemetry(self.telemetry)


class _Table:
    """One TOML table being read. Tracks which keys were consumed so the
    leftovers can be reported as unknown."""

    def __init__(self, path: Path, data: Mapping[str, Any], prefix: str = "") -> None:
        self.path = path
        self.data = data
        self.prefix = prefix
        self._seen: set[str] = set()

    def key(self, name: str) -> str:
        return f"{self.prefix}{name}"

    def error(self, name: str | None, message: str) -> ProfileError:
        return ProfileError(self.path, self.key(name) if name else self.prefix.rstrip(".") or None,
                            message)

    def has(self, name: str) -> bool:
        return name in self.data

    def raw(self, name: str) -> Any:
        self._seen.add(name)
        return self.data[name]

    def get(self, name: str, kind: str, default: Any = None, required: bool = False) -> Any:
        if name not in self.data:
            if required:
                raise self.error(name, "missing")
            return default
        return _check_type(self, name, self.raw(name), kind)

    def choice(self, name: str, options: tuple[str, ...], default: str) -> str:
        value = self.get(name, "str", default)
        if value not in options:
            raise self.error(name, f"{value!r} is not one of {', '.join(options)}")
        return value

    def table(self, name: str) -> _Table:
        value = self.get(name, "table", {})
        return _Table(self.path, value, f"{self.key(name)}.")

    def finish(self) -> None:
        unknown = [k for k in self.data if k not in self._seen]
        if unknown:
            raise self.error(unknown[0], "unknown key")


_TYPE_NAMES = {"str": "a string", "bool": "true or false", "int": "an integer",
               "float": "a number", "table": "a table", "str_list": "a list of strings",
               "scalar": "a string, number or boolean"}


def _check_type(t: _Table, name: str, value: Any, kind: str) -> Any:
    ok = {
        "str": isinstance(value, str),
        "bool": isinstance(value, bool),
        "int": isinstance(value, int) and not isinstance(value, bool),
        "float": isinstance(value, (int, float)) and not isinstance(value, bool),
        "table": isinstance(value, dict),
        "str_list": isinstance(value, list) and all(isinstance(v, str) for v in value),
        "scalar": isinstance(value, (str, int, float, bool)),
    }[kind]
    if not ok:
        raise t.error(name, f"must be {_TYPE_NAMES[kind]}")
    return float(value) if kind == "float" else value


def _seed(t: _Table, name: str) -> tuple[float, float, float]:
    value = t.raw(name)
    if (not isinstance(value, list) or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
        raise t.error(name, "must be a list of three numbers [a, b, c]")
    return (float(value[0]), float(value[1]), float(value[2]))


def _telemetry(t: _Table) -> TelemetrySpec:
    if not t.data:
        raise t.error(None, "missing")
    kind = t.get("kind", "str", required=True)
    spec = KINDS.get(kind)
    if spec is None:
        raise t.error("kind", f"{kind!r} is not one of {', '.join(KINDS)}")
    params = {}
    for name, typ in spec.params.items():
        if t.has(name):
            params[name] = typ(_check_type(t, name, t.raw(name), typ.__name__))
    t.finish()
    return TelemetrySpec(kind, MappingProxyType(params))


def _camera(t: _Table) -> Camera:
    fov = t.get("fov_h_deg", "float")
    if fov is not None and not 10.0 <= fov <= 170.0:
        raise t.error("fov_h_deg", "must be between 10 and 170")
    hint = t.get("hint", "str", "")
    source = None
    if t.has("fov_source"):
        s = t.table("fov_source")
        source = FovSource(file=s.get("file", "str", required=True),
                           key=s.get("key", "str", required=True),
                           axis=s.choice("axis", FOV_AXES, "horizontal"))
        s.finish()
    t.finish()
    return Camera(fov_h_deg=fov, hint=hint, fov_source=source)


def _liveparams(t: _Table) -> LiveParamsSpec:
    gates = "strict"
    if t.has("gates"):
        g = t.table("gates")
        gates = g.choice("profile", GATE_PROFILES, "strict")
        g.finish()
    seeds = {}
    for device in DEVICE_KINDS:
        if t.has(device):
            d = t.table(device)
            if d.has("seed"):
                seeds[device] = _seed(d, "seed")
            d.finish()
    t.finish()
    return LiveParamsSpec(gates=gates, seeds=MappingProxyType(seeds))


def _setup(t: _Table) -> tuple[SetupCheck, ...]:
    if not t.has("setup"):
        return ()
    entries = t.raw("setup")
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise t.error("setup", "must be [[setup]] tables")
    checks = []
    for i, entry in enumerate(entries):
        s = _Table(t.path, entry, f"setup[{i}].")
        name = s.get("check", "str", required=True)
        params = {k: s.get(k, "scalar") for k in entry if k != "check"}
        s.finish()
        checks.append(SetupCheck(name, MappingProxyType(params)))
    return tuple(checks)


def parse_profile(text: str, path: Path | str, builtin: bool = False) -> GameProfile:
    path = Path(path)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(path, None, f"not valid TOML ({e})") from None
    t = _Table(path, data)

    game_id = t.get("id", "str", required=True)
    if not _ID_RE.match(game_id) or game_id in _RESERVED_IDS:
        raise t.error("id", f"{game_id!r} must be lowercase letters, digits, - or _ "
                            f"and not {' or '.join(_RESERVED_IDS)}")
    display_name = t.get("display_name", "str", required=True)
    process_names = tuple(p.lower() for p in t.get("process_names", "str_list", []))
    tested = t.get("tested", "bool", False)
    steam_app_id = t.get("steam_app_id", "int")

    c = t.table("controller")
    controller = {k: c.get(k, "float") for k in CONTROLLER_KEYS if c.has(k)}
    c.finish()

    caps = t.table("capabilities")
    capabilities = Capabilities(
        longitudinal=caps.get("longitudinal", "bool", True),
        speed_source=caps.choice("speed_source", SPEED_SOURCES, "telemetry"),
        synth_wheel_angle=caps.get("synth_wheel_angle", "bool", False),
        passive_fit_safe=caps.get("passive_fit_safe", "bool", False))
    caps.finish()

    profile = GameProfile(
        id=game_id,
        display_name=display_name,
        process_names=process_names,
        tested=tested,
        steam_app_id=steam_app_id,
        controller=MappingProxyType(controller),
        camera=_camera(t.table("camera")),
        telemetry=_telemetry(t.table("telemetry")),
        capabilities=capabilities,
        liveparams=_liveparams(t.table("liveparams")),
        setup=_setup(t),
        path=path,
        builtin=builtin,
    )
    t.finish()
    return profile


def load_profile(path: Path, builtin: bool = False) -> GameProfile:
    try:
        # utf-8-sig: PowerShell 5.1 and older Notepad write a BOM.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise ProfileError(path, None, f"cannot read ({e})") from None
    return parse_profile(text, path, builtin=builtin)
