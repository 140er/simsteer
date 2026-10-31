from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import pytest

from simsteer.app.settings import Settings


@dataclass
class TypedSettings(Settings):
    # Exercise supported types that have no fields on Settings yet.
    gain: float = 0.5
    limit: Optional[int] = 5
    label: str | None = "default"
    axes: dict[str, int] = field(default_factory=dict)


def write_settings(data_dir, values):
    (data_dir / "settings.json").write_text(json.dumps(values), encoding="utf-8")


@pytest.mark.parametrize("name,value,expected", [
    ("passive_fit", True, True),
    ("passive_fit", False, False),
    ("forza_port", 9000, 9000),
    ("forza_port", 9000.0, 9000),
    ("gain", 2, 2.0),
    ("gain", 0.25, 0.25),
    ("game", "ets2", "ets2"),
    ("disabled_hotkeys", [], []),
    ("disabled_hotkeys", ["engage", "hud"], ["engage", "hud"]),
    ("limit", None, None),
    ("limit", 10, 10),
    ("label", None, None),
    ("label", "custom", "custom"),
    ("axes", {"steer": 1}, {"steer": 1}),
])
def test_load_valid_types(data_dir, capsys, name, value, expected):
    write_settings(data_dir, {name: value})

    actual = getattr(TypedSettings.load(), name)

    assert actual == expected
    assert type(actual) is type(expected)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name,value", [
    (name, value) for name, values in [
        ("passive_fit", ["true", "false", 0, 1, 0.0, None, [], {}]),
        ("forza_port", ["90", True, False, 1.5, None, [], {}]),
        ("gain", ["0.5", True, False, None, [], {}, 10**400]),
        ("game", [90, 0.5, True, None, [], {}]),
        ("disabled_hotkeys", ["engage", 1, None, {}, ["engage", 1], [None]]),
        ("limit", ["10", True, 1.5, [], {}]),
        ("label", [1, True, [], {}]),
        ("axes", [[], None, "steer", {"steer": True}, {"steer": "1"}]),
    ] for value in values
])
def test_load_invalid_types_keep_defaults(data_dir, capsys, name, value):
    # Other valid fields still load when one field is invalid.
    write_settings(data_dir, {name: value, "max_width": 1200})

    loaded = TypedSettings.load()

    assert getattr(loaded, name) == getattr(TypedSettings(), name)
    assert loaded.max_width == 1200
    warning = capsys.readouterr().out.splitlines()
    assert len(warning) == 1
    assert "warning" in warning[0].lower()
    assert name in warning[0]


@pytest.mark.parametrize("cls", [Settings, TypedSettings])
def test_save_load_round_trip(data_dir, capsys, cls):
    settings = cls(game="ats", passive_fit=True, forza_port=9000,
                   max_width=1200, disabled_hotkeys=["engage", "hud"])
    if isinstance(settings, TypedSettings):
        settings.gain = 1.25
        settings.limit = None
        settings.label = None
        settings.axes = {"steer": 2}

    settings.save()

    assert cls.load() == settings
    assert capsys.readouterr().out == ""


def test_unknown_fields_still_ignored(data_dir, capsys):
    write_settings(data_dir, {"unknown": {"anything": None}, "game": "ets2"})

    assert Settings.load() == Settings(game="ets2")
    assert capsys.readouterr().out == ""
