from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from simsteer.app.main import main
from simsteer.app.manual import ManualInputs
from simsteer.app.settings import Settings
from simsteer.core.calibration import Calibration
from simsteer.core.control import ControllerConfig
from simsteer.ui.state import toggle_developer_mode
from simsteer.ui.tuner import Tuner


@pytest.mark.parametrize("launch", ["cli", "replay"])
@pytest.mark.parametrize("save", ["H", "developer", "game", "flag", "save_all"])
def test_cli_overrides_are_not_persisted_by_ui_saves(
    tmp_path, monkeypatch, launch, save
):
    import simsteer.runtime.loop as loop
    from test_replay import _make_clip

    baseline = Settings(game="ats", max_width=1234, vjoy_device=2)
    expected = asdict(baseline)
    baseline.save()
    args = ["--game", "ets2"]
    if launch == "cli":
        args += ["--force-engage", "--passive-fit", "--no-gamepad", "--no-tuner",
                 "--device", "wheel", "--vjoy-device", "3", "--max-width", "900", "--no-passthrough"]
    else:
        clip = _make_clip(tmp_path / "clip", frames=1)
        (clip / "state").mkdir()
        (clip / "state" / "settings.json").write_text(json.dumps(expected))
        args += ["--source", str(clip), "--headless", "--max-frames", "1",
                 "--log-jsonl", str(tmp_path / "log.jsonl"), "--log-timing"]

    calls = []

    def run(settings, game, **kwargs):
        calls.append(True)
        assert game.id == "ets2"
        assert settings.no_gamepad
        if launch == "cli":
            assert settings.force_engage and settings.passive_fit and settings.no_tuner
            assert not settings.wheel_passthrough
            assert (settings.device, settings.vjoy_device, settings.max_width) == ("wheel", 3, 900)
        else:
            assert kwargs["headless"] and kwargs["source"] is not None
        tuner = Tuner(Calibration(), ControllerConfig(), ManualInputs(),
                      game=game.id, settings=settings, start_thread=False)
        if save == "H":
            toggle_developer_mode(settings)
            expected["hud_mode"] = "dev"
        elif save == "developer":
            tuner._set_developer(True)
            expected["hud_mode"] = "dev"
        elif save == "game":
            tuner._on_game_change(next(k for k, v in tuner._game_choices.items() if v == "ets2"))
            expected["game"] = "ets2"
        elif save == "flag":
            tuner._on_probe_toggle(False)
            expected["no_probe"] = True
        else:
            tuner._save_all()
        assert json.loads(Path(settings.path()).read_text(encoding="utf-8")) == expected
        return 0

    monkeypatch.setattr(loop, "run", run)
    assert main(args) == 0
    assert calls == [True]
    if launch == "replay":
        assert asdict(Settings.load()) == asdict(baseline)


@pytest.mark.parametrize("field,value,expected", [
    ("wheel_passthrough", False, False), ("wheel_passthrough", "false", True),
    ("rim_range_deg", 100, 900), ("rim_range_deg", 3000, 900),
    ("rim_range_deg", 1080, 1080), ("rim_range_deg", "1080", 900),
    ("rim_range_deg", True, 900), ("rim_range_deg", 900.5, 900),
    ("rim_range_deg", float("inf"), 900), ("wheel_instance", 1, ""),
    ("wheel_instance", "invalid", ""), ("wheel_axes", [], {}),
    ("wheel_axes", {"throttle": {"slot": 1, "rest": 1, "full": 1}}, {}),
])
def test_wheel_saved_types_ranges_and_shapes(tmp_path, field, value, expected):
    (tmp_path / "settings.json").write_text(json.dumps({field: value}))
    assert getattr(Settings.load(), field) == expected


def test_wheel_identity_and_explicit_pedal_map_round_trip():
    identity = "00000001-0000-0000-0000-000000000000"
    axes = {"steer": 0, "throttle": {"slot": 1, "rest": 32767, "full": 0}}
    settings = Settings(wheel_instance=identity, wheel_axes=axes, rim_range_deg=1080)
    settings.save()
    loaded = Settings.load()
    assert (loaded.wheel_instance, loaded.wheel_axes, loaded.rim_range_deg) == (identity, axes, 1080)


@pytest.mark.parametrize("field,value", [
    ("rim_range_deg", 179), ("rim_range_deg", 2521), ("rim_range_deg", 10**400),
    ("wheel_instance", "bad-guid"), ("wheel_instance", "x" * 64),
    ("wheel_axes", {"steer": 8}), ("wheel_axes", {"steer": True}),
    ("wheel_passthrough", "false"),
    *[pytest.param("wheel_axes", {
        "steer": 0, pedal: {"slot": slot, "rest": 0, "full": 65535, endpoint: value}},
        id=f"{pedal}-{endpoint}-{i}")
      for pedal, slot in (("throttle", 1), ("brake", 2), ("clutch", 3))
      for endpoint in ("rest", "full")
      for i, value in enumerate((2**31, -(2**31) - 1, 10**400))],
])
def test_corrupt_wheel_settings_use_defaults_and_banner(data_dir, field, value):
    from simsteer.state import take_state_warnings

    raw = {"wheel_instance": "00000001-0000-0000-0000-000000000000", "max_width": 1200, field: value}
    (data_dir / "settings.json").write_text(json.dumps(raw))
    loaded = Settings.load()
    assert getattr(loaded, field) == getattr(Settings(), field)
    assert loaded.max_width == 1200
    assert "Settings" in take_state_warnings(data_dir)
    loaded.save()


@pytest.mark.parametrize("rim", [180, 2520])
@pytest.mark.parametrize("rest,full", [(-(2**31), 2**31 - 1), (2**31 - 1, -(2**31))])
def test_wheel_setting_boundaries_round_trip(rim, rest, full):
    axes = {"steer": 0, "throttle": dict(slot=1, rest=rest, full=full)}
    settings = Settings(wheel_instance="AAAAAAAA-0000-0000-0000-000000000000",
                        wheel_axes=axes, rim_range_deg=rim)
    settings.save()
    assert Settings.load() == settings


def test_pedal_map_saves_atomically_in_user_data_dir(tmp_path, monkeypatch):
    from simsteer.paths import data_dir
    from simsteer.state import take_state_warnings

    source = tmp_path / "source"
    source.mkdir()
    legacy = source / "settings.json"
    legacy.write_text('{"max_width": 999}')
    monkeypatch.chdir(source)
    monkeypatch.delenv("SIMSTEER_DATA_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user"))
    settings = Settings(wheel_instance="00000001-0000-0000-0000-000000000000",
                        wheel_axes={"steer": 0, "brake": dict(slot=2, rest=0, full=65535)})
    settings.save()
    target = data_dir() / "settings.json"
    assert target == tmp_path / "user" / "SimSteer" / "settings.json"
    assert Settings.load() == settings
    before = target.read_bytes()

    def reject(*args):
        raise OSError("replacement refused")

    monkeypatch.setattr("simsteer.state.os.replace", reject)
    settings.set_persistent(wheel_instance="00000002-0000-0000-0000-000000000000", wheel_axes={"steer": 1})
    settings.save()
    assert target.read_bytes() == before
    assert legacy.read_text() == '{"max_width": 999}'
    assert "not saved" in take_state_warnings(data_dir())
    assert not list(target.parent.glob(".*.tmp"))


def test_queued_pedal_map_and_instance_load_as_one_snapshot():
    from simsteer.state import background_saves

    settings = Settings()
    with background_saves():
        settings.save()
        settings.set_persistent(wheel_instance="00000002-0000-0000-0000-000000000000",
                                wheel_axes={"steer": 1})
        settings.save()
        assert Settings.load() == settings
