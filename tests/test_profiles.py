"""Game profiles: TOML parsing, the registry, and how a game is picked."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

import simsteer
from simsteer.app.main import _apply_settings, _pick_game, _remember_game, main
from simsteer.app.settings import Settings
from simsteer.core.calibration import Calibration
from simsteer.core.control import ControllerConfig
from simsteer.core.learners.liveparams import GATE_PROFILES, LiveParams
from simsteer.games.base import BUILTIN_DIR, load_registry, reload
from simsteer.games.profile import (
    CONTROLLER_KEYS, DEVICE_KINDS, ProfileError, load_profile, parse_profile,
)
from simsteer.runtime import telemetry

MINIMAL = '''
id = "mygame"
display_name = "My Game"
process_names = ["MyGame.exe"]
telemetry = { kind = "none" }
'''

# `_INITIAL_SEEDS` in liveparams.py before profiles replaced it.
OLD_ETS2_SEEDS = {"gamepad": (3.5, 0.0015, 0.0), "wheel": (4.0, 0.0, 0.0)}


def _write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_every_builtin_profile_parses():
    reg = load_registry(user=Path("/nonexistent"))
    assert reg.errors == ()
    assert {"ets2", "ats"} <= set(reg.ids)
    for p in reg.profiles:
        assert p.builtin and p.process_names, p.id
        assert p.liveparams.gates in GATE_PROFILES


def test_ets2_profile_keeps_the_old_hard_coded_values():
    ets2 = load_registry(user=Path("/nonexistent")).get("ets2")
    assert dict(ets2.liveparams.seeds) == OLD_ETS2_SEEDS
    assert ets2.liveparams.gates == "strict"
    assert ets2.process_names == ("eurotrucks2.exe",)
    assert not ets2.capabilities.passive_fit_safe
    assert ets2.tested
    # The loop's effective values before profiles, so ETS2 replays match.
    assert ControllerConfig(**ets2.controller) == ControllerConfig()
    assert ets2.camera.fov_h_deg == Calibration().fov_h_deg


def test_ats_is_data_only():
    assert importlib.util.find_spec("simsteer.games.ats") is None
    pkg = Path(simsteer.__file__).parent
    assert not [p for p in pkg.rglob("*.py") if "ats" in p.stem.split("_")]
    assert (BUILTIN_DIR / "ats.toml").is_file()


def test_minimal_profile_gets_defaults():
    p = parse_profile(MINIMAL, "mygame.toml")
    assert p.id == "mygame"
    assert p.process_names == ("mygame.exe",)
    assert not p.tested
    assert p.capabilities.longitudinal and p.capabilities.speed_source == "telemetry"
    assert p.liveparams.gates == "strict" and dict(p.liveparams.seeds) == {}
    assert p.camera.fov_h_deg is None and p.camera.fov_source is None
    assert dict(p.controller) == {} and p.setup == ()


def test_full_profile_round_trip():
    p = parse_profile(MINIMAL + '''
tested = true
steam_app_id = 123

[controller]
wheelbase_m = 2
steer_max = 0.6

[camera]
fov_h_deg = 70.5
hint = "Bonnet view"
[camera.fov_source]
file = "~/Documents/My Game/settings.ini"
key = "FieldOfView"
axis = "vertical"

[capabilities]
longitudinal = false
speed_source = "model"
synth_wheel_angle = true
passive_fit_safe = true

[liveparams.gates]
profile = "loose"
[liveparams.wheel]
seed = [3, 0.001, 0]

[[setup]]
check = "camera_fov"
where = "Options > Video"
[[setup]]
check = "data_out"
port = 7777
''', "full.toml")
    assert p.controller == {"wheelbase_m": 2.0, "steer_max": 0.6}
    assert isinstance(p.controller["wheelbase_m"], float)
    assert p.camera.fov_source.axis == "vertical"
    assert p.capabilities.speed_source == "model" and p.capabilities.passive_fit_safe
    assert p.liveparams.gates == "loose"
    assert p.liveparams.seeds == {"wheel": (3.0, 0.001, 0.0)}
    assert [(s.check, dict(s.params)) for s in p.setup] == [
        ("camera_fov", {"where": "Options > Video"}), ("data_out", {"port": 7777})]


@pytest.mark.parametrize("extra,key,message", [
    ('telemetri = 1', "telemetri", "unknown key"),
    ('[controller]\nwheelbase = 4.0', "controller.wheelbase", "unknown key"),
    ('[controller]\nwheelbase_m = "4"', "controller.wheelbase_m", "must be a number"),
    ('tested = "yes"', "tested", "must be true or false"),
    ('[capabilities]\nspeed_source = "gps"', "capabilities.speed_source", "not one of"),
    ('[liveparams.gamepad]\nsed = [1, 0, 0]', "liveparams.gamepad.sed", "unknown key"),
    ('[liveparams.gamepad]\nseed = [1, 0]', "liveparams.gamepad.seed", "three numbers"),
    ('[liveparams.keyboard]\nseed = [1, 0, 0]', "liveparams.keyboard", "unknown key"),
    ('[liveparams.gates]\nprofile = "medium"', "liveparams.gates.profile", "not one of"),
    ('[camera]\nfov_h_deg = 400', "camera.fov_h_deg", "between"),
    ('[camera.fov_source]\nfile = "x.cfg"', "camera.fov_source.key", "missing"),
    ('[[setup]]\nwhere = "x"', "setup[0].check", "missing"),
    ('[[setup]]\ncheck = "x"\nopts = {a = 1}', "setup[0].opts", "must be a string"),
])
def test_bad_key_is_named(extra, key, message):
    with pytest.raises(ProfileError) as e:
        parse_profile(MINIMAL + extra, "games/bad.toml")
    assert e.value.key == key
    assert message in e.value.message
    assert str(e.value).startswith(f"{Path('games/bad.toml')}: {key}: ")


@pytest.mark.parametrize("text,key", [
    (MINIMAL.replace('"mygame"', '"My Game"'), "id"),
    (MINIMAL.replace('"mygame"', '"auto"'), "id"),
    (MINIMAL.replace('"mygame"', '"default"'), "id"),
    (MINIMAL.replace('display_name = "My Game"', ""), "display_name"),
    (MINIMAL.replace('kind = "none"', 'kind = "carrier_pigeon"'), "telemetry.kind"),
    (MINIMAL.replace('telemetry = { kind = "none" }', ""), "telemetry"),
    (MINIMAL.replace('kind = "none"', 'kind = "none", port = 1'), "telemetry.port"),
    ("id = [", None),
])
def test_bad_profile_rejected(text, key):
    with pytest.raises(ProfileError) as e:
        parse_profile(text, "bad.toml")
    assert e.value.key == key


def test_telemetry_params_are_typed(monkeypatch):
    monkeypatch.setitem(telemetry.KINDS, "udp", telemetry.Kind({"port": int}, dict))
    p = parse_profile(MINIMAL.replace('kind = "none"', 'kind = "udp", port = 9999'), "u.toml")
    assert dict(p.telemetry.params) == {"port": 9999}
    assert p.make_telemetry() == {"port": 9999}
    with pytest.raises(ProfileError, match="telemetry.port: must be an integer"):
        parse_profile(MINIMAL.replace('kind = "none"', 'kind = "udp", port = "1"'), "u.toml")


def test_none_kind_has_no_telemetry():
    tel = parse_profile(MINIMAL, "x.toml").make_telemetry()
    assert not tel.available
    assert tel.speed_mps() is None


def test_user_profile_is_listed(data_dir, capsys):
    ats = (BUILTIN_DIR / "ats.toml").read_text(encoding="utf-8")
    _write(data_dir / "games", "mygame.toml",
           ats.replace('id = "ats"', 'id = "mygame"')
              .replace("American Truck Simulator", "My Game"))
    reload()
    assert main(["--list-games"]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("mygame "))
    assert "My Game" in line and "amtrucks.exe" in line and "user:" in line
    assert any(ln.startswith("ets2 ") for ln in out.splitlines())


def test_bad_user_profile_is_skipped_and_named(data_dir, capsys):
    bad = _write(data_dir / "games", "typo.toml", MINIMAL + "testd = true\n")
    reload()
    assert main(["--list-games"]) == 0
    out, err = capsys.readouterr()
    assert f"{bad}: testd: unknown key" in err
    assert {"ets2", "ats"} <= {ln.split()[0] for ln in out.splitlines()[1:]}
    assert "mygame" not in out


def test_profile_with_utf8_bom_loads(tmp_path):
    path = tmp_path / "bom.toml"
    path.write_bytes(b"\xef\xbb\xbf" + MINIMAL.encode())
    assert load_profile(path).id == "mygame"


def test_user_profile_replaces_builtin_in_place(tmp_path):
    user = tmp_path / "games"
    _write(user, "ets2.toml", MINIMAL.replace('"mygame"', '"ets2"'))
    _write(user, "zz.toml", MINIMAL)
    reg = load_registry(user=user)
    builtin = load_registry(user=tmp_path / "none")
    assert reg.ids == builtin.ids + ["mygame"]
    ets2 = reg.get("ets2")
    assert not ets2.builtin and ets2.display_name == "My Game"


def test_duplicate_id_in_one_folder_keeps_the_first(tmp_path):
    user = tmp_path / "games"
    _write(user, "a.toml", MINIMAL)
    _write(user, "b.toml", MINIMAL.replace("My Game", "Other"))
    reg = load_registry(builtin=tmp_path / "none", user=user)
    assert reg.ids == ["mygame"] and reg.get("mygame").display_name == "My Game"
    assert [(e.path.name, e.key) for e in reg.errors] == [("b.toml", "id")]


def test_registry_does_not_create_the_user_dir(data_dir):
    reload()
    assert not (data_dir / "games").exists()


def test_detect_and_fallback(tmp_path):
    user = tmp_path / "games"
    _write(user, "a.toml", MINIMAL)
    _write(user, "b.toml", MINIMAL.replace('"mygame"', '"tested"')
           .replace("MyGame.exe", "Tested.exe") + "tested = true\n")
    reg = load_registry(builtin=tmp_path / "none", user=user)
    assert reg.detect({"explorer.exe", "TESTED.EXE"}).id == "tested"
    assert reg.detect({"explorer.exe"}) is None
    assert reg.fallback().id == "tested"
    _write(tmp_path / "untested", "a.toml", MINIMAL)
    untested = load_registry(builtin=tmp_path / "none", user=tmp_path / "untested")
    assert untested.fallback().id == "mygame"
    assert load_registry(builtin=tmp_path / "none", user=tmp_path / "none").fallback() is None


@pytest.fixture
def games(tmp_path):
    return load_registry(user=tmp_path / "none")


def test_pick_saved_unknown_game_falls_back_to_auto(games, monkeypatch, capsys):
    monkeypatch.setattr("simsteer.games.processes.running_process_names", lambda: set())
    assert _pick_game(games, None, "removed", None).id == "ets2"
    out = capsys.readouterr().out
    assert "saved game 'removed' has no profile, using auto" in out
    assert "no known game running" in out


def test_pick_auto_finds_the_running_game(games, monkeypatch):
    monkeypatch.setattr("simsteer.games.processes.running_process_names",
                        lambda: {"amtrucks.exe"})
    assert _pick_game(games, None, "auto", None).id == "ats"
    assert _pick_game(games, "auto", "ets2", None).id == "ats"
    assert _pick_game(games, "ets2", "auto", None).id == "ets2"


def test_pick_replay_uses_the_clip_game_and_never_scans(games, tmp_path, monkeypatch):
    def scan():
        raise AssertionError("replay must not look at running processes")
    monkeypatch.setattr("simsteer.games.processes.running_process_names", scan)
    clip = tmp_path / "clip"
    _write(clip, "clip.json", json.dumps({"game": "ats"}))
    assert _pick_game(games, None, "ets2", clip).id == "ats"
    assert _pick_game(games, "ets2", "auto", clip).id == "ets2"
    _write(clip, "clip.json", json.dumps({"game": "forza"}))
    assert _pick_game(games, None, "ats", clip).id == "ets2"


def test_replay_with_saved_unknown_game_runs(tmp_path, monkeypatch, capsys):
    from test_replay import FakeModel, _make_clip

    import simsteer.runtime.loop as loop
    monkeypatch.setattr(loop, "DrivingModel", FakeModel)
    clip = _make_clip(tmp_path / "clip")
    _write(clip / "state", "settings.json", json.dumps({"game": "removed"}))
    log = tmp_path / "log.jsonl"
    assert main(["--source", str(clip), "--headless", "--log-jsonl", str(log)]) == 0
    assert "saved game 'removed' has no profile, using auto" in capsys.readouterr().out
    assert log.read_text().count("\n") > 0


def test_settings_migrate_passive_fit_ets2(data_dir):
    (data_dir / "settings.json").write_text(json.dumps({"passive_fit_ets2": True}))
    assert Settings.load().passive_fit
    assert "passive_fit_ets2" not in {f.name for f in fields(Settings)}


def test_profile_seeds_reach_the_learners(games, data_dir):
    ats = games.get("ats")
    assert ControllerConfig.load(ats.id, seed=ats.controller).wheelbase_m == 5.5
    assert Calibration.load(ats.id, fov_h_deg=72.0).fov_h_deg == 72.0
    for device, seed in ats.liveparams.seeds.items():
        lp = LiveParams(game=ats.id, device_kind=device, seeds=ats.liveparams.seeds,
                        gates=ats.liveparams.gates)
        assert tuple(lp.x) == pytest.approx(seed)
        lp.reset()
        assert tuple(lp.x) == pytest.approx(seed)
    # A saved file wins over the seed.
    ControllerConfig(wheelbase_m=6.1).save(game="ats")
    assert ControllerConfig.load("ats", seed=ats.controller).wheelbase_m == 6.1


def test_schema_names_match_the_code():
    from simsteer.runtime.output.base import DeviceKind
    assert set(DEVICE_KINDS) == set(get_args(DeviceKind))
    assert set(CONTROLLER_KEYS) <= {f.name for f in fields(ControllerConfig)}
    assert set(GATE_PROFILES) == {"strict", "loose"}


def test_running_process_names_sees_this_process():
    from simsteer.games.processes import running_process_names
    names = running_process_names()
    assert Path(sys.executable).name.lower() in names
    if Path("/proc/self/comm").exists():
        assert Path("/proc/self/comm").read_text().strip().lower() in names


def test_ac_and_forza_profiles_keep_the_pilot_learner_values(games):
    for game_id in ("ac", "acc", "fh4", "fh5"):
        p = games.get(game_id)
        assert p is not None
        assert p.telemetry.kind == ("ac_shm" if game_id in ("ac", "acc") else "forza_udp")
        assert p.liveparams.gates == "loose"
        assert dict(p.liveparams.seeds) == {
            "gamepad": (3.0, 0.0010, 0.0), "wheel": (3.0, 0.0010, 0.0)}
        assert p.capabilities.passive_fit_safe and p.capabilities.synth_wheel_angle
    assert not games.get("acc").tested
    assert not games.get("fh4").tested


def test_auto_remembers_last_game_then_falls_back_to_ets2(games, monkeypatch):
    monkeypatch.setattr("simsteer.games.processes.running_process_names", lambda: set())
    assert _pick_game(games, None, "auto", None, "fh5").id == "fh5"
    assert _pick_game(games, None, "auto", None, "removed").id == "ets2"
    assert _pick_game(games, None, "auto", None).id == "ets2"
    monkeypatch.setattr("simsteer.games.processes.running_process_names", lambda: {"acs.exe"})
    assert _pick_game(games, None, "auto", None, "fh5").id == "ac"


def test_remembered_game_does_not_persist_cli_overrides(data_dir):
    Settings(game="auto", device="gamepad", forza_port=9000).save()
    _remember_game("fh5")
    saved = Settings.load()
    assert saved.last_game == "fh5"
    assert (saved.game, saved.device, saved.forza_port) == ("auto", "gamepad", 9000)


def test_forza_port_override_leaves_the_profile_unchanged(games):
    fh5 = games.get("fh5")
    assert _apply_settings(fh5, Settings()).telemetry.params["port"] == 7777
    assert _apply_settings(fh5, Settings(forza_port=9000)).telemetry.params["port"] == 9000
    assert fh5.telemetry.params["port"] == 7777
    ets2 = games.get("ets2")
    assert _apply_settings(ets2, Settings(forza_port=9000)) is ets2


@pytest.mark.parametrize("port", [0, -1, 65536, True, "bad", []])
def test_bad_forza_port_override_uses_profile(games, port, capsys):
    p = games.get("fh5")
    assert _apply_settings(p, Settings(forza_port=port)) is p
    assert "not a UDP port" in capsys.readouterr().out


def test_live_telemetry_replay_uses_user_port_and_leaves_user_state_alone(
        tmp_path, monkeypatch, data_dir):
    from test_replay import FakeModel, _make_clip

    import simsteer.runtime.loop as loop
    monkeypatch.setattr(loop, "DrivingModel", FakeModel)
    opened = []

    def open_reader(**params):
        opened.append(params)
        return telemetry.NoTelemetry()

    monkeypatch.setitem(telemetry.KINDS, "forza_udp",
                        telemetry.Kind({"port": int, "host": str}, open_reader))
    clip = _make_clip(tmp_path / "clip")
    Settings(last_game="ac", forza_port=9000).save()
    before = (data_dir / "settings.json").read_bytes()
    assert main(["--source", str(clip), "--game", "fh5", "--live-telemetry",
                 "--headless"]) == 0
    assert opened == [{"port": 9000}]
    assert (data_dir / "settings.json").read_bytes() == before


def test_live_telemetry_requires_a_clip(capsys):
    assert main(["--live-telemetry"]) == 2
    assert "needs --source" in capsys.readouterr().out
