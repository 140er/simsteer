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
                 "--device", "wheel", "--vjoy-device", "3", "--max-width", "900"]
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
