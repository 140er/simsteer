"""Per-game state files for calibration, controller and LiveParams."""
from __future__ import annotations

import pytest

from simsteer.core.calibration import Calibration
from simsteer.core.control import ControllerConfig
from simsteer.core.learners.liveparams import LiveParams
from simsteer.paths import state_path


def test_state_path_names(data_dir):
    assert state_path("liveparams") == data_dir / "liveparams.json"
    assert state_path("liveparams", "default") == data_dir / "liveparams.json"
    assert state_path("liveparams", "ets2") == data_dir / "liveparams_ets2.json"
    assert state_path("controller", "forza") == data_dir / "controller_forza.json"
    assert state_path("calibration", "ac") == data_dir / "calibration_ac.json"


def test_controller_config_is_per_game(data_dir):
    ets2 = ControllerConfig.load(game="ets2")
    ets2.steer_max = 0.8
    ets2.save(game="ets2")
    forza = ControllerConfig.load(game="forza")
    forza.steer_max = 0.5
    forza.save(game="forza")

    assert (data_dir / "controller_ets2.json").exists()
    assert (data_dir / "controller_forza.json").exists()
    assert ControllerConfig.load(game="ets2").steer_max == 0.8
    assert ControllerConfig.load(game="forza").steer_max == 0.5


def test_fresh_game_falls_back_to_legacy_file():
    legacy = ControllerConfig()
    legacy.steer_max = 0.65
    legacy.save()
    assert ControllerConfig.load(game="ac").steer_max == 0.65

    Calibration(fov_h_deg=55.0).save()
    assert Calibration.load(game="ac").fov_h_deg == 55.0


@pytest.mark.parametrize("game,a", [("ets2", 0.123), ("forza", -0.456)])
def test_liveparams_reload_equals_saved_fit(game, a, data_dir):
    lp = LiveParams(game=game)
    lp.a_linear = a
    lp.b_quad = 0.0012 if a > 0 else -0.0012
    lp.c_bias = 0.01
    lp.samples = 321
    lp.save()

    assert (data_dir / f"liveparams_{game}.json").exists()
    back = LiveParams(game=game)
    assert back.x == pytest.approx(lp.x)
    assert back.samples == 321
    # x[0] is the inverse-rack slope `a`; `scale` is the derived 1/a.
    assert back.scale == pytest.approx(1.0 / a)


def test_liveparams_is_per_device(data_dir):
    pad = LiveParams(game="ets2", device_kind="gamepad")
    pad.a_linear = 3.7
    pad.save()
    wheel = LiveParams(game="ets2", device_kind="wheel")
    wheel.a_linear = -4.2
    wheel.save()

    assert (data_dir / "liveparams_ets2_gamepad.json").exists()
    assert LiveParams(game="ets2", device_kind="gamepad").a_linear == pytest.approx(3.7)
    assert LiveParams(game="ets2", device_kind="wheel").a_linear == pytest.approx(-4.2)
