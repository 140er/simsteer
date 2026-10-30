from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.app.calibration_routine import CalibrationRoutine
from simsteer.app.manual import ManualInputs
from simsteer.app.settings import Settings
from simsteer.app.wizard import Wizard
from simsteer.core.calibration import Calibration
from simsteer.core.control import ControllerConfig, LongitudinalController
from simsteer.core.learners.livecalib import CalStatus, LiveCalib
from simsteer.core.learners.liveparams import LiveParams
from simsteer.ui.hud import HudRenderer, HudState, bar_height
from simsteer.ui.state import (
    CalibrationView,
    calibration_view,
    developer_mode,
    restart_required,
    toggle_developer_mode,
)
from simsteer.ui.tuner import Tuner, _NumberSpec


def test_calibration_view_bounds_and_unknown_eta():
    view = CalibrationView("Camera", "Drive straight", 1.8, -3)
    assert view.progress == 1
    assert view.time_left_s == 0
    assert CalibrationView("FOV", "Drive straight").time_left_s is None


def test_legacy_wizard_has_only_its_actual_steps():
    lc, lp = LiveCalib(), LiveParams()
    wizard = Wizard("ets2", lc, lp)
    view = calibration_view(wizard=wizard, v_ego=20)
    assert view.step_name == "Camera"
    assert view.steps == ("Camera", "Steering")
    assert view.step_index == 0
    assert view.instruction
    assert "camera calibrating" not in view.instruction
    lc.cal_status = CalStatus.CALIBRATED
    view = calibration_view(wizard=wizard, v_ego=20)
    assert view.step_name == "Steering"
    assert view.key_prompt == "Insert  Engage"
    assert "warming up" not in view.instruction
    assert calibration_view(wizard=wizard, v_ego=20, engaged=True).key_prompt == ""
    wizard._active = False
    assert calibration_view(wizard=wizard) is None


def test_routine_takes_precedence_and_retains_countdown(monkeypatch):
    lc, lp = LiveCalib(), LiveParams()
    routine = CalibrationRoutine(lp)
    routine.active, routine.current_step, routine.step_started_ts = True, 1, 100
    monkeypatch.setattr("simsteer.ui.state.time.monotonic", lambda: 105)
    view = calibration_view(wizard=Wizard("ets2", lc, lp), routine=routine)
    assert view.step_name == "Slalom"
    assert view.step_index == 1
    assert view.progress == 0.25
    assert view.time_left_s == 15
    assert "SLALOM" in view.instruction


def test_guided_flow_adapter_fov_keys_and_score():
    # This is #11's public Panel/flow surface; no flow import is needed.
    steps = [SimpleNamespace(value=v) for v in ("fov", "camera", "steering", "check")]
    panel = SimpleNamespace(title="FOV", progress=1, tone="calibrating")
    flow = SimpleNamespace(
        panel=lambda v, e: panel,
        steps=steps,
        step=steps[0],
        fov_proposal=72.2,
        calib=SimpleNamespace(fov_h_deg=90),
        hint=lambda v, e: "FOV",
        eta_s=lambda: None,
        result={"score": 0.93},
    )
    view = calibration_view(flow=flow, v_ego=20)
    assert view.steps == ("FOV", "Camera", "Steering", "Check")
    assert view.instruction == "FOV 72 measured, set 90"
    assert view.key_prompt == "Insert  Use 72    Backspace  Keep 90"
    assert view.score == 0.93
    panel.tone = "failed"
    assert calibration_view(flow=flow).tone == "failed"
    panel.tone, panel.title = "passed", "CALIBRATED score 0.93"
    passed = calibration_view(flow=flow)
    assert passed.score == 0.93
    assert "score" not in passed.instruction.lower()
    flow.panel = lambda v, e: None
    assert calibration_view(flow=flow) is None


def test_tuner_and_hotkey_share_persisted_mode():
    settings = Settings()
    tuner = Tuner(
        Calibration(), ControllerConfig(), ManualInputs(), settings=settings, start_thread=False
    )
    tuner._set_developer(True)
    assert developer_mode(settings)
    assert Settings.load().hud_mode == "dev"
    toggle_developer_mode(settings)
    assert not developer_mode(settings)
    assert Settings.load().hud_mode == "user"
    tuner._set_developer(True)
    toggle_developer_mode(settings)
    assert settings.hud_mode == "user"


def test_mirror_key_is_developer_only():
    from simsteer.app.hotkeys import hk_enabled

    settings = Settings()
    assert not hk_enabled("mirror_sign", settings)
    toggle_developer_mode(settings)
    assert hk_enabled("mirror_sign", settings)
    settings.disabled_hotkeys.append("mirror_sign")
    assert not hk_enabled("mirror_sign", settings)


def test_both_speed_units_and_acc_write_existing_controller_state():
    config = ControllerConfig()
    long = LongitudinalController(config)
    tuner = Tuner(Calibration(), config, ManualInputs(), game="ets2", start_thread=False)
    normal = _NumberSpec(config, "max_speed_mps", "Max", 5, 50, 1 / 3.6, "{:.0f}", 3.6)
    dev = _NumberSpec(config, "max_speed_mps", "Max speed", 5, 50, 0.1)
    tuner._set_number(normal, "108")
    assert long.cfg.max_speed_mps == pytest.approx(30)
    assert ControllerConfig.load("ets2").max_speed_mps == pytest.approx(30)
    tuner._set_number(dev, "25")
    assert config.max_speed_mps * normal.scale == pytest.approx(90)
    for enabled in (False, True):
        tuner._set_acc(enabled)
        assert long.cfg.lead_follow_enabled is enabled
        assert ControllerConfig.load("ets2").lead_follow_enabled is enabled


@pytest.mark.parametrize("value", ["NaN", "inf", "hello", "0", "181"])
def test_invalid_normal_speed_does_not_change_state(value):
    config = ControllerConfig()
    tuner = Tuner(Calibration(), config, ManualInputs(), start_thread=False)
    spec = _NumberSpec(config, "max_speed_mps", "Max", 5, 50, 1 / 3.6, scale=3.6)
    with pytest.raises(ValueError):
        tuner._set_number(spec, value)
    assert config.max_speed_mps == 25


def test_restart_only_for_a_changed_game_and_reverting_hides_it():
    settings = Settings(game="auto")
    tuner = Tuner(
        Calibration(),
        ControllerConfig(),
        ManualInputs(),
        game="ets2",
        settings=settings,
        start_thread=False,
    )
    assert not restart_required(tuner._startup_game, settings)
    tuner._on_game_change(next(k for k, v in tuner._game_choices.items() if v == "ets2"))
    assert restart_required(tuner._startup_game, settings)
    assert Settings.load().game == "ets2"
    tuner._on_game_change("Auto")
    assert not restart_required(tuner._startup_game, settings)
    # Normal speed and ACC changes never request a restart.
    tuner._set_acc(False)
    assert not restart_required(tuner._startup_game, settings)


def test_hud_bar_grows_for_calibration_and_developer_only():
    s = HudState(v_ego_mps=22, max_speed_mps=25)
    assert bar_height(s) == 42
    assert bar_height(s, "dev") == 68
    s.calibration = CalibrationView("Steering", "Drive a straight stretch", 0.5, 5)
    assert bar_height(s) == 84
    assert bar_height(s, "dev") == 110
    for shape in [(720, 1280, 3), (180, 320, 3)]:
        img = np.zeros(shape, np.uint8)
        HudRenderer().draw(img, s)
        assert img[:84].any()
        assert not img[84:].any()


def test_ttf_cache_reuses_fonts_and_glyphs_for_changing_numbers():
    from simsteer.ui import text

    frame = np.zeros((80, 800, 3), np.uint8)
    text.draw_text(frame, "0123456789 Max ACC On Off →", 10, 10)
    font_misses = text._font.cache_info().misses
    glyph_misses = text._glyph.cache_info().misses
    text.draw_text(frame, "Max 89 ACC On →", 10, 35)
    assert text._font.cache_info().misses == font_misses
    assert text._glyph.cache_info().misses == glyph_misses


@pytest.mark.parametrize("frozen", [False, True])
def test_restart_relaunches_selected_game_without_losing_module(monkeypatch, frozen):
    import os
    import subprocess
    import sys

    settings = Settings(game="auto")
    tuner = Tuner(
        Calibration(), ControllerConfig(), ManualInputs(), settings=settings, start_thread=False
    )
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda args: calls.append(args))
    monkeypatch.setattr(os, "_exit", lambda code: calls.append(code))
    monkeypatch.setattr(sys, "argv", ["simsteer", "--game", "ets2", "--no-gamepad"])
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    tuner._save_and_restart()
    assert not calls
    settings.game = "ac"
    tuner._save_and_restart()
    assert calls[-1] == 0
    assert calls[0][-2:] == ["--game", "ac"]
    assert "--no-gamepad" in calls[0]
    if not frozen:
        assert calls[0][1:3] == ["-m", "simsteer"]


def test_legacy_completion_banner_is_brief(monkeypatch):
    lc, lp = LiveCalib(), LiveParams()
    wizard = Wizard("ets2", lc, lp)
    lc.cal_status = CalStatus.CALIBRATED
    monkeypatch.setattr(lp, "trusted", lambda: True)
    wizard._ready_banner_until = 110
    monkeypatch.setattr("simsteer.ui.state.time.time", lambda: 105)
    view = calibration_view(wizard=wizard)
    assert view.tone == "passed"
    assert view.step_name == "Calibrated"
    monkeypatch.setattr("simsteer.ui.state.time.time", lambda: 111)
    assert calibration_view(wizard=wizard) is None
