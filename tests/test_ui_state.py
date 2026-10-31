from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.app.manual import ManualInputs
from simsteer.app.settings import Settings
from simsteer.core.calibration import Calibration
from simsteer.core.control import ControllerConfig, LongitudinalController
from simsteer.calib.flow import CalibFlow, Step
from simsteer.core.learners.livecalib import LiveCalib
from simsteer.core.learners.liveparams import LiveParams
from simsteer.ui.hud import HudRenderer, HudState, bar_height
from simsteer.ui.state import (
    CalibrationView,
    calibration_view,
    developer_mode,
    toggle_developer_mode,
)
from simsteer.ui.tuner import Tuner, _NumberSpec


def test_calibration_view_bounds_and_unknown_eta():
    view = CalibrationView("Camera", "Drive straight", 1.8, -3)
    assert view.progress == 1
    assert view.time_left_s == 0
    assert CalibrationView("FOV", "Drive straight").time_left_s is None


def legacy_wizard(phase="CAMERA"):
    return SimpleNamespace(
        active=True, phase=SimpleNamespace(name=phase), _ready_banner_until=0,
        _lc=None, _lp=None, _calib_frac=lambda obj: 0.2, _lp_frac=lambda obj: 0.3,
        hint=lambda v: "Drive manually — camera calibrating", eta_seconds=lambda: 15,
    )


def test_legacy_wizard_has_only_its_actual_steps():
    wizard = legacy_wizard()
    view = calibration_view(wizard=wizard, v_ego=20)
    assert view.step_name == "Camera"
    assert view.steps == ("Camera", "Steering")
    assert view.step_index == 0
    assert view.instruction == "Drive manually"
    wizard.phase.name = "STEERING"
    view = calibration_view(wizard=wizard, v_ego=20)
    assert view.step_name == "Steering"
    assert view.key_prompt == "Insert  Engage"
    assert calibration_view(wizard=wizard, v_ego=20, engaged=True).key_prompt == ""
    wizard.active = False
    assert calibration_view(wizard=wizard) is None


def test_routine_takes_precedence_and_retains_countdown(monkeypatch):
    routine = SimpleNamespace(active=True, current_step=1, step_started_ts=100, steps=[
        SimpleNamespace(name="straight", prompt="STRAIGHT", max_duration_s=30),
        SimpleNamespace(name="slalom", prompt="SLALOM", max_duration_s=20),
    ])
    monkeypatch.setattr("simsteer.ui.state.time.monotonic", lambda: 105)
    view = calibration_view(wizard=legacy_wizard(), routine=routine)
    assert view.step_name == "Slalom"
    assert view.step_index == 1
    assert view.progress == 0.25
    assert view.time_left_s == 15
    assert view.instruction == "SLALOM"


@pytest.fixture
def flow():
    return CalibFlow("ets2", "gamepad", live_calib=LiveCalib(), live_params=LiveParams(),
                     calib=Calibration(), persist=False)


def test_guided_flow_adapter_fov_keys_and_score(flow):
    flow.fov_proposal = 72.2
    view = calibration_view(flow=flow, v_ego=20)
    assert view.steps == ("FOV", "Camera", "Steering", "Direction", "Check", "Done")
    assert view.instruction == "FOV 72 measured, set 90"
    assert view.key_prompt == "Insert  Use 72    Backspace  Keep 90"
    flow.step = Step.DONE
    flow.result = {"score": 0.93, "passed": True}
    flow._banner_until = 10
    passed = calibration_view(flow=flow)
    assert passed.tone == "passed" and passed.step_name == "Calibrated"
    assert passed.step_index == 5 and passed.score == 0.93
    assert passed.instruction == ""
    flow._now = 11
    assert calibration_view(flow=flow) is None


def test_guided_flow_stage_names_and_engage_prompt(flow):
    for step, name, index in [(Step.FOV, "FOV", 0), (Step.CAMERA, "Camera", 1),
                              (Step.STEERING, "Steering", 2), (Step.CHECK, "Check", 4)]:
        flow.step = step
        view = calibration_view(flow=flow, v_ego=20)
        assert (view.step_name, view.step_index) == (name, index)
    assert view.key_prompt == "Insert  Engage"
    assert view.instruction == "Check drive"
    flow.step, flow.result = Step.FAILED, {"score": 0.1, "reason": "Poor fit", "redo": "steering"}
    failed = calibration_view(flow=flow)
    assert failed.step_name == "Not calibrated" and failed.tone == "failed"
    assert failed.instruction == "Poor fit" and failed.key_prompt == "R  Recalibrate"


def test_stopped_direction_check_and_no_response_are_explicit(flow):
    flow.step = Step.STEERING
    flow.direction.model_sign = flow.direction.axis_sign = 1
    moving = calibration_view(flow=flow, v_ego=22)
    assert (moving.step_name, moving.step_index) == ("Direction", 3)
    assert moving.instruction == "Stop to check steering direction"
    assert moving.key_prompt == "" and moving.time_left_s is None
    stopped = calibration_view(flow=flow, v_ego=0)
    assert stopped.instruction == "Check steering direction"
    assert stopped.key_prompt == "Insert  Start"
    output = flow.output_direction
    assert output.start(speed=0, observed=0)
    for _ in range(20):
        output.tick_stopped(speed=0, observed=0, dt=0.05)
    checking = calibration_view(flow=flow, v_ego=0)
    assert checking.instruction == "Checking steering direction: stay stopped"
    assert checking.key_prompt == "Insert  Cancel"
    assert checking.progress == output.progress
    assert checking.time_left_s == pytest.approx(11)
    for _ in range(41):
        output.tick_stopped(speed=0, observed=0, dt=0.05)
    no_response = calibration_view(flow=flow, v_ego=0)
    assert no_response.instruction == "No steering response: check binding/deadzone"
    assert no_response.tone == "failed" and no_response.key_prompt == "Insert  Retry"
    assert not output.ready and not flow.ai_steering_ready


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


@pytest.mark.parametrize("hotkey", [
    "mirror_sign", "save_calib", "reset_calib_defaults", "fov_minus", "fov_plus",
    "pitch_minus", "pitch_plus", "height_minus", "height_plus", "input_inset",
])
def test_calibration_override_and_input_keys_are_developer_only(hotkey):
    from simsteer.app.hotkeys import hk_enabled

    settings = Settings()
    assert not hk_enabled(hotkey, settings)
    toggle_developer_mode(settings)
    assert hk_enabled(hotkey, settings)
    settings.disabled_hotkeys.append(hotkey)
    assert not hk_enabled(hotkey, settings)


@pytest.mark.parametrize("hotkey", ["engage", "recalibrate", "keep_fov", "hud_mode"])
def test_normal_mode_keeps_guided_calibration_and_mode_keys(hotkey):
    from simsteer.app.hotkeys import hk_enabled

    assert hk_enabled(hotkey, Settings())


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


def test_restart_only_for_a_changed_game_and_reverting_hides_it(monkeypatch):
    import os
    import subprocess

    settings = Settings(game="auto")
    tuner = Tuner(
        Calibration(),
        ControllerConfig(),
        ManualInputs(),
        game="ets2",
        settings=settings,
        start_thread=False,
    )
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda args: calls.append(args))
    monkeypatch.setattr(os, "_exit", lambda code: None)
    tuner._save_and_restart()
    assert not calls
    tuner._on_game_change(next(k for k, v in tuner._game_choices.items() if v == "ets2"))
    tuner._save_and_restart()
    assert not calls
    assert Settings.load().game == "ets2"
    tuner._on_game_change(next(k for k, v in tuner._game_choices.items() if v == "ats"))
    tuner._save_and_restart()
    assert len(calls) == 1
    tuner._on_game_change(next(k for k, v in tuner._game_choices.items() if v == "ets2"))
    tuner._save_and_restart()
    assert len(calls) == 1
    # Normal speed and ACC changes never request a restart.
    tuner._set_acc(False)
    tuner._save_and_restart()
    assert len(calls) == 1


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
    tuner._on_game_change(next(k for k, v in tuner._game_choices.items() if v == "ats"))
    tuner._save_and_restart()
    assert calls[-1] == 0
    assert calls[0][-2:] == ["--game", "ats"]
    assert "--no-gamepad" in calls[0]
    if not frozen:
        assert calls[0][1:3] == ["-m", "simsteer"]


def test_legacy_completion_banner_is_brief(monkeypatch):
    wizard = legacy_wizard("DONE")
    wizard._ready_banner_until = 110
    monkeypatch.setattr("simsteer.ui.state.time.time", lambda: 105)
    view = calibration_view(wizard=wizard)
    assert view.tone == "passed" and view.step_name == "Calibrated"
    monkeypatch.setattr("simsteer.ui.state.time.time", lambda: 111)
    assert calibration_view(wizard=wizard) is None
