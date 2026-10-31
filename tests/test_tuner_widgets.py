"""Real Tk interaction checks. Run with xvfb-run on a headless machine."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.name == "nt" or os.environ.get("DISPLAY")), reason="Tk display required"
)


@pytest.fixture
def tuner(request):
    import customtkinter as ctk
    from simsteer.app.manual import ManualInputs
    from simsteer.app.settings import Settings
    from simsteer.core.calibration import Calibration
    from simsteer.core.control import ControllerConfig
    from simsteer.core.learners.liveparams import LiveParams
    from simsteer.ui.tuner import Tuner

    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    tuner = Tuner(
        Calibration(),
        ControllerConfig(),
        ManualInputs(),
        live_params=LiveParams(),
        game="ets2",
        settings=Settings(game=getattr(request, "param", "auto")),
        start_thread=False,
    )
    tuner._build_ui(root)
    root.update()
    yield tuner
    root.destroy()
    tuner._root = None


def test_actual_normal_controls_and_developer_controls_stay_in_sync(tuner):
    from simsteer.core.control import ControllerConfig
    from simsteer.ui.state import toggle_developer_mode

    normal = next(n for n in tuner._numbers if n.spec.attr == "max_speed_mps")
    normal.var.set("108")
    tuner._commit_number(normal)
    assert tuner.ctrl_cfg.max_speed_mps == pytest.approx(30)
    assert ControllerConfig.load("ets2").max_speed_mps == pytest.approx(30)
    dev_switch = next(sw for sw, _ in tuner._switches if sw.cget("text") == "Developer")
    dev_switch.toggle()
    tuner._root.update()
    assert tuner.settings.hud_mode == "dev"
    assert tuner._workbench.state() == "normal"
    fields = [n for n in tuner._numbers if n.spec.attr == "max_speed_mps"]
    assert len(fields) == 2
    dev = next(n for n in fields if n.spec.scale == 1)
    assert float(dev.var.get()) == 30
    dev.var.set("25")
    tuner._commit_number(dev)
    assert float(normal.var.get()) == 90
    normal_acc = next(sw for sw, _ in tuner._switches if sw.cget("text") == "ACC")
    normal_acc.toggle()
    assert not tuner.ctrl_cfg.lead_follow_enabled
    acc_switches = [
        (sw, get) for sw, get in tuner._switches if get() == tuner.ctrl_cfg.lead_follow_enabled
    ]
    assert all(bool(sw.get()) == get() for sw, get in acc_switches)
    toggle_developer_mode(tuner.settings)
    tuner._refresh()
    assert tuner._workbench.state() == "withdrawn"
    assert not dev_switch.get()


def test_live_poll_does_not_erase_uncommitted_typing(tuner):
    number = tuner._numbers[0]
    number.entry._entry.focus_force()
    tuner._root.update()
    number.var.set("12")
    tuner._refresh()
    assert number.var.get() == "12"
    assert tuner.ctrl_cfg.max_speed_mps == 25


def test_restart_visibility_and_complete_workbench_inventory(tuner):
    assert not tuner._restart_button.winfo_ismapped()
    game = next(k for k, v in tuner._game_choices.items() if v == "ats")
    tuner._on_game_change(game)
    tuner._root.update()
    assert tuner._restart_button.winfo_ismapped()
    tuner._on_game_change(next(k for k, v in tuner._game_choices.items() if v == "ets2"))
    tuner._root.update()
    assert not tuner._restart_button.winfo_ismapped()
    tuner._set_developer(True)
    tuner._root.update()
    attrs = {n.spec.attr for n in tuner._numbers}
    assert attrs == {
        "fov_v_deg",
        "pitch_deg",
        "yaw_deg",
        "height_m",
        "crop_top_pct",
        "crop_bottom_pct",
        "steer_max",
        "steer_authority",
        "axis_bias",
        "lookahead_s",
        "curvature_anticipation_s",
        "wheelbase_m",
        "a_linear",
        "b_quad",
        "c_bias",
        "wheel_trim_gain",
        "wheel_trim_clip",
        "lane_change_hold_s",
        "long_anticipation_s",
        "max_accel_mps2",
        "max_decel_mps2",
        "speed_p_gain",
        "accel_deadband_mps2",
        "max_speed_mps",
        "max_lat_accel_mps2",
        "lead_min_prob",
        "lead_time_headway_s",
        "lead_min_gap_m",
        "lead_gap_p_gain",
        "lead_ttc_brake_s",
        "throttle",
        "brake",
        "steer",
    }
    assert {g.title for g in tuner._groups} == {
        "Camera",
        "Steering",
        "Lane change",
        "Fit",
        "Trim",
        "Flags",
        "Manual",
        "Status",
        "Speed & ACC",
        "Hotkeys",
    }
    tuner._filter_var.set("headway")
    tuner._root.update()
    visible = [g.title for g in tuner._groups if g.frame.winfo_ismapped()]
    assert visible == ["Speed & ACC"]
    tuner._filter_var.set("")
    tuner._root.update()
    assert all(g.frame.winfo_ismapped() for g in tuner._groups)


def test_zero_manual_replaces_a_focused_edit(tuner):
    tuner._set_developer(True)
    number = next(
        n for n in tuner._numbers if n.spec.obj is tuner.manual and n.spec.attr == "steer"
    )
    number.entry._entry.focus_force()
    tuner._root.update()
    number.var.set("0.9")
    tuner.manual.steer = 0.8
    tuner.manual.steer_override = True
    tuner._zero_manual()
    assert float(number.var.get()) == 0
    assert tuner.manual.steer == 0
    assert not tuner.manual.steer_override
    tuner._commit_number(number)
    assert tuner.manual.steer == 0


def test_escape_restores_value_and_clears_invalid_input_color(tuner):
    from simsteer.ui.tuner import INK, RED

    number = tuner._numbers[0]
    number.entry._entry.focus_force()
    tuner._root.update()
    number.var.set("invalid")
    assert not tuner._commit_number(number)
    assert number.entry.cget("text_color") == RED
    number.entry._entry.event_generate("<Escape>")
    tuner._root.update()
    assert number.var.get() == "90"
    assert number.entry.cget("text_color") == INK
    assert tuner.ctrl_cfg.max_speed_mps == 25


@pytest.mark.parametrize("save", ["focus_out", "save_all"])
@pytest.mark.parametrize("attr, live_value", [
    ("a_linear", 3.5678), ("pitch_deg", 0.2734), ("max_speed_mps", 25.3),
])
def test_unedited_focused_number_does_not_write_stale_or_rounded_value(
    tuner, monkeypatch, attr, live_value, save
):
    tuner._set_developer(True)
    number = next(n for n in tuner._numbers if n.spec.attr == attr)
    number.entry._entry.focus_force()
    tuner._root.update()
    assert number.editing()
    saves = []
    monkeypatch.setattr(number.spec.obj, "save", lambda *a, **kw: saves.append(True))
    setattr(number.spec.obj, attr, live_value)
    tuner._refresh()
    if save == "focus_out":
        other = next(n for n in tuner._numbers if n is not number)
        other.entry._entry.focus_force()
        tuner._root.update()
        assert not saves
    else:
        tuner._save_all()
    assert getattr(number.spec.obj, attr) == live_value
    assert number.var.get() == number.spec.fmt.format(live_value * number.spec.scale)


def test_focused_number_shows_live_values_until_text_is_edited(tuner):
    tuner._set_developer(True)
    number = next(n for n in tuner._numbers if n.spec.attr == "a_linear")
    number.entry._entry.focus_force()
    tuner._root.update()
    tuner.live_params.a_linear = 3.5678
    tuner._refresh()
    assert number.var.get() == "+3.57"
    number.var.set("4.25")
    tuner.live_params.a_linear = 3.6789
    tuner._refresh()
    assert number.var.get() == "4.25"


@pytest.mark.parametrize("save", ["focus_out", "save_all"])
def test_edited_number_commits_after_a_live_update(tuner, save):
    from simsteer.core.learners.liveparams import LiveParams

    tuner._set_developer(True)
    number = next(n for n in tuner._numbers if n.spec.attr == "a_linear")
    number.entry._entry.focus_force()
    tuner._root.update()
    number.var.set("4.25")
    tuner.live_params.a_linear = 3.5678
    tuner._refresh()
    if save == "focus_out":
        tuner._numbers[0].entry._entry.focus_force()
        tuner._root.update()
    else:
        tuner._save_all()
    assert tuner.live_params.a_linear == 4.25
    assert LiveParams().a_linear == 4.25


@pytest.mark.parametrize("tuner", ["auto", "ats"], indirect=True)
def test_restart_compares_to_running_game_with_a_different_saved_game(tuner):
    running = next(k for k, v in tuner._game_choices.items() if v == "ets2")
    other = next(k for k, v in tuner._game_choices.items() if v == "ats")
    assert tuner._game_picker.get() == running
    assert not tuner._restart_button.winfo_ismapped()
    tuner._on_game_change(running)
    tuner._root.update()
    assert not tuner._restart_button.winfo_ismapped()
    tuner._on_game_change(other)
    tuner._root.update()
    assert tuner._restart_button.winfo_ismapped()
    tuner._on_game_change(running)
    tuner._root.update()
    assert not tuner._restart_button.winfo_ismapped()


def test_calibrated_label_is_green_and_resets_when_not_calibrated(tuner, monkeypatch):
    from types import SimpleNamespace
    from simsteer.ui.tuner import GREEN, MUTED

    tuner.live_calib = SimpleNamespace(cal_status=1)
    monkeypatch.setattr(tuner.live_params, "trusted", lambda: True)
    tuner._refresh()
    assert tuner._cal_label.cget("text") == "Calibrated"
    assert tuner._cal_label.cget("text_color") == GREEN
    tuner.live_calib.cal_status = 0
    tuner._refresh()
    assert tuner._cal_label.cget("text_color") == MUTED


def test_insufficient_calibration_can_display_without_a_numeric_score(tuner):
    from test_calib_flow import new_flow

    flow = new_flow()
    flow._invalidate_check("missing evidence")
    tuner.calib_flow = flow
    tuner.settings.hud_mode = "dev"
    tuner._refresh()
    assert tuner._status_rows["calibration"].cget("text") == "Check again"


@pytest.mark.parametrize("attr, label", [
    ("force_engage", "Force engage"), ("passive_fit", "Passive fit"),
])
def test_explicit_flag_edits_can_replace_cli_overrides(tuner, monkeypatch, attr, label):
    import customtkinter as ctk
    import simsteer.runtime.loop as loop
    from simsteer.app.main import main
    from simsteer.app.settings import Settings

    Settings().save()

    def run(settings, game, **kwargs):
        tuner.settings = settings
        tuner._set_developer(True)
        switch = next(
            sw for sw, _ in tuner._switches
            if any(isinstance(child, ctk.CTkLabel) and child.cget("text") == label
                   for child in sw.master.winfo_children())
        )
        for expected in (False, True):
            switch.toggle()
            saved = Settings.load()
            assert getattr(saved, attr) is expected
            assert not saved.no_gamepad
            other = "passive_fit" if attr == "force_engage" else "force_engage"
            assert not getattr(saved, other)
        return 0

    monkeypatch.setattr(loop, "run", run)
    assert main(["--game", "ets2", "--force-engage", "--passive-fit", "--no-gamepad"]) == 0
