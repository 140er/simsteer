"""Real Tk interaction checks. Run with xvfb-run on a headless machine."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.name == "nt" or os.environ.get("DISPLAY")), reason="Tk display required"
)


@pytest.fixture
def tuner():
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
        settings=Settings(),
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
    game = next(k for k, v in tuner._game_choices.items() if v == "ets2")
    tuner._on_game_change(game)
    tuner._root.update()
    assert tuner._restart_button.winfo_ismapped()
    tuner._on_game_change("Auto")
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
