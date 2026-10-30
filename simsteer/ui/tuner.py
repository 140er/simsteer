"""One-line normal tuner and a separate developer workbench.

Both windows edit the existing shared objects. Tk stays on its own
thread; the loop publishes an immutable HUD snapshot without calling Tk.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import customtkinter as ctk

from simsteer.app.hotkeys import groups_in_order, hotkeys_in_group
from simsteer.app.manual import ManualInputs
from simsteer.app.settings import Settings
from simsteer.core.calibration import Calibration
from simsteer.core.control import ControllerConfig
from simsteer.games.base import registry
from simsteer.ui.hud import HudState
from simsteer.ui.state import (
    calibration_view,
    developer_mode,
    restart_required,
    set_developer_mode,
    status_label,
)

WINDOW_W, WINDOW_H = 1280, 62
WORKBENCH_W, WORKBENCH_H = 1160, 900
POLL_MS = 250
BG = "#0b0c0d"
PANEL = "#151719"
FIELD = "#24272a"
INK = "#eeeeee"
MUTED = "#9b9d9f"
GREEN = "#63df71"
BLUE = "#40bddd"
RED = "#ff5555"


@dataclass
class _NumberSpec:
    obj: Any
    attr: str
    label: str
    lo: float
    hi: float
    step: float
    fmt: str = "{:.2f}"
    scale: float = 1.0  # display units per stored unit
    unit: str = ""


@dataclass
class _Number:
    spec: _NumberSpec
    entry: Any
    var: Any

    def editing(self):
        return self.entry.focus_get() in (self.entry, self.entry._entry)

    def sync(self, force=False):
        if force or not self.editing():
            self.var.set(
                self.spec.fmt.format(getattr(self.spec.obj, self.spec.attr) * self.spec.scale)
            )
            self.entry.configure(text_color=INK)


class _Group:
    """Always open; filtering hides rows and then empty groups."""

    def __init__(self, parent, title):
        self.title = title
        self.frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=12)
        self.frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(
            self.frame,
            text=title,
            anchor="w",
            text_color=INK,
            font=ctk.CTkFont(size=13, weight="bold"),
            height=28,
        ).pack(fill="x", padx=12, pady=(5, 2))
        self.body = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.body.pack(fill="x", padx=12, pady=(0, 10))
        self.rows = []

    def add(self, widget, label):
        self.rows.append((widget, label.casefold()))
        widget.pack(fill="x", pady=1)

    def filter(self, query):
        visible = False
        for row, label in self.rows:
            row.pack_forget()
            if not query or query in self.title.casefold() or query in label:
                row.pack(fill="x", pady=1)
                visible = True
        self.frame.pack_forget()
        if visible:
            self.frame.pack(fill="x", pady=(0, 10))


class Tuner:
    def __init__(
        self,
        calib: Calibration,
        ctrl_cfg: ControllerConfig,
        manual: ManualInputs,
        live_params=None,
        game=None,
        device=None,
        settings: Settings | None = None,
        probe=None,
        live_calib=None,
        wizard=None,
        cal_routine=None,
        calib_flow=None,
        *,
        start_thread: bool = True,
    ):
        self.calib, self.ctrl_cfg, self.manual = calib, ctrl_cfg, manual
        self.live_params, self.game, self.device = live_params, game, device
        self.settings = settings if settings is not None else Settings()
        self.probe, self.live_calib = probe, live_calib
        self.wizard, self.cal_routine, self.calib_flow = wizard, cal_routine, calib_flow
        self._startup_game = self.settings.game
        self._game_choices = {"Auto": "auto"}
        self._game_choices.update({p.display_name: p.id for p in registry().profiles})
        self._hud_state: HudState | None = None
        self._root = self._workbench = self._bind_window = None
        self._numbers: list[_Number] = []
        self._switches: list[tuple[Any, Callable]] = []
        self._groups: list[_Group] = []
        self._status_rows = {}
        self._save_status = self._live = None
        self._bind_frame = self._bind_status_label = None
        self._filter_var = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def update_state(self, state: HudState):
        self._hud_state = state

    def _run(self):
        ctk.set_appearance_mode("dark")
        root = ctk.CTk(fg_color=BG)
        self._build_ui(root)
        try:
            root.mainloop()
        finally:
            self._root = None

    def _build_ui(self, root):
        self._root = root
        ctk.ThemeManager.theme["CTkFont"]["family"] = (
            "Segoe UI" if os.name == "nt" else "DejaVu Sans"
        )
        root.title("SimSteer")
        root.geometry(f"{WINDOW_W}x{WINDOW_H}+30+30")
        root.minsize(1180, WINDOW_H)
        root.resizable(True, False)
        root.attributes("-topmost", True)
        bar = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
        bar.pack(fill="both", expand=True, padx=12, pady=10)
        self._state_label = ctk.CTkLabel(
            bar, text="Disengaged", text_color=INK, font=ctk.CTkFont(size=15, weight="bold")
        )
        self._state_label.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            bar,
            text="Insert",
            fg_color=FIELD,
            corner_radius=4,
            width=48,
            height=24,
            text_color=INK,
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(0, 10))
        current = next(
            (name for name, id_ in self._game_choices.items() if id_ == self.settings.game), "Auto"
        )
        self._game_picker = ctk.CTkOptionMenu(
            bar,
            values=list(self._game_choices),
            width=180,
            height=30,
            fg_color=FIELD,
            button_color=FIELD,
            button_hover_color="#393c3f",
            text_color=INK,
            command=self._on_game_change,
        )
        self._game_picker.set(current)
        self._game_picker.pack(side="left", padx=(0, 8))
        self._restart_button = self._button(bar, "Restart", self._save_and_restart, width=66)
        self._device_picker = ctk.CTkSegmentedButton(
            bar,
            values=["Gamepad", "Wheel"],
            width=138,
            height=30,
            fg_color=FIELD,
            selected_color="#44484c",
            selected_hover_color="#505458",
            unselected_color=FIELD,
            unselected_hover_color="#393c3f",
            command=self._on_device_change_seg,
        )
        self._device_picker.set("Wheel" if self.device and self.device.is_wheel else "Gamepad")
        self._device_picker.pack(side="left", padx=(0, 5))
        if self.device is None:
            self._device_picker.configure(state="disabled")
        self._button(bar, "Bind", self._open_bind, width=42).pack(side="left", padx=(0, 12))
        self._cal_label = ctk.CTkLabel(
            bar, text="Calibration", text_color=MUTED, font=ctk.CTkFont(size=12)
        )
        self._cal_label.pack(side="left", padx=(0, 6))
        self._button(bar, "Recalibrate", self._on_recalibrate_click, width=88).pack(
            side="left", padx=(0, 12)
        )
        self._add_number(
            bar,
            _NumberSpec(
                self.ctrl_cfg, "max_speed_mps", "Max", 5, 50, 1 / 3.6, "{:.0f}", 3.6, "km/h"
            ),
            inline=True,
        )
        self._add_switch(
            bar, "ACC", lambda: self.ctrl_cfg.lead_follow_enabled, self._set_acc, inline=True
        )
        self._add_switch(
            bar,
            "Developer",
            lambda: developer_mode(self.settings),
            self._set_developer,
            inline=True,
        )
        self._refresh()
        root.after(POLL_MS, self._poll)

    @staticmethod
    def _button(parent, text, command, width=80):
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            height=28,
            fg_color=FIELD,
            hover_color="#393c3f",
            text_color=INK,
            corner_radius=14,
            font=ctk.CTkFont(size=12),
        )

    def _make_group(self, parent, title):
        group = _Group(parent, title)
        self._groups.append(group)
        return group

    def _build_workbench(self):
        win = ctk.CTkToplevel(self._root, fg_color=BG)
        self._workbench = win
        win.title("SimSteer workbench")
        win.geometry(f"{WORKBENCH_W}x{WORKBENCH_H}+30+140")
        win.minsize(1000, 700)
        win.protocol("WM_DELETE_WINDOW", lambda: self._set_developer(False))
        header = ctk.CTkFrame(win, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=12)
        self._filter_var = ctk.StringVar()
        ctk.CTkLabel(header, text="Filter", text_color=MUTED, font=ctk.CTkFont(size=12)).pack(
            side="left", padx=(0, 8)
        )
        ctk.CTkEntry(
            header,
            textvariable=self._filter_var,
            width=120,
            height=30,
            fg_color=PANEL,
            border_width=0,
            text_color=INK,
        ).pack(side="left", padx=(0, 12))
        self._filter_var.trace_add("write", lambda *_: self._filter())
        self._live = ctk.CTkLabel(
            header, text="", anchor="w", text_color=INK, font=ctk.CTkFont(size=11)
        )
        self._live.pack(side="left", fill="x", expand=True)
        self._save_status = ctk.CTkLabel(
            header, text="", text_color=MUTED, font=ctk.CTkFont(size=11)
        )
        self._save_status.pack(side="left", padx=8)
        save = self._button(header, "Save all", self._save_all)
        save.configure(fg_color=INK, text_color=BG, hover_color="#d6d6d6")
        save.pack(side="right")
        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent", corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        # 3 equal columns, two 12 px gutters; group rows share label/value edges.
        columns = []
        for i in range(3):
            scroll.grid_columnconfigure(i, weight=1, uniform="workbench")
            col = ctk.CTkFrame(scroll, fg_color="transparent", corner_radius=0)
            col.grid(row=0, column=i, sticky="new", padx=6)
            columns.append(col)
        self._build_camera(columns[0])
        self._build_steering(columns[0])
        lane = self._make_group(columns[0], "Lane change")
        self._add_number(
            lane, _NumberSpec(self.ctrl_cfg, "lane_change_hold_s", "Hold (s)", 0.05, 6, 0.05)
        )
        self._build_fit(columns[1])
        self._build_trim(columns[1])
        self._build_flags(columns[1])
        self._build_manual(columns[1])
        self._build_status(columns[2])
        self._build_speed(columns[2])
        hotkeys = ctk.CTkFrame(scroll, fg_color="transparent")
        hotkeys.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6)
        self._build_hotkeys(hotkeys)

    def _build_camera(self, parent):
        group = self._make_group(parent, "Camera")
        for attr, label, lo, hi, step, fmt in [
            ("fov_v_deg", "Capture VFOV (°)", 20, 120, 1, "{:.0f}"),
            ("pitch_deg", "Pitch (°)", -20, 20, 0.1, "{:+.1f}"),
            ("yaw_deg", "Yaw (°)", -8, 8, 0.1, "{:+.1f}"),
            ("height_m", "Height (m)", 0.5, 5, 0.05, "{:.2f}"),
            ("crop_top_pct", "Crop top", 0, 0.45, 0.01, "{:.2f}"),
            ("crop_bottom_pct", "Crop bottom", 0, 0.45, 0.01, "{:.2f}"),
        ]:
            self._add_number(group, _NumberSpec(self.calib, attr, label, lo, hi, step, fmt))
        self._add_switch(
            group,
            "Mirror lateral sign",
            lambda: self.calib.lateral_sign < 0,
            lambda v: setattr(self.calib, "lateral_sign", -1.0 if v else 1.0),
            obj=self.calib,
        )
        self._add_switch(
            group,
            "Simple warp",
            lambda: self.calib.simple_warp,
            lambda v: setattr(self.calib, "simple_warp", v),
            obj=self.calib,
        )

    def _build_steering(self, parent):
        group = self._make_group(parent, "Steering")
        for attr, label, lo, hi, step, fmt in [
            ("steer_max", "Steer max", 0, 1, 0.01, "{:.2f}"),
            ("steer_authority", "Authority", 0.5, 3, 0.05, "{:.2f}"),
            ("axis_bias", "Axis bias", -0.1, 0.1, 0.005, "{:+.3f}"),
            ("lookahead_s", "Lookahead (s)", 0, 5, 0.01, "{:.2f}"),
            ("curvature_anticipation_s", "Anticipation (s)", -0.3, 0.5, 0.01, "{:+.2f}"),
            ("wheelbase_m", "Wheelbase (m)", 2.5, 6, 0.1, "{:.1f}"),
        ]:
            self._add_number(group, _NumberSpec(self.ctrl_cfg, attr, label, lo, hi, step, fmt))

    def _build_fit(self, parent):
        group = self._make_group(parent, "Fit")
        lp = self.live_params
        if lp is None:
            self._readout(group, "fit", "Steering fit", "—")
            return
        for attr, label, lo, hi, step, fmt in [
            ("a_linear", "Fit a", -30, 30, 0.01, "{:+.2f}"),
            ("b_quad", "Fit b", -0.1, 0.1, 0.001, "{:+.4f}"),
            ("c_bias", "Fit c", -0.1, 0.1, 0.001, "{:+.3f}"),
        ]:
            self._add_number(group, _NumberSpec(lp, attr, label, lo, hi, step, fmt))
        self._add_switch(group, "Lock fit", lambda: lp.frozen, lambda v: setattr(lp, "frozen", v))
        row = ctk.CTkFrame(group.body, fg_color="transparent")
        self._button(row, "Flip sign", self._flip_liveparams_sign).pack(side="left", padx=(0, 6))
        self._button(row, "Reset fit", self._reset_liveparams).pack(side="left")
        group.add(row, "Flip sign Reset fit")
        if self.cal_routine is not None:
            row = ctk.CTkFrame(group.body, fg_color="transparent")
            self._button(row, "Start calibration", self.cal_routine.start, width=130).pack(
                side="left", padx=(0, 6)
            )
            self._button(row, "Cancel", self.cal_routine.cancel).pack(side="left")
            group.add(row, "Start calibration Cancel")

    def _build_trim(self, parent):
        group = self._make_group(parent, "Trim")
        self._add_switch(
            group,
            "Closed-loop trim",
            lambda: self.ctrl_cfg.wheel_trim_enabled,
            lambda v: setattr(self.ctrl_cfg, "wheel_trim_enabled", v),
            obj=self.ctrl_cfg,
        )
        self._add_number(
            group,
            _NumberSpec(self.ctrl_cfg, "wheel_trim_gain", "Trim gain", 0, 0.02, 0.0005, "{:.4f}"),
        )
        self._add_number(
            group,
            _NumberSpec(self.ctrl_cfg, "wheel_trim_clip", "Trim clip", 0, 0.2, 0.005, "{:.3f}"),
        )

    def _build_flags(self, parent):
        group = self._make_group(parent, "Flags")
        self._add_switch(
            group, "Steering probe", lambda: not self.settings.no_probe, self._on_probe_toggle
        )
        self._add_switch(
            group,
            "Passive fit",
            lambda: self.settings.passive_fit,
            lambda v: setattr(self.settings, "passive_fit", v),
            obj=self.settings,
        )
        self._add_switch(
            group,
            "Force engage",
            lambda: self.settings.force_engage,
            lambda v: setattr(self.settings, "force_engage", v),
            obj=self.settings,
        )
        label = ctk.CTkLabel(
            group.body,
            text=self.settings.path(),
            text_color=MUTED,
            font=ctk.CTkFont(size=10),
            anchor="w",
            justify="left",
            wraplength=310,
        )
        group.add(label, "Settings file path")

    def _build_manual(self, parent):
        group = self._make_group(parent, "Manual")
        for attr, label, lo in [
            ("throttle", "Throttle", 0),
            ("brake", "Brake", 0),
            ("steer", "Steer", -1),
        ]:
            self._add_number(
                group,
                _NumberSpec(
                    self.manual, attr, label, lo, 1, 0.01, "{:+.2f}" if lo < 0 else "{:.2f}"
                ),
            )
        self._add_switch(
            group,
            "Manual steer",
            lambda: self.manual.steer_override,
            lambda v: setattr(self.manual, "steer_override", v),
        )
        self._add_switch(
            group,
            "Manual throttle/brake",
            lambda: self.manual.long_override,
            lambda v: setattr(self.manual, "long_override", v),
        )
        group.add(self._button(group.body, "Zero all", self._zero_manual), "Zero all")

    def _build_status(self, parent):
        group = self._make_group(parent, "Status")
        for key, name in [
            ("telemetry", "Telemetry"),
            ("device", "Device"),
            ("camera", "Camera"),
            ("steering", "Steering fit"),
            ("fov", "FOV"),
            ("calibration", "Calibration"),
        ]:
            self._readout(group, key, name, "—")

    def _build_speed(self, parent):
        group = self._make_group(parent, "Speed & ACC")
        self._add_switch(group, "ACC", lambda: self.ctrl_cfg.lead_follow_enabled, self._set_acc)
        for attr, label, lo, hi, step, fmt in [
            ("long_anticipation_s", "Speed anticipation (s)", 0, 6, 0.05, "{:.2f}"),
            ("max_accel_mps2", "Max accel (m/s²)", 0.5, 5, 0.1, "{:.1f}"),
            ("max_decel_mps2", "Max decel (m/s²)", 1, 9, 0.1, "{:.1f}"),
            ("speed_p_gain", "Speed gain", 0, 1.5, 0.05, "{:.2f}"),
            ("accel_deadband_mps2", "Accel deadband (m/s²)", 0, 0.5, 0.01, "{:.2f}"),
            ("max_speed_mps", "Max speed (m/s)", 5, 50, 0.1, "{:.1f}"),
            ("max_lat_accel_mps2", "Corner lateral accel (m/s²)", 0, 8, 0.1, "{:.1f}"),
            ("lead_min_prob", "ACC lead probability", 0, 1, 0.05, "{:.2f}"),
            ("lead_time_headway_s", "ACC headway (s)", 0.5, 4, 0.05, "{:.2f}"),
            ("lead_min_gap_m", "ACC min gap (m)", 1, 15, 0.5, "{:.1f}"),
            ("lead_gap_p_gain", "ACC gap gain", 0, 1, 0.05, "{:.2f}"),
            ("lead_ttc_brake_s", "AEB time to collision (s)", 0.5, 5, 0.1, "{:.1f}"),
        ]:
            self._add_number(group, _NumberSpec(self.ctrl_cfg, attr, label, lo, hi, step, fmt))

    def _build_hotkeys(self, parent):
        group = self._make_group(parent, "Hotkeys")
        for name in groups_in_order():
            for hk in hotkeys_in_group(name):
                self._add_switch(
                    group,
                    f"{hk.key}   {hk.action}",
                    lambda id_=hk.id: id_ not in self.settings.disabled_hotkeys,
                    lambda v, id_=hk.id: self._set_hotkey(id_, v),
                )

    def _row(self, parent, label, inline=False):
        container = parent.body if isinstance(parent, _Group) else parent
        row = ctk.CTkFrame(container, fg_color="transparent", height=24)
        if isinstance(parent, _Group):
            parent.add(row, label)
        else:
            row.pack(side="left" if inline else "top", padx=(0, 10), fill="x")
        return row

    def _add_number(self, parent, spec, inline=False):
        row = self._row(parent, spec.label, inline)
        ctk.CTkLabel(
            row, text=spec.label, anchor="w", text_color=INK, height=23, font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 6))
        var = ctk.StringVar(value=spec.fmt.format(getattr(spec.obj, spec.attr) * spec.scale))
        if spec.unit:
            ctk.CTkLabel(row, text=spec.unit, text_color=MUTED, font=ctk.CTkFont(size=11)).pack(
                side="right", padx=(4, 0)
            )
        entry = ctk.CTkEntry(
            row,
            textvariable=var,
            width=54 if inline else 82,
            height=23,
            border_width=0,
            corner_radius=4,
            fg_color=FIELD,
            text_color=INK,
            justify="right",
            font=ctk.CTkFont(size=12),
        )
        entry.pack(side="right")
        field = _Number(spec, entry, var)
        self._numbers.append(field)
        entry.bind("<Return>", lambda _: self._commit_number(field))
        entry.bind("<FocusOut>", lambda _: self._commit_number(field))
        entry.bind("<Escape>", lambda _: field.sync(force=True))

    def _commit_number(self, field):
        try:
            self._set_number(field.spec, field.var.get())
        except (ValueError, OverflowError):
            field.entry.configure(text_color=RED)
            return False
        field.entry.configure(text_color=INK)
        field.var.set(
            field.spec.fmt.format(getattr(field.spec.obj, field.spec.attr) * field.spec.scale)
        )
        self._sync_controls()
        return True

    def _set_number(self, spec, text):
        value = float(text) / spec.scale
        if not math.isfinite(value) or not spec.lo <= value <= spec.hi:
            raise ValueError("Out of range")
        value = max(spec.lo, min(spec.hi, round(value / spec.step) * spec.step))
        if getattr(spec.obj, spec.attr) != value:
            setattr(spec.obj, spec.attr, value)
            self._persist(spec.obj)

    def _add_switch(self, parent, label, get, set_, *, obj=None, inline=False):
        row = self._row(parent, label, inline)

        def changed():
            set_(bool(sw.get()))
            if obj is not None:
                self._persist(obj)
            self._sync_controls()

        if not inline:
            ctk.CTkLabel(
                row, text=label, text_color=INK, anchor="w", height=23, font=ctk.CTkFont(size=12)
            ).pack(side="left", fill="x", expand=True)
        sw = ctk.CTkSwitch(
            row,
            text=label if inline else "",
            command=changed,
            width=38,
            switch_width=30,
            switch_height=17,
            progress_color=INK,
            button_color="#a6aaae",
            button_hover_color="#b7bbbf",
            fg_color="#45484c",
            text_color=INK,
            font=ctk.CTkFont(size=12),
        )
        sw.pack(side="right")
        self._switches.append((sw, get))
        if get():
            sw.select()

    def _readout(self, group, key, label, value):
        row = self._row(group, label)
        ctk.CTkLabel(row, text=label, text_color=INK, height=24, font=ctk.CTkFont(size=12)).pack(
            side="left"
        )
        val = ctk.CTkLabel(
            row, text=value, anchor="e", text_color=MUTED, font=ctk.CTkFont(size=11), height=24
        )
        val.pack(side="right", fill="x", expand=True)
        self._status_rows[key] = val

    def _set_acc(self, enabled):
        self.ctrl_cfg.lead_follow_enabled = bool(enabled)
        self._persist(self.ctrl_cfg)

    def _set_developer(self, enabled):
        set_developer_mode(self.settings, enabled)
        if self._root is not None:
            self._sync_mode()
            self._sync_controls()

    def _on_game_change(self, name):
        self.settings.game = self._game_choices[name]
        self.settings.save()
        self._refresh_restart()

    def _refresh_restart(self):
        if self._root is None:
            return
        if restart_required(self._startup_game, self.settings):
            self._restart_button.pack(side="left", after=self._game_picker, padx=(0, 8))
        else:
            self._restart_button.pack_forget()

    def _sync_mode(self):
        if developer_mode(self.settings):
            if self._workbench is None:
                self._build_workbench()
            if self._workbench.state() == "withdrawn":
                self._workbench.deiconify()
        elif self._workbench is not None:
            self._workbench.withdraw()

    def _sync_controls(self, force_obj=None):
        for field in self._numbers:
            field.sync(force=field.spec.obj is force_obj)
        for sw, get in self._switches:
            if get():
                sw.select()
            else:
                sw.deselect()

    def _filter(self):
        query = self._filter_var.get().strip().casefold()
        for group in self._groups:
            group.filter(query)

    def _poll(self):
        if self._root is not None:
            self._refresh()
            self._root.after(POLL_MS, self._poll)

    def _refresh(self):
        self._sync_mode()
        self._sync_controls()
        self._refresh_restart()
        s = self._hud_state
        engaged = bool(self.device and self.device.engaged)
        panel = (
            s.calibration
            if s
            else calibration_view(
                wizard=self.wizard, routine=self.cal_routine, flow=self.calib_flow, engaged=engaged
            )
        )
        label = (
            status_label(s)
            if s
            else (
                "Engaged"
                if engaged
                else "Disengaged"
                if self.device and self.device.kind
                else "No gamepad"
            )
        )
        if panel and not engaged and label == "Disengaged":
            label = "Calibrating" if panel.tone == "calibrating" else panel.step_name
        color = GREEN if engaged else BLUE if panel else INK
        self._state_label.configure(text=label, text_color=color)
        calibrated = (
            bool(self.calib_flow.calibrated)
            if self.calib_flow is not None
            else bool(
                self.live_params
                and self.live_params.trusted()
                and self.live_calib
                and self.live_calib.cal_status == 1
            )
        )
        self._cal_label.configure(
            text="Calibrated" if calibrated else panel.step_name if panel else "Not calibrated"
        )
        if self._live is not None:
            fit = ""
            if self.live_params:
                lp = self.live_params
                fit = f"a {lp.a_linear:+.2f}  b {lp.b_quad:+.4f}  c {lp.c_bias:+.3f}  n {lp.samples:,}"
                if lp.frozen:
                    fit += "  LOCKED"
            self._live.configure(
                text=f"{label}    Look {self.ctrl_cfg.lookahead_s:.2f} s    {fit}"
                + (f"    {s.fps:.0f} fps" if s else "")
            )
            values = {
                "telemetry": (s.telemetry_status if s else "—"),
                "device": (
                    f"{self.device.kind} ready"
                    if self.device and self.device.kind
                    else "Unavailable"
                ),
                "camera": f"{self.calib.pitch_deg:+.1f}°  {self.calib.yaw_deg:+.1f}°  {self.calib.height_m:.2f} m",
                "steering": f"a {self.live_params.a_linear:+.2f}  n {self.live_params.samples:,}"
                if self.live_params
                else "—",
                "fov": f"V {self.calib.fov_v_deg:.1f}°  H {self.calib.fov_h_deg:.1f}°",
                "calibration": (
                    f"Score {panel.score:.2f}"
                    if panel and panel.score is not None
                    else self.cal_routine.summary_text
                    if self.cal_routine and self.cal_routine.summary_text
                    else f"{panel.step_name} {panel.progress:.0%}"
                    if panel
                    else "Calibrated"
                    if calibrated
                    else "—"
                ),
            }
            if self.calib_flow and self.calib_flow.result:
                values["calibration"] = f"Score {self.calib_flow.result.get('score', 0):.2f}"
            for key, value in values.items():
                if key in self._status_rows:
                    self._status_rows[key].configure(text=value)

    def _persist(self, obj):
        if obj is self.manual:
            return True
        try:
            if obj is self.calib or obj is self.ctrl_cfg:
                obj.save(game=self.game)
            else:
                obj.save()
            self._saved()
            return True
        except OSError:
            self._saved("Save failed", RED)
            return False

    def _saved(self, text=None, color=MUTED):
        if self._save_status is not None:
            self._save_status.configure(
                text=text or datetime.now().strftime("Saved %H:%M:%S"), text_color=color
            )

    def _save_all(self):
        ok = True
        for field in self._numbers:
            if field.editing():
                ok = self._commit_number(field) and ok
        for obj in (self.calib, self.ctrl_cfg, self.live_params, self.settings):
            if obj is not None:
                ok = self._persist(obj) and ok
        if not ok:
            self._saved("Save failed", RED)

    def _set_hotkey(self, id_, enabled):
        disabled = self.settings.disabled_hotkeys
        if enabled and id_ in disabled:
            disabled.remove(id_)
        elif not enabled and id_ not in disabled:
            disabled.append(id_)
        self.settings.save()

    def _on_probe_toggle(self, enabled):
        self.settings.no_probe = not enabled
        if self.probe is not None:
            self.probe.enabled = enabled
        self._persist(self.settings)

    def _reset_liveparams(self):
        self.live_params.reset()
        self._persist(self.live_params)
        self._sync_controls(force_obj=self.live_params)

    def _flip_liveparams_sign(self):
        self.live_params.a_linear = -self.live_params.a_linear
        self.live_params.b_quad = -self.live_params.b_quad
        self._persist(self.live_params)
        self._sync_controls(force_obj=self.live_params)

    def _zero_manual(self):
        self.manual.throttle = self.manual.brake = self.manual.steer = 0.0
        self.manual.steer_override = self.manual.long_override = False
        self._sync_controls(force_obj=self.manual)

    def _on_recalibrate_click(self):
        from tkinter import messagebox

        if not messagebox.askyesno(
            "Recalibrate?", "Reset this game's camera and steering calibration?", parent=self._root
        ):
            return
        if self.calib_flow is not None:
            self.calib_flow.request_restart("Recalibrate clicked")
            return
        if self.device and self.device.engaged:
            self.device.disengage()
        if self.live_calib:
            self.live_calib.reset(reason="user clicked Recalibrate")
        if self.live_params:
            self.live_params.reset(wipe_disk=True)
        if self.wizard:
            self.wizard.reset()
        if self.probe:
            self.probe.reset()
        defaults = Calibration()
        for attr in ("pitch_deg", "yaw_deg", "lateral_sign", "height_m"):
            setattr(self.calib, attr, getattr(defaults, attr))
        self._persist(self.calib)
        self._sync_controls()

    def _save_and_restart(self):
        if not restart_required(self._startup_game, self.settings):
            return
        import os
        import subprocess
        import sys

        self.settings.save()
        args = [sys.executable]
        if not getattr(sys, "frozen", False):
            args += ["-m", "simsteer"]
        args += [*sys.argv[1:], "--game", self.settings.game]
        try:
            subprocess.Popen(args)
            os._exit(0)
        except OSError:
            self._restart_button.configure(text="Retry")

    def _on_device_change_seg(self, value):
        if self.device is None:
            return
        requested = "wheel" if value == "Wheel" else "gamepad"
        if not self.device.set_kind(requested):
            self._device_picker.set("Wheel" if self.device.is_wheel else "Gamepad")
            return
        self.settings.device = self.device.kind
        self.settings.save()
        if self.live_params:
            self.live_params.switch_device(self.device.kind)
        self.ctrl_cfg.lookahead_s = 0.10 if self.device.is_wheel else 0.25
        self._persist(self.ctrl_cfg)
        self._sync_controls()
        self._rebuild_bind_buttons()

    def _open_bind(self):
        if self._bind_window is not None and self._bind_window.winfo_exists():
            self._bind_window.lift()
            return
        win = ctk.CTkToplevel(self._root, fg_color=BG)
        self._bind_window = win
        win.title("Bind controller")
        win.geometry("560x260")
        self._bind_frame = ctk.CTkFrame(win, fg_color="transparent")
        self._bind_frame.pack(fill="both", expand=True, padx=12, pady=12)
        self._bind_status_label = ctk.CTkLabel(win, text="", text_color=INK)
        self._bind_status_label.pack(fill="x", padx=12, pady=(0, 12))
        self._rebuild_bind_buttons()

    def _bind_inputs_for_active_device(self):
        if not self.device:
            return []
        if self.device.is_wheel:
            from simsteer.runtime.output.wheel import Wheel

            return list(Wheel.WIGGLE_INPUTS)
        if self.device.is_gamepad:
            return [
                ("left_stick_x", "Left stick X"),
                ("right_stick_x", "Right stick X"),
                ("left_trigger", "Brake"),
                ("right_trigger", "Throttle"),
                *[("button_" + key.lower(), key) for key in ("A", "B", "X", "Y", "LB", "RB")],
            ]
        return []

    def _rebuild_bind_buttons(self):
        if self._bind_frame is None or not self._bind_frame.winfo_exists():
            return
        for child in self._bind_frame.winfo_children():
            child.destroy()
        inputs = self._bind_inputs_for_active_device()
        if not inputs:
            ctk.CTkLabel(self._bind_frame, text="No controller", text_color=INK).pack()
        for i, (kind, label) in enumerate(inputs):
            button = self._button(
                self._bind_frame, label, lambda k=kind, lbl=label: self._wiggle(k, lbl)
            )
            button.grid(row=i // 3, column=i % 3, sticky="ew", padx=3, pady=3)
        for i in range(3):
            self._bind_frame.grid_columnconfigure(i, weight=1)

    def _wiggle(self, kind, label):
        if not self.device or not self.device.kind:
            return
        if self.device.engaged:
            self._set_bind_status("Disengage with Insert")
            return

        def run():
            # Keep the selected backend stable if the toolbar changes device.
            backend, is_wheel, raw = self.device.device, self.device.is_wheel, self.device.raw
            for n in (4, 3, 2, 1):
                if self.device.engaged or self.device.device is not backend:
                    self._set_bind_status("Cancelled")
                    return
                self._set_bind_status(f"{label} in {n} s")
                threading.Event().wait(1)
            if self.device.engaged or self.device.device is not backend:
                self._set_bind_status("Cancelled")
                return
            self._set_bind_status(f"Binding {label}")
            try:
                if is_wheel:
                    backend.wiggle(kind)
                else:
                    from tools.bind_helper import wiggle

                    wiggle(raw, kind)
                self._set_bind_status("Done")
            except Exception as exc:
                self._set_bind_status(str(exc))

        threading.Thread(target=run, daemon=True).start()

    def _set_bind_status(self, text):
        if self._root is not None:

            def show():
                if self._bind_status_label is not None and self._bind_status_label.winfo_exists():
                    self._bind_status_label.configure(text=text)

            self._root.after(0, show)
