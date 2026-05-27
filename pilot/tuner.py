"""Customtkinter tuner window for live calibration + controller tuning.

Single-page layout with a persistent top bar (engage badge, device
picker, Save-all), a 2 Hz live-status strip (lookahead estimate +
FROZEN badge, LiveParams summary + LOCKED badge), and a vertical
stack of collapsible sections — Status, Setup, Camera, Lateral,
LiveParams + guided calibration, Closed-loop trim, Lane Change,
Longitudinal, Manual+bind, Hotkeys.

Runs in its own thread so the cv2 capture/inference loop in main.py is
unaffected. Sliders mutate the shared `Calibration`, `ControllerConfig`,
`ManualInputs`, and `LiveParams` instances directly — Python attribute
writes are atomic enough; the loop reads each field once per iteration
and we're not coordinating multi-field updates.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import customtkinter as ctk

from pilot.calibration import Calibration
from pilot.calibration_routine import CalibrationRoutine
from pilot.controller import ControllerConfig
from pilot.device import DeviceManager
from pilot.hotkeys import groups_in_order, hotkeys_in_group
from pilot.livecalib import LiveCalib
from pilot.liveparams import LiveParams
from pilot.manual import ManualInputs
from pilot.probe import SteeringProbe
from pilot.settings import Settings
from pilot.wizard import Wizard


WINDOW_W = 620
WINDOW_H = 820
MIN_W = 540
MIN_H = 620

POLL_MS = 500  # status refresh, 2 Hz — covers strip + dashboard + cal routine

COLOR_BAR_BG = "#1b2026"
COLOR_STRIP_BG = "#171b20"
COLOR_SECTION_HEADER = "#222a31"
COLOR_SECTION_HEADER_HOVER = "#2c3741"
COLOR_HEADER_TEXT = "#e8ecef"
COLOR_VALUE = "#dfe6ee"
COLOR_HINT = "#9aa3aa"
COLOR_ENGAGED = "#27ae60"
COLOR_DISENGAGED = "#4a525a"
COLOR_BADGE_FROZEN = "#e67e22"
COLOR_BADGE_LOCKED = "#c0392b"
COLOR_OK = "#3aaf5b"
COLOR_WARN = "#d6a93b"
COLOR_BAD = "#d65454"
COLOR_DANGER_KEY = "#e74c3c"
COLOR_SAVE_FLASH = "#27ae60"
COLOR_SAVE_ERR = "#e74c3c"
COLOR_DIVIDER = "#262e36"


@dataclass
class _SliderSpec:
    obj: Any
    attr: str
    label: str
    lo: float
    hi: float
    step: float
    fmt: str = "{:.2f}"


class _Section:
    """Collapsible section: header CTkButton + body CTkFrame."""

    def __init__(self, parent, title: str, *, expanded: bool = True) -> None:
        self.expanded = expanded
        self._title = title

        self.frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.frame.pack(fill="x", padx=2, pady=(2, 6))

        self.header = ctk.CTkButton(
            self.frame, text=self._header_text(), anchor="w",
            fg_color=COLOR_SECTION_HEADER,
            hover_color=COLOR_SECTION_HEADER_HOVER,
            text_color=COLOR_HEADER_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=6, height=30,
            command=self.toggle)
        self.header.pack(fill="x")

        self.body = ctk.CTkFrame(self.frame, fg_color="transparent")
        if self.expanded:
            self.body.pack(fill="x", padx=6, pady=(6, 2))

    def _header_text(self) -> str:
        return ("  ▾  " if self.expanded else "  ▸  ") + self._title

    def toggle(self) -> None:
        self.expanded = not self.expanded
        if self.expanded:
            self.body.pack(fill="x", padx=6, pady=(6, 2))
        else:
            self.body.pack_forget()
        self.header.configure(text=self._header_text())


class Tuner:
    def __init__(self, calib: Calibration, ctrl_cfg: ControllerConfig,
                 manual: ManualInputs,
                 live_params: LiveParams | None = None,
                 game: str | None = None,
                 device: DeviceManager | None = None,
                 settings: Settings | None = None,
                 probe: SteeringProbe | None = None,
                 live_calib: LiveCalib | None = None,
                 wizard: Wizard | None = None,
                 cal_routine: CalibrationRoutine | None = None) -> None:
        self.calib = calib
        self.ctrl_cfg = ctrl_cfg
        self.manual = manual
        self.live_params = live_params
        self.game = game
        self.device = device
        self.settings = settings
        self.probe = probe
        self.live_calib = live_calib
        self.wizard = wizard
        self.cal_routine = cal_routine

        # Slider widgets keyed by (id(obj), attr).
        self._sliders: dict[tuple[int, str],
                            tuple[ctk.CTkSlider, ctk.CTkLabel,
                                  _SliderSpec]] = {}
        # Switches keyed by stable id so _zero_manual / refresh can find them.
        self._switches: dict[str, ctk.CTkSwitch] = {}

        # Top-bar / status-strip / status-dashboard / bind handles.
        self._root: ctk.CTk | None = None
        self._engage_badge: ctk.CTkLabel | None = None
        self._device_seg: ctk.CTkSegmentedButton | None = None
        self._save_status_label: ctk.CTkLabel | None = None
        self._save_flash_after_id: str | None = None

        self._status_engage: ctk.CTkLabel | None = None
        self._status_device: ctk.CTkLabel | None = None
        self._status_lookahead: ctk.CTkLabel | None = None
        self._status_liveparams: ctk.CTkLabel | None = None
        self._status_liveparams_badge: ctk.CTkLabel | None = None

        # 5-row dashboard (colored dot + label) in the Status section.
        self._dash_rows: dict[str, tuple[ctk.CTkLabel, ctk.CTkLabel]] = {}

        # Setup section state.
        self._setup_game_var: "ctk.StringVar | None" = None
        self._setup_status_label: ctk.CTkLabel | None = None

        # Guided-calibration-routine status label.
        self._cal_status_label: ctk.CTkLabel | None = None

        self._bind_frame: ctk.CTkFrame | None = None
        self._bind_status_label: ctk.CTkLabel | None = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ----- thread target -----

    def _run(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        root = ctk.CTk()
        self._root = root
        root.title("SimSteer — tuner")
        root.geometry(f"{WINDOW_W}x{WINDOW_H}+40+40")
        root.attributes("-topmost", True)
        root.minsize(MIN_W, MIN_H)

        self._build_top_bar(root)
        self._build_status_strip(root)

        scroll = ctk.CTkScrollableFrame(root, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self._build_status_section(scroll)
        if self.settings is not None:
            self._build_setup_section(scroll)
        self._build_camera_section(scroll)
        self._build_lateral_section(scroll)
        if self.live_params is not None:
            self._build_liveparams_section(scroll)
        self._build_closed_loop_trim_section(scroll)
        self._build_lane_change_section(scroll)
        self._build_longitudinal_section(scroll)
        self._build_manual_section(scroll)
        if self.settings is not None:
            self._build_hotkeys_section(scroll)

        root.after(POLL_MS, self._poll_status)

        try:
            root.mainloop()
        except Exception:
            pass

    # ----- top bar -----

    def _build_top_bar(self, root) -> None:
        bar = ctk.CTkFrame(root, fg_color=COLOR_BAR_BG, corner_radius=0)
        bar.pack(side="top", fill="x")

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=10, pady=8)

        self._engage_badge = ctk.CTkLabel(
            inner, text="DISENGAGED", width=110, height=28,
            fg_color=COLOR_DISENGAGED, text_color="white",
            corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"))
        self._engage_badge.pack(side="left", padx=(0, 10))

        if self.device is not None:
            self._device_seg = ctk.CTkSegmentedButton(
                inner, values=["Gamepad", "Wheel"],
                command=self._on_device_change_seg, height=28)
            initial = "Wheel" if self.device.is_wheel else "Gamepad"
            self._device_seg.set(initial)
            self._device_seg.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(inner, text="").pack(side="left", expand=True, fill="x")

        self._save_status_label = ctk.CTkLabel(
            inner, text="", text_color=COLOR_SAVE_FLASH,
            font=ctk.CTkFont(size=11))
        self._save_status_label.pack(side="right", padx=(0, 8))

        ctk.CTkButton(inner, text="Save all", width=100, height=28,
                      command=self._save_all).pack(side="right")

    # ----- status strip -----

    def _build_status_strip(self, root) -> None:
        strip = ctk.CTkFrame(root, fg_color=COLOR_STRIP_BG, corner_radius=0)
        strip.pack(side="top", fill="x")
        inner = ctk.CTkFrame(strip, fg_color="transparent")
        inner.pack(fill="x", padx=10, pady=6)

        mono = ctk.CTkFont(family="Consolas", size=11)
        badge_font = ctk.CTkFont(size=9, weight="bold")

        self._status_engage = ctk.CTkLabel(
            inner, text="engage: —", font=mono, text_color=COLOR_VALUE)
        self._status_engage.pack(side="left", padx=(0, 12))

        self._status_device = ctk.CTkLabel(
            inner, text="device: —", font=mono, text_color=COLOR_VALUE)
        self._status_device.pack(side="left", padx=(0, 12))

        self._status_lookahead = ctk.CTkLabel(
            inner, text="look: —", font=mono, text_color=COLOR_VALUE)
        self._status_lookahead.pack(side="left", padx=(0, 4))

        self._status_liveparams = ctk.CTkLabel(
            inner, text="lp: —", font=mono, text_color=COLOR_VALUE)
        self._status_liveparams.pack(side="left", padx=(12, 4))
        self._status_liveparams_badge = ctk.CTkLabel(
            inner, text=" LOCKED ", fg_color=COLOR_BADGE_LOCKED,
            text_color="white", corner_radius=4, font=badge_font, height=18)

    # ----- section builders -----

    def _build_status_section(self, parent) -> None:
        sec = _Section(parent, "Status", expanded=True)
        for key, label in [("telemetry", "Telemetry"),
                           ("device", "Output device"),
                           ("camera", "Camera calibration"),
                           ("steering", "Steering fit"),
                           ("fov", "FOV calibration")]:
            row = ctk.CTkFrame(sec.body, fg_color="transparent")
            row.pack(fill="x", pady=1)
            dot = ctk.CTkLabel(row, text="", width=12, height=12,
                               fg_color=COLOR_BAD, corner_radius=6)
            dot.pack(side="left", padx=(0, 8))
            name = ctk.CTkLabel(row, text=f"{label}:", width=140,
                                anchor="w", text_color=COLOR_VALUE,
                                font=ctk.CTkFont(size=11))
            name.pack(side="left")
            value = ctk.CTkLabel(row, text="—", anchor="w",
                                 text_color=COLOR_HINT,
                                 font=ctk.CTkFont(size=11))
            value.pack(side="left", fill="x", expand=True)
            self._dash_rows[key] = (dot, value)

    def _build_setup_section(self, parent) -> None:
        sec = _Section(parent, "Setup", expanded=False)
        if self.settings is None:
            return

        # Game radio.
        ctk.CTkLabel(sec.body, text="Game",
                     anchor="w", text_color=COLOR_HEADER_TEXT,
                     font=ctk.CTkFont(size=12, weight="bold")
                     ).pack(anchor="w", pady=(0, 2))
        self._hint(sec.body,
                   "Which game's telemetry to read. 'auto' tries ETS2 "
                   "(shared memory), then AC, then Forza UDP. Game "
                   "changes require a restart.")
        self._setup_game_var = ctk.StringVar(value=self.settings.game)
        row = ctk.CTkFrame(sec.body, fg_color="transparent")
        row.pack(fill="x", pady=(0, 2))
        for label, val in [("Auto", "auto"), ("ETS2", "ets2"),
                           ("AC", "ac"), ("Forza", "forza")]:
            ctk.CTkRadioButton(row, text=label, value=val,
                               variable=self._setup_game_var,
                               font=ctk.CTkFont(size=11),
                               text_color=COLOR_VALUE
                               ).pack(side="left", padx=(0, 12))
        ctk.CTkLabel(sec.body,
                     text=(f"Currently running: "
                           f"{self.game if self.game else 'unknown'}"),
                     text_color=COLOR_HINT,
                     font=ctk.CTkFont(size=10), anchor="w"
                     ).pack(anchor="w", pady=(2, 8))

        # Advanced toggles.
        ctk.CTkLabel(sec.body, text="Advanced toggles (apply live)",
                     anchor="w", text_color=COLOR_HEADER_TEXT,
                     font=ctk.CTkFont(size=12, weight="bold")
                     ).pack(anchor="w", pady=(0, 4))

        self._add_switch(
            sec.body, "Active steering probing during wizard Phase B",
            get=lambda: not self.settings.no_probe,
            set_=self._on_probe_toggle, sid="probe_enabled")
        self._add_switch(
            sec.body, "Passive LiveParams fit on ETS2 while disengaged",
            get=lambda: self.settings.passive_fit_ets2,
            set_=self._on_passive_toggle, sid="passive_fit_ets2")
        self._add_switch(
            sec.body, "Force-engage (DEV — bypass calibration / FPS gate)",
            get=lambda: self.settings.force_engage,
            set_=self._on_force_toggle, sid="force_engage")

        self._divider(sec.body)

        # Recalibrate.
        ctk.CTkLabel(sec.body, text="Calibration",
                     anchor="w", text_color=COLOR_HEADER_TEXT,
                     font=ctk.CTkFont(size=12, weight="bold")
                     ).pack(anchor="w", pady=(0, 2))
        self._hint(sec.body,
                   "Recalibrate wipes camera pose + steering fit + "
                   "first-drive flag so the wizard starts over. Same "
                   "as the R hotkey.")
        ctk.CTkButton(sec.body,
                      text="Recalibrate (wipe LiveCalib + LiveParams)",
                      command=self._on_recalibrate_click,
                      fg_color="#7f3b3b", hover_color="#964545"
                      ).pack(fill="x")

        self._divider(sec.body)

        # Save & Restart (settings.json + relaunch).
        ctk.CTkButton(sec.body,
                      text="Save & Restart  (required for game changes)",
                      command=self._save_and_restart
                      ).pack(fill="x", pady=(0, 4))
        self._setup_status_label = ctk.CTkLabel(
            sec.body,
            text=f"settings file: {self.settings.path()}",
            text_color=COLOR_HINT, anchor="w", justify="left",
            wraplength=WINDOW_W - 80,
            font=ctk.CTkFont(size=10))
        self._setup_status_label.pack(anchor="w", pady=(4, 0))

    def _build_camera_section(self, parent) -> None:
        sec = _Section(parent, "Camera & Calibration", expanded=True)
        self._hint(sec.body,
                   "Capture VFOV must match your in-game vertical FOV. "
                   "ETS2: Options → Gameplay → Camera → Field of view. "
                   "AC: Options → Video → Camera FOV. Forza: Settings → "
                   "Difficulty → Camera FOV. After you set it, drive a "
                   "straight road ≥6 m/s and check the FOV row in the "
                   "HUD — `ratio vx_model/v_ego` should sit near 1.0; "
                   ">1.05 means VFOV too high, <0.95 means too low. "
                   "LiveCalib auto-writes pitch + yaw; override only "
                   "if the cyan horizon visibly drifts.")
        for spec in [
            _SliderSpec(self.calib, "fov_v_deg",
                        "Capture VFOV (deg)",
                        20, 120, 1, "{:.0f}"),
            _SliderSpec(self.calib, "pitch_deg",
                        "Pitch (deg, +down)",
                        -20, 20, 0.1, "{:+.1f}"),
            _SliderSpec(self.calib, "yaw_deg",
                        "Yaw (deg, +left)",
                        -8, 8, 0.1, "{:+.1f}"),
            _SliderSpec(self.calib, "height_m",
                        "Camera height (m)",
                        0.5, 5.0, 0.05, "{:.2f}"),
            _SliderSpec(self.calib, "crop_top_pct",
                        "Crop top (fraction)",
                        0.0, 0.45, 0.01, "{:.2f}"),
            _SliderSpec(self.calib, "crop_bottom_pct",
                        "Crop bottom (fraction)",
                        0.0, 0.45, 0.01, "{:.2f}"),
        ]:
            self._add_slider(sec.body, spec)
        self._add_switch(sec.body, "Mirror lateral sign",
                         get=lambda: self.calib.lateral_sign < 0,
                         set_=lambda v: setattr(
                             self.calib, "lateral_sign",
                             -1.0 if v else 1.0),
                         sid="mirror_lateral_sign")
        self._add_switch(sec.body,
                         "Simple warp (crop+resize, no perspective)",
                         get=lambda: bool(getattr(self.calib,
                                                  "simple_warp", True)),
                         set_=lambda v: setattr(
                             self.calib, "simple_warp", bool(v)),
                         sid="simple_warp")

    def _build_lateral_section(self, parent) -> None:
        sec = _Section(parent, "Lateral steering", expanded=True)
        self._hint(sec.body,
                   "TIMING: the controller reads the model's plan at "
                   "(Lookahead + Anticipation) seconds into the future "
                   "and steers toward THAT yaw. Lookahead = static "
                   "steerActuatorDelay (per-car constant in openpilot; "
                   "tune from the slider here). Anticipation = extra "
                   "lead time (openpilot uses a fixed +0.2 s). TOO "
                   "EARLY on gentle curves → lower Anticipation (go "
                   "negative). TOO LATE → raise it.\n\n"
                   "AUTHORITY is a flat multiplier on the model's "
                   "planned curvature. openpilot does NOT have this "
                   "knob — they take the plan as-is and close the loop "
                   "with a PID on lateral-accel error. Set to 1.0 for "
                   "comma-faithful behavior.")
        for spec in [
            _SliderSpec(self.ctrl_cfg, "steer_max",
                        "Steer max (axis cap)",
                        0.0, 1.0, 0.01, "{:.2f}"),
            _SliderSpec(self.ctrl_cfg, "steer_authority",
                        "Authority (flat curvature multiplier)",
                        0.5, 3.0, 0.05, "{:.2f}"),
            _SliderSpec(self.ctrl_cfg, "axis_bias",
                        "Axis bias (+R / -L)",
                        -0.1, 0.1, 0.005, "{:+.3f}"),
            _SliderSpec(self.ctrl_cfg, "lookahead_s",
                        "Lookahead (s) — actuator delay",
                        0.05, 5.0, 0.01, "{:.2f}"),
            _SliderSpec(self.ctrl_cfg, "curvature_anticipation_s",
                        "Anticipation (s) — +early / 0=default / −late",
                        -0.3, 0.5, 0.01, "{:+.2f}"),
            _SliderSpec(self.ctrl_cfg, "wheelbase_m",
                        "Wheelbase (m)",
                        2.5, 6.0, 0.1, "{:.1f}"),
        ]:
            self._add_slider(sec.body, spec)

    def _build_liveparams_section(self, parent) -> None:
        sec = _Section(parent, "LiveParams — RLS steering fit",
                       expanded=True)
        self._hint(sec.body,
                   "axis = a·wheel + b·wheel·v² + c. Drag a/b/c to seed "
                   "a starting point; RLS keeps refining unless locked.")
        for spec in [
            _SliderSpec(self.live_params, "a_linear",
                        "a  (axis per rad of wheel @ v=0)",
                        -30.0, 30.0, 0.01, "{:+.2f}"),
            _SliderSpec(self.live_params, "b_quad",
                        "b  (speed-stiffness)",
                        -0.10, 0.10, 0.001, "{:+.4f}"),
            _SliderSpec(self.live_params, "c_bias",
                        "c  (axis bias)",
                        -0.10, 0.10, 0.001, "{:+.3f}"),
        ]:
            self._add_slider(sec.body, spec)
        self._add_switch(
            sec.body,
            "Lock fit (freeze a/b/c — RLS and intervention off)",
            get=lambda: bool(self.live_params and self.live_params.frozen),
            set_=self._set_liveparams_lock, sid="liveparams_locked")
        btn_row = ctk.CTkFrame(sec.body, fg_color="transparent")
        btn_row.pack(fill="x", pady=(6, 0))
        ctk.CTkButton(btn_row, text="Flip sign (a, b → −a, −b)",
                      command=self._flip_liveparams_sign
                      ).pack(side="left", expand=True, fill="x",
                             padx=(0, 4))
        ctk.CTkButton(btn_row, text="Reset RLS (per-game seed)",
                      command=self._reset_liveparams
                      ).pack(side="left", expand=True, fill="x",
                             padx=(4, 0))

        # Guided calibration — sub-block within LiveParams since it
        # drives the RLS refit with widened gates.
        if self.cal_routine is not None:
            self._divider(sec.body)
            ctk.CTkLabel(sec.body, text="Guided calibration",
                         anchor="w", text_color=COLOR_HEADER_TEXT,
                         font=ctk.CTkFont(size=12, weight="bold")
                         ).pack(anchor="w", pady=(0, 2))
            self._hint(sec.body,
                       "4-step routine (straight / slalom / corner / "
                       "highway). Widens RLS gates and bumps Q so the "
                       "fit converges in ~60 s of driving. ETS2 strict "
                       "gates untouched (per-call override). Cancel "
                       "anytime; 90 s total cap.")
            row = ctk.CTkFrame(sec.body, fg_color="transparent")
            row.pack(fill="x", pady=(0, 4))
            ctk.CTkButton(row, text="Start guided calibration",
                          command=self._start_calibration
                          ).pack(side="left", fill="x", expand=True,
                                 padx=(0, 4))
            ctk.CTkButton(row, text="Cancel",
                          command=self._cancel_calibration,
                          fg_color="#7f3b3b", hover_color="#964545"
                          ).pack(side="left", fill="x", expand=True)
            self._cal_status_label = ctk.CTkLabel(
                sec.body, text="(idle)", text_color=COLOR_HINT,
                anchor="w", justify="left",
                wraplength=WINDOW_W - 80,
                font=ctk.CTkFont(size=11))
            self._cal_status_label.pack(anchor="w", pady=(2, 0))

    def _build_closed_loop_trim_section(self, parent) -> None:
        sec = _Section(parent, "Closed-loop trim (wheel-angle)",
                       expanded=True)
        self._hint(sec.body,
                   "Slow integrator on wheel-angle error. Frozen during "
                   "corners, lane changes, saturation, intervention, "
                   "cold RLS, and v<8 m/s. Clipped to ±wheel_trim_clip "
                   "so even a wrong trim can only nudge.")
        self._add_switch(
            sec.body, "Enable closed-loop trim",
            get=lambda: bool(self.ctrl_cfg.wheel_trim_enabled),
            set_=lambda v: setattr(self.ctrl_cfg,
                                   "wheel_trim_enabled", bool(v)),
            sid="wheel_trim_enabled")
        for spec in [
            _SliderSpec(self.ctrl_cfg, "wheel_trim_gain",
                        "Trim gain (axis per rad·s of error)",
                        0.0, 0.02, 0.0005, "{:.4f}"),
            _SliderSpec(self.ctrl_cfg, "wheel_trim_clip",
                        "Trim clip (max ± axis)",
                        0.0, 0.2, 0.005, "{:.3f}"),
        ]:
            self._add_slider(sec.body, spec)

    def _build_lane_change_section(self, parent) -> None:
        sec = _Section(parent, "Lane change", expanded=False)
        self._hint(sec.body,
                   "Press NumPad 4 / 6 in the overlay to trigger a "
                   "sustained desire pulse for `hold duration` seconds. "
                   "The model picks up the desire and commits the "
                   "maneuver on its own — no separate steering boost "
                   "(openpilot does the same).")
        for spec in [
            _SliderSpec(self.ctrl_cfg, "lane_change_hold_s",
                        "Hold duration (s)",
                        0.05, 6.0, 0.05, "{:.2f}"),
        ]:
            self._add_slider(sec.body, spec)

    def _build_longitudinal_section(self, parent) -> None:
        sec = _Section(parent, "Longitudinal / ACC / AEB", expanded=False)
        for spec in [
            _SliderSpec(self.ctrl_cfg, "long_anticipation_s",
                        "Anticipation on top of lookahead (s)",
                        0.0, 6.0, 0.05, "{:.2f}"),
            _SliderSpec(self.ctrl_cfg, "max_accel_mps2",
                        "Max accel @ throttle=1 (m/s²)",
                        0.5, 5.0, 0.1, "{:.1f}"),
            _SliderSpec(self.ctrl_cfg, "max_decel_mps2",
                        "Max decel @ brake=1 (m/s²)",
                        1.0, 9.0, 0.1, "{:.1f}"),
            _SliderSpec(self.ctrl_cfg, "speed_p_gain",
                        "Speed P gain (m/s² per m/s)",
                        0.0, 1.5, 0.05, "{:.2f}"),
            _SliderSpec(self.ctrl_cfg, "accel_deadband_mps2",
                        "Accel deadband (m/s²)",
                        0.0, 0.5, 0.01, "{:.2f}"),
            _SliderSpec(self.ctrl_cfg, "max_speed_mps",
                        "Max speed (m/s)",
                        5, 50, 1, "{:.0f}"),
            _SliderSpec(self.ctrl_cfg, "max_lat_accel_mps2",
                        "Corner: max lateral accel (m/s², 0=off)",
                        0.0, 8.0, 0.1, "{:.1f}"),
            _SliderSpec(self.ctrl_cfg, "lead_min_prob",
                        "ACC: min lead prob to engage",
                        0.0, 1.0, 0.05, "{:.2f}"),
            _SliderSpec(self.ctrl_cfg, "lead_time_headway_s",
                        "ACC: time headway (s)",
                        0.5, 4.0, 0.05, "{:.2f}"),
            _SliderSpec(self.ctrl_cfg, "lead_min_gap_m",
                        "ACC: min bumper gap (m)",
                        1.0, 15.0, 0.5, "{:.1f}"),
            _SliderSpec(self.ctrl_cfg, "lead_gap_p_gain",
                        "ACC: gap-error P gain",
                        0.0, 1.0, 0.05, "{:.2f}"),
            _SliderSpec(self.ctrl_cfg, "lead_ttc_brake_s",
                        "AEB: TTC threshold (s)",
                        0.5, 5.0, 0.1, "{:.1f}"),
        ]:
            self._add_slider(sec.body, spec)
        self._add_switch(
            sec.body, "Lead follow (ACC) enabled",
            get=lambda: bool(self.ctrl_cfg.lead_follow_enabled),
            set_=lambda v: setattr(self.ctrl_cfg,
                                   "lead_follow_enabled", bool(v)),
            sid="lead_follow_enabled")

    def _build_manual_section(self, parent) -> None:
        sec = _Section(parent, "Manual override + bind helper",
                       expanded=False)
        self._hint(sec.body,
                   "Manual overrides only fire while ENGAGED. The bind "
                   "helper wiggles a chosen input so ETS2's controller "
                   "wizard can detect it — disengage first.")
        for spec in [
            _SliderSpec(self.manual, "throttle", "Throttle (RT)",
                        0.0, 1.0, 0.01, "{:.2f}"),
            _SliderSpec(self.manual, "brake", "Brake (LT)",
                        0.0, 1.0, 0.01, "{:.2f}"),
            _SliderSpec(self.manual, "steer", "Steer (LX)",
                        -1.0, 1.0, 0.01, "{:+.2f}"),
        ]:
            self._add_slider(sec.body, spec)
        self._add_switch(
            sec.body, "Use manual steer (overrides AI)",
            get=lambda: self.manual.steer_override,
            set_=lambda v: setattr(self.manual,
                                   "steer_override", bool(v)),
            sid="manual_steer_override")
        self._add_switch(
            sec.body, "Use manual throttle/brake (overrides AI)",
            get=lambda: self.manual.long_override,
            set_=lambda v: setattr(self.manual,
                                   "long_override", bool(v)),
            sid="manual_long_override")
        ctk.CTkButton(sec.body, text="Zero all + drop overrides",
                      command=self._zero_manual
                      ).pack(fill="x", pady=(6, 0))

        if self.device is not None:
            self._divider(sec.body)
            ctk.CTkLabel(sec.body, text="Bind helper",
                         anchor="w", text_color=COLOR_HEADER_TEXT,
                         font=ctk.CTkFont(size=12, weight="bold")
                         ).pack(anchor="w", pady=(0, 2))
            self._hint(sec.body,
                       "Click an input → 4 s countdown → it wiggles. "
                       "Alt-tab to the game's binding slot during the "
                       "countdown.")
            self._bind_frame = ctk.CTkFrame(sec.body,
                                            fg_color="transparent")
            self._bind_frame.pack(fill="x", pady=(0, 4))
            self._bind_status_label = ctk.CTkLabel(
                sec.body, text="", text_color=COLOR_HINT, anchor="w",
                font=ctk.CTkFont(size=11))
            self._bind_status_label.pack(fill="x", pady=(0, 4))
            self._rebuild_bind_buttons()

    def _build_hotkeys_section(self, parent) -> None:
        sec = _Section(parent, "Hotkeys", expanded=False)
        if self.settings is None:
            return
        self._hint(sec.body,
                   "Toggle off any hotkey you don't want firing. "
                   "Changes persist to settings.json immediately.")
        for group in groups_in_order():
            ctk.CTkLabel(sec.body, text=group, anchor="w",
                         text_color=COLOR_HEADER_TEXT,
                         font=ctk.CTkFont(size=12, weight="bold")
                         ).pack(anchor="w", pady=(8, 2), padx=(2, 0))
            for hk in hotkeys_in_group(group):
                self._add_hotkey_row(sec.body, hk)

    def _add_hotkey_row(self, parent, hk) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=(2, 0), pady=2)

        is_enabled = hk.id not in self.settings.disabled_hotkeys

        def _toggle(hk_id=hk.id, sw=None) -> None:
            v = bool(sw.get()) if sw is not None else True
            if v:
                if hk_id in self.settings.disabled_hotkeys:
                    self.settings.disabled_hotkeys.remove(hk_id)
            else:
                if hk_id not in self.settings.disabled_hotkeys:
                    self.settings.disabled_hotkeys.append(hk_id)
            self.settings.save()

        sw = ctk.CTkSwitch(row, text="", width=40,
                           command=lambda: _toggle(sw=sw))
        if is_enabled:
            sw.select()
        else:
            sw.deselect()
        sw.pack(side="left", padx=(0, 4))

        key_color = COLOR_DANGER_KEY if hk.danger else COLOR_HEADER_TEXT
        ctk.CTkLabel(row, text=hk.key, width=110, anchor="w",
                     text_color=key_color,
                     font=ctk.CTkFont(family="Consolas", size=11,
                                      weight="bold")
                     ).pack(side="left", padx=(0, 6))

        text = ctk.CTkFrame(row, fg_color="transparent")
        text.pack(side="left", fill="x", expand=True)
        scope = "global" if hk.scope == "global" else "overlay"
        ctk.CTkLabel(text, text=f"{hk.action}   ({scope})",
                     anchor="w", text_color=COLOR_VALUE,
                     font=ctk.CTkFont(size=11, weight="bold")
                     ).pack(anchor="w")
        if hk.description:
            ctk.CTkLabel(text, text=hk.description, anchor="w",
                         justify="left", text_color=COLOR_HINT,
                         wraplength=WINDOW_W - 220,
                         font=ctk.CTkFont(size=10)
                         ).pack(anchor="w")

        # Hold a strong reference so the switch isn't GC'd.
        setattr(self, f"_hkey_sw_{hk.id}", sw)

    # ----- widget helpers -----

    def _add_slider(self, parent, spec: _SliderSpec) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(2, 0))

        ctk.CTkLabel(row, text=spec.label, anchor="w",
                     text_color=COLOR_VALUE,
                     font=ctk.CTkFont(size=11)).pack(side="left")
        value = ctk.CTkLabel(
            row, text=spec.fmt.format(getattr(spec.obj, spec.attr)),
            anchor="e", width=80, text_color=COLOR_VALUE,
            font=ctk.CTkFont(family="Consolas", size=11))
        value.pack(side="right")

        def on_change(raw, s=spec, lbl=value) -> None:
            snapped = round(float(raw) / s.step) * s.step
            setattr(s.obj, s.attr, snapped)
            lbl.configure(text=s.fmt.format(snapped))

        slider = ctk.CTkSlider(parent, from_=spec.lo, to=spec.hi,
                               command=on_change, height=18)
        slider.set(getattr(spec.obj, spec.attr))
        slider.pack(fill="x", pady=(0, 4))

        self._sliders[(id(spec.obj), spec.attr)] = (slider, value, spec)

    def _add_switch(self, parent, label: str, *,
                    get: Callable[[], bool],
                    set_: Callable[[bool], None],
                    sid: str | None = None) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        sw = ctk.CTkSwitch(
            row, text=label,
            command=lambda: set_(bool(sw.get())),
            font=ctk.CTkFont(size=11), text_color=COLOR_VALUE)
        if get():
            sw.select()
        else:
            sw.deselect()
        sw.pack(side="left", fill="x", expand=True)
        if sid:
            self._switches[sid] = sw
        # Hold a reference so the switch isn't GC'd.
        setattr(self, f"_sw_{id(sw)}", sw)

    def _hint(self, parent, text: str) -> None:
        ctk.CTkLabel(parent, text=text, text_color=COLOR_HINT,
                     anchor="w", justify="left",
                     wraplength=WINDOW_W - 80,
                     font=ctk.CTkFont(size=10)
                     ).pack(anchor="w", pady=(0, 6), padx=(2, 0))

    def _divider(self, parent) -> None:
        ctk.CTkFrame(parent, fg_color=COLOR_DIVIDER, height=1
                     ).pack(fill="x", pady=8)

    def _set_slider_value(self, obj, attr, value) -> None:
        """Mirror clamps / sign-preservation by re-reading after set."""
        setattr(obj, attr, value)
        actual = getattr(obj, attr)
        key = (id(obj), attr)
        if key in self._sliders:
            slider, value_label, spec = self._sliders[key]
            slider.set(float(actual))
            value_label.configure(text=spec.fmt.format(actual))

    # ----- status polling -----

    def _poll_status(self) -> None:
        if self._root is None:
            return
        try:
            self._refresh_strip()
            self._refresh_dashboard()
            self._refresh_cal_routine()
        except Exception:
            pass
        self._root.after(POLL_MS, self._poll_status)

    def _refresh_strip(self) -> None:
        engaged = bool(self.device and self.device.engaged)
        if self._engage_badge is not None:
            self._engage_badge.configure(
                text="ENGAGED" if engaged else "DISENGAGED",
                fg_color=COLOR_ENGAGED if engaged else COLOR_DISENGAGED)
        if self._status_engage is not None:
            self._status_engage.configure(
                text=f"engage: {'on' if engaged else 'off'}")

        if self._status_device is not None:
            kind = self.device.kind if self.device else None
            self._status_device.configure(text=f"device: {kind or '—'}")

        if self._status_lookahead is not None:
            # Static now — just mirror the slider value. No frozen
            # badge because there's nothing to freeze.
            self._status_lookahead.configure(
                text=f"look: {self.ctrl_cfg.lookahead_s:.2f}s")

        if self._status_liveparams is not None:
            if self.live_params is not None:
                a = self.live_params.a_linear
                b = self.live_params.b_quad
                c = self.live_params.c_bias
                n = self.live_params.samples
                self._status_liveparams.configure(
                    text=f"lp: a={a:+.2f} b={b:+.4f} c={c:+.3f} n={n}")
                if (self._status_liveparams_badge is not None
                        and self.live_params.frozen):
                    if not self._status_liveparams_badge.winfo_ismapped():
                        self._status_liveparams_badge.pack(
                            side="left", padx=(2, 0))
                elif (self._status_liveparams_badge is not None
                      and self._status_liveparams_badge.winfo_ismapped()):
                    self._status_liveparams_badge.pack_forget()
            else:
                self._status_liveparams.configure(text="lp: —")

    def _refresh_dashboard(self) -> None:
        if not self._dash_rows:
            return

        # Telemetry.
        if self.game and self.game != "auto":
            self._paint_dash("telemetry", "ok", f"{self.game} connected")
        else:
            self._paint_dash("telemetry", "warn", "auto-detecting")

        # Output device.
        if self.device is None:
            self._paint_dash("device", "bad", "not initialized")
        elif self.device.kind is None:
            self._paint_dash("device", "bad",
                             self.device.last_error or "unavailable")
        else:
            self._paint_dash("device", "ok", f"{self.device.kind} ready")

        # Camera calibration.
        if self.live_calib is None:
            self._paint_dash("camera", "warn", "not wired")
        else:
            try:
                from pilot.livecalib import CalStatus
                st = self.live_calib.cal_status
                if st == CalStatus.CALIBRATED:
                    pitch = self.live_calib.pitch_estimate or 0
                    yaw = self.live_calib.yaw_estimate or 0
                    self._paint_dash(
                        "camera", "ok",
                        f"pitch={pitch:+.2f}° yaw={yaw:+.2f}°")
                elif st == CalStatus.INVALID:
                    self._paint_dash("camera", "bad", "INVALID — drift")
                else:
                    pct = ((self.live_calib.blocks * 100
                            + self.live_calib._block_n) / 500.0) * 100
                    self._paint_dash("camera", "warn",
                                     f"warming up ({pct:.0f}%)")
            except Exception:
                self._paint_dash("camera", "warn", "—")

        # Steering fit.
        if self.live_params is None:
            self._paint_dash("steering", "warn", "not wired")
        else:
            try:
                trusted = self.live_params.trusted()
                n = max(self.live_params.samples,
                        getattr(self.live_params, "session_samples", 0))
                if trusted:
                    self._paint_dash(
                        "steering", "ok",
                        f"a={self.live_params.a_linear:+.2f} n={n}")
                else:
                    self._paint_dash("steering", "warn",
                                     f"warming up ({n}/200 samples)")
            except Exception:
                self._paint_dash("steering", "warn", "—")

        # FOV.
        self._paint_dash("fov", "ok",
                         f"{self.calib.fov_h_deg:.1f}° (static — match in-game)")

    def _refresh_cal_routine(self) -> None:
        """Mirror the guided-calibration routine's state into its label.
        Also pulls fresh a/b/c into the LiveParams sliders on completion
        so the UI reflects the new fit without the user dragging."""
        if self.cal_routine is None or self._cal_status_label is None:
            return
        try:
            if self.cal_routine.active:
                self._cal_status_label.configure(
                    text=self.cal_routine.prompt)
            elif self.cal_routine.summary_text:
                self._cal_status_label.configure(
                    text=self.cal_routine.summary_text)
                # The routine moved a/b/c — refresh sliders cheaply.
                if self.live_params is not None:
                    for attr in ("a_linear", "b_quad", "c_bias"):
                        key = (id(self.live_params), attr)
                        if key in self._sliders:
                            slider, value_label, spec = self._sliders[key]
                            current = getattr(self.live_params, attr)
                            if abs(slider.get() - current) > 1e-5:
                                slider.set(float(current))
                                value_label.configure(
                                    text=spec.fmt.format(current))
            else:
                self._cal_status_label.configure(text="(idle)")
        except Exception:
            pass

    def _paint_dash(self, key: str, level: str, text: str) -> None:
        if key not in self._dash_rows:
            return
        dot, label = self._dash_rows[key]
        color = {"ok": COLOR_OK, "warn": COLOR_WARN,
                 "bad": COLOR_BAD}.get(level, COLOR_BAD)
        try:
            dot.configure(fg_color=color)
            label.configure(text=text)
        except Exception:
            pass

    # ----- save / action handlers -----

    def _save_calib(self) -> None:
        self.calib.save(game=self.game)
        suffix = f"_{self.game}" if self.game else ""
        print(f"saved calibration{suffix}.json")

    def _save_ctrl(self) -> None:
        self.ctrl_cfg.save(game=self.game)
        suffix = f"_{self.game}" if self.game else ""
        print(f"saved controller{suffix}.json")

    def _save_liveparams(self) -> None:
        if self.live_params is None:
            return
        self.live_params.save()
        suffix = f"_{self.game}" if self.game else ""
        print(f"saved liveparams{suffix}.json  "
              f"a={self.live_params.a_linear:.2f} "
              f"b={self.live_params.b_quad:.4f} "
              f"c={self.live_params.c_bias:+.3f}")

    def _save_settings(self) -> None:
        if self.settings is None:
            return
        if self._setup_game_var is not None:
            self.settings.game = self._setup_game_var.get()
        self.settings.save()
        if self._setup_status_label is not None:
            self._setup_status_label.configure(
                text=f"saved → {self.settings.path()}")

    def _save_all(self) -> None:
        """Top-bar action. Saves tuning data (calib + ctrl + liveparams)
        AND settings. Each independent — one failure doesn't block the rest."""
        errors: list[str] = []
        for label, fn in (("calib", self._save_calib),
                          ("ctrl", self._save_ctrl),
                          ("liveparams", self._save_liveparams),
                          ("settings", self._save_settings)):
            try:
                fn()
            except Exception as e:
                errors.append(f"{label}: {e}")
        if errors:
            self._flash_save("save errors: " + "; ".join(errors),
                             color=COLOR_SAVE_ERR)
        else:
            stamp = datetime.now().strftime("%H:%M:%S")
            self._flash_save(f"Saved {stamp}", color=COLOR_SAVE_FLASH)

    def _flash_save(self, text: str, color: str) -> None:
        if self._save_status_label is None or self._root is None:
            return
        self._save_status_label.configure(text=text, text_color=color)
        if self._save_flash_after_id is not None:
            try:
                self._root.after_cancel(self._save_flash_after_id)
            except Exception:
                pass
        self._save_flash_after_id = self._root.after(
            2000, lambda: self._save_status_label.configure(text=""))

    def _reset_liveparams(self) -> None:
        if self.live_params is None:
            return
        self.live_params.reset()
        self.live_params.save()
        for attr in ("a_linear", "b_quad", "c_bias"):
            self._set_slider_value(self.live_params, attr,
                                   getattr(self.live_params, attr))
        print(f"liveparams reset to seed for game={self.game!r}: "
              f"a={self.live_params.a_linear:.3f} "
              f"b={self.live_params.b_quad:.4f} "
              f"c={self.live_params.c_bias:+.4f} (samples=0)")

    def _set_liveparams_lock(self, v: bool) -> None:
        if self.live_params is None:
            return
        self.live_params.frozen = bool(v)
        state = "LOCKED" if v else "unlocked"
        print(f"liveparams {state}  "
              f"a={self.live_params.a_linear:+.2f} "
              f"b={self.live_params.b_quad:+.4f} "
              f"c={self.live_params.c_bias:+.3f}")

    def _flip_liveparams_sign(self) -> None:
        if self.live_params is None:
            return
        self._set_slider_value(self.live_params, "a_linear",
                               -self.live_params.a_linear)
        self._set_slider_value(self.live_params, "b_quad",
                               -self.live_params.b_quad)
        self.live_params.save()
        print(f"liveparams sign flipped  "
              f"a={self.live_params.a_linear:+.2f} "
              f"b={self.live_params.b_quad:+.4f}")

    def _start_calibration(self) -> None:
        if self.cal_routine is None:
            return
        self.cal_routine.start()

    def _cancel_calibration(self) -> None:
        if self.cal_routine is None:
            return
        self.cal_routine.cancel()
        # If the routine moved a/b/c before cancel, the strip will pick
        # that up on the next poll — but refresh the sliders here too
        # so they don't sit at pre-routine values.
        if self.live_params is not None:
            for attr in ("a_linear", "b_quad", "c_bias"):
                key = (id(self.live_params), attr)
                if key in self._sliders:
                    slider, value_label, spec = self._sliders[key]
                    current = getattr(self.live_params, attr)
                    slider.set(float(current))
                    value_label.configure(text=spec.fmt.format(current))

    def _zero_manual(self) -> None:
        for attr in ("throttle", "brake", "steer"):
            self._set_slider_value(self.manual, attr, 0.0)
        self.manual.steer_override = False
        self.manual.long_override = False
        for sid in ("manual_steer_override", "manual_long_override"):
            sw = self._switches.get(sid)
            if sw is not None:
                sw.deselect()

    # ----- setup-section callbacks -----

    def _on_probe_toggle(self, v: bool) -> None:
        if self.settings is None:
            return
        self.settings.no_probe = not bool(v)
        if self.probe is not None:
            self.probe.enabled = bool(v)
        if self._setup_status_label is not None:
            self._setup_status_label.configure(
                text=f"probe → {'enabled' if v else 'disabled'}")

    def _on_passive_toggle(self, v: bool) -> None:
        if self.settings is None:
            return
        self.settings.passive_fit_ets2 = bool(v)
        if self._setup_status_label is not None:
            self._setup_status_label.configure(
                text=f"passive_fit_ets2 → {bool(v)}")

    def _on_force_toggle(self, v: bool) -> None:
        if self.settings is None:
            return
        self.settings.force_engage = bool(v)
        if self._setup_status_label is not None:
            self._setup_status_label.configure(
                text=f"force_engage → {bool(v)}")

    def _on_recalibrate_click(self) -> None:
        from tkinter import messagebox
        ok = messagebox.askyesno(
            "Recalibrate?",
            "This wipes camera calibration AND steering fit AND camera "
            "pose (pitch/yaw/sign) for the current game, then restarts "
            "the first-drive wizard.\n\n"
            "You'll need to drive ~10-15 min on a highway before "
            "engagement is allowed again.\n\n"
            "Continue?")
        if not ok:
            if self._setup_status_label is not None:
                self._setup_status_label.configure(
                    text="recalibrate cancelled")
            return
        if self.device is not None and self.device.engaged:
            self.device.disengage()
        if self.live_calib is not None:
            self.live_calib.reset(reason="user clicked Recalibrate")
        if self.live_params is not None:
            self.live_params.reset(wipe_disk=True)
            for attr in ("a_linear", "b_quad", "c_bias"):
                self._set_slider_value(self.live_params, attr,
                                       getattr(self.live_params, attr))
        if self.wizard is not None:
            self.wizard.reset()
        if self.probe is not None:
            self.probe.reset()
        defaults = Calibration()
        for attr in ("pitch_deg", "yaw_deg", "lateral_sign", "height_m"):
            self._set_slider_value(self.calib, attr,
                                   getattr(defaults, attr))
        try:
            self.calib.save(game=self.game)
        except Exception:
            pass
        # Refresh the mirror-sign switch in case sign changed.
        mirror = self._switches.get("mirror_lateral_sign")
        if mirror is not None:
            if self.calib.lateral_sign < 0:
                mirror.select()
            else:
                mirror.deselect()
        if self._setup_status_label is not None:
            self._setup_status_label.configure(
                text="calibration wiped — wizard restarted, drive ~15 min")

    def _save_and_restart(self) -> None:
        if self.settings is None:
            return
        if self._setup_game_var is not None:
            self.settings.game = self._setup_game_var.get()
        self.settings.save()
        if self._setup_status_label is not None:
            self._setup_status_label.configure(text="saved — relaunching...")
        import os
        import subprocess
        import sys
        try:
            subprocess.Popen(
                [sys.executable] + sys.argv[1:],
                cwd=os.path.dirname(os.path.abspath(sys.argv[0]))
                    if sys.argv[0] else None)
            os._exit(0)
        except Exception as e:
            if self._setup_status_label is not None:
                self._setup_status_label.configure(
                    text=f"restart failed: {e}")

    # ----- output device -----

    def _on_device_change_seg(self, value: str) -> None:
        if self.device is None:
            return
        requested = "wheel" if value == "Wheel" else "gamepad"
        ok = self.device.set_kind(requested)
        if (not ok and self.device.kind is not None
                and self._device_seg is not None):
            self._device_seg.set(
                "Wheel" if self.device.is_wheel else "Gamepad")

        if ok and self.settings is not None and self.device.kind is not None:
            self.settings.device = self.device.kind
            self.settings.save()

        if (ok and self.live_params is not None
                and self.device.kind is not None):
            self.live_params.switch_device(self.device.kind)
            for attr in ("a_linear", "b_quad", "c_bias"):
                self._set_slider_value(self.live_params, attr,
                                       getattr(self.live_params, attr))

        # On device swap, seed lookahead_s to a sensible per-device
        # default — wheel skips ETS2's input smoothing (~250 ms) so a
        # shorter delay reads correctly; gamepad goes through it.
        if ok:
            seed = 0.10 if (self.device and self.device.is_wheel) else 0.25
            self._set_slider_value(self.ctrl_cfg, "lookahead_s", seed)

        self._rebuild_bind_buttons()

        if ok:
            if self.live_params is not None:
                print(f"output device -> {self.device.kind}  "
                      f"liveparams: a={self.live_params.a_linear:+.2f} "
                      f"b={self.live_params.b_quad:+.4f} "
                      f"c={self.live_params.c_bias:+.3f} "
                      f"samples={self.live_params.samples}")
            else:
                print(f"output device -> {self.device.kind}")
        else:
            print(f"output device switch FAILED: {self.device.last_error}")

    # ----- bind-wiggle helper -----

    def _bind_inputs_for_active_device(self) -> list[tuple[str, str]]:
        if self.device is None:
            return []
        if self.device.is_wheel:
            from pilot.wheel import Wheel
            return list(Wheel.WIGGLE_INPUTS)
        if self.device.is_gamepad:
            return [
                ("left_stick_x",  "Left stick X (steer)"),
                ("right_stick_x", "Right stick X"),
                ("left_trigger",  "Left trigger (brake)"),
                ("right_trigger", "Right trigger (throttle)"),
                ("button_a",      "A"),
                ("button_b",      "B"),
                ("button_x",      "X"),
                ("button_y",      "Y"),
                ("button_lb",     "LB"),
                ("button_rb",     "RB"),
            ]
        return []

    def _rebuild_bind_buttons(self) -> None:
        frame = self._bind_frame
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        inputs = self._bind_inputs_for_active_device()
        if not inputs:
            ctk.CTkLabel(frame, text="(no device active)",
                         text_color=COLOR_HINT,
                         font=ctk.CTkFont(size=11)
                         ).pack(anchor="w")
            return
        for i, (kind, label) in enumerate(inputs):
            r, c = divmod(i, 3)
            btn = ctk.CTkButton(
                frame, text=label, height=28,
                command=lambda k=kind, lbl=label: self._wiggle(k, lbl))
            btn.grid(row=r, column=c, sticky="ew", padx=2, pady=2)
        for c in range(3):
            frame.grid_columnconfigure(c, weight=1)

    def _wiggle(self, kind: str, label: str) -> None:
        if self.device is None or self.device.kind is None:
            return
        if self.device.engaged:
            self._set_bind_status(
                "disengage first (INSERT) before binding inputs")
            return

        def _run() -> None:
            for n in (4, 3, 2, 1):
                self._set_bind_status(
                    f"wiggling {label} in {n} s — "
                    f"alt-tab to the game now")
                threading.Event().wait(1.0)
            self._set_bind_status(f"wiggling {label}…")
            try:
                if self.device.is_wheel:
                    self.device.device.wiggle(kind)
                elif self.device.is_gamepad:
                    from tools.bind_helper import wiggle as gamepad_wiggle
                    gamepad_wiggle(self.device.raw, kind)
            except Exception as e:
                self._set_bind_status(f"error: {e}")
                return
            self._set_bind_status(
                "done. Click the next binding slot, or pick another input.")
        threading.Thread(target=_run, daemon=True).start()

    def _set_bind_status(self, text: str) -> None:
        if self._bind_status_label is None or self._root is None:
            return
        self._root.after(
            0, lambda: self._bind_status_label.configure(text=text))
