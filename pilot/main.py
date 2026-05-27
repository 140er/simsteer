"""Closed-loop driving entrypoint.

    python -m pilot.main [--max-width 1600] [--device gamepad|wheel]

Default `--device gamepad` uses ViGEm Xbox 360 emulation (works in any
game that accepts XInput). Use `--device wheel` to emulate a sim wheel
via vJoy — this bypasses ETS2's speed-sensitive gamepad rack assist
(the `b · v²` term in LiveParams) and produces a linear axis→wheel
response. Requires the vJoy driver + `pip install pyvjoy`.

Controls. INSERT, ← / →, F1/F2/F3 fire GLOBALLY (game can stay focused).
Everything else needs the overlay window focused:
    INSERT     toggle ENGAGE / DISENGAGE                 (global)
    NumPad 4/6 command lane change LEFT / RIGHT (now)    (global)
    PgUp/PgDn  NAV: queue LEFT / RIGHT @ default dist    (global)
    End        NAV: clear queue                          (global)
    Q       quit                                         (overlay)
    V       toggle model-view / capture-view overlay     (overlay)
    I       toggle the model-input thumbnail             (overlay)
    M       mirror the lateral sign                      (overlay)
    [ / ]   widen / narrow FOV                           (overlay)
    , / .   tilt pitch                                   (overlay)
    ; / '   raise / lower camera height                  (overlay)
    C       save calibration.json                        (overlay)
    0       reset calibration to defaults                (overlay)
    1..0    wiggle a controller input for ETS2 binding   (overlay)

Safety: starts DISENGAGED. The virtual gamepad is centered until you
hit space. Engagement is also dropped automatically if the model can't
keep up (FPS below MIN_HEALTHY_FPS).

Calibration (pitch + height) and steering scale (curvature per gamepad
axis) self-tune from telemetry; there are no Shift+ toggles for it.
"""

from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np

from pilot import audio
from pilot.calibration import Calibration, model_view_calib
from pilot.calibration_routine import CalibrationRoutine
from pilot.capture import Capture, CaptureConfig
from pilot.constants import DESIRE_LEN, T_IDXS
from pilot.controller import ControllerConfig, LateralController, LongitudinalController
from pilot.device import DeviceManager
from pilot.gamepad import Gamepad
from pilot.wheel import Wheel
from pilot.global_keys import (
    GlobalKeys, KEY_END, KEY_INSERT, KEY_NUMPAD4, KEY_NUMPAD6,
    KEY_PAGEDOWN, KEY_PAGEUP,
)
from pilot.hotkeys import hk_enabled as _hk_enabled
from pilot.hud import (
    COL_ACCENT_BLUE, COL_ACCENT_GREEN, COL_ACCENT_RED, COL_ACCENT_VIOLET,
    COL_ACCENT_YELLOW, HudRenderer, HudState,
)
from pilot.livecalib import CalStatus, LiveCalib
from pilot.liveparams import LiveParams
from pilot.manual import ManualInputs
from pilot.model import DrivingModel
from pilot.nav import ManeuverDir, NavManager
from pilot.postprocess import decode
from pilot.preflight import (
    run_game_preflight, run_global_preflight, show_preflight_dialog,
    warnings_for_hud,
)
from pilot.preprocess import FrameQueue, yuv6_to_bgr
from pilot.probe import SteeringProbe
from pilot.settings import Settings
from pilot.version import __version__
from pilot.wizard import Wizard, WizardPhase
from pilot.telemetry import Telemetry
from pilot.telemetry_ac import ACTelemetry
from pilot.telemetry_forza import ForzaTelemetry
from pilot.tuner import Tuner
from tools.bind_helper import wiggle as wiggle_input
from debug.overlay import (
    WINDOW_NAME, draw_calibration_hud, draw_model_input_inset, draw_overlay,
    handle_calibration_key, setup_window, show_scaled,
)


# Desire one-hot indices (matches openpilot's order). Pulse one of
# these into model.step(desire=...) for a single frame and the model's
# 5-second context window holds the rising edge — its internal state
# then drives the lane change planning. We don't run openpilot's full
# DesireHelper state machine; the user just hits the key and the model
# picks it up.
DESIRE_LANE_CHANGE_LEFT = 3
DESIRE_LANE_CHANGE_RIGHT = 4


# Number-key wiggle bindings — used while DISENGAGED to drive an input
# at full deflection so ETS2's binding wizard can detect it.
WIGGLE_HOTKEYS: dict[int, str] = {
    ord("1"): "left_stick_x",
    ord("2"): "right_stick_x",
    ord("3"): "left_trigger",
    ord("4"): "right_trigger",
    ord("5"): "button_a",
    ord("6"): "button_b",
    ord("7"): "button_x",
    ord("8"): "button_y",
    ord("9"): "button_lb",
    ord("0"): "button_rb",
}

# Per-key hotkey-id lookups — used to gate handlers against
# `settings.disabled_hotkeys`. The hotkey-tab disable flow flips bits
# in that list; these tables let the keypress dispatcher consult it
# in O(1).
WIGGLE_HOTKEY_IDS: dict[int, str] = {
    ord("1"): "wiggle_lx",  ord("2"): "wiggle_rx",
    ord("3"): "wiggle_lt",  ord("4"): "wiggle_rt",
    ord("5"): "wiggle_a",   ord("6"): "wiggle_b",
    ord("7"): "wiggle_x",   ord("8"): "wiggle_y",
    ord("9"): "wiggle_lb",  ord("0"): "wiggle_rb",
}
CALIB_KEY_HOTKEY_IDS: dict[int, str] = {
    ord("["): "fov_minus",   ord("]"): "fov_plus",
    ord(","): "pitch_minus", ord("."): "pitch_plus",
    ord(";"): "height_minus", ord("'"): "height_plus",
    ord("c"): "save_calib",  ord("m"): "mirror_sign",
    ord("0"): "reset_calib_defaults",
}


MIN_HEALTHY_FPS = 8.0

# Re-center the virtual pad every N disengaged frames (~20 Hz capture).
# Defense vs ViGEm/ETS2 holding a stale non-zero axis value after we
# disengaged. 20 frames ≈ 1 s, low enough that ETS2's active-source
# logic shouldn't latch onto our zeros.
DISENGAGED_RECENTER_EVERY_N = 20


def hud(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 30
    for txt, color in lines:
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 1, cv2.LINE_AA)
        y += 28


def open_telemetry(prefer: str,
                   forza_port: int = 7777) -> tuple[object, str, str]:
    """Pick a telemetry source based on which game is producing data.

    `prefer` is "auto" / "ets2" / "ac" / "forza". For Forza we bind a
    UDP socket immediately; for the others we just open the shared-
    memory map. Returns `(instance, display_label, game_key)` — the
    game_key is the short canonical name used to suffix per-game state
    files (liveparams_<game_key>.json etc.). The display_label is for
    HUD / log output.
    """
    if prefer == "ets2":
        return Telemetry(), "ets2", "ets2"
    if prefer == "ac":
        return ACTelemetry(), "ac", "ac"
    if prefer == "forza":
        return (ForzaTelemetry(port=forza_port),
                f"forza (UDP :{forza_port})", "forza")
    # auto: ETS2 first, then AC, then Forza. ETS2 + AC are passive
    # (they just check if shared memory exists), Forza requires a
    # bind so we only try it if the others fail to attach.
    t = Telemetry()
    if t.available:
        return t, "ets2", "ets2"
    t.close()
    t = ACTelemetry()
    if t.available:
        return t, "ac", "ac"
    t.close()
    t = ForzaTelemetry(port=forza_port)
    return t, f"forza (UDP :{forza_port}) — also tried ets2 + ac", "forza"


def main() -> int:
    # Load persistent settings BEFORE argparse so saved values become
    # the argparse defaults. CLI flags then override settings for the
    # current run; the tuner's Setup tab persists changes the other
    # way (next launch picks them up). First launch with no settings
    # file uses dataclass defaults; we save them out at the end of
    # setup so users see a settings.json they can edit by hand.
    settings = Settings.load()

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-width", type=int, default=settings.max_width)
    ap.add_argument("--no-gamepad", action="store_true",
                    default=settings.no_gamepad,
                    help="run the loop and overlay but don't open any input "
                         "device (useful when ViGEm/vJoy isn't installed)")
    ap.add_argument("--device", choices=["gamepad", "wheel"],
                    default=settings.device,
                    help="output device kind. 'gamepad' = ViGEm Xbox 360 "
                         "(speed-sensitive in ETS2). 'wheel' = vJoy "
                         "(linear; bypasses ETS2's gamepad rack assist). "
                         "Requires the vJoy driver + pyvjoy.")
    ap.add_argument("--vjoy-device", type=int, default=settings.vjoy_device,
                    help="vJoy device index when --device=wheel (default 1).")
    ap.add_argument("--no-tuner", action="store_true",
                    default=settings.no_tuner,
                    help="don't open the slider GUI window")
    ap.add_argument("--game", choices=["auto", "ets2", "ac", "forza"],
                    default=settings.game,
                    help="which game's telemetry to read. 'auto' tries ETS2, "
                         "then AC, then Forza UDP.")
    ap.add_argument("--forza-port", type=int, default=settings.forza_port,
                    help="UDP port Forza is sending Data Out to. Must match "
                         "the port set in-game (Settings -> HUD and Gameplay "
                         "-> Data Out -> Data Out IP Port).")
    ap.add_argument("--passive-fit-ets2", action="store_true",
                    default=settings.passive_fit_ets2,
                    help="Allow LiveParams to fit while disengaged on ETS2 "
                         "(default off). ETS2's gamepad rack assist (b·v² "
                         "term) only applies to virtual-pad input, so a "
                         "passive fit while the user drives with a wheel "
                         "or keyboard can pollute the fit. AC and Forza "
                         "always passive-fit when disengaged.")
    ap.add_argument("--force-engage", action="store_true",
                    default=settings.force_engage,
                    help="Bypass the engagement readiness gate (camera/FPS/"
                         "telemetry checks). For dev/testing only — without "
                         "calibration the truck will not drive correctly.")
    ap.add_argument("--reset-calib", action="store_true",
                    help="Wipe LiveCalib + LiveParams state for the active "
                         "game at startup (camera pose + steering fit + "
                         "first-drive flag). Use when calibration has gone "
                         "bad and you want the wizard to start over. "
                         "Equivalent to deleting livecalib_state_<game>.json, "
                         "liveparams_<game>_<device>.json, and "
                         "first_drive_done_<game>.flag manually.")
    ap.add_argument("--no-probe", action="store_true",
                    default=settings.no_probe,
                    help="Disable active steering probing during wizard "
                         "Phase B. By default, the wizard superimposes a "
                         "small (~0.03 axis) sinusoidal perturbation on "
                         "top of the AI command so LiveParams converges "
                         "in minutes instead of tens of minutes of "
                         "straight-line driving.")
    args = ap.parse_args()

    # Materialize settings.json on first launch with the LOADED values
    # (dataclass defaults if no file). Do this BEFORE applying CLI
    # overrides so a one-off `--no-gamepad` doesn't get persisted as
    # the new default. If the user wants to lock in CLI choices, they
    # click Save in the tuner's Setup tab.
    settings_path = settings.path()
    from pathlib import Path as _Path
    if not _Path(settings_path).exists():
        settings.save()
        print(f"settings: created {settings_path} (edit via tuner Setup tab "
              f"or directly)")
    else:
        print(f"settings: loaded {settings_path}")

    # Now mirror the resolved CLI values into Settings so the tuner's
    # Setup tab sees the live state. The tuner's Save button persists
    # this back to disk — but we don't auto-save here.
    settings.max_width = args.max_width
    settings.no_gamepad = args.no_gamepad
    settings.device = args.device
    settings.vjoy_device = args.vjoy_device
    settings.no_tuner = args.no_tuner
    settings.game = args.game
    settings.forza_port = args.forza_port
    settings.passive_fit_ets2 = args.passive_fit_ets2
    settings.force_engage = args.force_engage
    settings.no_probe = args.no_probe

    # Preflight (global): drivers, model files, DML. Fatal misses abort
    # before we open any cv2 window — Tk dialogs + cv2 deadlock on
    # Windows if the cv2 window opens first.
    pre = run_global_preflight(device=args.device)
    if not args.no_gamepad and not show_preflight_dialog(pre):
        return 2

    # Open telemetry FIRST so we know which game we're targeting, then
    # load that game's persistent state. Switching games therefore
    # automatically loads a different calibration / controller config /
    # liveparams fit — none of those bleed across games.
    fq = FrameQueue()
    model = DrivingModel()
    tel, tel_label, game = open_telemetry(args.game,
                                          forza_port=args.forza_port)
    # Per-game preflight (after telemetry resolves the active game).
    game_pre = run_game_preflight(game)
    # Show install-offer dialogs for warnings that can self-fix (e.g.
    # the bundled SCS plugin). No fatals expected at this stage; if any
    # appear, treat them as warnings rather than aborting — the user
    # already passed the global gate.
    if not args.no_gamepad and game_pre.warnings:
        show_preflight_dialog(game_pre)
    preflight_warnings = warnings_for_hud(pre) + warnings_for_hud(game_pre)
    for w in preflight_warnings:
        print(f"preflight: WARN — {w}")
    tel_status = ("available" if tel.available
                  else "not detected — game not running, plugin missing, "
                       "or shared memory disabled. Using --default-speed.")
    print(f"telemetry [{tel_label}]: {tel_status}")
    print(f"per-game state: loading from *_{game}.json (legacy fallback if absent)")
    calib = Calibration.load(game=game)
    ctrl_cfg = ControllerConfig.load(game=game)
    manual = ManualInputs()
    print(f"vision: {model.active_provider}, policy: {model.policy_provider}")
    print(f"calibration: VFOV={calib.fov_v_deg:.1f}° (HFOV={calib.fov_h_deg:.1f}°) "
          f"pitch={calib.pitch_deg:.1f}° h={calib.height_m:.2f}m")

    pad: DeviceManager | None = None
    if not args.no_gamepad:
        pad = DeviceManager(initial_kind=args.device,
                            vjoy_device_id=args.vjoy_device)
        if pad.kind is None and args.device == "wheel":
            print(f"wheel: NOT AVAILABLE — {pad.last_error}")
            print("  falling back to gamepad...")
            pad.set_kind("gamepad")
        if pad.kind == "gamepad":
            print("output: ViGEm Xbox 360 gamepad ready")
        elif pad.kind == "wheel":
            print(f"output: vJoy wheel device #{args.vjoy_device} ready — "
                  f"set ETS2 controller as 'wheel', non-linearity = 0")
        else:
            print(f"output: NO DEVICE — {pad.last_error}")
            print("  continuing without steering output")

    # LiveParams is per-(game, device): the rack response and sign
    # differ between ViGEm gamepad and vJoy wheel, so each device has
    # its own persisted fit. The tuner's device-switch handler calls
    # `live_params.switch_device(new_kind)` to swap between them
    # without losing either set.
    live_params = LiveParams(game=game,
                             device_kind=pad.kind if pad else None)
    print(f"liveparams [{pad.kind if pad else 'no device'}]: "
          f"a={live_params.a_linear:.2f} b={live_params.b_quad:.4f} "
          f"c={live_params.c_bias:+.3f} (scale@0={live_params.scale:.4f}) "
          f"samples={live_params.samples}")

    live_calib = LiveCalib(game=game)
    print(f"livecalib [{game}]: pitch/yaw/height via block-aggregated openpilot "
          f"calibrationd (5 blocks x 100 samples; needs v>6.7 m/s, |yaw_rate|<2°/s) "
          f"— restored {live_calib.blocks}/5 blocks "
          f"({live_calib.samples} samples) from disk")

    # FOV is a static tuner slider — `calib.fov_h_deg` / `fov_v_deg`.
    # openpilot treats camera intrinsics as factory-calibrated, not
    # learned from driving. The old LiveFov auto-correction read
    # `vx_model / v_ego` and walked fov_h_deg to match, but the model's
    # vx is itself a function of the warp's FOV, so the loop fed itself
    # and could converge to wildly wrong values (44.5° when the in-game
    # FOV was actually 90°). Removed. User sets VFOV manually in the
    # Camera tab to match the in-game setting.

    # `lookahead_s` is a static tuner slider — equivalent to
    # openpilot's per-car STEER_ACTUATOR_DELAY. The auto-tuner that
    # walked it from NCC of game_steer vs yaw_rate was removed; comma
    # ships steerActuatorDelay as a fixed per-vehicle constant.
    print(f"steerActuatorDelay: lookahead_s={ctrl_cfg.lookahead_s:.2f}s (static)")

    # `curvature_anticipation_s` is a static tuner slider now — same
    # role openpilot's `+ 0.2 s for other delays` plays in
    # `get_lag_adjusted_curvature`. The auto-tuner that used to walk
    # it from yaw error was removed because it oscillated against
    # LiveParams and the rate-limit in desired_curvature_lag_adjusted.

    # `steer_authority` is a static tuner slider — openpilot uses a
    # per-car STEER_RATIO constant rather than walking the multiplier
    # online. The auto-tuner that compared lane_k vs plan_k was
    # removed because it oscillated against LiveParams and could
    # latch onto a wrong value (e.g. 1.7x when implied was 0.63x).

    wizard = Wizard(game=game, live_calib=live_calib, live_params=live_params)
    probe = SteeringProbe(enabled=not args.no_probe)
    print(f"probe: active-steering perturbation "
          f"{'ENABLED' if probe.enabled else 'DISABLED'} "
          f"(±{probe.BASE_AMPLITUDE_AXIS} axis sine, "
          f"{probe.PERIOD_S}s cycle, "
          f"only during wizard Phase B / not yet LP-trusted)")
    # Manual guided-calibration routine — user-triggered from the
    # tuner, independent of the first-drive wizard. The wizard runs
    # passively on first launch; cal_routine is for explicit re-runs
    # after a setup change (different rig, new game, etc).
    cal_routine = CalibrationRoutine(live_params)
    if args.reset_calib:
        live_calib.reset(reason="--reset-calib on startup")
        live_params.reset(wipe_disk=True)
        wizard.reset()
        print(f"reset-calib [{game}]: wiped LiveCalib + LiveParams + "
              f"wizard flag. Restarting calibration from zero.")
    if wizard.active:
        print(f"wizard [{game}]: first-drive mode — drive ~10-15 min on "
              f"highway to converge calibration. INSERT will be blocked "
              f"until camera calibration completes.")
    else:
        print(f"wizard [{game}]: already completed (flag at "
              f"{wizard._flag_path}); skipping.")

    nav = NavManager()
    print("nav: NOOP-style manager (PgUp=queue LEFT, PgDn=queue RIGHT, End=clear)")

    ctrl = LateralController(cfg=ctrl_cfg, live_params=live_params)
    long_ctrl = LongitudinalController(cfg=ctrl_cfg)

    if not args.no_tuner:
        Tuner(calib, ctrl_cfg, manual, live_params=live_params, game=game,
              device=pad, settings=settings, probe=probe,
              live_calib=live_calib, wizard=wizard,
              cal_routine=cal_routine)
        print("tuner: slider window opened (use --no-tuner to skip)")

    hud_renderer = HudRenderer()
    print(f"hud: {settings.hud_mode!r} mode (H to toggle)")

    print("\nDISENGAGED — press INSERT to engage (works while the game is focused). Q to quit.\n")

    with Capture(CaptureConfig(target_fps=20)) as cap:
        for _ in range(50):
            if cap.grab() is not None:
                break
            time.sleep(0.05)

        first = cap.grab()
        if first is not None:
            h, w = first.shape[:2]
            win_w = min(w, args.max_width)
            win_h = int(round(h * (win_w / w)))
            setup_window(win_w, win_h)
            cv2.setWindowTitle(WINDOW_NAME,
                               f"SimSteer {__version__} — {tel_label}")
        else:
            setup_window(args.max_width, args.max_width * 9 // 16)

        # Global key poller — Win32 GetAsyncKeyState. We deliberately
        # restrict this to ONLY the driving controls (engage +
        # lane changes) so they work while the game is focused. The
        # tuning / view / wiggle keys are NOT globally polled — they
        # need overlay-window focus via cv2.waitKey. Reason: pitch (`,`
        # `.`), height (`;` `'`), FOV (`[` `]`), view toggle, etc.
        # would otherwise fire any time the user hits them in-game.
        watched_keys: set[int] = {
            KEY_INSERT,                              # engage/disengage
            KEY_NUMPAD4, KEY_NUMPAD6,                # lane change L/R (now)
            KEY_PAGEUP, KEY_PAGEDOWN, KEY_END,       # NAV: queue L / R / clear
        }
        global_keys = GlobalKeys(watched_keys)

        last = time.perf_counter()
        show_input = True
        view_mode = "model"  # "model" or "capture"
        # Lane-change desire is *sustained* for ctrl_cfg.lane_change_hold_s
        # after the user presses A/D — a single-frame pulse washes out
        # of the model's 5 s temporal buffer before the maneuver commits.
        # `lane_change_idx` is None when no command is active, else the
        # one-hot index to feed each frame until `lane_change_until`.
        lane_change_idx: int | None = None
        lane_change_until = 0.0
        desire_msg = ""
        desire_msg_until = 0.0
        frame_idx = 0   # bumped at the top of each loop iteration
        # Engagement banner state. Three sources update these:
        #   - successful engage  ->  green "ENGAGED" for 1.5 s
        #   - successful engage in wizard phase B  ->  yellow persistent
        #   - blocked engage     ->  red "CANNOT ENGAGE — <reason>" for 3 s
        #   - auto-disengage     ->  red "DISENGAGED — <reason>" for 3 s
        banner_text = ""
        banner_color = (200, 200, 200)
        banner_until = 0.0
        banner_persistent = False    # True for the wizard-phase-B warning
        # Auto-disengage tracking — when pad.disengage() fires from FPS
        # drop, surface the reason in the banner (the user didn't ask).
        was_engaged_last_frame = False

        def _engage_check() -> tuple[bool, str]:
            """Returns (allowed, reason). reason='' if allowed."""
            if settings.force_engage:
                return True, ""
            if pad is None or pad.kind is None:
                return False, "no virtual gamepad (install ViGEm)"
            if not tel.available:
                return False, "game telemetry not detected"
            if fps < MIN_HEALTHY_FPS:
                return False, f"frame rate too low ({fps:.1f} < {MIN_HEALTHY_FPS:.0f})"
            ok, why = wizard.allows_engage()
            if not ok:
                return False, why
            if live_calib.cal_status != CalStatus.CALIBRATED:
                return False, "camera calibration not done — drive first"
            return True, ""
        try:
            while True:
                frame_idx += 1
                frame = cap.grab()
                if frame is None:
                    if (cv2.waitKey(1) & 0xFF) == ord("q") or ord("q") in global_keys.poll():
                        break
                    continue

                v_real = tel.speed_mps()
                v_ego = v_real if v_real is not None else ctrl_cfg.default_speed
                actual_yaw = tel.yaw_rate_rad_s()
                # Pass the controller's wheelbase so AC's bicycle-model
                # synthesis stays consistent with the controller's own
                # math (cancels out cleanly). ETS2 ignores the arg.
                wheel_angle = tel.wheel_angle_rad(ctrl_cfg.wheelbase_m)
                # Use ETS2's smoothed `gameSteer` rather than the raw
                # `userSteer` — keyboard input shows up as ±1 in
                # userSteer but as a smooth analog in gameSteer (after
                # ETS2's input filter), which is what we need to fit
                # against the wheel angle.
                game_steer = tel.game_steer()

                img_narrow, img_wide = fq.push(frame, calib)
                # Tick the navigation manager BEFORE composing the desire.
                # It returns a one-hot index (3 or 4) when an enqueued
                # maneuver crosses its trigger distance; we OR that with
                # the manual ← / → pulse. Manual wins when both fire at
                # once (last-write-wins on the one-hot).
                #
                # IMPORTANT: dt must be computed against the same clock
                # as `last` (which is perf_counter at the end of the
                # previous iteration). Mixing time.time() (unix epoch
                # = ~1.7e9) with perf_counter (process-relative, near 0)
                # makes dt huge on every frame, ages the maneuver past
                # expiry instantly, and silently drops it. This bug
                # made the nav hotkeys appear non-functional.
                now_perf = time.perf_counter()
                dt_nav = max(1e-3, now_perf - last)
                nav_desire_idx = nav.tick(
                    v_ego=v_ego, dt=dt_nav, now_t=now_perf)
                now_t = time.time()   # wall time, used downstream for the
                                       # human-facing lane-change-hold timer
                # Pull ETS2 destination distance/time if telemetry exposes
                # it (informational only — doesn't drive maneuvers).
                if hasattr(tel, "nav_distance_m"):
                    nav.update_destination(
                        tel.nav_distance_m(), tel.nav_time_s())

                # Sustain the lane-change desire one-hot for the
                # configured hold window. The model sees the same
                # one-hot every frame for ~2.5 s, which builds up a
                # consistent signal in its temporal context buffer.
                lane_change_active = (lane_change_idx is not None
                                      and now_t < lane_change_until)
                # Pick desire source: manual ← / → wins if active, else
                # use nav-fired one. Either way the model gets one
                # consistent one-hot per frame.
                effective_idx = (lane_change_idx if lane_change_active
                                 else nav_desire_idx)
                if effective_idx is not None:
                    desire_for_step = np.zeros(DESIRE_LEN, dtype=np.float32)
                    desire_for_step[effective_idx] = 1.0
                else:
                    desire_for_step = None
                    lane_change_idx = None  # clear when expired
                vision_out, policy_out = model.step(
                    img_narrow, img_wide, desire=desire_for_step)
                decoded = decode(vision_out, policy_out)

                live_calib.update(calib, decoded.pose, decoded.road_transform,
                                  actual_yaw, v_real,
                                  pose_std=decoded.pose_std,
                                  road_transform_std=decoded.road_transform_std,
                                  wide_from_device_euler=decoded.wide_from_device_euler,
                                  game_steer=game_steer)
                wizard.tick()

                ai_steer = ctrl.compute(decoded, v_ego,
                                        actual_wheel_angle=wheel_angle,
                                        lane_change_command_active=lane_change_active,
                                        dt=dt_nav)

                # Apply lateral_sign at the output. LiveParams fits the
                # in-game relationship between gameSteer and wheel angle
                # (or synthesized wheel angle for AC) — that fit is
                # internally consistent. But the model's plan_k uses
                # openpilot's left-positive convention, and game yaw
                # conventions don't always match. `lateral_sign = -1`
                # flips the output so left-of-model-frame becomes
                # left-of-game-frame. Toggle via tuner checkbox or M key.
                ai_steer *= calib.lateral_sign
                # Manual override: tuner has a steer slider + checkbox so
                # we can drive ourselves without disengaging the pad
                # (necessary for AC, which only accepts one input source
                # at a time — disengaging the pad doesn't hand control
                # back, the human just gets nothing).
                steer = manual.steer if manual.steer_override else ai_steer

                # Active steering probe (wizard Phase B only). Superimpose
                # a small sine perturbation so LiveParams gets useful
                # (axis, wheel) excitation even on a straight road, where
                # the AI's natural command is zero and the fit would
                # otherwise stall on `rej_small_axis`. The probe stops
                # itself once LP is trusted; until then it adds ~±0.03
                # axis on top of `steer`. Manual override disables it
                # (the user is in control). Compute dt off the wall clock
                # since `last`/`now` are perf_counter elsewhere in the
                # loop; we want monotonic seconds for sine phasing.
                probe_offset = 0.0
                if (pad is not None and not manual.steer_override):
                    probe_offset = probe.tick(
                        dt=dt_nav,
                        ai_axis=steer,
                        v_ego=v_real,
                        in_lane_change=lane_change_active,
                        engaged=pad.engaged,
                        lp_trusted=live_params.trusted(),
                        wizard_steering_phase=(
                            wizard.phase == WizardPhase.STEERING),
                    )
                    steer = steer + probe_offset

                ai_throttle, ai_brake = long_ctrl.compute(decoded, v_ego)
                # Manual override mirrors the steering one: when the
                # tuner checkbox is on, the sliders drive the pedals;
                # otherwise the AI does. Defaults to AI-on so the pad
                # actually drives the car when engaged.
                if manual.long_override:
                    throttle_out, brake_out = manual.throttle, manual.brake
                else:
                    throttle_out, brake_out = ai_throttle, ai_brake

                # Feed (gameSteer, wheel_angle) into LiveParams. The
                # fit lives in the game's frame: `game_steer` is what
                # telemetry reports was applied, `wheel_angle` is the
                # resulting front wheel angle, and the regression
                # `gameSteer = a*wheel + b*wheel*v² + c` is internally
                # consistent for that game's sign convention —
                # `lateral_sign` then maps the controller's output
                # into the same game frame at the OUTPUT stage
                # (`ai_steer *= calib.lateral_sign` above). Do NOT
                # pre-flip the fit inputs by lateral_sign — that
                # double-corrects and breaks convergence on every
                # setup with non-+1 lateral_sign.
                #
                # Passive-fit (disengaged): AC and Forza synthesize
                # their wheel angle from yaw rate; the game's input
                # filter behaves the same regardless of whether the
                # axis came from our virtual pad or the user's wheel,
                # so the fit is valid. ETS2 has gamepad-specific rack
                # assist (b·v² in the inverse), so a passive fit on
                # ETS2 while the user drives with a wheel/keyboard
                # would pollute the gamepad fit — gated behind
                # --passive-fit-ets2.
                #
                # The guided-calibration routine ticks first so it can
                # widen LiveParams' gates via the per-call override.
                # The override is None when the routine is inactive,
                # so the per-game profile in LiveParams is never
                # mutated.
                cal_routine.tick(v_real, wheel_angle, dt_nav)
                allow_passive = (
                    game in ("ac", "forza")
                    or (game == "ets2" and settings.passive_fit_ets2)
                    or wizard.in_camera_phase)
                if pad is not None and (pad.engaged or allow_passive):
                    # Disengaged: there's no AI command — the user IS
                    # the commander. Pass game_steer as the commanded
                    # axis so the assist/intervention check sees them
                    # as matched (no "user fighting AI" to detect).
                    cmd = steer if pad.engaged else game_steer
                    live_params.update(game_steer, v_real, wheel_angle,
                                       commanded_axis=cmd,
                                       gate_override=cal_routine.gate_override())

                if pad is not None:
                    if pad.engaged:
                        # Engaged: AI drives (with per-axis manual
                        # override already folded into `steer` and
                        # `throttle_out` / `brake_out` above).
                        pad.set_steering(steer)
                        pad.set_throttle_brake(throttle_out, brake_out)
                    else:
                        # Disengaged: periodic re-center keeps any
                        # ViGEm drift (driver glitches, lost updates,
                        # the game's input filter holding a stale
                        # value after we centered once at disengage-
                        # time) from leaving the truck slowly turning.
                        # ~1 s cadence is slow enough that ETS2's
                        # "active controller" logic doesn't latch onto
                        # our zeros (which would suppress the user's
                        # keyboard fallback).
                        if frame_idx % DISENGAGED_RECENTER_EVERY_N == 0:
                            pad.center()
                        # Manual override ALWAYS fires when its toggle
                        # is on — even while disengaged. Per-axis: the
                        # steer override drives the joystick, the long
                        # override drives the triggers, and each is
                        # forced through the engaged-gate inside the
                        # device. Non-overridden axes stay at center
                        # (just zeroed above).
                        if manual.steer_override:
                            pad.set_steering(manual.steer, force=True)
                        if manual.long_override:
                            pad.set_throttle_brake(
                                manual.throttle, manual.brake, force=True)

                now = time.perf_counter()
                fps = 1.0 / max(now - last, 1e-3)
                last = now

                if pad is not None and pad.engaged and fps < MIN_HEALTHY_FPS:
                    pad.disengage()
                    ctrl.reset()    # drop PID wind-up on the FPS-drop disengage
                    audio.play("disengage")
                    banner_text = f"DISENGAGED — frame rate dropped ({fps:.1f} fps)"
                    banner_color = (80, 80, 255)
                    banner_until = time.time() + 3.0
                    banner_persistent = False

                # Engaged-warning banner clears itself once steering becomes
                # trusted (transition from "warming" to "OK"). Also clears
                # if the user disengaged for any reason.
                if banner_persistent and pad is not None:
                    if (not pad.engaged) or live_params.trusted():
                        banner_persistent = False
                        banner_text = ""
                # Detect auto-disengage we didn't explicitly catch above
                # (e.g. pad backend dropped its connection). The state
                # transition from engaged -> disengaged outside the
                # INSERT handler is informative.
                pad_engaged_now = bool(pad is not None and pad.engaged)
                if was_engaged_last_frame and not pad_engaged_now and not banner_text:
                    audio.play("disengage")
                    banner_text = "DISENGAGED"
                    banner_color = (80, 80, 255)
                    banner_until = time.time() + 2.0
                was_engaged_last_frame = pad_engaged_now

                engaged_color = (80, 255, 80) if (pad and pad.engaged) else (80, 80, 255)
                state_text = "ENGAGED" if (pad and pad.engaged) else "DISENGAGED"
                if pad is None:
                    state_text = "NO GAMEPAD"

                if view_mode == "model" and fq.last_yuv is not None:
                    mv_bgr = yuv6_to_bgr(fq.last_yuv)
                    mv_h_disp = 720
                    mv_w_disp = mv_h_disp * mv_bgr.shape[1] // mv_bgr.shape[0]
                    mv_bgr = cv2.resize(mv_bgr, (mv_w_disp, mv_h_disp),
                                        interpolation=cv2.INTER_LINEAR)
                    # Use the model's own per-frame height estimate (the
                    # third component of `road_transform.trans`) for the
                    # overlay projection. The model predicts lane / plan
                    # positions as if the camera were at that height —
                    # using our tall truck-cab `calib.height_m` here
                    # would draw lanes at the wrong row. Fall back to
                    # the live-calib smoothed value, then to the manual
                    # calib value if neither is ready yet.
                    rt = decoded.road_transform
                    model_h = (float(rt[2]) if rt is not None and len(rt) >= 3
                               and 0.5 < float(rt[2]) < 4.0
                               else (live_calib.height_estimate or calib.height_m))
                    mv_calib = model_view_calib(
                        calib,
                        captured_shape=fq.last_captured_shape or frame.shape,
                        cropped_shape=fq.last_cropped_shape or frame.shape,
                        view_w=mv_w_disp, view_h=mv_h_disp,
                        model_height_m=model_h,
                    )
                    overlay = draw_overlay(mv_bgr, decoded, mv_calib)
                    if show_input:
                        small_full = cv2.resize(frame, (mv_w_disp // 3,
                                                        mv_h_disp // 3),
                                                interpolation=cv2.INTER_AREA)
                        H, W = overlay.shape[:2]
                        sh, sw = small_full.shape[:2]
                        overlay[10:10 + sh, W - 10 - sw:W - 10] = small_full
                else:
                    overlay = draw_overlay(frame, decoded, calib)
                    if show_input:
                        draw_model_input_inset(overlay, fq.last_yuv_narrow,
                                               fq.last_yuv_wide)

                pred_yaw = float(np.interp(ctrl_cfg.lookahead_s,
                                           np.asarray(T_IDXS, dtype=np.float32),
                                           decoded.plan[:, 14]))

                steer_label = "MANUAL OVERRIDE" if manual.steer_override else "AI"
                steer_color = ((255, 200, 80) if manual.steer_override
                               else (255, 255, 255))
                long_label = "MANUAL OVERRIDE" if manual.long_override else "AI"
                long_color = ((255, 200, 80) if manual.long_override
                              else (255, 255, 255))
                # Routine prompt — yellow when active, green-ish when
                # finishing (summary still visible). Inserted near the
                # top of dev_lines so the user sees it without
                # scanning; user-mode HUD pulls it via the renderer.
                routine_line: list[tuple[str, tuple[int, int, int]]] = []
                if cal_routine.active:
                    routine_line.append((cal_routine.prompt, (80, 255, 255)))
                elif cal_routine.summary_text:
                    routine_line.append(
                        (cal_routine.summary_text, (160, 255, 160)))

                # ----- DEV-mode payload -----
                # Same compact stack the old `hud()` would render. Built
                # every frame regardless of mode so toggling H is
                # immediate. User mode picks just the highlights below.
                dev_lines = [
                    (f"{state_text}", engaged_color),
                    *routine_line,
                    (f"steer cmd: {steer:+.3f} [{steer_label}]   "
                     f"target wheel: {ctrl.last_target_wheel:+.4f} rad "
                     f"(AI {ai_steer:+.3f})   "
                     f"auth={ctrl.last_authority:.2f}"
                     f"{'  +LC' if ctrl.last_in_lane_change else ''}",
                     steer_color),
                    (f"  axis: trim={ctrl.axis_trim_state:+.4f} "
                     f"bias={ctrl_cfg.axis_bias:+.3f} "
                     f"lpf_err={ctrl.lpf_wheel_error:+.4f} rad   "
                     f"trim: "
                     f"{('FROZEN[' + ctrl.last_trim_frozen_reason + ']') if ctrl.last_trim_frozen_reason else 'ACTIVE'}",
                     (180, 200, 255) if ctrl.last_trim_frozen_reason in ("", "off")
                     else (200, 200, 200)),
                    (f"throttle: {throttle_out:.2f}  brake: {brake_out:.2f} "
                     f"[{long_label}]   (AI t={ai_throttle:.2f} b={ai_brake:.2f})",
                     long_color),
                    (f"long: v_tgt={long_ctrl.last_v_target:5.1f}  "
                     f"a_tgt={long_ctrl.last_a_target:+.2f}  "
                     f"a_cmd={long_ctrl.last_a_cmd:+.2f} m/s^2  "
                     f"dv={long_ctrl.last_v_target - v_ego:+.1f}  "
                     f"t_long={ctrl_cfg.lookahead_s + ctrl_cfg.long_anticipation_s:.2f}s",
                     (200, 200, 255)),
                    (f"corner: v_safe="
                     f"{'inf' if long_ctrl.last_v_safe_corner == float('inf') else f'{long_ctrl.last_v_safe_corner:5.1f}'}"
                     f"  k={long_ctrl.last_corner_k:.4f} 1/m"
                     f"  @t={long_ctrl.last_corner_t:.2f}s"
                     f"{'   BRAKING TO CORNER' if long_ctrl.last_v_safe_corner < long_ctrl.last_v_target + 0.5 and long_ctrl.last_v_safe_corner != float('inf') else ''}",
                     (255, 180, 80) if (long_ctrl.last_v_safe_corner != float('inf')
                                        and long_ctrl.last_v_safe_corner < long_ctrl.last_v_target + 0.5)
                     else (160, 200, 255)),
                    (f"ACC: "
                     f"{'engaged' if long_ctrl.last_lead_prob > ctrl_cfg.lead_min_prob else 'no lead'}  "
                     f"prob={long_ctrl.last_lead_prob:.2f}  "
                     f"lead_x={('inf' if long_ctrl.last_lead_x == float('inf') else f'{long_ctrl.last_lead_x:5.1f}m')}  "
                     f"lead_v={long_ctrl.last_lead_v:+.1f} m/s  "
                     f"ttc={('inf' if long_ctrl.last_lead_ttc == float('inf') else f'{long_ctrl.last_lead_ttc:.1f}s')}"
                     f"{'   AEB!' if long_ctrl.last_aeb else ''}",
                     (255, 80, 80) if long_ctrl.last_aeb else
                     (255, 180, 80) if long_ctrl.last_lead_prob > ctrl_cfg.lead_min_prob else
                     (160, 200, 255)),
                    (nav.hud_line(),
                     (80, 255, 255) if nav.next_maneuver and nav.next_maneuver.fired
                     else (180, 220, 180) if nav.queue_len > 0
                     else (140, 140, 140)),
                    (f"v_ego: {v_ego:.1f} m/s   plan k: {ctrl.last_curvature:+.5f} 1/m",
                     (255, 255, 255)),
                    (f"yaw pred/actual: {pred_yaw:+.3f} / "
                     f"{actual_yaw if actual_yaw is not None else 0:+.3f} rad/s",
                     (255, 255, 255)),
                    (f"wheel: {wheel_angle if wheel_angle is not None else 0:+.3f} rad   "
                     f"gameSteer: {game_steer if game_steer is not None else 0:+.3f}",
                     (255, 255, 255)),
                    (f"LIVEPARAMS  a={live_params.a_linear:.2f}  "
                     f"b={live_params.b_quad:.4f}  c={live_params.c_bias:+.3f}   "
                     f"scale@v={live_params.effective_scale_at(v_ego):.4f} rad/axis  "
                     f"n={live_params.session_samples} (life {live_params.samples})  "
                     f"{live_params.trust_level}"
                     f"{'  [SEEDED]' if live_params._was_seeded else ''}",
                     (120, 200, 255) if live_params.locked else
                     (255, 80, 80) if live_params._consecutive_bad_fit > 10 else
                     (255, 150, 80) if (live_params.trusted()
                                        and live_params.innov_ratio > 0.30) else
                     (80, 255, 80) if (live_params.trusted()
                                       and live_params.innov_ratio < 0.05) else
                     (160, 220, 160) if live_params.trusted() else
                     (200, 200, 80)),
                    (f"  innov_ema={live_params.innov_ema:.4f}  "
                     f"ratio={live_params.innov_ratio:.2f}  "
                     f"wd={live_params.watchdog_resets}  "
                     f"last upd "
                     f"{(time.monotonic() - live_params.last_update_ts):4.1f}s ago"
                     if live_params.last_update_ts > 0 else
                     f"  innov_ema={live_params.innov_ema:.4f}  "
                     f"ratio={live_params.innov_ratio:.2f}  "
                     f"wd={live_params.watchdog_resets}  "
                     f"last upd: never",
                     (180, 180, 180)),
                    (f"  rej: slow={live_params.rej_slow} "
                     f"no_wheel={live_params.rej_no_wheel} "
                     f"small_axis={live_params.rej_small_axis} "
                     f"small_wheel={live_params.rej_small_wheel} "
                     f"transient={live_params.rej_transient} "
                     f"wheel={live_params.rej_wheel} "
                     f"innov={live_params.rej_innovation} "
                     f"assist={live_params.rej_assist} "
                     f"sat={live_params.rej_saturated} "
                     f"wd={live_params.watchdog_resets}",
                     (180, 180, 180)),
                    (f"  pred axis={live_params.last_predicted:+.4f}  "
                     f"actual axis={live_params.last_actual:+.4f}  "
                     f"err={live_params.last_innovation:+.4f}",
                     (180, 180, 180)),
                    (f"LIVECALIB  pitch={live_calib.pitch_estimate or 0:+.2f}deg  "
                     f"yaw={live_calib.yaw_estimate or 0:+.2f}deg  "
                     f"h={live_calib.height_estimate or 0:.2f}m  "
                     f"blocks={live_calib.blocks}/5  "
                     f"block_fill={live_calib.block_progress*100:.0f}%  "
                     f"n={live_calib.samples} rej={live_calib.rejected}  "
                     f"{live_calib.status_label}",
                     (80, 255, 80) if live_calib.cal_status.value == 1 else
                     (255, 100, 100) if live_calib.cal_status.value == 2 else
                     (200, 200, 80)),
                    (f"  wide_from_device r/p/y="
                     f"{(live_calib.wide_from_device_estimate[0] if live_calib.wide_from_device_estimate is not None else 0):+.3f}/"
                     f"{(live_calib.wide_from_device_estimate[1] if live_calib.wide_from_device_estimate is not None else 0):+.3f}/"
                     f"{(live_calib.wide_from_device_estimate[2] if live_calib.wide_from_device_estimate is not None else 0):+.3f} rad",
                     (180, 180, 180)),
                    (f"PROBE  offset={probe.last_offset:+.3f}  "
                     f"gate={probe.last_gate}  "
                     f"cycles={probe.cycles_done} active={probe.active_frames}",
                     (80, 255, 80) if probe.last_gate == "active"
                     else (160, 160, 160)),
                    (f"  rej breakdown: speed={live_calib.rej_speed} "
                     f"turning={live_calib.rej_turning} "
                     f"lat_acc={live_calib.rej_lat_acc} "
                     f"human={live_calib.rej_human} "
                     f"vel_std={live_calib.rej_vel_std} "
                     f"h_std={live_calib.rej_height_std} "
                     f"pitch={live_calib.rej_pitch} yaw={live_calib.rej_yaw} "
                     f"jump={live_calib.rej_block_jump}  "
                     f"raw pitch/yaw={live_calib.last_raw_pitch:+.1f}/"
                     f"{live_calib.last_raw_yaw:+.1f}",
                     (180, 180, 180)),
                    # vx_model / v_ego is a self-check on FOV: model's
                    # forward velocity (from `pose[0]`) should equal
                    # telemetry speed when the warp's FOV is right.
                    # >1.05 = FOV set too HIGH (warp over-crops, world
                    # appears zoomed, vx over-reads). <0.95 = too LOW
                    # (warp under-crops, world looks far, vx under-
                    # reads). Drive ≥6 m/s on a straight to read it.
                    # Diagnostic only — never auto-applied.
                    *([(
                        f"FOV  h={calib.fov_h_deg:.1f}°  v={calib.fov_v_deg:.1f}°"
                        f"   ratio vx_model/v_ego="
                        f"{(float(decoded.pose[0]) / v_real):.2f}  "
                        f"{'OK' if 0.95 < (float(decoded.pose[0]) / v_real) < 1.05 else ('FOV too HIGH — narrow it' if (float(decoded.pose[0]) / v_real) > 1.05 else 'FOV too LOW — widen it')}",
                        (80, 255, 80) if (decoded.pose is not None
                                          and v_real is not None and v_real > 6.0
                                          and 0.95 < (float(decoded.pose[0]) / v_real) < 1.05)
                        else (255, 180, 80)
                    )] if (decoded.pose is not None and len(decoded.pose) > 0
                           and v_real is not None and v_real > 6.0)
                       else [(
                        f"FOV  h={calib.fov_h_deg:.1f}°  v={calib.fov_v_deg:.1f}°"
                        f"   (drive ≥6 m/s on a straight to read vx ratio)",
                        (160, 200, 255)
                    )]),
                    (f"{fps:5.1f} fps   SimSteer v{__version__}", (255, 255, 255)),
                ]
                if pad is not None and not pad.engaged:
                    dev_lines.append(
                        ("BIND: 1 LX  2 RX  3 LT  4 RT  5-8 ABXY  9/0 LB/RB",
                         (180, 180, 180)))
                # Lane-change command flash (DEV mode only — user mode
                # shows it as a top-right chip).
                if desire_msg and time.time() < desire_msg_until:
                    dev_lines.append((desire_msg, (80, 255, 255)))
                # Dev mode keeps the preflight warnings + wizard +
                # banner as text rows (legacy behavior). User mode
                # surfaces them via the HudState fields below.
                if settings.hud_mode == "dev":
                    if preflight_warnings:
                        dev_lines = [(f"! {w}", (80, 220, 255))
                                     for w in preflight_warnings] + dev_lines
                    wiz_banner = wizard.banner()
                    if wiz_banner is not None:
                        wiz_rows = [wiz_banner] + wizard.progress_lines()
                        hint_txt = wizard.hint(v_real)
                        if hint_txt:
                            wiz_rows.append((f"  {hint_txt}", (200, 200, 200)))
                        dev_lines = wiz_rows + dev_lines
                    if banner_text and (banner_persistent
                                        or time.time() < banner_until):
                        dev_lines = [(banner_text, banner_color)] + dev_lines

                # ----- USER-mode HudState -----
                if pad is None:
                    eng_label = "NO GAMEPAD"
                    eng_col = COL_ACCENT_RED
                elif pad.engaged:
                    eng_label = "ENGAGED"
                    eng_col = COL_ACCENT_GREEN
                else:
                    eng_label = "DISENGAGED"
                    eng_col = COL_ACCENT_BLUE

                # Calibration fracs — pass None when N/A so the panel
                # can hide rows individually. We always show camera +
                # steering while the wizard is active; once both are
                # ready (post-wizard) we hide the dashboard entirely
                # to declutter — the user trusts the system at that
                # point.
                from pilot.livecalib import BLOCK_SIZE as LC_BLOCK, INPUTS_NEEDED as LC_INPUTS_NEEDED, CalStatus as _CalStatus
                from pilot.liveparams import TRUSTED_MIN_SAMPLES as LP_TRUSTED
                cam_frac = min(1.0,
                               (live_calib.blocks * LC_BLOCK + live_calib._block_n)
                               / float(LC_INPUTS_NEEDED * LC_BLOCK))
                lp_frac = min(1.0,
                              max(live_params.samples, live_params.session_samples)
                              / float(LP_TRUSTED))
                cam_ready = live_calib.cal_status == _CalStatus.CALIBRATED
                lp_ready = live_params.trusted()
                # Hide the dashboard once everything is ready AND the
                # wizard is done — at that point the user just wants
                # to see the model preview.
                dashboard_visible = (wizard.active
                                     or not cam_ready
                                     or not lp_ready)
                cal_camera_arg = cam_frac if dashboard_visible else None
                cal_steer_arg = lp_frac if dashboard_visible else None
                cal_fov_arg = None

                # Wizard banner text — strip the "FIRST DRIVE — N% — "
                # prefix because the renderer draws its own progress
                # bar; the hint covers the rest of the message.
                wiz_text = ""
                wiz_color = COL_ACCENT_VIOLET
                wiz_hint = ""
                if wizard.active:
                    wb = wizard.banner()
                    if wb is not None:
                        wiz_text = wb[0]
                        wiz_color = wb[1]
                    wiz_hint = wizard.hint(v_real) or ""

                # Lane-change chip — show while the desire is sustained.
                lc_chip = None
                if lane_change_active:
                    if lane_change_idx == DESIRE_LANE_CHANGE_LEFT:
                        lc_chip = "L"
                    elif lane_change_idx == DESIRE_LANE_CHANGE_RIGHT:
                        lc_chip = "R"

                # Live warnings the user should see right now (in
                # addition to the preflight set). Currently just one:
                # the steering fit collapsed to zero.
                live_warnings: list[str] = list(preflight_warnings)
                if live_params.a_below_floor and live_params.trusted():
                    live_warnings.append(
                        "steering fit collapsed (|a|<0.5) — using "
                        "floor; click Recalibrate")

                hud_state = HudState(
                    engaged=bool(pad and pad.engaged),
                    engaged_label=eng_label,
                    engaged_color=eng_col,
                    banner_text=(banner_text if banner_text and (
                        banner_persistent or time.time() < banner_until)
                        else ""),
                    banner_color=banner_color,
                    wizard_active=wizard.active,
                    wizard_text=wiz_text,
                    wizard_color=wiz_color,
                    wizard_progress=wizard.progress_pct() if wizard.active else 0.0,
                    wizard_hint=wiz_hint,
                    warnings=live_warnings,
                    v_ego_mps=v_real,
                    cal_camera_frac=cal_camera_arg,
                    cal_camera_status=("READY" if cam_ready else "WARMING"),
                    cal_steering_frac=cal_steer_arg,
                    cal_steering_status=("READY" if lp_ready else "WARMING"),
                    cal_fov_frac=cal_fov_arg,
                    cal_fov_status="READY",  # FOV is a static slider now
                    probe_active=(probe.last_gate == "active"),
                    manual_override=(manual.steer_override or manual.long_override),
                    lane_change=lc_chip,
                    lead_following=(long_ctrl.last_lead_prob > ctrl_cfg.lead_min_prob),
                    aeb=bool(long_ctrl.last_aeb),
                    fps=fps,
                    version=__version__,
                    mode_hint="H: dev view" if settings.hud_mode == "user"
                              else "H: user view",
                    dev_lines=dev_lines,
                )
                hud_renderer.draw(overlay, hud_state, mode=settings.hud_mode)
                draw_calibration_hud(overlay, calib)

                show_scaled(overlay, args.max_width)
                # Drain pressed keys from BOTH sources every frame:
                #   - cv2.waitKey: only fires when the overlay window
                #     has focus. Still useful for tuning + binding-mode
                #     where the user IS looking at the overlay.
                #   - GlobalKeys: Win32 GetAsyncKeyState; fires when the
                #     GAME has focus, which is the common case during
                #     real driving. Edge-triggered so a held key only
                #     reports once.
                pressed: list[int] = []
                cv_key = cv2.waitKey(1) & 0xFF
                if cv_key != 0xFF:
                    pressed.append(cv_key)
                pressed.extend(global_keys.poll())

                quit_loop = False
                def hk(hk_id: str) -> bool:
                    return _hk_enabled(hk_id, settings)
                for key in pressed:
                    if key == ord("q") and hk("quit"):
                        quit_loop = True
                        break
                    if key == ord("v") and hk("view_toggle"):
                        view_mode = "capture" if view_mode == "model" else "model"
                        print(f"view: {view_mode}")
                    elif key == ord("i") and hk("input_inset"):
                        show_input = not show_input
                    elif key == ord("h") and hk("hud_mode"):
                        # Toggle HUD mode (user ↔ dev). Persist so the
                        # choice carries across launches.
                        settings.hud_mode = (
                            "dev" if settings.hud_mode == "user" else "user")
                        settings.save()
                        print(f"hud: {settings.hud_mode!r}")
                    elif key == ord("r") and hk("recalibrate"):
                        # Hard reset: wipe LiveCalib + LiveParams + wizard
                        # flag so the user can start calibration over
                        # without quitting + manually deleting JSONs. Force-
                        # disengage first so we don't drive on the partial
                        # state mid-reset.
                        if pad is not None and pad.engaged:
                            pad.disengage()
                            audio.play("disengage")
                        live_calib.reset(reason="user pressed R")
                        live_params.reset(wipe_disk=True)
                        wizard.reset()
                        probe.reset()
                        ctrl.reset()
                        banner_text = "CALIBRATION RESET — wizard restarted"
                        banner_color = (80, 220, 255)
                        banner_until = time.time() + 4.0
                        banner_persistent = False
                        print("-> CALIBRATION RESET (LiveCalib + LiveParams "
                              "+ wizard flag + probe wiped)")
                    elif key == KEY_INSERT and pad is not None and hk("engage"):
                        if pad.engaged:
                            # User-initiated disengage — always allowed.
                            pad.disengage()
                            ctrl.reset()
                            audio.play("disengage")
                            banner_text = "DISENGAGED"
                            banner_color = (80, 80, 255)
                            banner_until = time.time() + 1.5
                            banner_persistent = False
                            print("-> DISENGAGED")
                        else:
                            allowed, why = _engage_check()
                            if not allowed:
                                audio.play("denied")
                                banner_text = f"CANNOT ENGAGE — {why}"
                                banner_color = (80, 80, 255)
                                banner_until = time.time() + 3.0
                                banner_persistent = False
                                print(f"-> ENGAGE BLOCKED: {why}")
                            else:
                                pad.engage()
                                ctrl.reset()
                                audio.play("engage")
                                if not live_params.trusted():
                                    banner_text = ("ENGAGED — steering fit "
                                                   "warming up, drive gently")
                                    banner_color = (80, 220, 200)
                                    banner_persistent = True
                                else:
                                    banner_text = "ENGAGED"
                                    banner_color = (80, 255, 80)
                                    banner_until = time.time() + 1.5
                                    banner_persistent = False
                                print("-> ENGAGED")
                    elif (key in WIGGLE_HOTKEYS and pad is not None
                          and hk(WIGGLE_HOTKEY_IDS.get(key, ""))):
                        if pad.engaged:
                            print("disengage first (INSERT) before binding inputs")
                        elif not pad.is_gamepad:
                            print("wiggle bind-helper is gamepad-only — use "
                                  "the in-game controller config to bind the "
                                  "vJoy wheel device directly.")
                        else:
                            kind = WIGGLE_HOTKEYS[key]
                            print(f"wiggling {kind} for ETS2 binding...")
                            wiggle_input(pad.raw, kind)
                            print(f"  done.")
                    elif key == KEY_NUMPAD4 and hk("lane_change_left"):
                        lane_change_idx = DESIRE_LANE_CHANGE_LEFT
                        lane_change_until = time.time() + ctrl_cfg.lane_change_hold_s
                        desire_msg = f"-> LANE CHANGE LEFT (holding {ctrl_cfg.lane_change_hold_s:.1f}s)"
                        desire_msg_until = lane_change_until
                        print(f"lane change LEFT commanded ({ctrl_cfg.lane_change_hold_s:.1f}s)")
                    elif key == KEY_NUMPAD6 and hk("lane_change_right"):
                        lane_change_idx = DESIRE_LANE_CHANGE_RIGHT
                        lane_change_until = time.time() + ctrl_cfg.lane_change_hold_s
                        desire_msg = f"-> LANE CHANGE RIGHT (holding {ctrl_cfg.lane_change_hold_s:.1f}s)"
                        desire_msg_until = lane_change_until
                        print(f"lane change RIGHT commanded ({ctrl_cfg.lane_change_hold_s:.1f}s)")
                    elif key == KEY_PAGEUP and hk("nav_queue_left"):
                        nav.queue(ManeuverDir.LEFT)
                        m = nav.next_maneuver
                        d = m.distance_m if m else 0
                        print(f"NAV: queued LEFT in {d:.0f}m (queue {nav.queue_len})")
                    elif key == KEY_PAGEDOWN and hk("nav_queue_right"):
                        nav.queue(ManeuverDir.RIGHT)
                        m = nav.next_maneuver
                        d = m.distance_m if m else 0
                        print(f"NAV: queued RIGHT in {d:.0f}m (queue {nav.queue_len})")
                    elif key == KEY_END and hk("nav_clear"):
                        nav.clear()
                        print("NAV: queue cleared")
                    elif key == ord("t") and pad is not None and hk("test_fire"):
                        # Test-fire works on either device — uses the
                        # public set_steering, which both Gamepad and
                        # Wheel implement. Force engaged for the test.
                        print("TEST FIRE: -0.5 (left) -> +0.5 (right) -> 0")
                        was_engaged = pad.engaged
                        if not was_engaged:
                            pad.engage()
                        pad.set_steering(-0.5)
                        time.sleep(0.5)
                        pad.set_steering(+0.5)
                        time.sleep(0.5)
                        pad.set_steering(0.0)
                        if not was_engaged:
                            pad.disengage()
                        print("  done. did the truck steer?")
                    else:
                        # Calibration-tweak keys: gate against the
                        # registry before calling overlay's handler so
                        # disabled FOV/pitch/etc nudges are ignored.
                        calib_hk_id = CALIB_KEY_HOTKEY_IDS.get(key)
                        if calib_hk_id and not hk(calib_hk_id):
                            continue
                        msg = handle_calibration_key(calib, key)
                        if msg is not None:
                            print(msg)
                if quit_loop:
                    break
        finally:
            if pad is not None:
                pad.disengage()
            live_params.save()
            calib.save(game=game)
            tel.close()
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
