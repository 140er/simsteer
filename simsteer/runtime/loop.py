"""The driving loop: capture -> warp -> model -> control -> device, with
the full overlay/HUD and the live tuning controls.

This is the game-agnostic heart of SimSteer. It owns the per-frame
pipeline, the engagement state machine, the model/capture overlay views,
the HUD, and the keyboard controls. The game-specific bits (telemetry,
wheelbase) come in through a `GameProfile`; the driving math lives in
`core/`. Capture, the model, and the output device are constructed once
and persist for the session.

Two things this loop fixes vs a naive port of v1's monolith:

  - **Overlay alignment.** The default view draws the model's predicted
    path onto the *warped model view* (`model_view_calib`), where the
    image and the model output share one calibrated frame, so the path
    lines up by construction. Drawing onto the raw capture (press `v`)
    depends on the static FOV/pitch being exactly right, which is the
    historical "path too small / not lined up" failure.
  - **Calibration.** `CalibFlow` walks FOV, Camera, Steering and a
    scored Check per game and device, and gates engagement until the
    camera is calibrated. See `simsteer/calib/flow.py`.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import replace

import cv2
import numpy as np

from simsteer.app import audio
from simsteer.app.global_keys import (
    GlobalKeys, KEY_END, KEY_INSERT, KEY_NUMPAD4, KEY_NUMPAD6,
    KEY_PAGEDOWN, KEY_PAGEUP,
)
from simsteer.app.hotkeys import hk_enabled as _hk_enabled
from simsteer.app.manual import ManualInputs
from simsteer.app.nav import ManeuverDir, NavManager
from simsteer.app.probe import SteeringProbe
from simsteer.app.settings import Settings
from simsteer.calib.flow import CalibFlow, Obs, Step, steering_fit_for
from simsteer.calib.fov_source import profile_reader
from simsteer.core.calibration import Calibration, model_view_calib
from simsteer.core.constants import DESIRE_LEN, T_IDXS
from simsteer.core.control import (
    ControllerConfig, LateralController, LongitudinalController,
)
from simsteer.core.learners.livecalib import CalStatus, LiveCalib
from simsteer.core.learners.liveparams import LiveParams
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue, model_views, yuv6_to_bgr
from simsteer.games.base import GameProfile, get_profile
from simsteer.runtime.capture import Capture, CaptureConfig
from simsteer.runtime.fov import FovSolver, model_vision
from simsteer.runtime.output.base import DeviceManager
from simsteer.runtime.replay import LiveClock
from simsteer.runtime.telemetry import Telemetry
from simsteer.ui.hud import (
    COL_ACCENT_BLUE, COL_ACCENT_GREEN, COL_ACCENT_RED, COL_ACCENT_VIOLET,
    HudRenderer, HudState,
)
from simsteer.ui.overlay import (
    WINDOW_NAME, draw_calibration_hud, draw_model_input_inset, draw_overlay,
    handle_calibration_key, setup_window, show_scaled,
)
from simsteer.ui.tuner import Tuner

try:
    from simsteer.version import __version__
except Exception:  # pragma: no cover - version file optional
    __version__ = "0.2.0"

# Desire one-hot indices (openpilot's order). Pulse one into
# model.step(desire=...) and the model's 5 s context buffer holds the
# rising edge, which drives its lane-change planning.
DESIRE_LANE_CHANGE_LEFT = 3
DESIRE_LANE_CHANGE_RIGHT = 4

KEY_BACKSPACE = 8
CALIB_TONE_COLORS = {"calibrating": COL_ACCENT_VIOLET,
                     "passed": COL_ACCENT_GREEN, "failed": COL_ACCENT_RED}

MIN_HEALTHY_FPS = 8.0
DISENGAGED_RECENTER_EVERY_N = 20
# DirectML compiles its kernels on the first few inferences (<5 fps);
# don't enforce the fps auto-disengage until past this many frames.
WARMUP_FRAMES = 20
# Until the frame queue holds distinct frames the model sees no motion.
FRAME_QUEUE_FILL = 5
# Telemetry detection is a one-shot handshake; retry this often (frames)
# while unavailable so it recovers without a restart.
TELEMETRY_RETRY_EVERY_N = 40

# Per-key hotkey-id lookups for the calib-tweak keys (gate against the
# disabled-hotkeys registry before calling the overlay handler).
CALIB_KEY_HOTKEY_IDS: dict[int, str] = {
    ord("["): "fov_minus", ord("]"): "fov_plus",
    ord(","): "pitch_minus", ord("."): "pitch_plus",
    ord(";"): "height_minus", ord("'"): "height_plus",
    ord("c"): "save_calib", ord("m"): "mirror_sign",
    ord("0"): "reset_calib_defaults",
}


class WindowKeys:
    """Keys from the overlay window, plus global keys when given."""

    def __init__(self, global_keys: GlobalKeys | None = None) -> None:
        self.global_keys = global_keys

    def poll(self) -> list[int]:
        pressed: list[int] = []
        cv_key = cv2.waitKey(1) & 0xFF
        if cv_key != 0xFF:
            pressed.append(cv_key)
        if self.global_keys is not None:
            pressed.extend(self.global_keys.poll())
        return pressed


class NoKeys:
    def poll(self) -> list[int]:
        return []


def _setup_window_or_exit(w: int, h: int) -> None:
    try:
        setup_window(w, h)
    except cv2.error as e:
        raise SystemExit(
            "this OpenCV build cannot open a window (opencv-python-headless?). "
            f"Run with --headless or install opencv-python. ({e.err})") from None


def run(settings: Settings, game: GameProfile | str, *,
        source=None,
        make_telemetry: Callable[[], Telemetry] | None = None,
        clock=None,
        keys=None,
        model=None,
        headless: bool = False,
        max_frames: int | None = None,
        log_path: str | None = None,
        log_timing: bool = False) -> int:
    """Run the driving loop for one pinned game until the user quits (q),
    the source runs out, or `max_frames` frames are done.

    Defaults are the live game: screen capture, the profile's telemetry,
    the wall clock, window plus global keys, and the ONNX model. Replay
    passes a `ReplaySource`, `ReplayTelemetry` and `ReplayClock` instead.
    `headless` opens no window, no tuner and reads no keys.

    Pinned-game only for now: the live `GameDetector` + `SessionManager`
    hot-swap arrives in Phase 2. Per-game state (calibration, controller
    config, liveparams, livecalib) is read from the same files v1 uses,
    so an existing v1 calibration carries over."""
    profile = get_profile(game) if isinstance(game, str) else game
    if profile is None:
        print(f"unknown game: {game!r}")
        return 2
    game_id = profile.id
    if source is None:
        source = Capture(CaptureConfig(target_fps=20))
    if make_telemetry is None:
        make_telemetry = profile.make_telemetry
    if clock is None:
        clock = LiveClock()

    # --- Per-game state (keyed by game/device, shared with v1) ---
    calib = Calibration.load(game_id, fov_h_deg=profile.camera.fov_h_deg)
    ctrl_cfg = ControllerConfig.load(game_id, seed=profile.controller)
    live_params = LiveParams(game=game_id, device_kind=settings.device,
                             seeds=profile.liveparams.seeds,
                             gates=profile.liveparams.gates,
                             clock=clock.now)
    lat_ctrl = LateralController(ctrl_cfg, live_params)
    long_ctrl = LongitudinalController(ctrl_cfg)
    live_calib = LiveCalib(game=game_id, clock=clock.now)

    # --- Session-wide, game-agnostic resources ---
    fq = FrameQueue()
    if model is None:
        model = DrivingModel()
    tel = make_telemetry()
    manual = ManualInputs()
    nav = NavManager()
    probe = SteeringProbe(enabled=not settings.no_probe)
    hud_renderer = HudRenderer()

    def passive_fit_allowed() -> bool:
        return profile.capabilities.passive_fit_safe or settings.passive_fit

    vision = model_vision(model)
    flow = CalibFlow(
        game_id, settings.device, live_calib=live_calib,
        live_params=live_params, calib=calib,
        steering_fit=steering_fit_for(profile),
        passive_fit=passive_fit_allowed(),
        make_fov_solver=lambda: FovSolver(vision, model_views),
        fov_config=profile_reader(profile.camera.fov_source))

    pad: DeviceManager | None = None
    if not settings.no_gamepad:
        pad = DeviceManager(initial_kind=settings.device,
                            vjoy_device_id=settings.vjoy_device)
        if pad.kind is None:
            print(f"device init failed: {pad.last_error} — overlay-only")

    # Tuner: the customtkinter slider window (FOV/pitch/height/control
    # knobs, Setup + Hotkeys tabs, Recalibrate). Runs in its own daemon
    # Tk thread; mutates calib/ctrl_cfg/manual which the loop reads each
    # frame. Set settings.no_tuner to skip.
    if not settings.no_tuner and not headless:
        try:
            Tuner(calib, ctrl_cfg, manual, live_params=live_params,
                  game=game_id, device=pad, settings=settings, probe=probe,
                  live_calib=live_calib, calib_flow=flow)
            print("tuner: slider window opened (set no_tuner to skip)")
        except Exception as e:
            print(f"tuner: failed to open ({e}) — continuing without it")

    print(f"vision: {getattr(model, 'active_provider', '?')}   "
          f"policy: {getattr(model, 'policy_provider', '?')}")
    print(f"game: {profile.display_name}   telemetry: "
          f"{'available' if tel.available else 'NOT detected'}")
    print(f"calibration: {flow.step.value}  camera {live_calib.status_label}  "
          f"(FOV={calib.fov_h_deg:.1f} pitch={calib.pitch_deg:+.1f} "
          f"h={calib.height_m:.2f})")
    print(f"hud: {settings.hud_mode!r} (H toggles)   "
          f"probe: {'on' if probe.enabled else 'off'}")
    print("keys: INSERT engage (accepts a measured FOV)  BACKSPACE keep FOV  "
          "v model/capture  i input  h hud  r recalibrate  "
          "k/l turn-in later/earlier  NumPad4/6 lane L/R  "
          "PgUp/PgDn/End nav  t test  q quit")

    show_input = True
    view_mode = "model"   # default to the self-aligning model view
    last = t_start = clock.now()
    fps = 0.0
    frame_idx = 0
    n_frames = 0
    window_open = False
    log_f = open(log_path, "w", encoding="utf-8") if log_path else None
    blocked_reason = ""
    lane_change_idx: int | None = None
    lane_change_until = 0.0
    banner_text = ""
    banner_color = (200, 200, 200)
    banner_until = 0.0
    was_engaged_last_frame = False
    last_device_axis: float | None = None
    was_measuring = False

    def engage_check() -> tuple[bool, str]:
        if not flow.ai_steering_ready:
            return False, flow.engage_block()
        if settings.force_engage:
            return True, ""
        if pad is None or pad.kind is None:
            return False, "no virtual gamepad (install ViGEm)"
        if not tel.available:
            return False, "game telemetry not detected"
        if fps < MIN_HEALTHY_FPS:
            return False, f"frame rate too low ({fps:.1f} < {MIN_HEALTHY_FPS:.0f})"
        why = flow.engage_block()
        if why:
            return False, why
        if live_calib.cal_status != CalStatus.CALIBRATED:
            return False, "camera calibration not done — drive manually first"
        return True, ""

    def set_banner(text: str, color: tuple[int, int, int], secs: float) -> None:
        nonlocal banner_text, banner_color, banner_until
        banner_text = text
        banner_color = color
        banner_until = clock.now() + secs

    def hk(hk_id: str) -> bool:
        return _hk_enabled(hk_id, settings)

    def restart_calibration(reason: str) -> None:
        if pad is not None and pad.engaged:
            pad.disengage()
            audio.play("disengage")
        elif pad is not None and flow.output_direction.active:
            pad.center()
        probe.reset()
        lat_ctrl.reset()
        flow.restart(reason)
        set_banner("CALIBRATION RESET", (80, 220, 255), 4.0)

    with source as cap:
        if keys is None and headless:
            keys = NoKeys()
        elif keys is None:
            watched = {KEY_INSERT, KEY_NUMPAD4, KEY_NUMPAD6,
                       KEY_PAGEUP, KEY_PAGEDOWN, KEY_END}
            keys = WindowKeys(GlobalKeys(watched)
                              if sys.platform == "win32" else None)

        if settings.force_engage and pad is not None and flow.ai_steering_ready:
            pad.engage()
            print("force-engage: pad engaged at startup (DEV — needs "
                  "calibration to actually track)")

        try:
            while max_frames is None or n_frames < max_frames:
                frame_idx += 1
                t_grab = time.perf_counter()
                frame = cap.grab()
                if frame is None:
                    if cap.done:
                        break
                    if ord("q") in keys.poll():
                        break
                    if not window_open:
                        # waitKey only sleeps once a window exists.
                        time.sleep(0.002)
                    continue
                n_frames += 1
                t_warp = time.perf_counter()
                restart_reason = flow.take_restart_request()
                if restart_reason:
                    restart_calibration(restart_reason)
                if live_params.device_kind != flow.device:
                    if pad is not None and pad.engaged:
                        pad.disengage()
                    flow.switch_device(live_params.device_kind)
                    lat_ctrl.reset()
                    long_ctrl.reset()
                    probe.reset()
                    last_device_axis = None
                    was_measuring = False
                if not headless and not window_open:
                    h, w = frame.shape[:2]
                    win_w = min(w, settings.max_width)
                    _setup_window_or_exit(win_w, int(round(h * win_w / w)))
                    cv2.setWindowTitle(
                        WINDOW_NAME,
                        f"SimSteer {__version__} — {profile.display_name}")
                    window_open = True

                # Recover telemetry if it wasn't live at launch.
                if (not tel.available
                        and frame_idx % TELEMETRY_RETRY_EVERY_N == 0):
                    tel.close()
                    tel = make_telemetry()

                # --- Telemetry reads ---
                v_real = tel.speed_mps()
                v_ego = v_real if v_real is not None else ctrl_cfg.default_speed
                actual_yaw = tel.yaw_rate_rad_s()
                wheel_angle = tel.wheel_angle_rad(ctrl_cfg.wheelbase_m)
                game_steer = tel.game_steer()
                user_steer = getattr(tel, "user_steer", lambda: None)()

                # --- Compose desire (nav + manual lane-change), step model ---
                img_narrow, img_wide = fq.push(frame, calib)
                now_perf = clock.now()
                dt = max(1e-3, now_perf - last)
                nav_desire_idx = nav.tick(v_ego=v_ego, dt=dt, now_t=now_perf)
                now_wall = now_perf
                if hasattr(tel, "nav_distance_m"):
                    nav.update_destination(tel.nav_distance_m(), tel.nav_time_s())
                lane_change_active = (lane_change_idx is not None
                                      and now_wall < lane_change_until)
                effective_idx = (lane_change_idx if lane_change_active
                                 else nav_desire_idx)
                if effective_idx is not None:
                    desire_vec = np.zeros(DESIRE_LEN, dtype=np.float32)
                    desire_vec[effective_idx] = 1.0
                else:
                    desire_vec = None
                    lane_change_idx = None
                t_model = time.perf_counter()
                vision_out, policy_out = model.step(img_narrow, img_wide,
                                                    desire=desire_vec)
                t_ctrl = time.perf_counter()
                decoded = decode(vision_out, policy_out)

                vx = (float(decoded.pose[0])
                      if decoded.pose is not None and len(decoded.pose) > 0
                      else None)

                # --- Calibration: FOV measurement, then camera pose ---
                fov_vision_s = 0.0
                if flow.step is Step.FOV and frame_idx > FRAME_QUEUE_FILL:
                    before = flow.fov_solver.vision_s
                    flow.feed_fov(frame, vx, v_real, actual_yaw,
                                  time.perf_counter)
                    fov_vision_s = flow.fov_solver.vision_s - before
                if flow.camera_learning:
                    live_calib.update(
                        calib, decoded.pose, decoded.road_transform,
                        actual_yaw, v_real,
                        pose_std=decoded.pose_std,
                        road_transform_std=decoded.road_transform_std,
                        wide_from_device_euler=decoded.wide_from_device_euler,
                        game_steer=game_steer)

                # --- Lateral + longitudinal control ---
                output_fault = flow.observe_output(
                    command=last_device_axis,
                    observed=game_steer if tel.available else None,
                    speed=v_real, engaged=bool(pad and pad.engaged),
                    manual_override=manual.steer_override,
                    user_axis=user_steer,
                    lane_change=lane_change_active or lat_ctrl.last_in_lane_change,
                    dt=dt)
                if output_fault and pad is not None:
                    pad.disengage()
                    lat_ctrl.reset()
                    long_ctrl.reset()
                    probe.reset()
                    audio.play("disengage")
                    set_banner("DISENGAGED — output direction changed", COL_ACCENT_BLUE, 4.0)
                ai_steer = lat_ctrl.compute(
                    decoded, v_ego, actual_wheel_angle=wheel_angle,
                    lane_change_command_active=lane_change_active, dt=dt,
                    wheel_sign=flow.steering_sign)
                ai_steer = flow.device_axis(ai_steer)
                steer = manual.steer if manual.steer_override else ai_steer
                if pad is not None and flow.ai_steering_ready and not manual.steer_override:
                    steer += probe.tick(
                        dt=dt, ai_axis=steer, v_ego=v_real,
                        in_lane_change=lane_change_active,
                        engaged=pad.engaged,
                        lp_trusted=live_params.trusted(),
                        calibrating=flow.probe_wanted)
                ai_throttle, ai_brake = long_ctrl.compute(decoded, v_ego)
                if not flow.ai_steering_ready:
                    ai_throttle = ai_brake = 0.0
                if manual.long_override:
                    throttle_out, brake_out = manual.throttle, manual.brake
                else:
                    throttle_out, brake_out = ai_throttle, ai_brake

                # --- Feed LiveParams (after the command is known) ---
                # Disengaged, the observed axis is the player's, so only
                # profiles whose rack response doesn't depend on the input
                # device fit from it. Replay has no pad and fits here too.
                engaged_now = bool(pad is not None and pad.engaged)
                flow.passive_fit = passive_fit_allowed()
                if (engaged_now and flow.ai_steering_ready) or (not engaged_now and flow.passive_fit):
                    # LiveParams fits game input units, never device units.
                    cmd = (steer * (flow.output_direction.sign if flow.steering_fit else 1)
                           if engaged_now else game_steer)
                    live_params.update(game_steer, v_real, wheel_angle,
                                       commanded_axis=cmd,
                                       gate_override=flow.gate_override)

                # --- Device output ---
                last_device_axis = None
                if pad is not None:
                    measuring = flow.output_direction.active
                    if measuring:
                        # Verification is steering only, with AI disengaged.
                        # INSERT explicitly starts it; --no-probe only disables
                        # the later rack-fitting perturbation.
                        steer = flow.measure_output(
                            speed=v_real if tel.available else None,
                            observed=game_steer, user_axis=user_steer, dt=dt,
                            manual_override=manual.steer_override or pad.engaged
                            or lane_change_active)
                        pad.set_steering(steer, force=True)
                        throttle_out = brake_out = 0.0
                        if not flow.output_direction.active:
                            pad.center()
                            lat_ctrl.reset()
                    elif was_measuring:
                        pad.center()
                    elif pad.engaged:
                        pad.set_steering(steer)
                        last_device_axis = steer
                        pad.set_throttle_brake(throttle_out, brake_out)
                    else:
                        if frame_idx % DISENGAGED_RECENTER_EVERY_N == 0:
                            pad.center()
                        if manual.steer_override:
                            pad.set_steering(manual.steer, force=True)
                        if manual.long_override:
                            pad.set_throttle_brake(manual.throttle,
                                                   manual.brake, force=True)
                    was_measuring = flow.output_direction.active

                # --- FPS + auto-disengage (with DML-warmup grace) ---
                now = clock.now()
                fps = 1.0 / max(now - last, 1e-3)
                last = now
                if (pad is not None and pad.engaged and fps < MIN_HEALTHY_FPS
                        and frame_idx > WARMUP_FRAMES):
                    pad.disengage()
                    lat_ctrl.reset()
                    long_ctrl.reset()
                    audio.play("disengage")
                    set_banner(f"DISENGAGED — frame rate dropped ({fps:.1f} fps)",
                               COL_ACCENT_BLUE, 3.0)

                engaged_now = bool(pad is not None and pad.engaged)
                if was_engaged_last_frame and not engaged_now and not banner_text:
                    audio.play("disengage")
                    set_banner("DISENGAGED", COL_ACCENT_BLUE, 2.0)
                was_engaged_last_frame = engaged_now
                direction_was_ready = flow.direction.ready
                flow.tick(Obs(
                    now=now, dt=dt, v=v_real, yaw_rate=actual_yaw,
                    wheel=wheel_angle, vx_model=vx,
                    observed_axis=game_steer,
                    model_curvature=lat_ctrl.last_curvature,
                    lane_offset=_lane_offset(decoded), axis=float(steer),
                    driving=engaged_now or pad is None, engaged=engaged_now,
                    lane_change=(lane_change_active
                                 or lat_ctrl.last_in_lane_change),
                    telemetry_ok=tel.available))
                if flow.direction.ready and not direction_was_ready:
                    lat_ctrl.reset()
                if (not flow.ai_steering_ready
                        and pad is not None and pad.engaged):
                    pad.disengage()
                    lat_ctrl.reset()
                    long_ctrl.reset()
                allowed, blocked_reason = engage_check()
                t_render = time.perf_counter()

                # --- Keys (game-focused global + overlay-focused cv2) ---
                # Read before rendering, so headless replay with scripted
                # keys takes the same path as the window.
                quit_loop = False
                for key in keys.poll():
                    if key == ord("q") and hk("quit"):
                        quit_loop = True
                        break
                    if key == ord("v") and hk("view_toggle"):
                        view_mode = "capture" if view_mode == "model" else "model"
                        print(f"view: {view_mode}")
                    elif key == ord("i") and hk("input_inset"):
                        show_input = not show_input
                    elif key == ord("h") and hk("hud_mode"):
                        settings.hud_mode = ("dev" if settings.hud_mode == "user"
                                             else "user")
                        settings.save()
                        print(f"hud: {settings.hud_mode!r}")
                    elif key == ord("r") and hk("recalibrate"):
                        restart_calibration("R pressed")
                    elif key == KEY_INSERT and flow.accept_fov():
                        set_banner(f"FOV {calib.fov_h_deg:.0f}", COL_ACCENT_GREEN, 3.0)
                    elif key == KEY_BACKSPACE and hk("keep_fov") and flow.keep_fov():
                        set_banner(f"FOV {calib.fov_h_deg:.0f} kept",
                                   COL_ACCENT_BLUE, 3.0)
                    elif key == KEY_INSERT and pad is not None and hk("engage"):
                        if (flow.step is Step.STEERING and flow.direction.ready
                                and not flow.output_direction.ready and not pad.engaged):
                            output = flow.output_direction
                            if output.active:
                                output.cancel("Press INSERT to check steering direction")
                                pad.center()
                            elif pad.kind is not None:
                                output.start(speed=v_real if tel.available else None,
                                             observed=game_steer, user_axis=user_steer,
                                             manual_override=manual.steer_override)
                                pad.center()
                            probe.reset()
                            lat_ctrl.reset()
                        elif pad.engaged:
                            pad.disengage()
                            lat_ctrl.reset()
                            long_ctrl.reset()
                            audio.play("disengage")
                            set_banner("DISENGAGED", COL_ACCENT_BLUE, 1.5)
                            print("-> DISENGAGED")
                        else:
                            ok, why = engage_check()
                            if not ok:
                                audio.play("denied")
                                set_banner(f"CANNOT ENGAGE — {why}",
                                           COL_ACCENT_BLUE, 3.0)
                                print(f"-> ENGAGE BLOCKED: {why}")
                            else:
                                pad.engage()
                                lat_ctrl.reset()
                                audio.play("engage")
                                set_banner("ENGAGED", COL_ACCENT_GREEN, 1.5)
                                print("-> ENGAGED")
                    elif key == KEY_NUMPAD4 and hk("lane_change_left"):
                        lane_change_idx = DESIRE_LANE_CHANGE_LEFT
                        lane_change_until = now + ctrl_cfg.lane_change_hold_s
                        print("lane change LEFT commanded")
                    elif key == KEY_NUMPAD6 and hk("lane_change_right"):
                        lane_change_idx = DESIRE_LANE_CHANGE_RIGHT
                        lane_change_until = now + ctrl_cfg.lane_change_hold_s
                        print("lane change RIGHT commanded")
                    elif key == KEY_PAGEUP and hk("nav_queue_left"):
                        nav.queue(ManeuverDir.LEFT)
                        print(f"NAV: queued LEFT (queue {nav.queue_len})")
                    elif key == KEY_PAGEDOWN and hk("nav_queue_right"):
                        nav.queue(ManeuverDir.RIGHT)
                        print(f"NAV: queued RIGHT (queue {nav.queue_len})")
                    elif key == KEY_END and hk("nav_clear"):
                        nav.clear()
                        print("NAV: queue cleared")
                    elif key == ord("t") and pad is not None and hk("test_fire"):
                        was = pad.engaged
                        if not was:
                            pad.engage()
                        pad.set_steering(-0.5)
                        time.sleep(0.4)
                        pad.set_steering(+0.5)
                        time.sleep(0.4)
                        pad.set_steering(0.0)
                        if not was:
                            pad.disengage()
                        print("test-fire done — did the truck steer?")
                    elif key == ord("k"):
                        ctrl_cfg.curvature_anticipation_s = round(max(
                            -0.4, ctrl_cfg.curvature_anticipation_s - 0.05), 2)
                        print("turn-in LATER: anticipation="
                              f"{ctrl_cfg.curvature_anticipation_s:+.2f}s "
                              f"(lead {ctrl_cfg.lookahead_s + ctrl_cfg.curvature_anticipation_s:.2f}s)")
                    elif key == ord("l"):
                        ctrl_cfg.curvature_anticipation_s = round(min(
                            0.6, ctrl_cfg.curvature_anticipation_s + 0.05), 2)
                        print("turn-in EARLIER: anticipation="
                              f"{ctrl_cfg.curvature_anticipation_s:+.2f}s "
                              f"(lead {ctrl_cfg.lookahead_s + ctrl_cfg.curvature_anticipation_s:.2f}s)")
                    else:
                        calib_hk_id = CALIB_KEY_HOTKEY_IDS.get(key)
                        if calib_hk_id and not hk(calib_hk_id):
                            continue
                        msg = handle_calibration_key(calib, key)
                        if msg is not None:
                            print(msg)

                if log_f is not None:
                    row = {
                        "frame": n_frames,
                        "t": round(now - t_start, 4),
                        "v_ego": round(float(v_ego), 4),
                        "steer_axis": round(float(steer), 6),
                        "ai_steer_axis": round(float(ai_steer), 6),
                        "curvature": round(float(lat_ctrl.last_curvature), 7),
                        "target_wheel": round(float(lat_ctrl.last_target_wheel), 7),
                        "wheel_angle": wheel_angle,
                        "observed_axis": game_steer,
                        "throttle": round(float(throttle_out), 4),
                        "brake": round(float(brake_out), 4),
                        "engaged": engaged_now,
                        "telemetry": "ok" if tel.available else "lost",
                        **flow.log_fields(),
                    }
                    if log_timing:
                        # The FOV step's extra vision runs are model time.
                        row.update(
                            cap_ms=round((t_warp - t_grab) * 1e3, 3),
                            warp_ms=round((t_model - t_warp) * 1e3, 3),
                            model_ms=round((t_ctrl - t_model + fov_vision_s) * 1e3, 3),
                            ctrl_ms=round((t_render - t_ctrl - fov_vision_s) * 1e3, 3),
                            frame_ms=round((t_render - t_grab) * 1e3, 3))
                    log_f.write(json.dumps(row) + "\n")

                if quit_loop:
                    break
                if headless:
                    continue

                # --- Render: model view (self-aligning) or capture view ---
                # BOTH views project the model's path using the height the
                # MODEL assumes (road_transform[2], ~1.2 m), NOT our physical
                # cab height. comma's model predicts road geometry at its
                # trained height; projecting with the taller calib.height_m
                # (~2.1 m, what LiveCalib estimates for the real mount) pushes
                # the near path off the bottom of the frame and squishes the
                # rest toward the horizon — the "path too small / converging
                # low" bug. Fall back to LiveCalib's estimate, then calib.
                rt = decoded.road_transform
                model_h = (float(rt[2]) if rt is not None and len(rt) >= 3
                           and 0.5 < float(rt[2]) < 4.0
                           else (live_calib.height_estimate or calib.height_m))
                if view_mode == "model" and fq.last_yuv is not None:
                    mv_bgr = yuv6_to_bgr(fq.last_yuv)
                    mv_h = 720
                    mv_w = mv_h * mv_bgr.shape[1] // mv_bgr.shape[0]
                    mv_bgr = cv2.resize(mv_bgr, (mv_w, mv_h),
                                        interpolation=cv2.INTER_LINEAR)
                    mv_calib = model_view_calib(
                        calib,
                        captured_shape=fq.last_captured_shape or frame.shape,
                        cropped_shape=fq.last_cropped_shape or frame.shape,
                        view_w=mv_w, view_h=mv_h, model_height_m=model_h)
                    overlay = draw_overlay(mv_bgr, decoded, mv_calib)
                    if show_input:
                        small = cv2.resize(frame, (mv_w // 3, mv_h // 3),
                                           interpolation=cv2.INTER_AREA)
                        H, W = overlay.shape[:2]
                        sh, sw = small.shape[:2]
                        overlay[10:10 + sh, W - 10 - sw:W - 10] = small
                else:
                    cap_calib = replace(calib, height_m=model_h)
                    overlay = draw_overlay(frame, decoded, cap_calib)
                    if show_input:
                        draw_model_input_inset(overlay, fq.last_yuv_narrow,
                                               fq.last_yuv_wide)

                # --- HUD ---
                hud_state = _build_hud_state(
                    pad=pad, view_mode=view_mode, fps=fps, v_real=v_real,
                    v_ego=v_ego, settings=settings, banner_text=banner_text,
                    banner_color=banner_color, banner_until=banner_until,
                    blocked_reason=blocked_reason, lane_change_idx=lane_change_idx,
                    lane_change_active=lane_change_active, manual=manual,
                    probe=probe, lat_ctrl=lat_ctrl, long_ctrl=long_ctrl,
                    live_params=live_params, live_calib=live_calib,
                    ctrl_cfg=ctrl_cfg, flow=flow,
                    decoded=decoded, actual_yaw=actual_yaw, wheel_angle=wheel_angle,
                    game_steer=game_steer, steer=steer, ai_steer=ai_steer,
                    ai_throttle=ai_throttle, ai_brake=ai_brake,
                    throttle_out=throttle_out, brake_out=brake_out, nav=nav,
                    now=now)
                hud_renderer.draw(overlay, hud_state, mode=settings.hud_mode)
                draw_calibration_hud(overlay, calib)
                show_scaled(overlay, settings.max_width)

        finally:
            if pad is not None:
                pad.disengage()
            live_params.save()
            live_calib.save_state(game_id)
            calib.save(game=game_id)
            flow.save()
            tel.close()
            if log_f is not None:
                log_f.close()
            if window_open:
                cv2.destroyAllWindows()
    print(f"frames: {n_frames}")
    return 0


def _build_hud_state(*, pad, view_mode, fps, v_real, v_ego, settings,
                     banner_text, banner_color, banner_until, blocked_reason,
                     lane_change_idx, lane_change_active, manual, probe,
                     lat_ctrl, long_ctrl, live_params, live_calib, ctrl_cfg,
                     flow, decoded, actual_yaw, wheel_angle, game_steer,
                     steer, ai_steer, ai_throttle, ai_brake, throttle_out,
                     brake_out, nav, now) -> HudState:
    """Assemble both the clean user-mode HudState fields and the dev-mode
    text stack. Kept out of the hot loop body for readability."""
    if pad is None:
        eng_label, eng_col = "NO GAMEPAD", COL_ACCENT_RED
    elif pad.engaged:
        eng_label, eng_col = "ENGAGED", COL_ACCENT_GREEN
    else:
        eng_label, eng_col = "DISENGAGED", COL_ACCENT_BLUE

    cam_ready = live_calib.cal_status == CalStatus.CALIBRATED

    banner = (banner_text if banner_text and now < banner_until else "")

    lc_chip = None
    if lane_change_active:
        lc_chip = "L" if lane_change_idx == DESIRE_LANE_CHANGE_LEFT else "R"

    # Compact dev stack — the diagnostics power users expect (H toggles).
    sc = (255, 200, 80) if manual.steer_override else (255, 255, 255)
    dev_lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"{eng_label}   view={view_mode}   {fps:5.1f} fps   "
         f"v={v_ego * 3.6:5.1f} km/h", eng_col),
    ]
    if flow.calibrating:
        dev_lines.append((f"CALIB {flow.step.value} {flow.progress * 100:.0f}%  "
                          f"{flow.hint(v_real, bool(pad and pad.engaged))}",
                          (80, 255, 255)))
    dev_lines += [
        (f"steer cmd: {steer:+.3f} [{'MANUAL' if manual.steer_override else 'AI'}]"
         f"   target wheel: {lat_ctrl.last_target_wheel:+.4f} rad   "
         f"auth={lat_ctrl.last_authority:.2f}"
         f"{'  +LC' if lat_ctrl.last_in_lane_change else ''}", sc),
        (f"  axis trim={lat_ctrl.axis_trim_state:+.4f} bias={ctrl_cfg.axis_bias:+.3f}"
         f"  trim={'FROZEN[' + lat_ctrl.last_trim_frozen_reason + ']' if lat_ctrl.last_trim_frozen_reason else 'ACTIVE'}",
         (180, 200, 255)),
        (f"throttle {throttle_out:.2f} brake {brake_out:.2f}   "
         f"v_tgt={long_ctrl.last_v_target:5.1f} a_cmd={long_ctrl.last_a_cmd:+.2f}"
         f"{'  AEB!' if long_ctrl.last_aeb else ''}", (200, 200, 255)),
        (f"corner v_safe="
         f"{'inf' if long_ctrl.last_v_safe_corner == float('inf') else f'{long_ctrl.last_v_safe_corner:.1f}'}"
         f"  ACC prob={long_ctrl.last_lead_prob:.2f}", (160, 200, 255)),
        (nav.hud_line(), (180, 220, 180) if nav.queue_len > 0 else (140, 140, 140)),
        (f"plan k={lat_ctrl.last_curvature:+.5f}  yaw pred/act "
         f"{pred_yaw_safe(decoded, ctrl_cfg):+.3f}/"
         f"{actual_yaw if actual_yaw is not None else 0:+.3f}", (255, 255, 255)),
        (f"wheel {wheel_angle if wheel_angle is not None else 0:+.3f} rad   "
         f"gameSteer {game_steer if game_steer is not None else 0:+.3f}",
         (255, 255, 255)),
        (f"LIVEPARAMS a={live_params.a_linear:.2f} b={live_params.b_quad:.4f} "
         f"c={live_params.c_bias:+.3f}  n={live_params.session_samples}  "
         f"{live_params.trust_level}",
         (80, 255, 80) if live_params.trusted() else (200, 200, 80)),
        (f"LIVECALIB {live_calib.status_label}  blocks={live_calib.blocks}  "
         f"fill={live_calib.block_progress * 100:.0f}%  "
         f"pitch={live_calib.pitch_estimate or 0:+.2f} "
         f"h={live_calib.height_estimate or 0:.2f}m",
         (80, 255, 80) if cam_ready else (200, 200, 80)),
        (f"FOV {calib_fov_line(flow)}",
         (160, 200, 255) if flow.step is Step.FOV else (80, 255, 80)),
        (f"{fps:5.1f} fps   SimSteer v{__version__}", (255, 255, 255)),
    ]
    if blocked_reason and not (pad and pad.engaged):
        dev_lines.insert(1, (f"cannot engage: {blocked_reason}", COL_ACCENT_BLUE))

    panel = flow.panel(v_real, engaged=bool(pad and pad.engaged))

    return HudState(
        engaged=bool(pad and pad.engaged),
        engaged_label=eng_label,
        engaged_color=eng_col,
        banner_text=banner,
        banner_color=banner_color,
        calib_text=panel.title if panel else "",
        calib_color=(CALIB_TONE_COLORS[panel.tone] if panel
                     else COL_ACCENT_VIOLET),
        calib_progress=panel.progress if panel else 0.0,
        calib_hint=panel.hint if panel else "",
        warnings=[],
        v_ego_mps=v_real,
        probe_active=(probe.last_gate == "active"),
        manual_override=(manual.steer_override or manual.long_override),
        lane_change=lc_chip,
        lead_following=(long_ctrl.last_lead_prob > ctrl_cfg.lead_min_prob),
        aeb=bool(long_ctrl.last_aeb),
        fps=fps,
        version=__version__,
        mode_hint="H: dev view" if settings.hud_mode == "user" else "H: user view",
        dev_lines=dev_lines,
    )


def calib_fov_line(flow: CalibFlow) -> str:
    solver = flow.fov_solver
    if flow.step is Step.FOV:
        if flow.fov_proposal is not None:
            return f"proposed {flow.fov_proposal:.1f}"
        est = f" est={solver.estimate:.1f}" if solver.estimate else ""
        cands = (" cand=" + ",".join(f"{c:.0f}" for c in solver.candidates)
                 if solver.candidates else "")
        return f"{solver.stage} {solver.samples} samples{est}{cands}"
    return f"{flow.calib.fov_h_deg:.1f} ({flow.fov_source or 'saved'})"


def _lane_offset(decoded) -> float | None:
    """Metres from the ego lane centre, when both ego lines are seen."""
    prob = decoded.lane_lines_prob
    if prob is None or len(prob) < 3 or prob[1] < 0.5 or prob[2] < 0.5:
        return None
    return -float(decoded.lane_lines[1, 0, 0] + decoded.lane_lines[2, 0, 0]) / 2.0


def pred_yaw_safe(decoded, ctrl_cfg) -> float:
    return float(np.interp(ctrl_cfg.lookahead_s,
                           np.asarray(T_IDXS, dtype=np.float32),
                           decoded.plan[:, 14]))
