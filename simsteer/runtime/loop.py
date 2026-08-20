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
  - **Auto-FOV.** `FovResolver` solves the true FOV once from
    `vx_model / v_ego` on a straight (open-loop, then frozen) so a wrong
    stored FOV self-corrects instead of needing a manual slider.

Onboarding wizard, the Tk tuner, and the preflight installer dialogs are
the remaining Phase 4 surfaces and are intentionally not wired here yet;
everything the user needs to drive and tune live is.
"""

from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np

from simsteer.app import audio
from simsteer.app.calibration_routine import CalibrationRoutine
from simsteer.app.global_keys import (
    GlobalKeys, KEY_END, KEY_INSERT, KEY_NUMPAD4, KEY_NUMPAD6,
    KEY_PAGEDOWN, KEY_PAGEUP,
)
from simsteer.app.hotkeys import hk_enabled as _hk_enabled
from simsteer.app.manual import ManualInputs
from simsteer.app.nav import ManeuverDir, NavManager
from simsteer.app.probe import SteeringProbe
from simsteer.app.settings import Settings
from simsteer.app.wizard import Wizard
from simsteer.core.calibration import Calibration, model_view_calib
from simsteer.core.constants import DESIRE_LEN, T_IDXS
from simsteer.core.control import (
    ControllerConfig, LateralController, LongitudinalController,
)
from simsteer.core.learners.livecalib import (
    BLOCK_SIZE, INPUTS_NEEDED, CalStatus, LiveCalib,
)
from simsteer.core.learners.liveparams import LiveParams, TRUSTED_MIN_SAMPLES
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue, yuv6_to_bgr
from simsteer.games.base import GameProfile, get_profile
from simsteer.runtime.capture import Capture, CaptureConfig
from simsteer.runtime.fov import FovResolver
from simsteer.runtime.output.base import DeviceManager
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

MIN_HEALTHY_FPS = 8.0
DISENGAGED_RECENTER_EVERY_N = 20
# DirectML compiles its kernels on the first few inferences (<5 fps);
# don't enforce the fps auto-disengage until past this many frames.
WARMUP_FRAMES = 20
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


def run(settings: Settings, game_id: str = "ets2") -> int:
    """Run the driving loop for one pinned game until the user quits (q).

    Pinned-game only for now: the live `GameDetector` + `SessionManager`
    hot-swap arrives in Phase 2. Per-game state (calibration, controller
    config, liveparams, livecalib) is read from the same files v1 uses,
    so an existing v1 calibration carries over."""
    profile: GameProfile | None = get_profile(game_id)
    if profile is None:
        print(f"unknown game: {game_id!r}")
        return 2

    # --- Per-game state (keyed by game/device, shared with v1) ---
    calib = Calibration.load(game_id)
    ctrl_cfg = ControllerConfig.load(game_id)
    live_params = LiveParams(game=game_id, device_kind=settings.device)
    lat_ctrl = LateralController(ctrl_cfg, live_params)
    long_ctrl = LongitudinalController(ctrl_cfg)
    live_calib = LiveCalib(game=game_id)

    # --- Session-wide, game-agnostic resources ---
    fq = FrameQueue()
    model = DrivingModel()
    tel = profile.make_telemetry()
    manual = ManualInputs()
    nav = NavManager()
    probe = SteeringProbe(enabled=not settings.no_probe)
    cal_routine = CalibrationRoutine(live_params)
    # Auto-FOV: enabled by default. Solves FOV once from vx_model/v_ego
    # on a straight (60 samples, ~30-60s of highway), then freezes.
    # One-shot open-loop solve is safe; it cannot run away like the old
    # continuous LiveFov. User can disable via settings.auto_fov = False.
    enable_auto_fov = getattr(settings, "auto_fov", True)
    fov_resolver = FovResolver(enabled=enable_auto_fov)
    hud_renderer = HudRenderer()
    # First-drive onboarding: gates engage until camera calibration is
    # done and drives the HUD's wizard panel. Inert once a game is
    # already calibrated (wizard.active == False).
    wizard = Wizard(game=game_id, live_calib=live_calib, live_params=live_params)

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
    if not settings.no_tuner:
        try:
            Tuner(calib, ctrl_cfg, manual, live_params=live_params,
                  game=game_id, device=pad, settings=settings, probe=probe,
                  live_calib=live_calib, wizard=wizard, cal_routine=cal_routine)
            print("tuner: slider window opened (set no_tuner to skip)")
        except Exception as e:
            print(f"tuner: failed to open ({e}) — continuing without it")

    print(f"vision: {model.active_provider}   policy: {model.policy_provider}")
    print(f"game: {profile.display_name}   telemetry: "
          f"{'available' if tel.available else 'NOT detected'}")
    print(f"calibration: {live_calib.status_label}  "
          f"(FOV={calib.fov_h_deg:.1f} pitch={calib.pitch_deg:+.1f} "
          f"h={calib.height_m:.2f})")
    print(f"auto-FOV: {'enabled (will measure on highway)' if enable_auto_fov else 'disabled'}")
    print(f"hud: {settings.hud_mode!r} (H toggles)   "
          f"probe: {'on' if probe.enabled else 'off'}")
    print("keys: INSERT engage  v model/capture  i input  h hud  r recal  "
          "k/l turn-in later/earlier  NumPad4/6 lane L/R  "
          "PgUp/PgDn/End nav  t test  q quit")

    show_input = True
    view_mode = "model"   # default to the self-aligning model view
    last = time.perf_counter()
    fps = 0.0
    frame_idx = 0
    blocked_reason = ""
    lane_change_idx: int | None = None
    lane_change_until = 0.0
    banner_text = ""
    banner_color = (200, 200, 200)
    banner_until = 0.0
    was_engaged_last_frame = False

    def engage_check() -> tuple[bool, str]:
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
            return False, "camera calibration not done — drive manually first"
        return True, ""

    def set_banner(text: str, color: tuple[int, int, int], secs: float) -> None:
        nonlocal banner_text, banner_color, banner_until
        banner_text = text
        banner_color = color
        banner_until = time.time() + secs

    with Capture(CaptureConfig(target_fps=20)) as cap:
        first = None
        for _ in range(50):
            first = cap.grab()
            if first is not None:
                break
            time.sleep(0.05)
        if first is not None:
            h, w = first.shape[:2]
            win_w = min(w, settings.max_width)
            setup_window(win_w, int(round(h * win_w / w)))
            cv2.setWindowTitle(WINDOW_NAME,
                               f"SimSteer {__version__} — {profile.display_name}")
        else:
            setup_window(settings.max_width, settings.max_width * 9 // 16)

        watched = {KEY_INSERT, KEY_NUMPAD4, KEY_NUMPAD6,
                   KEY_PAGEUP, KEY_PAGEDOWN, KEY_END}
        global_keys = GlobalKeys(watched)

        if settings.force_engage and pad is not None:
            pad.engage()
            print("force-engage: pad engaged at startup (DEV — needs "
                  "calibration to actually track)")

        try:
            while True:
                frame_idx += 1
                frame = cap.grab()
                if frame is None:
                    keys = [cv2.waitKey(1) & 0xFF, *global_keys.poll()]
                    if ord("q") in keys:
                        break
                    continue

                # Recover telemetry if it wasn't live at launch.
                if (not tel.available
                        and frame_idx % TELEMETRY_RETRY_EVERY_N == 0):
                    tel.close()
                    tel = profile.make_telemetry()

                # --- Telemetry reads ---
                v_real = tel.speed_mps()
                v_ego = v_real if v_real is not None else ctrl_cfg.default_speed
                actual_yaw = tel.yaw_rate_rad_s()
                wheel_angle = tel.wheel_angle_rad(ctrl_cfg.wheelbase_m)
                game_steer = tel.game_steer()

                # --- Compose desire (nav + manual lane-change), step model ---
                img_narrow, img_wide = fq.push(frame, calib)
                now_perf = time.perf_counter()
                dt = max(1e-3, now_perf - last)
                nav_desire_idx = nav.tick(v_ego=v_ego, dt=dt, now_t=now_perf)
                now_wall = time.time()
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
                vision_out, policy_out = model.step(img_narrow, img_wide,
                                                    desire=desire_vec)
                decoded = decode(vision_out, policy_out)

                # --- Online camera-pose calibration ---
                live_calib.update(
                    calib, decoded.pose, decoded.road_transform,
                    actual_yaw, v_real,
                    pose_std=decoded.pose_std,
                    road_transform_std=decoded.road_transform_std,
                    wide_from_device_euler=decoded.wide_from_device_euler,
                    game_steer=game_steer)
                wizard.tick()

                # --- Auto-FOV (one-shot, open-loop, then frozen) ---
                if frame_idx > WARMUP_FRAMES:
                    vx = (float(decoded.pose[0])
                          if decoded.pose is not None and len(decoded.pose) > 0
                          else None)
                    fov_msg = fov_resolver.update(calib, vx, v_real, actual_yaw)
                    if fov_msg:
                        calib.save(game=game_id)
                        set_banner(fov_msg, (80, 220, 255), 5.0)
                        print(f"-> {fov_msg} (saved)")

                # --- Lateral + longitudinal control ---
                ai_steer = lat_ctrl.compute(
                    decoded, v_ego, actual_wheel_angle=wheel_angle,
                    lane_change_command_active=lane_change_active, dt=dt)
                ai_steer *= calib.lateral_sign
                steer = manual.steer if manual.steer_override else ai_steer
                if pad is not None and not manual.steer_override:
                    steer += probe.tick(
                        dt=dt, ai_axis=steer, v_ego=v_real,
                        in_lane_change=lane_change_active,
                        engaged=pad.engaged, lp_trusted=live_params.trusted(),
                        wizard_steering_phase=False)
                ai_throttle, ai_brake = long_ctrl.compute(decoded, v_ego)
                if manual.long_override:
                    throttle_out, brake_out = manual.throttle, manual.brake
                else:
                    throttle_out, brake_out = ai_throttle, ai_brake

                # --- Feed LiveParams (after the command is known) ---
                cal_routine.tick(v_real, wheel_angle, dt)
                allow_passive = (game_id in ("ac", "forza")
                                 or (game_id == "ets2"
                                     and settings.passive_fit_ets2)
                                 or wizard.in_camera_phase)
                if pad is not None and (pad.engaged or allow_passive):
                    cmd = steer if pad.engaged else game_steer
                    live_params.update(game_steer, v_real, wheel_angle,
                                       commanded_axis=cmd,
                                       gate_override=cal_routine.gate_override())

                # --- Device output ---
                if pad is not None:
                    if pad.engaged:
                        pad.set_steering(steer)
                        pad.set_throttle_brake(throttle_out, brake_out)
                    else:
                        if frame_idx % DISENGAGED_RECENTER_EVERY_N == 0:
                            pad.center()
                        if manual.steer_override:
                            pad.set_steering(manual.steer, force=True)
                        if manual.long_override:
                            pad.set_throttle_brake(manual.throttle,
                                                   manual.brake, force=True)

                # --- FPS + auto-disengage (with DML-warmup grace) ---
                now = time.perf_counter()
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
                allowed, blocked_reason = engage_check()

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
                    ctrl_cfg=ctrl_cfg, fov_resolver=fov_resolver,
                    decoded=decoded, actual_yaw=actual_yaw, wheel_angle=wheel_angle,
                    game_steer=game_steer, steer=steer, ai_steer=ai_steer,
                    ai_throttle=ai_throttle, ai_brake=ai_brake,
                    throttle_out=throttle_out, brake_out=brake_out, nav=nav,
                    cal_routine=cal_routine, wizard=wizard)
                hud_renderer.draw(overlay, hud_state, mode=settings.hud_mode)
                draw_calibration_hud(overlay, calib)
                show_scaled(overlay, settings.max_width)

                # --- Keys (game-focused global + overlay-focused cv2) ---
                pressed: list[int] = []
                cv_key = cv2.waitKey(1) & 0xFF
                if cv_key != 0xFF:
                    pressed.append(cv_key)
                pressed.extend(global_keys.poll())

                def hk(hk_id: str) -> bool:
                    return _hk_enabled(hk_id, settings)

                quit_loop = False
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
                        settings.hud_mode = ("dev" if settings.hud_mode == "user"
                                             else "user")
                        settings.save()
                        print(f"hud: {settings.hud_mode!r}")
                    elif key == ord("r") and hk("recalibrate"):
                        if pad is not None and pad.engaged:
                            pad.disengage()
                            audio.play("disengage")
                        live_calib.reset(reason="user pressed R")
                        live_params.reset(wipe_disk=True)
                        probe.reset()
                        lat_ctrl.reset()
                        wizard.reset()
                        fov_resolver = FovResolver(enabled=False)
                        set_banner("CALIBRATION RESET", (80, 220, 255), 4.0)
                        print("-> CALIBRATION RESET (LiveCalib + LiveParams "
                              "+ probe + auto-FOV)")
                    elif key == KEY_INSERT and pad is not None and hk("engage"):
                        if pad.engaged:
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
                        lane_change_until = time.time() + ctrl_cfg.lane_change_hold_s
                        print("lane change LEFT commanded")
                    elif key == KEY_NUMPAD6 and hk("lane_change_right"):
                        lane_change_idx = DESIRE_LANE_CHANGE_RIGHT
                        lane_change_until = time.time() + ctrl_cfg.lane_change_hold_s
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
                if quit_loop:
                    break
        finally:
            if pad is not None:
                pad.disengage()
            live_params.save()
            calib.save(game=game_id)
            tel.close()
            cv2.destroyAllWindows()
    return 0


def _build_hud_state(*, pad, view_mode, fps, v_real, v_ego, settings,
                     banner_text, banner_color, banner_until, blocked_reason,
                     lane_change_idx, lane_change_active, manual, probe,
                     lat_ctrl, long_ctrl, live_params, live_calib, ctrl_cfg,
                     fov_resolver, decoded, actual_yaw, wheel_angle, game_steer,
                     steer, ai_steer, ai_throttle, ai_brake, throttle_out,
                     brake_out, nav, cal_routine, wizard) -> HudState:
    """Assemble both the clean user-mode HudState fields and the dev-mode
    text stack. Kept out of the hot loop body for readability."""
    if pad is None:
        eng_label, eng_col = "NO GAMEPAD", COL_ACCENT_RED
    elif pad.engaged:
        eng_label, eng_col = "ENGAGED", COL_ACCENT_GREEN
    else:
        eng_label, eng_col = "DISENGAGED", COL_ACCENT_BLUE

    cam_ready = live_calib.cal_status == CalStatus.CALIBRATED
    lp_ready = live_params.trusted()
    fov_ready = fov_resolver.done or not fov_resolver.enabled
    dashboard = (not cam_ready) or (not lp_ready) or (not fov_ready)
    cam_frac = min(1.0, (live_calib.blocks * BLOCK_SIZE + live_calib._block_n)
                   / float(INPUTS_NEEDED * BLOCK_SIZE))
    lp_frac = min(1.0, max(live_params.samples, live_params.session_samples)
                  / float(TRUSTED_MIN_SAMPLES))

    banner = (banner_text if banner_text and time.time() < banner_until else "")

    lc_chip = None
    if lane_change_active:
        lc_chip = "L" if lane_change_idx == DESIRE_LANE_CHANGE_LEFT else "R"

    # Compact dev stack — the diagnostics power users expect (H toggles).
    sc = (255, 200, 80) if manual.steer_override else (255, 255, 255)
    dev_lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"{eng_label}   view={view_mode}   {fps:5.1f} fps   "
         f"v={v_ego * 3.6:5.1f} km/h", eng_col),
    ]
    if cal_routine.active:
        dev_lines.append((cal_routine.prompt, (80, 255, 255)))
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
        (f"AUTO-FOV {fov_resolver.status}"
         + (f"  ratio={fov_resolver.last_ratio:.2f}"
            if fov_resolver.last_ratio else ""),
         (80, 255, 80) if fov_resolver.done else (160, 200, 255)),
        (f"{fps:5.1f} fps   SimSteer v{__version__}", (255, 255, 255)),
    ]
    if blocked_reason and not (pad and pad.engaged):
        dev_lines.insert(1, (f"cannot engage: {blocked_reason}", COL_ACCENT_BLUE))

    wiz_text, wiz_color, wiz_hint = "", COL_ACCENT_VIOLET, ""
    if wizard.active:
        wb = wizard.banner()
        if wb is not None:
            wiz_text, wiz_color = wb[0], wb[1]
        wiz_hint = wizard.hint(v_real) or ""

    return HudState(
        engaged=bool(pad and pad.engaged),
        engaged_label=eng_label,
        engaged_color=eng_col,
        banner_text=banner,
        banner_color=banner_color,
        wizard_active=wizard.active,
        wizard_text=wiz_text,
        wizard_color=wiz_color,
        wizard_progress=wizard.progress_pct() if wizard.active else 0.0,
        wizard_hint=wiz_hint,
        warnings=[],
        v_ego_mps=v_real,
        cal_camera_frac=cam_frac if dashboard else None,
        cal_camera_status=("READY" if cam_ready else "WARMING"),
        cal_steering_frac=lp_frac if dashboard else None,
        cal_steering_status=("READY" if lp_ready else "WARMING"),
        cal_fov_frac=fov_resolver.progress if dashboard else None,
        cal_fov_status=("READY" if fov_ready else "MEASURING"),
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


def pred_yaw_safe(decoded, ctrl_cfg) -> float:
    return float(np.interp(ctrl_cfg.lookahead_s,
                           np.asarray(T_IDXS, dtype=np.float32),
                           decoded.plan[:, 14]))
