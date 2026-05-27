# SimSteer — Architecture

How a screen-captured game frame becomes a steering command. The driving
model is comma.ai's openpilot; everything around it (capture, warp,
controller, output, UX) is the bridge.

## Pipeline

```
   game window
        │  DXGI desktop duplication (dxcam), ~20 Hz
        ▼
   capture.py ──► preprocess.py ──► warp.py
        │            (frame queue,      (reproject onto the model's fixed
        │             YUV convert)       virtual cameras: medmodel ~31° HFOV
        │                                narrow, sbigmodel ~59° HFOV wide)
        ▼
   model.py
     two-stage ONNX:
       driving_vision.onnx   (DirectML / GPU)   frame  → vision features
       driving_policy.onnx   (CPU)              feats  → plan/lanes/pose/leads
        │
        ▼
   postprocess.py  ── decode tensors into a typed `Decoded`
        │             (plan, lane_lines, road_edges, pose, leads, desire)
        ▼
   controller.py
     LateralController:   plan curvature → target wheel → axis (via LiveParams) → +trim
     LongitudinalController: plan accel + speed-P + corner-braking + ACC/AEB → throttle/brake
        │
        ▼
   device.py → gamepad.py (ViGEm Xbox 360)  OR  wheel.py (vJoy)
        │
        ▼
   game reads the virtual controller
```

Telemetry (speed, yaw rate, wheel angle) comes back from the game through
a per-game adapter — `telemetry.py` (ETS2 SCS shared memory),
`telemetry_ac.py` (AC shared memory), `telemetry_forza.py` (Forza Data
Out UDP) — and feeds the online learners and the controller.

`main.py` is the loop that wires it all together. `hud.py` +
`debug/overlay.py` draw the overlay; `tuner.py` is the live-knob window.

## The lateral control law

```
k        = desired_curvature_lag_adjusted(plan, v_ego,
                                          steer_actuator_delay = lookahead_s,
                                          extra_buffer_s       = curvature_anticipation_s)
k        = k · steer_authority
wheel    = atan(k · wheelbase)                       # bicycle model
axis_ff  = LiveParams.axis_for_wheel_angle(wheel, v) # invert the rack fit
axis     = clip(axis_ff + axis_trim + axis_bias, ±steer_max)
```

`axis_trim` is a slow leaky integrator on `target_wheel − actual_wheel`
(see [TUNING.md](TUNING.md#closed-loop-trim-wheel-angle)).

## Online learners

| Module | Learns | openpilot analog |
|---|---|---|
| `liveparams.py` | inverse steering rack `axis = a·wheel + b·wheel·v² + c`, via RLS, per (game, device) | `selfdrive/locationd/paramsd.py` (`steerRatio`, `stiffnessFactor`) |
| `livecalib.py` | camera mount pitch / yaw / height, block-aggregated | `selfdrive/locationd/calibrationd.py` |

Both are slow, long-horizon estimators. LiveParams' watchdog reset and
adaptive forgetting are **disabled once the fit is trusted**, so a good
highway fit isn't corrupted by a short off-regime detour (the failure we
hit and fixed).

## Direct openpilot ports

- **Frame warp** (`warp.py`) — reproduces `common/transformations/model.py`:
  `K_cam · view_from_device · R(rpy) · K_model⁻¹` onto medmodel/sbigmodel
  intrinsics.
- **`desired_curvature_lag_adjusted`** (`postprocess.py`) — openpilot's
  `get_lag_adjusted_curvature`: lag-corrected pure-pursuit + rate limit,
  including the fixed `+0.2 s for other delays` buffer.
- **LiveParams / LiveCalib** — see table above.
- **Decode math** — plan/lane/lead tensor layout matches openpilot's
  model output heads.

## Project-specific additions (NOT openpilot)

These exist because we drive a *game* off a *screen capture* with a
*virtual controller*, which openpilot never does:

- **`steer_authority`** — a flat multiplier on commanded curvature.
  openpilot trusts the plan and closes the loop with a PID on
  lateral-acceleration error; we keep a flat gain knob. `1.0` is the
  faithful value.
- **Closed-loop trim** — a heavily-gated slow integrator on wheel-angle
  error. The conservative spiritual cousin of openpilot's
  `latcontrol_torque` / `latcontrol_pid` feedback, tuned to *not* fight
  LiveParams.
- **Screen capture + game telemetry adapters** — comma reads a real
  camera and CAN; we read a window and a telemetry plugin/UDP/shared-mem.
- **Virtual device output** — ViGEm pad / vJoy wheel.
- **UX layer** — `wizard.py` (first-drive calibration), `preflight.py`
  (driver/plugin checks), `settings.py`, `paths.py`, `hotkeys.py`,
  `audio.py`, `tuner.py`, `nav.py` (NOOP-style lane-change queue),
  `probe.py` (active steering perturbation to excite the rack fit during
  calibration).

## Removed auto-tuners (and why)

Four online auto-tuners were deleted. Each read a signal that was itself
a function of the parameter it tuned, so the loop fed itself and could
run away. The replacements are static knobs (set once) or comma's own
slow learners.

| Removed | Tuned | Failure | Replacement |
|---|---|---|---|
| `LiveLookahead` | `lookahead_s` from NCC of game_steer vs yaw | drifted; openpilot ships a fixed per-car delay anyway | static `lookahead_s` (= `steerActuatorDelay`) |
| `LiveFov` | `fov_h_deg` from `vx_model / v_ego` | vx is a function of the warp's FOV → positive feedback → FOV ran to 44° when the real in-game FOV was ~120° | static manual FOV + a read-only HUD ratio to tune by hand |
| `LiveAuthority` | `steer_authority` from `lane_k / plan_k` | oscillated against LiveParams; latched wrong | static `steer_authority` |
| `LiveAnticipation` | `curvature_anticipation_s` from yaw tracking error | oscillated against the rate limit + LiveParams | static `curvature_anticipation_s` slider |

The principle: only close a loop on a signal that's **independent** of
the thing you're adjusting. Camera intrinsics, actuator delay, and gain
are set once (openpilot treats them as per-car constants); only the rack
fit and camera *pose* are learned online, because those have clean,
independent observations (telemetry wheel angle, model pose vector).

## Files at a glance

| Area | Files |
|---|---|
| Loop / entry | `pilot/main.py`, `pilot/__main__.py` |
| Capture / preprocess | `pilot/capture.py`, `pilot/preprocess.py`, `pilot/warp.py` |
| Model | `pilot/model.py`, `pilot/postprocess.py`, `pilot/constants.py` |
| Control | `pilot/controller.py`, `pilot/liveparams.py`, `pilot/calibration.py`, `pilot/livecalib.py` |
| Output | `pilot/device.py`, `pilot/gamepad.py`, `pilot/wheel.py` |
| Telemetry | `pilot/telemetry.py`, `pilot/telemetry_ac.py`, `pilot/telemetry_forza.py` |
| UX | `pilot/tuner.py`, `pilot/hud.py`, `pilot/wizard.py`, `pilot/preflight.py`, `pilot/settings.py`, `pilot/paths.py`, `pilot/hotkeys.py`, `pilot/audio.py`, `pilot/nav.py`, `pilot/probe.py` |
| Overlay (standalone) | `debug/overlay.py` |
| Tools | `tools/fetch_model.py`, `tools/bind_helper.py` |
