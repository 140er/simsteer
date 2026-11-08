# SimSteer — Tuning & calibration reference

Everything in the tuner window, what it does, and how to fix specific
bad-driving symptoms. The tuner is a single scrollable window of
collapsible sections; sliders apply live, per-section Save buttons
persist to disk.

> **Philosophy**: SimSteer tries to stay close to how openpilot actually
> drives. Several "auto-tuners" that used to walk parameters online were
> removed because each one read a signal that was itself a function of
> the value being tuned — so the loops fed themselves and ran away. What
> remains is mostly static knobs + comma's own online learners
> (LiveParams ≈ `paramsd`, LiveCalib ≈ `calibrationd`). See
> [ARCHITECTURE.md](ARCHITECTURE.md).

---

## The fast path

A fresh setup, in order:

1. **Calibrate** in one guided drive: FOV, Camera, Steering, then a
   two minute Check. The FOV step measures your FOV once and asks you
   to accept it; wrong FOV means wrong geometry and a plan that veers
   off the road. Camera converges LiveCalib (pitch/yaw/height) and
   Steering first verifies direction from manual left and right bends,
   then measures and freezes LiveParams (the rack fit).
2. The Check scores the drive. It passes, or names the step to redo:
   weaving, full-lock steering or a steering model that mispredicts the
   wheel point at Steering; model speed that disagrees with the game
   points at FOV.
3. If driving feels off after that, reach for the knobs below, but
   change **one at a time** and watch the HUD.

---

## Camera & Calibration

The default simple warp uses the original capture width and HFOV to
compute focal length. Top/bottom crops shift the optical center without
changing that focal length. Pitch translates the forward horizon onto
the model's expected row; yaw is ignored. Use the perspective warp for
full pitch/yaw rotation, including rays away from the vanishing point.

Simple-warp geometry version 2 invalidates older saved pitch/yaw and
Camera sample history. On upgrade, a finished FOV resumes at **Camera**;
an unfinished FOV restarts that step. FOV, crops and steering fits are
preserved, and the Check must run again. Current-version Camera progress
still resumes after quitting. Saved perspective calibrations are unchanged.

| Knob | What it does |
|---|---|
| **Capture VFOV** | Vertical FOV of your capture. Set by the calibration's FOV step; change it here to enter it by hand. HFOV is derived from aspect. Never tuned continuously. |
| **Pitch / Yaw** | Camera mount angles. Auto-written by LiveCalib; only override if the cyan horizon line visibly drifts off the real horizon. |
| **Camera height** | Height above road (m). Auto-refined by LiveCalib. |
| **Crop top / bottom** | Hide cab roof / hood from the model input. |
| **Mirror road overlay** | Mirror the road projection. With wheel telemetry, steering direction is measured automatically. Without it, also flip steering. |
| **Simple warp** | Crop+resize (default) vs full perspective warp. Leave on. |

**FOV health check** (HUD `FOV` line, dev HUD): drive straight at highway
speed and read `ratio vx_model/v_ego`:
- `~1.00` → correct.
- `> 1.05` → FOV set too low → widen VFOV.
- `< 0.95` → FOV set too high → narrow VFOV.

The FOV step measures this ratio at fixed candidate warps and asks you
to accept the answer. It never adjusts FOV continuously. A known
in-game setting can also be entered by hand.

---

## Lateral steering

The lateral pipeline:

```
model plan curvature k
  → desired_curvature_lag_adjusted(k, lookahead_s + anticipation)   # openpilot's pure-pursuit + rate limit
  → k × steer_authority
  → bounded tire-slip feed-forward (learned from wheel/yaw telemetry)
  → atan(k · understeer_ff · wheelbase) = target wheel angle
  → LiveParams.axis_for_wheel_angle()  = feed-forward axis
  → + closed-loop trim + axis_bias
  → clip to ±steer_max
```

| Knob | What it does | Reach for it when… |
|---|---|---|
| **Steer max** | Hard cap on output axis. | The truck oversteers at full lock; lower to e.g. 0.6. |
| **Authority** | Flat multiplier on the model's commanded curvature. openpilot has no such knob (it trusts the plan); we keep a flat one. **1.0 is the faithful value.** | The car systematically under- or over-commits everywhere. |
| **Axis bias** | Constant left/right trim added to output. | A persistent steady drift to one side. |
| **Lookahead (s)** | Static actuator delay = openpilot's per-car `steerActuatorDelay`. The controller reads the plan this far ahead so the command lines up with the wheel by the time it lands. | — usually leave at default 0.3 (range 0.10–0.40). |
| **Anticipation (s)** | Extra lead time on top of lookahead (openpilot uses a fixed +0.2 s). | Truck turns in **too early** → lower it (negative steers later). **Too late** → raise it. |
| **Wheelbase (m)** | Bicycle-model wheelbase. A constant error here just rolls into the LiveParams fit. | Rarely. |

**Timing intuition**: the controller steers toward the plan's yaw at
`lookahead + anticipation` seconds ahead. Lookahead is the physical
delay: set it from measured command-to-wheel lag, rather than using it
to hide a steady curvature deficit. The existing 0.30 s default is
retained; no fork-specific measured delay is installed. Anticipation
is where you dial early/late feel. With `curvature_anticipation_cap`
(default `true`), the anticipation magnitude stays within the plan's
peak curvature over 1.5 times the effective lookahead, including the
interpolated endpoint. `curvature_anticipation_cap_min_curvature`
(default 0.0015 m⁻¹) restricts this to curves, preserving heading
recovery on straight roads and heading-only plans. The jerk limiter follows the cap, so an abrupt
plan change still unwinds at the allowed rate.

### Tire-slip feed-forward

The wheel can reach its commanded angle while the vehicle still turns
less than expected. This is tire understeer, separate from the rack's
axis-to-wheel mapping. On sustained curves the controller estimates:

```
|tan(measured_wheel) / wheelbase| / |measured_yaw / speed|
    = 1 + speed_gate * understeer_gradient * speed²
```

The gradient starts at zero on every engagement and is never persisted
to a vehicle constant file. Missing wheel/yaw telemetry, a cold rack fit,
or a disabled estimator gives a multiplier of 1.0. A wheel synthesized
from yaw yields a neutral gradient; it cannot identify tire slip.
The estimator can decrease its boost wherever the speed gate is active,
including partial-gate speeds. It never writes LiveParams a/b/c.

| ControllerConfig field | Default | Meaning |
|---|---|---|
| `understeer_enabled` | `true` | Enable telemetry-based tire-slip estimation/application. |
| `understeer_v_lo` / `understeer_v_hi` | 10 / 22 m/s | Linear speed gate; no boost at/below lo. |
| `understeer_gradient_max` | 0.003 s²/m² | Hard bound on the learned gradient. |
| `understeer_cap` | 1.8 | Hard cap on the applied curvature multiplier. |
| `understeer_steady_s` | 1.0 s | Required sustained-curve window before updating. |
| `understeer_learn_s` | 8.0 s | Estimator exponential time constant. |
| `understeer_min_curvature` | 0.0015 m⁻¹ | Minimum demand, measured wheel curvature and yaw curvature. |
| `understeer_steady_fraction` | 0.15 | Maximum window range relative to mean curvature or speed. |
| `feedback_min_speed` | 8 m/s | Minimum speed for both feedback learners. |

Lane changes, saturation (including trim/bias and the previous output),
intervention, low speed and cold/bad rack fits freeze learning and clear
the sustained-curve window. Transient or reversing curves also freeze
the gradient; an existing estimate can still supply feed-forward while
the curve settles. Wheel/yaw ratios outside 0.4–2.5 are rejected as
implausible. Estimates and trim reset on disengagement. Updates run
once per model frame using explicit elapsed time; the 100 Hz executor
continues to limit the complete output independently.

Diagnostics on `LateralController`: `last_understeer_gradient`,
`last_understeer_ff`, `last_understeer_delivery` (yaw/wheel curvature
magnitude ratio), `last_understeer_frozen_reason`,
`last_curvature_error` (demand magnitude minus yaw-curvature magnitude),
and `last_lookahead_s`. `last_curvature` remains the pre-compensation
model demand; `last_target_wheel` includes compensation.

---

## LiveParams — measured steering fit

LiveParams uses the inverse steering-rack mapping:

```
axis = a·wheel + b·wheel·v² + c
```

The guided **Steering** step is the only automatic scale writer. It collects
lag-matched observations, then solves `axis - b·wheel·v² - c = a·wheel`
by least squares through the origin. It requires at least 100 samples,
at least 10 excited samples in each turn direction, wheel RMS of at least
0.003 rad, a plausible signed gain, and residual RMS no greater than
0.035 axis and 25% of signal RMS. Rejected fits keep collecting and show why.
The bounded sample window lets poor data age out.

**b** keeps the per-device profile seed's magnitude, with the sign of **a**: at one speed `wheel`
and `wheel·v²` are collinear, so a short drive cannot reliably separate
two gains. **c** remains the separate axis bias; the solve never estimates
an intercept. A successful measurement installs **a** and **b** once and
persists their freeze per game/device. Engaged driving, manual driving
outside Steering, interventions, the probe, Check and session saves cannot
adapt the scale. The bounded, leaky trim integrator still corrects residuals.

| Control | Effect |
|---|---|
| **a / b / c sliders** | Explicit manual overrides, saved as chosen. No online refinement. |
| **Lock fit** | Session hold during collection; clearing it cannot unlock a measured fit. |
| **Flip sign** | Explicitly negate a and b. Recalibrate to measure again. |
| **Reset fit** | Start a new guided calibration when the flow is running. |
| **Save liveparams** | Persist the current mapping and its freeze. |

Valid older fits load frozen as-is, preserving existing per-device settings
and state imports. Loading them does not invent new measurement evidence:
the existing direction, coverage and scored Check requirements still apply.
If a saved fit is wrong, use **R / Recalibrate**. A failed Check that identifies
Steering also clears the fit and collects a new measurement.

The guided flow's **Steering** step first asks for manual left and right
bends. It compares model curvature and the observed input axis with
measured wheel angle, including the input delay. Both directions and
strong correlation are required before engagement. Inverted input
reseeds the rack fit with the measured sign. The correction saves per
game and device; restarting calibration verifies it again.

After those manual measurements, stop on level ground with the parking brake
on (below 0.5 m/s), release steering and press **INSERT**. While disengaged,
four alternating ±0.10 steering pulses measure device-to-game polarity;
no throttle is sent. Each holds for three seconds, averaging the final second
to allow input lag and smoothing.
Movement or player steering aborts the check. No response, unclear evidence
or a timeout produces a one-line reason; AI remains disengaged. This check
also works with **Steering probe** off in the tuner. The profile convention
(−1 for SCS) is only a prior. Press **INSERT** again after verification to
drive. Output polarity saves per game and device; recalibration and device
swaps clear it. A sustained
contradiction while driving disengages and clears only output verification;
the rack fit and manual direction remain. Driver corrections are excluded
from this guard. A replay has no device response
and therefore stops at this measurement.

After output direction is verified, follow the prompts for gentle left and
right bends. Profiles that permit passive fitting can use the player's
telemetry during this step. Otherwise the existing engaged Steering probe
can supply excitation, subject to all its speed, lane-change, command and
direction gates. The sample target is always the observed game input, never
a replacement with SimSteer's command. No game scripting or teleportation
is involved. Sample-admission gates widen only during Steering; the profile
stays unchanged. **Recalibrate** restarts the whole
flow, including camera calibration and the scored Check.

---

## Closed-loop trim (wheel-angle)

A slow leaky integrator on the wheel-angle error `target_wheel −
actual_wheel`, producing a small `axis_trim` added downstream of
LiveParams. It corrects steady-state offsets the feed-forward stack
can't see (rack-fit residual, model plan bias).

| Knob | Default | Notes |
|---|---|---|
| **Enable** | on | Master toggle. |
| **Trim gain** | 0.5 | Axis per (rad·s) of LPF'd error; corrects milliradian residuals in seconds. |
| **Trim clip** | 0.05 | Hard cap on the trim — it can only ever *nudge*. |
| `wheel_trim_max_curvature` | 0.0015 m⁻¹ | Freeze when either desired or measured wheel curvature exceeds this. |

Trim learns bias only on straights, freezing even in gentle sweepers
(`|curvature| > 0.0015 m⁻¹`). This prevents lag-induced windup from
returning as curve-exit overshoot. It also freezes during lane changes,
saturation, uncalibrated steering, intervention, and below
`feedback_min_speed`. The error LPF's `wheel_trim_error_lpf_alpha=0.05`
is specified at 20 Hz and converted using elapsed time; the existing
`wheel_trim_leak_s=60` and clip remain in force. The HUD `axis:` line shows
`trim`, `bias`, and whether the integrator is `ACTIVE` or
`FROZEN[reason]`.

---

## Lane change

Press **NumPad 4 / 6** (overlay or global) to trigger a left/right lane
change. Each request sends one rising-edge desire pulse to the model;
the model decides when the maneuver ends. **State window** controls how
long SimSteer treats the request as mid-lane-change for steering,
calibration and HUD state. It does not extend the model input.
There is no separate steering boost — that matches openpilot (lane
changes are model-driven).

| Knob | Notes |
|---|---|
| **State window (s)** | How long SimSteer keeps its lane-change state active for steering, calibration and HUD. Increasing it does not strengthen the model's lane change. |

---

## Longitudinal / ACC / AEB

Throttle/brake from the plan's accel + a light P term on speed error,
plus corner-anticipation braking and a time-headway lead follower.
Collapsed by default; the defaults are sane. Key knobs: `max_speed_mps`
(speed cap), `max_lat_accel_mps2` (how hard it brakes for corners),
`lead_time_headway_s` (ACC following gap), `lead_ttc_brake_s` (AEB
threshold).

---

## Manual override + bind helper

Manual sliders for steer / throttle / brake, each with an override
toggle.

Manual override works per axis, including while disengaged, when capture
and telemetry are healthy and all control data is finite. Source loss
neutralizes output while preserving slider values and selections; manual
output resumes when those gates recover. Non-overridden axes stay centered
while disengaged.

The **bind helper** wiggles a chosen input so a game's controller-binding
wizard can detect it. Stop and disengage first. Binding works in paused
menus and on static screens, including when speed telemetry is unavailable.
Known motion, engagement or a manual override cancels it. The tuner shows
the countdown, completion or cancellation.

---

## Calibration recovery

If calibration goes bad (truck wandering, lanes drawn wrong, persistent
INVALID in the LIVECALIB HUD line):

1. **Press `R`** in the overlay, or **Recalibrate** in the tuner, to
   wipe LiveCalib and LiveParams and restart calibration at FOV.
2. **Manual**: delete `%LOCALAPPDATA%\SimSteer\calibflow_<game>_<device>.json`,
   `livecalib_state_<game>.json` and `liveparams_<game>_<device>.json`.

LiveCalib also self-recovers if its estimate goes INVALID for 3+
consecutive blocks.

---

## Symptom → knob cheat sheet

| Symptom | First thing to try |
|---|---|
| Plan veers off the road, won't hold lane | **FOV is wrong** — fix Capture VFOV (ratio ≈ 1.0). |
| Steers in too early on gentle curves | Lower **Anticipation** (toward 0 or negative). |
| Steers in too late | Raise **Anticipation**. |
| Persistent drift to one side | **Axis bias**, or let closed-loop trim settle. |
| Oversteers / wobbles in lane | Lower **Steer max** or **Authority**; check LiveParams `a` isn't inflated. |
| Truck stopped steering / wild oversteer after off-highway | Saved steering fit is unsuitable → **Recalibrate**, then repeat Steering and Check. |
| Lane-change state ends too soon | Raise **State window** to extend steering, calibration and HUD gating. |
| Won't engage | Calibration not trusted yet (drive more), low FPS (DirectML), or no telemetry. |
