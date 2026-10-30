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
   then converges LiveParams (the rack fit) while SimSteer drives.
2. The Check scores the drive. It passes, or names the step to redo:
   weaving, full-lock steering or a steering model that mispredicts the
   wheel point at Steering; model speed that disagrees with the game
   points at FOV.
3. If driving feels off after that, reach for the knobs below, but
   change **one at a time** and watch the HUD.

---

## Camera & Calibration

| Knob | What it does |
|---|---|
| **Capture VFOV** | Vertical FOV of your capture. Set by the calibration's FOV step; change it here to enter it by hand. HFOV is derived from aspect. Never tuned continuously. |
| **Pitch / Yaw** | Camera mount angles. Auto-written by LiveCalib; only override if the cyan horizon line visibly drifts off the real horizon. |
| **Camera height** | Height above road (m). Auto-refined by LiveCalib. |
| **Crop top / bottom** | Hide cab roof / hood from the model input. |
| **Mirror lateral sign** | Mirror the road projection. With wheel telemetry, calibration learns steering direction separately. |
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
  → atan(k · wheelbase)                = target wheel angle
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
delay (don't fiddle); anticipation is where you dial early/late feel.

---

## LiveParams — RLS steering fit

LiveParams learns the inverse steering-rack model online:

```
axis = a·wheel + b·wheel·v² + c
```

- **a** — axis per radian of wheel at low speed (the base ratio).
- **b** — speed-stiffness; how much more axis the same wheel needs at
  highway speed (ETS2's variable-ratio rack). For the vJoy wheel device
  this is ~0 (linear).
- **c** — small constant axis bias.

It's our analog of openpilot's `paramsd` (which estimates `steerRatio` +
`stiffnessFactor`). Per-(game, device) — a gamepad fit and a wheel fit
are stored separately.

| Control | Effect |
|---|---|
| **a / b / c sliders** | Manually seed the fit. RLS keeps refining unless locked. |
| **Lock fit** | Freeze a/b/c — both RLS and intervention learning stop. |
| **Flip sign** | Negate a and b. Use if the truck steers the wrong way (alternative to mirror lateral sign). |
| **Reset RLS** | Back to the per-(game, device) **seed** (e.g. ETS2 gamepad a=3.5). |
| **Save liveparams** | Persist current a/b/c. |

**Trust + off-regime protection**: once the fit has enough samples it
flips to *trusted* (HUD `trust_level`). Trusted fits are deliberately
stubborn — the watchdog covariance-reset and adaptive forgetting are
**disabled when trusted**. This fixes a known runaway: a highway-trained
fit used to corrupt itself the moment you drove off-highway (low speed,
big wheel angles → huge innovation → watchdog reset → `a` ran from ~4 to
~19 and never recovered). Now off-regime samples are absorbed slowly and
wash out when you're back on the highway.

**If your fit is already corrupted** (a way off its seed, truck won't
steer or wildly oversteers): hit **Reset RLS**, then drive highway until
trusted again.

The guided flow's **Steering** step first asks for manual left and right
bends. It compares model curvature and the observed input axis with
measured wheel angle, including the input delay. Both directions and
strong correlation are required before engagement. Inverted input
reseeds the rack fit with the measured sign. The correction saves per
game and device; restarting calibration verifies it again.

After direction is verified, the step prompts for a straight stretch,
gentle curves, lane changes and slow turns until the fit is trusted.
It widens the RLS gates with a per-call override during this step;
the profile's gates stay unchanged. **Recalibrate** restarts the whole
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
| **Trim gain** | 0.005 | Axis per (rad·s) of LPF'd error. Higher = faster correction. |
| **Trim clip** | 0.05 | Hard cap on the trim — it can only ever *nudge*. |

This is the conservative cousin of openpilot's PID-on-lateral-accel: it
runs ~40× slower than the wheel actuator (~10 s integrator + 60 s leak)
and **freezes itself during every transient regime** — corners
(`|target_wheel| > 0.05`), lane changes, saturation, cold/recovering
RLS, intervention, and below 8 m/s. That's why it doesn't re-create the
old PID's habit of fighting LiveParams. The HUD `axis:` line shows
`trim`, `bias`, and whether the integrator is `ACTIVE` or
`FROZEN[reason]`.

---

## Lane change

Press **NumPad 4 / 6** (overlay or global) to trigger a left/right lane
change. This feeds a sustained desire pulse to the model for
**Hold duration** seconds; the model commits the maneuver on its own.
There is no separate steering boost — that matches openpilot (lane
changes are model-driven).

| Knob | Notes |
|---|---|
| **Hold duration (s)** | How long the desire pulse is held. A single-frame pulse washes out of the model's 5 s context; ~2.5 s gives it enough to commit. Bump to 4–5 s if changes feel half-hearted. |

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

> **Manual override always fires when its toggle is on — including while
> DISENGAGED.** This is per-axis: you can drive steering manually while
> leaving throttle/brake to your keyboard/wheel, engaged or not. Useful
> for testing the output path, forcing a specific axis, or driving the
> car by slider. Non-overridden axes stay centered so the AI doesn't
> ghost-drive.

The **bind helper** wiggles a chosen input so a game's controller-binding
wizard can detect it. Disengage first.

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
| Truck stopped steering / wild oversteer after off-highway | LiveParams corrupted → **Reset RLS**, reconverge on highway. |
| Lane change stalls mid-maneuver | Raise **Hold duration**. |
| Won't engage | Calibration not trusted yet (drive more), low FPS (DirectML), or no telemetry. |
