# Steering execution and curvature jerk

The capture reader supplies frames at 20 Hz. The driving loop blocks in
`Capture.grab()` (up to 100 ms without a sample), runs inference on each
fresh frame, and previously wrote steering only inside that fresh-frame
branch. Its normal output cadence was therefore about 20 Hz, with no
intermediate steering updates while inference or capture was running.

Live sessions now submit complete axis targets to a separate 100 Hz
steering writer. It uses a 50 ms exponential response and limits changes
in the curvature implied by the final axis command. Immutable rack
snapshots convert between axis and wheel angle; this includes authority,
trim, bias and probe contributions without running those controllers on
every output tick. Replay clocks retain frame-driven output.

The default steering contribution to lateral jerk is bounded by
`v² * abs(delta_curvature) / dt <= 5 m/s³`, with speed floored at 1 m/s.
This is a constant-speed command bound, not a measurement of physical
vehicle jerk: longitudinal acceleration, rack errors, game input filtering
and vehicle dynamics also matter. `lat_jerk_max_mps3 = null` restores the
legacy model curvature-rate table and disables the executor's jerk bound;
the executor still smooths targets. No lateral-acceleration magnitude cap
is added, so the existing corner-speed policy remains unchanged.

Model control and trim use the elapsed time between inference completions.
A nonpositive interval or gap above 250 ms resets lateral/probe state.
The writer waits on an event between deadlines, skips missed deadlines,
and never catches up with bursts. An expired plan or executor stall above
250 ms clears its target and disengages every channel. Its writes use the
existing output lock and shutdown latch, recheck engagement, binding,
manual override and calibration readiness, and reject a replaced device.
The ordinary stop hook cancels the target before handing output back.
Disengaged recentering, manual commands and stopped direction checks keep
their direct model-loop paths. No writer is created without an output
device. Output observation and rack fitting use the applied axis rather
than the outstanding target.

Ideas ported from SpysyWeeb/Beamng-onnx (MIT):

- [100 Hz executor, c12995b](https://github.com/SpysyWeeb/Beamng-onnx/commit/c12995ba453f438ae639ed4e0ca4bc5033b13cf0)
- [Jerk limits, 48419a4](https://github.com/SpysyWeeb/Beamng-onnx/commit/48419a4f3d4f9282c17cb64847fb0b5727277616)

The existing LICENSE and attribution are preserved. The longitudinal
comfort clamp is omitted: V0.12's acceleration/deceleration values are
pedal scales and its corner braking, speed cap and AEB bypass would need
separate validation before changing pedal slew or defaults.

## Deterministic measurement

Run these from the repository root:

```sh
.venv/bin/python -m tools.measure_steering --ref 65856cb
.venv/bin/python -m tools.measure_steering
```

The baseline loads the actual control and postprocess source from the
specified git revision in an isolated process; no other checkout is used.
Both runs use a four-second fake clock, a 20 Hz plan, a curvature step
from 0 to 0.006 at 0.5 seconds, then a ramp through zero to -0.006 from
1.5 to 3.5 seconds. Rack gain is 4 axis/rad, wheelbase is 4 m, and trim
and bias are zero. The reported rate counts steering writes, including
unchanged values. Maximum steps cover all adjacent writes; implied jerk
uses each pair's actual elapsed time and the inverse rack/bicycle mapping.

| Speed (m/s) | Writes/s before → after | Max axis step before → after | Max implied jerk before → after (m/s³) |
| ---: | ---: | ---: | ---: |
| 10 | 20 → 100 | 0.02199523 → 0.00799957 | 2.74943 → 5.00000 |
| 20 | 20 → 100 | 0.01186779 → 0.00199999 | 5.93391 → 5.00000 |
| 30 | 20 → 100 | 0.00275296 → 0.00088889 | 3.09708 → 5.00000 |

This establishes deterministic command cadence and bounded increments.
It does not establish real Windows scheduling precision or driving
stability. A real-rig check is pending; in particular, verify response
lag, corner tracking, probe learning and driver takeover.

[PR #28](https://github.com/140er/simsteer/pull/28) targets V0.12 and
changes the model/FOV time base. It overlaps in `runtime/loop.py`,
`core/postprocess.py` and safety tests. This change leaves capture,
model history and FOV timing alone; integration must retain both its
frame timing and this executor's inference-completion control dt.
