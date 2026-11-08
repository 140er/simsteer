# Session diagnostics

Live sessions keep a numeric flight recorder automatically. It holds up to
15 seconds and at most 600 samples, evicting older samples in memory. It does
not retain frames, screenshots, free-form fault messages, game names, paths,
or device identifiers.

Incidents write JSON under the app data directory's `logs/incidents/` folder.
Names contain only a UTC timestamp, such as `20261108T120000123456Z.json`.
Each file contains `schema: 1`, numeric `events`, and `samples` with elapsed
session seconds. Event codes are:

| Code | Event |
| --- | --- |
| 1 | User disengagement, manual override, or learned intervention |
| 2 | Output direction changed |
| 3 | Invalid numeric input |
| 4 | Frame-rate auto-disengagement |
| 5 | Steering executor fault or expired steering plan |
| 6 | Other `stop_output` reason |
| 7 | Confident lane offset crosses 1.5 m from lane centre |
| 8 | Capture unavailable |
| 9 | Telemetry unavailable |
| 10 | Output device changed |
| 11 | Steering calibration required |
| 12 | Session restarting |

The triggering tick is included. At most one dump is accepted per ten seconds;
events during the cooldown are suppressed. Lane excursion and override triggers
fire on entry. The writer runs in the background with one pending dump, and
session shutdown waits for it after releasing output. A slow writer can hold
one active and one queued snapshot in addition to the live ring: at most 1,800
retained sample references. Directory/write failures disable diagnostics and
emit one generic warning. Diagnostics do not engage, disengage, or alter output.

The existing `--log-jsonl` run log gains model belief, lane width/offset,
plan and pose speed, camera geometry, and steering decomposition fields.
`desire_state` uses the model's eight desire indices (lane changes at 3 and 4),
and `lane_line_probs` uses its four lane indices. `road_edge_probs` is `null`
for the pinned model, which has no road-edge probability output. Missing
measurements stay `null`. `model_fresh` distinguishes a new healthy inference
from the display fallback on a fault or missing capture.

Steering fields are `plan_curvature` (instantaneous model yaw rate / plan
speed), `raw_k` (lag-adjusted and jerk-limited curvature before authority),
`steer_authority`, `feedforward_axis`, `trim_axis`, `probe_axis`, `final_axis`,
and `executor_axis`. Feed-forward and trim use game input units before the
output sign; probe and final axis use device units. `output_sign` records the
measured mapping, while `executor_axis` is the last submitted executor output
or direct device command; it is `null` when no output was applied. These
snapshots preserve the computation before a later stop resets the controller.
The existing log fields and their meanings remain unchanged.

Per-tick capture-wait, warp, model, control, and draw durations, five-second
rolling means (`timing_stats`), and effective model frequency are available in
live logs and developer HUD text. Draw includes key handling and preparation
after the control stage; headless runs include only that preparation. FOV
solver inference remains charged to model time. Legacy timing columns retain
their definitions; `frame_ms` still ends before drawing. Untimed replays omit
wall-clock timing fields to preserve byte-for-byte repeatability; use
`--log-timing` to include them.

The model assumes 20 Hz. After a five-second startup window, effective inference
cadence below 18 Hz for three seconds enables the existing HUD banner. Recovery
requires at least 19 Hz for three seconds. Missing frames count toward elapsed
time without counting as inferences. The banner waits for higher-priority
messages to expire and does not change control or the existing low-FPS stop.

## Offline geometry audit

Replay a clip through the normal model with `--log-jsonl run.jsonl`, then run:

```sh
python tools/geometry_check.py run.jsonl
python tools/geometry_check.py logs/incidents/20261108T120000123456Z.json
```

A replay folder containing `run.jsonl` or `replay.jsonl` is accepted too. No
model, capture backend, game integration, or screenshots are loaded by the audit.
It reports sample counts and mean/median/p05/p95 for confident model lane width,
pose/telemetry speed ratio, residual pitch from `-atan2(pose_vz, pose_vx)`,
calibration and implied horizon pixels, and model-implied camera height versus
the calibration used for that frame. Slow and missing-telemetry samples are
excluded. Old run logs supply whichever measurements they contain; unavailable
checks report `insufficient`.

Checks warn on median lane width more than 0.8 m from the reference
(`--lane-width-m`, default 3.7 m), speed ratio more than 10% from unity,
residual pitch over one degree, or height mismatch over 0.5 m. Exit status is
0 for no warnings, 1 for warnings, and 2 for unreadable input. Road slope,
camera motion, and different lane widths can explain warnings. Height is a
model estimate, and the model warp does not depend on physical camera height;
a height difference alone does not establish a projection error.

## Reproducible benchmark

`python tests/bench_diagnostics.py --frames 5000` runs the real loop with a
deterministic 20 Hz clock and numeric model/telemetry fakes. It reports mean
and p99 tick time after 100 warmup frames; `--log` includes JSONL serialization.
Use identical harness files and the same dev environment on the base and head.
No camera, model inference, or output device is required.

Thread CPU mean/p99 are reported alongside wall time to distinguish work from
scheduler delays. `--instrument` times the diagnostics calls within the loop;
`--component` alternates a tick's numeric diagnostics calls with the base's
absence of those calls using identical fake-clock inputs. The latter also fills
the ring to its 600-sample cap and measures its retained Python object size.

Ideas ported from SpysyWeeb/Beamng-onnx (MIT), commits 1e717af, f9921fe,
02209e6, 6865aa6, and 4323812. The MIT license and copyright notice are
preserved in `LICENSE`.
