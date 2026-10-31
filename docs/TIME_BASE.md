# Model time base

The pinned model runs at 20 Hz and uses 5 Hz context. Its vision input is
two frames 200 ms apart. The policy input has 25 feature samples spanning
4.8 seconds and 25 desire windows spanning five seconds.

Sources at openpilot revision `0f17a9879364c6f410df7ddf990380ad623a4c92`:

- [Timing constants](https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/constants.py#L13-L16).
- [Frame/feature sampling and desire pooling](https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/compile_modeld.py#L105-L169).
- [Desire rising edges](https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/modeld.py#L103-L116).
- [Dropped-frame evaluation handling](https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/modeld.py#L244-L264).

Upstream's queues count frames on its fixed-rate camera stream. SimSteer
extends that contract to variable capture rates using frame timestamps:

- Capture stamps frames before inference; replay supplies recorded times.
- Vision selects the frame nearest 200 ms earlier. Decoding scales its
  linear/angular velocity and uncertainty by `0.2 / measured_span`.
- Features select the nearest sample to each 200 ms history target, with
  zeros before the available history. Desire rising edges are max-pooled
  into 200 ms windows and expire after five seconds. At exactly 20 Hz the
  tensors match upstream's skip-four and pool-four operations.
- A gap longer than 250 ms (twice the supported 8 Hz period), a non-finite,
  backwards/repeated timestamp, a pause or a crop/resolution/FOV/warp-mode
  change clears temporal inputs. A flow reset clears them explicitly.
  Engagement waits for a fresh 200 ms history. Resuming telemetry also
  clears frames collected while paused. Earlier-adapter Checks are invalidated.
- Ordinary reader jitter preserves history. A main vision pair is eligible
  for camera learning and the FOV estimate only when its span is within
  60 ms of 200 ms and both buffered frames carry the same pitch/yaw/height
  tag. Pose edits from every writer preserve engagement and policy history;
  straddling pairs are skipped until the tags match again.

The FOV solver receives raw vision speed and the measured vision-pair
span. Each candidate warp also records its own previous-frame timestamp;
both stages apply `0.2 / measured_span` to the speed ratio. Previously
saved ratios without this timing contract are discarded.
Candidate pairs retain their four-tick schedule and are independent of the
main pair's quality flag. A pose change clears their cached frame pairs,
while preserving accumulated ratios. A completed solve stays completed.
At zero frame span, decoded pose and its uncertainty are both absent.

Live runs with rendering print the mean `live_frame_period_ms` per native
capture resolution and HUD mode. With a JSONL log, the same run receipt is
saved beside it as `<name>.metrics.json`. The period includes processing,
HUD rendering, display and capture wait between delivered frames. It is
informational; existing Windows replay p95 budgets are unchanged.

`tools/measure_time_base.py` creates a dense synthetic clip (90° by default,
or another known horizontal FOV with `--fov`) and
resamples its saved frames at 10, 15, 18 and 20 Hz. It records real ONNX
speed and FOV results, plus a separate exact-flow FOV probe that isolates
time-span math. Exact-flow accuracy does not certify a network's absolute
FOV accuracy: scene-dependent speed bias can remain at the trained rate.
`tools/measure_capture_timing.py` separately probes a jittered 20 Hz reader
sampled by 8/9/10/12/15/18/20 Hz consumers for 60 seconds per rate. It records
resets, ready frames and eligible calibration pairs without live capture.

## Absolute FOV accuracy

Absolute accuracy and candidate-selection changes are deferred to the
recorded-game evaluation. This timing change preserves the selection rule,
acceptance thresholds and the 20 Hz results on both synthetic clips.

The full-model solver has an unresolved scale bias on the synthetic
calibration scenes. At the true FOV, its learned speed can be lower than
telemetry speed. Inverting that ratio then yields a narrower FOV. This
occurs on both 75° and 90° clips, including with the exact perspective
warp, lossless native-resolution images and independently verified source
intrinsics. Changing only camera height or speed changes the ratio;
a fixed correction factor is not supported by these measurements.

Candidate responses can also be non-monotonic. The solver rejects an
inconsistent crossing and falls back to its initial estimate; that
estimate inherits the same speed bias. A finer grid alone does not fix it.
The exact [upstream wide-camera principal point](https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/common/transformations/model.py#L26-L34)
is 151.8; correcting the previous 152.0 approximation does not remove the
observed bias. Earlier-adapter Check results are invalidated for this
input change.

Sampling and raster controls also remain outside the accuracy target.
Warping YUV planes with the pinned upstream nearest-neighbour/clipped-border
convention yields 87.06° on the 90° clip and 72.30° on the 75° clip at 20 Hz.
The renderer's original dashed paint snaps to its depth grid. Giving those
boundaries exact world coordinates still yields 87.68° and 72.28° on
lossless diagnostic renders. The original saved comparison clips are
unchanged. These controls do not establish a sampling or rendering fix for
the model's absolute speed scale.

`tools/diagnose_fov.py` measures candidate responses on identical frame
pairs and compares warp/pitch/principal-point variants.
`tools/diagnose_fov_render.py` independently checks the renderer projection
and probes compression, resolution, speed, height and pair span. Its
`--dash-audit` mode compares the original and world-anchored paint. The
measurement summary's `real_model_fov_target_met` uses every full-model
result; an exact-flow probe cannot make it pass.

### Verification with recorded game frames

Synthetic results do not establish whether real game footage avoids this
bias. Verify that separately before claiming ±1.5° absolute accuracy:

1. Record manual drives in ETS2 and AC with native capture dimensions,
   frame timestamps and synchronized speed/yaw telemetry. Record the
   actual game FOV setting and whether it is horizontal or vertical;
   convert vertical FOV using the captured aspect ratio. Include two known
   FOVs, two speed ranges, and low/high camera positions. Keep camera
   settings fixed within each clip and record pauses or geometry changes.
2. Establish and freeze camera pitch using a separate segment at the
   known FOV. Keep that segment separate from the evaluation frames. Do
   not treat the model's estimated FOV as ground truth or adjust pitch to
   improve the FOV score.
3. Resample evaluation frames to 10/15/18/20 Hz by selecting the nearest
   original timestamp, retaining actual timestamps and telemetry. Do not
   invent intermediate images or timing. Compare candidates on identical
   frame pairs and run the full solver from the true FOV and seeds 10°
   above/below it. Use straight, moving segments accepted by the existing
   speed/yaw gates; preserve rejected samples and missing results.
4. Report each clip, camera, speed range and frame rate separately. Every
   full solve must be within ±1.5° of the independently known FOV; missing
   results fail. Keep held-out clips from both games. Report the speed
   ratio at the true FOV alongside the solve to distinguish model scale
   error from selection error. If that bias persists, use independently
   known/configured FOV or reconsider the estimator; do not fit a scalar
   to the evaluation clips.
