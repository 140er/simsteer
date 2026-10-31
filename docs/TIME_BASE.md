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
- A gap longer than 100 ms, a backwards/repeated timestamp, a pause or a
  geometry/FOV change clears temporal inputs. Engagement waits for a fresh
  200 ms history. Resuming telemetry also clears frames collected while
  paused. Calibration Checks from the previous adapter are invalidated.

The FOV solver receives raw vision speed and the measured vision-pair
span. Each candidate warp also records its own previous-frame timestamp;
both stages apply `0.2 / measured_span` to the speed ratio. Previously
saved ratios without this timing contract are discarded.

Live runs with rendering print the mean `live_frame_period_ms` per native
capture resolution and HUD mode. With a JSONL log, the same run receipt is
saved beside it as `<name>.metrics.json`. The period includes processing,
HUD rendering, display and capture wait between delivered frames. It is
informational; existing Windows replay p95 budgets are unchanged.

`tools/measure_time_base.py` creates one dense synthetic 90-degree clip and
resamples its saved frames at 10, 15, 18 and 20 Hz. It records real ONNX
speed and FOV results, plus a separate exact-flow FOV probe that isolates
time-span math. Exact-flow accuracy does not certify a network's absolute
FOV accuracy: scene-dependent speed bias can remain at the trained rate.
