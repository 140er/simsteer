"""Time intervals, independent of frame counts or inference wall time."""

from collections import deque
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.core.calibration import Calibration
from simsteer.core.constants import FEATURE_LEN, VISION_SLICES
from simsteer.core.finite import NonFiniteError
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue
from simsteer.runtime.fov import FovSolver


@pytest.fixture
def small_views(monkeypatch):
    import simsteer.core.preprocess as preprocess
    monkeypatch.setattr(preprocess, "crop_for_model", lambda image, calib: (image, image))
    monkeypatch.setattr(preprocess, "bgr_to_yuv6",
                        lambda image: np.full((6, 1, 1), image.flat[0], np.uint8))


def pixels(value):
    return np.full((2, 2, 3), value, np.uint8)


@pytest.mark.parametrize("hz", [10, 15, 18, 20, 40])
def test_frame_pair_and_pose_use_elapsed_time(small_views, hz):
    queue = FrameQueue()
    vision, policy = np.zeros(1576, np.float32), np.zeros(1000, np.float32)
    for n in range(hz):
        image, _ = queue.push(pixels(n), timestamp=n / hz)
        if n < math.ceil(.2 * hz):
            assert not queue.ready
            continue
        assert queue.ready
        assert queue.span_s == pytest.approx((n - image[0, 0, 0, 0]) / hz)
        assert abs(queue.span_s - .2) <= .5 / hz + 1e-8
        vision[87:93] = np.array([22., .3, .4, .1, .2, .3]) * queue.span_s / .2
        vision[93:99] = math.log(2.)
        decoded = decode(vision, policy, frame_span_s=queue.span_s)
        np.testing.assert_allclose(decoded.pose, [22., .3, .4, .1, .2, .3], rtol=1e-6)
        np.testing.assert_allclose(decoded.pose_std, 2 * .2 / queue.span_s, rtol=1e-6)


@pytest.mark.parametrize("change", [
    {"fov_h_deg": 80.},
    {"crop_top_pct": .1}, {"crop_bottom_pct": .1}, {"simple_warp": False},
    {"cx_frac": .4}, {"cy_frac": .4}, {"image_w": 800}, {"image_h": 600},
])
def test_frame_geometry_change_discards_history(small_views, change):
    queue, calib = FrameQueue(), Calibration()
    for n in range(6):
        queue.push(pixels(n), calib, timestamp=n * .05)
    before = queue.resets
    image, _ = queue.push(pixels(99), replace(calib, **change), timestamp=.3)
    assert queue.resets == before + 1
    assert not queue.ready and queue.span_s == 0
    assert (image == 99).all()


@pytest.mark.parametrize("next_time", [.55, .25, .1])
def test_gap_duplicate_or_reversed_time_discards_frame_pairs(small_views, next_time):
    queue = FrameQueue()
    for n in range(6):
        queue.push(pixels(n), timestamp=n * .05)
    image, _ = queue.push(pixels(99), timestamp=next_time)
    assert queue.resets == 1
    assert not queue.ready and queue.span_s == 0
    assert (image == 99).all()


def test_resolution_change_resets_pairs(small_views):
    queue = FrameQueue()
    queue.push(pixels(1), timestamp=0.)
    queue.push(np.full((3, 4, 3), 99, np.uint8), timestamp=.05)
    assert queue.resets == 1 and queue.span_s == 0


@pytest.fixture
def model_inputs(monkeypatch):
    import simsteer.core.model as adapter

    class Session:
        inputs = None

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, outputs, inputs):
            self.inputs = inputs
            width = 1576 if "img" in inputs else 1000
            out = np.zeros((1, width), np.float32)
            if width == 1576:
                start, end = VISION_SLICES["hidden_state"]
                out[0, start:end] = inputs["img"].flat[0]
            return [out]

    vision, policy = Session(), Session()
    monkeypatch.setattr(adapter, "_make_session",
                        lambda path, providers: vision if path == adapter.VISION_PATH else policy)
    monkeypatch.setattr(adapter, "actual_model_identity", lambda *args: {})
    return DrivingModel(), policy


@pytest.mark.parametrize("hz", [10, 15, 18, 20])
def test_features_cover_4800ms_and_desire_expires_after_5s(model_inputs, hz):
    model, policy = model_inputs
    left = np.eye(8, dtype=np.float32)[3]
    times = np.arange(6 * hz + 1) / hz
    for i, t in enumerate(times):
        model.step(np.full((1, 12, 1, 1), i + 1, np.uint8), desire=left, timestamp=t)
        pulse = policy.inputs["desire_pulse"][0]
        expected = np.zeros((25, 8))
        if t < 5.0 - 1e-6:
            expected[24 - math.floor((t + 1e-6) / .2), 3] = 1
        np.testing.assert_array_equal(pulse, expected)
    # An independent nearest-time reference; frame skipping would stretch
    # this history to 9.6 seconds on a 10 Hz stream.
    targets = times[-1] - np.arange(24, -1, -1) * .2
    expected = [np.argmin(abs(times - target)) + 1 for target in targets]
    np.testing.assert_array_equal(policy.inputs["features_buffer"][0, :, 0], expected)
    assert policy.inputs["features_buffer"].dtype == np.float16


def test_features_match_pinned_skip_four_at_20hz(model_inputs):
    model, policy = model_inputs
    upstream = deque([np.zeros(FEATURE_LEN)] * 97, maxlen=97)
    for i in range(120):
        upstream.append(np.full(FEATURE_LEN, i + 1))
        model.step(np.full((1, 12, 1, 1), i + 1, np.uint8), timestamp=i / 20)
        np.testing.assert_array_equal(policy.inputs["features_buffer"][0],
                                      np.stack(upstream)[::4])


@pytest.mark.parametrize("reset", ["gap", "pause", "backwards"])
def test_model_reset_clears_features_and_held_desire(model_inputs, reset):
    model, policy = model_inputs
    left = np.eye(8, dtype=np.float32)[3]
    for i in range(10):
        model.step(pixels(1), desire=left, timestamp=i / 20)
    t = .8 if reset == "gap" else 0.
    if reset == "pause":
        model.reset()
        t = .5
    model.step(pixels(2), desire=left, timestamp=t)
    features = policy.inputs["features_buffer"][0]
    assert not features[:-1].any() and (features[-1] == 2).all()
    pulse = policy.inputs["desire_pulse"][0]
    assert pulse[-1, 3] == pulse.sum() == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_timestamps_and_spans_are_rejected(small_views, model_inputs, bad):
    with pytest.raises(NonFiniteError):
        FrameQueue().push(pixels(1), timestamp=bad)
    with pytest.raises(NonFiniteError):
        model_inputs[0].step(pixels(1), timestamp=bad)
    with pytest.raises(NonFiniteError):
        decode(np.zeros(1576), np.zeros(1000), frame_span_s=bad)


def flow_views(frame, calib):
    out = np.zeros((6, 1, 2))
    out[:, 0, 0] = calib.fov_h_deg
    out[:, 0, 1] = frame.item()
    return out, out


def flow_vision(img, big):
    span = img[0, 6, 0, 1] - img[0, 0, 0, 1]
    fov = img[0, 6, 0, 0]
    return 22 * span / .2 / math.tan(math.radians(fov / 2)), .1


@pytest.mark.parametrize("hz", [10, 15, 18, 20])
@pytest.mark.parametrize("start", [75., 90., 120.])
def test_fov_normalises_main_and_candidate_spans(hz, start):
    solver = FovSolver(flow_vision, flow_views)
    calib = Calibration(fov_h_deg=start)
    for i in range(1000):
        t = i / hz
        raw_vx = 22 * (4 / hz) / .2 / math.tan(math.radians(start / 2))
        solver.update(np.array([t]), calib, raw_vx, 22., 0.,
                      timestamp=t, frame_span_s=4 / hz)
        if solver.done:
            break
    assert solver.result == pytest.approx(90., abs=1.5)


def test_fov_uses_individual_spans_under_jitter():
    solver = FovSolver(flow_vision, flow_views)
    calib = Calibration(fov_h_deg=100.)
    times = [0.]
    for i in range(800):
        times.append(times[-1] + (.035, .06, .045, .07, .05)[i % 5])
        t = times[-1]
        span = t - times[max(0, len(times) - 5)]
        solver.update(np.array([t]), calib, 22 * span / .2 / math.tan(math.radians(50)),
                      22., 0., timestamp=t, frame_span_s=span)
        if solver.done:
            break
    assert solver.result == pytest.approx(90., abs=1.5)


@pytest.mark.parametrize("interruption", ["gap", "pause", "geometry"])
def test_fov_discontinuity_discards_ratios_and_candidate_pairs(interruption):
    solver = FovSolver(flow_vision, flow_views)
    calib = Calibration(fov_h_deg=90.)
    for i in range(55):
        solver.update(np.array([i / 20]), calib, 22., 22., 0., timestamp=i / 20)
    assert solver.stage == "bracket" and solver._prev
    solver.vision_s = 12.
    t = 3. if interruption == "gap" else 2.75
    if interruption == "pause":
        solver.reset()
    if interruption == "geometry":
        calib.crop_top_pct = .1
    solver.update(np.array([t]), calib, 22., 22., 0., timestamp=t)
    assert solver.stage == "estimate" and solver.samples == 1
    assert not solver._prev and not solver._prev_times
    assert solver.vision_s == 12.


def test_old_untimed_fov_evidence_is_invalidated():
    solver = FovSolver()
    solver.load_dict({"ratios": [1.] * 40, "result": 90.})
    assert not solver.done and solver.samples == 0


@pytest.mark.parametrize("rendered", [False, True])
def test_loop_resets_on_capture_gap_pause_and_geometry(monkeypatch, tmp_path, small_views, rendered):
    import simsteer.runtime.loop as loop
    from test_replay import FakeModel

    state = SimpleNamespace(i=-1, t=0.)
    events, stamps = [], []
    holder = {}

    class Queue(FrameQueue):
        def reset(self):
            events.append((state.i, "frames"))
            super().reset()

    class Model(FakeModel):
        def reset(self):
            events.append((state.i, "model"))

        def step(self, *args, timestamp=None, **kwargs):
            stamps.append(timestamp)
            return super().step(*args, timestamp=timestamp, **kwargs)

    class Solver(FovSolver):
        def reset(self):
            events.append((state.i, "fov"))
            super().reset()

    class Flow(loop.CalibFlow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["flow"] = self

    class Source:
        done = False
        frame_timestamp = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def grab(self):
            state.i += 1
            if state.i == 40:
                self.done = True
                return None
            t = state.i / 20 + (.3 if state.i >= 9 else 0.)
            self.frame_timestamp = t
            state.t = 20. + t * 2  # Processing time is a different clock.
            if state.i == 30:
                holder["flow"].calib.fov_h_deg = 80.
            return np.full((2, 3 if state.i >= 24 else 2, 3), state.i, np.uint8)

    class Telemetry:
        available = property(lambda self: state.i not in (15, 16, 17))
        speed_mps = lambda self: 22. if self.available else None
        yaw_rate_rad_s = lambda self: 0.
        wheel_angle_rad = lambda self, wheelbase: 0.
        game_steer = lambda self: 0.
        close = lambda self: None

    monkeypatch.setattr(loop, "FrameQueue", Queue)
    monkeypatch.setattr(loop, "FovSolver", Solver)
    monkeypatch.setattr(loop, "CalibFlow", Flow)
    draws = []
    if rendered:
        # Exercise the live receipt without a desktop or a real capture API.
        monkeypatch.setattr(loop, "Capture", Source)
        monkeypatch.setattr(loop, "_setup_window_or_exit", lambda *args: None)
        monkeypatch.setattr(loop.cv2, "setWindowTitle", lambda *args: None)
        monkeypatch.setattr(loop.cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(loop, "show_scaled", lambda *args: None)
        monkeypatch.setattr(loop, "yuv6_to_bgr", lambda image: np.zeros((256, 512, 3), np.uint8))
        monkeypatch.setattr(loop, "draw_overlay", lambda frame, *args, **kwargs: frame)
        monkeypatch.setattr(loop, "HudRenderer", lambda: SimpleNamespace(
            draw=lambda *args, **kwargs: draws.append(state.i)))
    log = tmp_path / "frames.jsonl"
    assert loop.run(loop.Settings(no_gamepad=True, no_tuner=True), "ets2", source=Source(),
                    make_telemetry=Telemetry, model=Model(), headless=not rendered, keys=loop.NoKeys(),
                    clock=SimpleNamespace(now=lambda: state.t), log_path=str(log)) == 0
    for i in (9, 15, 18, 24, 30):
        assert {(i, kind) for kind in ("frames", "model", "fov")} <= set(events)
    assert stamps == pytest.approx([i / 20 + (.3 if i >= 9 else 0.) for i in range(40)])
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for i in (0, 9, 18, 24, 30):
        assert rows[i]["vision_span_s"] == 0.
        assert rows[i]["engage_blocked_reason"] == "Frame history warming up"
        assert not rows[i]["engaged"]
    if rendered:
        assert draws == list(range(40))
        receipt = json.loads(log.with_suffix(".metrics.json").read_text())
        assert receipt["metric"] == "live_frame_period_ms"
        runs = receipt["runs"]
        assert [(r["capture_width"], r["capture_height"], r["samples"]) for r in runs] == [
            (2, 2, 23), (3, 2, 15)]
        assert [r["mean_ms"] for r in runs] == pytest.approx([2900 / 23, 100])
        assert {r["hud_mode"] for r in runs} == {"user"}
    else:
        assert not log.with_suffix(".metrics.json").exists()
