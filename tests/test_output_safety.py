"""Fault injection at the submitted controller-report boundary."""

from __future__ import annotations

import json
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.app.global_keys import KEY_INSERT
from simsteer.app.settings import Settings
from simsteer.core.calibration import Calibration
from simsteer.core.control import LateralController, LongitudinalController
from simsteer.core.learners.livecalib import LiveCalib
from simsteer.core.learners.liveparams import LiveParams
from simsteer.runtime.capture import Capture
from simsteer.runtime.output.gamepad import Gamepad
from simsteer.runtime.output.wheel import Wheel, VJOY_AXIS_CTR, VJOY_AXIS_MIN
from simsteer.runtime.telemetry import scs_shm as scs


BAD = [float("nan"), float("inf"), -float("inf")]


class Reports:
    def __init__(self):
        self.visible = (0., 0., 0.)
        self.pending = [0., 0., 0.]
        self.sent = []
        self.writers = []

    def reset(self):
        self.pending[:] = [0., 0., 0.]

    def left_joystick_float(self, x_value_float, y_value_float):
        self.pending[0] = x_value_float

    def right_trigger_float(self, value_float):
        self.pending[1] = value_float

    def left_trigger_float(self, value_float):
        self.pending[2] = value_float

    def update(self):
        self.visible = tuple(self.pending)
        self.sent.append(self.visible)
        self.writers.append(threading.get_ident())


def gamepad():
    pad = Gamepad.__new__(Gamepad)
    pad._pad = Reports()
    pad._engaged = False
    return pad


def attempt(call):
    # Rejection may raise after neutralizing, or just submit neutral.
    try:
        return call()
    except ValueError:
        return None


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("channel", ["steer", "throttle", "brake"])
@pytest.mark.parametrize("backend", ["gamepad", "wheel"])
def test_nonfinite_device_command_neutralizes_every_channel(bad, channel, backend):
    if backend == "gamepad":
        pad = gamepad()
        visible = lambda: pad.raw.visible
        neutral = (0., 0., 0.)
    else:
        pad = Wheel.__new__(Wheel)
        axes = {}
        pad._dev = SimpleNamespace(set_axis=lambda axis, value: axes.update({axis: value}))
        pad._pyvjoy = SimpleNamespace(HID_USAGE_X=1, HID_USAGE_Y=2, HID_USAGE_Z=3)
        pad._engaged = False
        visible = lambda: tuple(axes[i] for i in (1, 2, 3))
        neutral = (VJOY_AXIS_CTR, VJOY_AXIS_MIN, VJOY_AXIS_MIN)
    pad.engage()
    pad.set_steering(.4)
    pad.set_throttle_brake(.6, .2)
    if channel == "steer":
        attempt(lambda: pad.set_steering(bad))
    else:
        attempt(lambda: pad.set_throttle_brake(
            bad if channel == "throttle" else .6, bad if channel == "brake" else .2))
    assert visible() == neutral
    assert not pad.engaged


def drive(monkeypatch, *, fault=None, frames=4, key=None, no_frames=False,
          mode="player", force=True, source=None, on_grab=None, speed=20.,
          telemetry_source=None, tuner_factory=None):
    import simsteer.runtime.loop as loop
    import simsteer.runtime.output.base as devices

    st = SimpleNamespace(i=-1, t=0., checkpoints=[], polls=0, capture_closed=False, telemetry_closed=False)
    pad = gamepad()
    monkeypatch.setattr(devices, "Gamepad", lambda: pad)
    manager = devices.DeviceManager()
    monkeypatch.setattr(loop, "DeviceManager", lambda **kw: manager)
    monkeypatch.setattr(loop.audio, "play", lambda *a: None)

    class Flow(loop.CalibFlow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steering_fit = False
            self.step = loop.Step.DONE
            self.lc.cal_status = loop.CalStatus.CALIBRATED
            st.flow = self

    monkeypatch.setattr(loop, "CalibFlow", Flow)
    monkeypatch.setattr(loop, "FrameQueue", lambda: SimpleNamespace(push=lambda *a: (None, None)))

    class Source:
        done = False

        def __enter__(self):
            if source is not None:
                source.__enter__()
            return self

        def __exit__(self, *args):
            if source is not None:
                source.__exit__(*args)
            st.capture_closed = True

        def grab(self):
            st.checkpoints.append((pad.engaged, pad.raw.visible))
            st.i += 1
            st.t += .05
            if on_grab:
                on_grab(st, manager)
            if st.i >= frames:
                self.done = True
                return None
            if source is not None:
                return source.grab()
            return None if no_frames and st.i > 0 else np.full((8, 8, 3), st.i, np.uint8)

    class Model:
        def step(self, *args, **kwargs):
            vision, policy = np.zeros(1576, np.float32), np.zeros(1000, np.float32)
            plan = policy[:495].reshape(33, 15)
            plan[:, 3] = 20.
            plan[:, 11] = .02
            if st.i == 1 and fault in ("vision", "policy"):
                (vision if fault == "vision" else policy)[11] = np.nan
            if st.i == 1 and fault == "decode_overflow":
                vision[93] = 1000.
            return vision, policy

    def value(name, normal):
        return np.nan if st.i == 1 and fault == name else normal

    tel = SimpleNamespace(
        available=True, speed_mps=lambda: value("speed", speed),
        yaw_rate_rad_s=lambda: value("yaw", 0.),
        wheel_angle_rad=lambda wb: value("wheel", .01),
        game_steer=lambda: value("axis", .2),
        user_steer=lambda: value("user", 0.), close=lambda: setattr(st, "telemetry_closed", True))
    if fault == "lost":
        class Telemetry:
            available = property(lambda self: st.i < 1)
        for name, method in vars(tel).items():
            if name != "available":
                setattr(Telemetry, name, staticmethod(method))
        tel = Telemetry()
    if telemetry_source is not None:
        tel = telemetry_source

    class Keys:
        def poll(self):
            st.polls += 1
            if key and st.i == 1:
                return [key]
            return []

    if fault == "lateral":
        original = loop.LateralController.compute
        monkeypatch.setattr(loop.LateralController, "compute", lambda self, *a, **k:
                            np.nan if st.i == 1 else original(self, *a, **k))
    if fault == "longitudinal":
        original = loop.LongitudinalController.compute
        monkeypatch.setattr(loop.LongitudinalController, "compute", lambda self, *a, **k:
                            (np.nan, .5) if st.i == 1 else original(self, *a, **k))
    if fault == "learner":
        original = loop.LiveParams.update

        def update(self, *a, **k):
            if st.i == 1:
                self.x[0] = np.nan
            else:
                original(self, *a, **k)
        monkeypatch.setattr(loop.LiveParams, "update", update)
    if fault == "decoded":
        original = loop.decode

        def decode(*a):
            d = original(*a)
            if st.i == 1:
                d.plan[5, 11] = np.nan
            return d
        monkeypatch.setattr(loop, "decode", decode)
    if fault == "camera_state":
        original = Model.step

        def step(self, *a, **kw):
            if st.i == 1:
                st.flow.lc._wide_sum[0] = np.nan
            return original(self, *a, **kw)
        monkeypatch.setattr(Model, "step", step)
    if tuner_factory:
        monkeypatch.setattr(loop, "Tuner", lambda *a, **kw: tuner_factory(st, manager))
        monkeypatch.setattr(loop, "FrameQueue", lambda: SimpleNamespace(
            push=lambda *a: (None, None), last_yuv=None))
        monkeypatch.setattr(loop, "_setup_window_or_exit", lambda *a: None)
        monkeypatch.setattr(loop.cv2, "setWindowTitle", lambda *a: None)
        monkeypatch.setattr(loop.cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(loop, "draw_overlay", lambda frame, *a, **kw: frame)
        monkeypatch.setattr(loop, "show_scaled", lambda *a: None)
        monkeypatch.setattr(loop, "HudRenderer", lambda: SimpleNamespace(draw=lambda *a, **kw: None))
    loop.run(Settings(no_tuner=not tuner_factory, no_probe=True, force_engage=force, hud_mode=mode),
             "ets2", source=Source(), make_telemetry=lambda: tel,
             clock=SimpleNamespace(now=lambda: st.t), model=Model(), keys=Keys(), headless=not tuner_factory)
    return st, pad, manager


@pytest.mark.parametrize("fault", ["vision", "policy", "decoded", "speed", "yaw", "wheel",
                                       "axis", "user", "lateral", "longitudinal", "learner", "lost",
                                       "decode_overflow", "camera_state"])
def test_fault_at_each_stage_stops_submitted_output(monkeypatch, fault):
    st, pad, _ = drive(monkeypatch, fault=fault)
    assert st.checkpoints[1][0]  # A genuinely engaged command precedes the fault.
    assert any(st.checkpoints[1][1])
    assert st.checkpoints[2] == (False, (0., 0., 0.))
    assert all(not engaged and report == (0., 0., 0.)
               for engaged, report in st.checkpoints[2:])


def test_insert_stops_output_without_a_frame(monkeypatch):
    st, _, _ = drive(monkeypatch, no_frames=True, key=KEY_INSERT)
    assert st.checkpoints[1][0]
    assert st.checkpoints[2] == (False, (0., 0., 0.))


@pytest.mark.parametrize("mode", ["player", "dev"])
def test_t_never_submits_test_fire_at_speed(monkeypatch, mode):
    _, pad, _ = drive(monkeypatch, key=ord("t"), mode=mode, force=False)
    assert all(report == (0., 0., 0.) for report in pad.raw.sent)


def test_capture_stall_neutralizes_within_bounded_wait(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class Camera:
        n = 0

        def start(self, **kwargs):
            pass

        def get_latest_frame(self):
            self.n += 1
            if self.n == 1:
                return np.ones((8, 8, 3), np.uint8)
            entered.set()
            release.wait(2.)
            return None

        def stop(self):
            release.set()

    monkeypatch.setitem(__import__("sys").modules, "dxcam",
                        SimpleNamespace(create=lambda **kw: Camera()))
    cap = Capture()
    elapsed = []
    result = []

    def grab():
        start = time.monotonic()
        result.append(drive(monkeypatch, source=cap, frames=3))
        elapsed.append(time.monotonic() - start)

    thread = threading.Thread(target=grab, daemon=True)
    thread.start()
    try:
        assert entered.wait(1.)
        thread.join(.6)
        assert not thread.is_alive(), "capture blocked the stop path"
        assert elapsed[0] < .6
        st, pad, _ = result[0]
        assert st.checkpoints[1][0] and any(st.checkpoints[1][1])
        assert st.checkpoints[2] == (False, (0., 0., 0.))
        assert pad.raw.visible == (0., 0., 0.) and not pad.engaged
    finally:
        release.set()
        thread.join(1.)


def test_video_mode_identical_frames_stop_output(monkeypatch):
    import sys

    class Camera:
        def start(self, **kwargs):
            pass

        def get_latest_frame(self):
            time.sleep(.01)
            return np.ones((8, 8, 3), np.uint8)

        def stop(self):
            pass

    monkeypatch.setitem(sys.modules, "dxcam", SimpleNamespace(create=lambda **kw: Camera()))
    with Capture() as cap:
        pad = gamepad()
        pad.engage()
        pad.set_steering(.4)
        first = cap.grab()
        assert first is not None
        for _ in range(30):
            if cap.grab() is None:
                pad.disengage()
                break
        assert not pad.engaged and pad.raw.visible == (0., 0., 0.)


def telemetry(monkeypatch):
    clock = SimpleNamespace(t=0.)
    monkeypatch.setattr(time, "monotonic", lambda: clock.t)
    class Mapping(bytearray):
        def close(self):
            pass
    buf = Mapping(scs.SHARED_MEM_SIZE)
    buf[scs.OFF_SDK_ACTIVE] = 1
    struct.pack_into("<Q", buf, scs.OFF_TIME, 100)
    struct.pack_into("<f", buf, scs.OFF_TRUCK_SPEED, 20.)
    # Construct normally on all platforms with a fake named mapping.
    monkeypatch.setattr(scs, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(scs.mmap, "mmap", lambda *a, **kw: buf)
    return scs.ScsTelemetry(), buf, clock


@pytest.mark.parametrize("reason", ["paused", "inactive", "stale"])
def test_scs_loss_neutralizes_device(monkeypatch, reason):
    tel, buf, clock = telemetry(monkeypatch)
    assert tel.available

    def lose(st, manager):
        if st.i == 1:
            if reason == "paused":
                buf[scs.OFF_PAUSED] = 1
            elif reason == "inactive":
                buf[scs.OFF_SDK_ACTIVE] = 0
            else:
                clock.t = 2.

    st, _, _ = drive(monkeypatch, telemetry_source=tel, on_grab=lose)
    assert st.checkpoints[1][0] and any(st.checkpoints[1][1])
    assert st.checkpoints[2] == (False, (0., 0., 0.))


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("field", ["axis", "speed", "wheel", "command"])
def test_liveparams_rejects_poison_before_mutating_fit(bad, field):
    lp = LiveParams(lag_frames=1)
    lp.x[:] = [-.5, -.004, .008]
    lp.samples = 10000
    clean = dict(axis=-.2, speed=20., wheel=.02, command=-.2)
    for _ in range(3):
        lp.update(clean["axis"], clean["speed"], clean["wheel"], clean["command"])
    x, P, samples = lp.x.copy(), lp.P.copy(), lp.samples
    clean[field] = bad
    attempt(lambda: lp.update(clean["axis"], clean["speed"], clean["wheel"], clean["command"]))
    assert np.array_equal(lp.x, x) and np.array_equal(lp.P, P)
    assert lp.samples == samples
    assert all(np.isfinite(v) for v in lp._axis_buf)
    assert all(np.isfinite(v) for v in lp._v_buf)


@pytest.mark.parametrize("field", ["pose", "pose_std", "road_transform", "road_transform_std",
                                     "wide_from_device_euler", "actual_yaw_rate", "v_ego", "game_steer"])
def test_livecalib_rejects_nonfinite_samples(field):
    lc = LiveCalib()
    values = dict(pose=np.array([20., 0., 0., 0., 0., 0.]), pose_std=np.zeros(6),
                  road_transform=np.array([0., 0., 1.2, 0., 0., 0.]),
                  road_transform_std=np.zeros(6), wide_from_device_euler=np.zeros(3),
                  actual_yaw_rate=0., v_ego=20., game_steer=0.)
    if isinstance(values[field], np.ndarray):
        values[field][-1] = np.nan
    else:
        values[field] = np.nan
    attempt(lambda: lc.update(Calibration(), **values))
    assert lc.samples == lc._block_n == 0
    assert np.all(np.isfinite(lc._wide_sum))
    assert lc._last_game_steer is None


@pytest.mark.parametrize("learner", ["params", "calib"])
def test_learner_never_saves_nonfinite_state(tmp_path, learner):
    if learner == "params":
        obj = LiveParams()
        path = tmp_path / "params.json"
        obj.save(path)
        before = path.read_bytes()
        obj.x[0] = np.nan
        attempt(lambda: obj.save(path))
    else:
        from simsteer.paths import state_path
        obj = LiveCalib()
        path = state_path("livecalib_state", "ets2")
        obj.save_state("ets2")
        before = path.read_bytes()
        obj._wide_sum[0] = np.nan
        attempt(lambda: obj.save_state("ets2"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["rpys", "heights", "wide_sum", "block_rpy_sum", "block_height_sum"])
def test_livecalib_rejects_entire_nonfinite_snapshot(field):
    from simsteer.paths import state_path
    data = dict(rpys=[[0., .01, .01]], heights=[1.2], wide_sum=[0., 0., 0.],
                block_rpy_sum=[0., 0., 0.], block_height_sum=0., samples=600)
    data[field] = ([[0., np.nan, 0.]] if field == "rpys" else
                   [np.nan] if field == "heights" else
                   np.nan if field == "block_height_sum" else [0., np.nan, 0.])
    state_path("livecalib_state", "ets2").write_text(json.dumps(data))
    lc = LiveCalib(game="ets2")
    assert lc.blocks == lc.samples == 0
    assert np.all(np.isfinite(lc._wide_sum))
    assert np.all(np.isfinite(lc._block_rpy_sum))


@pytest.mark.parametrize("bad", BAD)
def test_liveparams_rejects_nonfinite_snapshot(tmp_path, bad):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(dict(a=-.5, b=0., c=0., samples=bad)))
    loaded = LiveParams._load(path=path)
    assert loaded.samples == 0


def test_controller_fault_does_not_latch_full_steering():
    from simsteer.core.postprocess import decode
    d = decode(np.zeros(1576, np.float32), np.zeros(1000, np.float32))
    lat, lon = LateralController(), LongitudinalController()
    d.plan[:, 11] = np.nan
    result = attempt(lambda: lat.compute(d, 20.))
    assert result in (None, 0.)
    d.plan[:] = 0.
    assert lat.compute(d, 20.) == 0.
    result = attempt(lambda: lon.compute(d, np.nan))
    assert result in (None, (0., 0.))


def test_tuner_restart_only_requests_orderly_shutdown(monkeypatch):
    import os
    import subprocess
    from simsteer.ui.tuner import Tuner
    calls = []
    monkeypatch.setattr(os, "_exit", lambda *a: calls.append("exit"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a: calls.append("spawn"))
    tuner = Tuner(Calibration(), SimpleNamespace(), SimpleNamespace(), game="ets2", start_thread=False)
    tuner._selected_game = "ats"
    tuner._save_and_restart()
    assert calls == []
    assert tuner.take_restart_request()[-2:] == ["--game", "ats"]


def test_tuner_binding_only_enqueues_request(monkeypatch):
    from simsteer.ui.tuner import Tuner
    calls = []
    monkeypatch.setattr(threading, "Thread", lambda *a, **kw: SimpleNamespace(start=lambda: calls.append("thread")))
    device = SimpleNamespace(kind="gamepad", engaged=False, request_bind=lambda kind: calls.append(kind))
    tuner = Tuner(Calibration(), SimpleNamespace(), SimpleNamespace(), device=device, start_thread=False)
    tuner._wiggle("left_stick_x", "Steering")
    assert calls == ["left_stick_x"]


@pytest.mark.parametrize("exit_kind", ["atexit", 0, 1, 2, 5, 6])
def test_exit_hooks_neutralize_and_prevent_later_output(monkeypatch, exit_kind):
    import atexit
    import ctypes
    hooks = {}
    monkeypatch.setattr(atexit, "register", lambda fn: hooks.update(atexit=fn))
    monkeypatch.setattr(atexit, "unregister", lambda fn: None)
    # Fake just the output module's platform via os.name inside the hook
    # installer; no real Windows API or global platform mutation is needed.
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", lambda *a: lambda fn: fn, raising=False)
    kernel = SimpleNamespace(SetConsoleCtrlHandler=lambda fn, add:
                             hooks.update(console=fn) or 1)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: kernel, raising=False)

    def trigger(st, manager):
        if st.i == 1:
            callback = hooks.get("atexit" if exit_kind == "atexit" else "console")
            assert callback is not None, "no neutral-on-exit hook installed"
            callback() if exit_kind == "atexit" else callback(exit_kind)

    st, _, _ = drive(monkeypatch, on_grab=trigger)
    assert st.checkpoints[1][0]
    assert all(report == (0., 0., 0.) and not engaged
               for engaged, report in st.checkpoints[2:])


def test_binding_is_cancelled_if_engagement_changes_during_countdown(monkeypatch):
    def request(st, manager):
        if st.i == 0:
            # The manager is the public endpoint used by the tuner.
            assert hasattr(manager, "request_bind"), "binding bypasses the output owner"
            manager.request_bind("left_stick_x")
        if st.i == 50:
            manager.engage()

    st, pad, _ = drive(monkeypatch, force=False, frames=120, on_grab=request, speed=0.)
    assert not any(abs(report[0]) > .2 for report in pad.raw.sent)


def test_active_binding_uses_loop_thread_and_cancels_on_engagement(monkeypatch):
    cut = []

    def request(st, manager):
        if st.i == 0:
            manager.request_bind("left_stick_x")
        if st.i == 100:
            cut.append(len(manager.raw.sent))
            manager.engage()

    _, pad, _ = drive(monkeypatch, force=False, frames=145, on_grab=request, speed=0.)
    assert any(abs(report[0]) == 1. for report in pad.raw.sent[:cut[0]])
    assert all(report[0] == 0. for report in pad.raw.sent[cut[0]:])
    assert set(pad.raw.writers) == {threading.get_ident()}


def test_restart_spawns_only_after_neutral_save_and_capture_close(monkeypatch, data_dir):
    import subprocess
    calls = []

    def tuner(st, manager):
        def spawn(args):
            assert not manager.engaged and manager.raw.visible == (0., 0., 0.)
            assert st.telemetry_closed and st.capture_closed
            assert (data_dir / "livecalib_state_ets2.json").exists()
            calls.append(args)
        monkeypatch.setattr(subprocess, "Popen", spawn)
        return SimpleNamespace(update_state=lambda *a: None,
                               take_restart_request=lambda: ["simsteer", "--game", "ats"]
                               if st.i >= 0 else None)

    drive(monkeypatch, tuner_factory=tuner)
    assert calls == [["simsteer", "--game", "ats"]]
