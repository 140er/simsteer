"""Deterministic model/executor clocks and the live serialized writer."""
from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

import pytest

from simsteer.core.finite import NonFiniteError
from simsteer.core.steering import SteeringExecutor, SteeringTarget, limit_curvature
from simsteer.runtime.output.base import DeviceManager
from simsteer.runtime.output.steering import SteeringOutput
from test_output_safety import gamepad
from test_steering_direction import _plan


def target(axis=.4, speed=20., gain=4., bias=0., jerk=5.):
    return SteeringTarget(axis, speed, gain, bias, 4., jerk)


def curvature(axis, gain=4., bias=0.):
    return math.tan((axis - bias) / gain) / 4.


def test_executor_moves_between_model_frames_without_overshooting():
    executor = SteeringExecutor()
    executor.submit(target(), 0.)
    axes = [executor.tick(i * .01) for i in range(5)]
    assert all(0 < a < b < .4 for a, b in zip(axes, axes[1:]))
    executor.submit(target(), .05)
    assert executor.tick(.05) > axes[-1]


@pytest.mark.parametrize("speed", [0., 1., 10., 20., 30.])
@pytest.mark.parametrize("gain,bias", [(4., 0.), (-4., .01), (12., -.02)])
def test_final_command_jerk_bound_including_bias_and_reversals(speed, gain, bias):
    executor = SteeringExecutor()
    previous, last = 0., -.01
    for tick in range(100):
        now = tick * .01
        if tick % 5 == 0:
            executor.submit(target(.4 if tick < 50 else -.4, speed, gain, bias), now)
        axis = executor.tick(now)
        jerk = (max(speed, 1.) ** 2
                * abs(curvature(axis, gain, bias) - curvature(previous, gain, bias))
                / (now - last))
        assert jerk <= 5. + 1e-9
        previous, last = axis, now


def test_low_rack_gain_clamps_wheel_instead_of_faulting():
    # gain .5 maps a full axis to 2 rad, past tan()'s usable range.
    executor = SteeringExecutor(smoothing_s=0.)
    axes = []
    for i in range(400):
        if i % 10 == 0:  # fresh 10 Hz plans keep the target alive
            executor.submit(target(axis=1., speed=1., gain=.5), i * .01)
        axes.append(executor.tick((i + 1) * .01))
    assert all(axis is not None and math.isfinite(axis) for axis in axes)
    assert axes[-1] == pytest.approx(.5 * 1.5)


def test_variable_frame_intervals_and_missed_frame_do_not_change_executor_clock():
    executor = SteeringExecutor()
    updates = {0: .3, 4: -.3, 11: .3, 21: -.3}
    previous, last = 0., -.01
    for tick in range(35):
        now = tick * .01
        if tick in updates:
            executor.submit(target(updates[tick]), now)
        axis = executor.tick(now)
        assert axis is not None
        assert 400 * abs(curvature(axis) - curvature(previous)) <= 5 * (now - last) + 1e-12
        previous, last = axis, now


@pytest.mark.parametrize("time", [.3, -.01])
def test_stall_or_reversed_time_clears_state_even_after_a_fresh_submit(time):
    executor = SteeringExecutor()
    executor.submit(target(), 0.)
    executor.tick(0.)
    executor.submit(target(-.4), time)
    assert executor.tick(time) is None
    assert executor.target is None and executor.axis == 0.
    executor.submit(target(), 1.)
    assert 0 < executor.tick(1.) < .01


def test_expired_plan_does_not_continue_or_reappear():
    executor = SteeringExecutor()
    executor.submit(target(), 0.)
    for tick in range(26):
        executor.tick(tick * .01)
    assert executor.tick(.26) is None
    assert executor.tick(.27) is None


def test_reset_on_disengage_discards_target_and_history():
    executor = SteeringExecutor()
    executor.submit(target(), 0.)
    executor.tick(0.)
    executor.reset()
    assert executor.tick(.01) is None
    executor.submit(target(-.4), .02)
    assert -.01 < executor.tick(.02) < 0


@pytest.mark.parametrize("dt", [0., -.1, .01, .037, .05, .12])
def test_curvature_limit_and_controller_use_explicit_dt(dt):
    from simsteer.core.control import ControllerConfig, LateralController
    from simsteer.core.learners.liveparams import LiveParams
    assert limit_curvature(.1, 0., 20., dt) == pytest.approx(5 * max(dt, 0.) / 400)
    controller = LateralController(ControllerConfig(wheel_trim_enabled=False), LiveParams())
    controller.compute(_plan(.1, 20.), 20., dt=dt)
    assert controller.last_desired_k_raw == pytest.approx(5 * max(dt, 0.) / 400)


@pytest.mark.parametrize("field", ["axis", "speed", "rack_gain", "rack_bias", "wheelbase", "max_jerk"])
def test_nonfinite_target_is_rejected(field):
    executor = SteeringExecutor()
    values = vars(target()) | {field: float("nan")}
    with pytest.raises(NonFiniteError):
        executor.submit(SteeringTarget(**values), 0.)


def writer(monkeypatch, *, allowed=lambda: True, threaded=False):
    import simsteer.runtime.output.base as devices
    pad = gamepad()
    monkeypatch.setattr(devices, "Gamepad", lambda: pad)
    manager = DeviceManager()
    pad.engage()
    output = SteeringOutput(manager, SimpleNamespace(now=time.monotonic), allowed,
                            threaded=threaded)
    return output, pad, manager


def test_writer_stall_neutralizes_pedals_and_requires_reengagement(monkeypatch):
    output, pad, manager = writer(monkeypatch)
    pad.set_throttle_brake(.6, 0.)
    output.submit(target(), 0.)
    output.tick(0.)
    assert any(pad.raw.visible)
    output.tick(.3)
    assert pad.raw.visible == (0., 0., 0.) and not pad.engaged
    assert output.take_fault() == "Steering plan expired"
    count = len(pad.raw.sent)
    output.submit(target(), .31)
    output.tick(.31)
    assert len(pad.raw.sent) == count
    manager.shutdown_output()


@pytest.mark.parametrize("action", ["reset", "disengage", "shutdown", "override", "binding", "replacement"])
def test_writer_cannot_revive_cancelled_or_unqualified_output(monkeypatch, action):
    state = SimpleNamespace(allowed=True)
    output, pad, manager = writer(monkeypatch, allowed=lambda: state.allowed)
    output.submit(target(), 0.)
    output.tick(0.)
    if action == "reset":
        output.reset()
    elif action == "disengage":
        manager.disengage()
    elif action == "shutdown":
        manager.shutdown_output()
    elif action == "override":
        state.allowed = False
    elif action == "binding":
        manager._binding.request = ("steer", pad)
    else:
        manager._device = gamepad()
        manager._device.engage()
    count = len(pad.raw.sent)
    output.tick(.01)
    assert len(pad.raw.sent) == count
    manager.shutdown_output()


def test_worker_paces_writes_while_model_is_blocked_and_joins_on_close(monkeypatch):
    output, pad, manager = writer(monkeypatch, threaded=True)
    enough = threading.Event()
    original = pad.set_steering
    axes = []

    def record(axis):
        original(axis)
        axes.append(axis)
        if len(axes) == 5:
            enough.set()

    monkeypatch.setattr(pad, "set_steering", record)
    try:
        output.submit(target(), time.monotonic())
        assert enough.wait(.5)
        output.close()
        count = len(axes)
        assert 5 <= count <= 15
        assert all(a < b for a, b in zip(axes, axes[1:]))
        assert not output._thread.is_alive()
    finally:
        output.close()
        manager.shutdown_output()


@pytest.mark.parametrize("fault", [None, "vision", "lateral", "learner", "lost"])
def test_live_loop_keeps_fault_paths_and_model_rate_trim(monkeypatch, fault):
    import simsteer.runtime.loop as loop
    from test_output_safety import drive

    # Exercise the live output branch with deterministic 10 ms substeps.
    monkeypatch.setattr(loop, "LiveClock", SimpleNamespace)
    outputs, calls = [], []
    original = loop.LateralController.compute

    def compute(self, *args, **kwargs):
        calls.append(kwargs["dt"])
        return original(self, *args, **kwargs)

    class Output(SteeringOutput):
        def __init__(self, pad, clock, allowed):
            super().__init__(pad, clock, allowed, threaded=False)
            outputs.append(self)

        def submit(self, target, now):
            super().submit(target, now)
            for i in range(5):
                self.tick(now + i * .01)

    monkeypatch.setattr(loop, "SteeringOutput", Output)
    monkeypatch.setattr(loop.LateralController, "compute", compute)
    st, pad, _ = drive(monkeypatch, fault=fault)
    assert len(outputs) == 1
    assert len(calls) <= 4  # Five output ticks never rerun trim/model control.
    assert all(dt == pytest.approx(.05) for dt in calls)
    assert st.checkpoints[1][0] and any(st.checkpoints[1][1])
    if fault:
        assert st.checkpoints[2] == (False, (0., 0., 0.))
    assert not pad.engaged and pad.raw.visible == (0., 0., 0.)


def test_legacy_rate_limit_opt_out_and_zero_jerk():
    from simsteer.core.postprocess import desired_curvature_lag_adjusted
    plan = _plan(.1, 20.).plan
    default = desired_curvature_lag_adjusted(plan, 20., .3)
    assert desired_curvature_lag_adjusted(plan, 20., .3, dt=.1) == pytest.approx(2 * default)
    assert limit_curvature(.1, .01, 20., .05, max_jerk=0.) == .01


def test_smoothing_is_time_based_when_jerk_limit_is_disabled():
    regular, variable = SteeringExecutor(), SteeringExecutor()
    for executor in (regular, variable):
        executor.submit(target(jerk=None), 0.)
    for now in (0., .01, .02, .03, .04):
        regular.tick(now)
    for now in (0., .007, .025, .04):
        variable.tick(now)
    assert regular.axis == pytest.approx(variable.axis)
    assert regular.axis == pytest.approx(.4 * (1 - math.exp(-1)))


def test_loop_control_dt_includes_inference_and_resets_after_stall(monkeypatch):
    import simsteer.runtime.loop as loop
    from test_output_safety import drive

    state, intervals = {}, []
    original_decode, original_compute = loop.decode, loop.LateralController.compute

    def decode(*args):
        # Inference completion advances the same explicit clock as capture.
        st = state["st"]
        st.t += .3 if st.i == 2 else .02
        return original_decode(*args)

    def compute(self, *args, **kwargs):
        intervals.append(kwargs["dt"])
        return original_compute(self, *args, **kwargs)

    monkeypatch.setattr(loop, "decode", decode)
    monkeypatch.setattr(loop.LateralController, "compute", compute)
    drive(monkeypatch, on_grab=lambda st, manager: state.update(st=st))
    assert intervals == pytest.approx([.05, .07, .05, .07])


def test_concurrent_disengage_cannot_be_followed_by_worker_write(monkeypatch):
    output, pad, manager = writer(monkeypatch, threaded=True)
    written = threading.Event()
    original = pad.set_steering

    def record(axis):
        original(axis)
        written.set()

    monkeypatch.setattr(pad, "set_steering", record)
    try:
        output.submit(target(), time.monotonic())
        assert written.wait(.5)
        manager.disengage()
        count = len(pad.raw.sent)
        # Run an explicitly scheduled late tick while the worker is alive.
        output.tick(time.monotonic())
        output.close()
        assert len(pad.raw.sent) == count
        assert pad.raw.visible == (0., 0., 0.)
    finally:
        output.close()
        manager.shutdown_output()


@pytest.mark.parametrize("failure", ["command", "permission"])
def test_writer_exceptions_stop_all_output(monkeypatch, failure):
    def fail(*args):
        raise RuntimeError("output unavailable")

    output, pad, manager = writer(monkeypatch, allowed=fail if failure == "permission" else lambda: True)
    pad.set_throttle_brake(.6, 0.)
    if failure == "command":
        monkeypatch.setattr(pad, "set_steering", fail)
    output.submit(target(), 0.)
    output.tick(0.)
    assert not pad.engaged and pad.raw.visible == (0., 0., 0.)
    assert output.executor.target is None
    assert output.take_fault() == "Steering executor failed"
    manager.shutdown_output()
