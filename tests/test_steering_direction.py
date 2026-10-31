"""Calibration must change actual control for an inverted input axis."""

from __future__ import annotations

import json
import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.calib.flow import CalibFlow, Obs, Step, flow_path
from simsteer.calib.steering import SteeringDirection
from simsteer.core.calibration import Calibration
from simsteer.core.constants import IDX_N, PLAN_WIDTH, T_IDXS
from simsteer.core.control import ControllerConfig, LateralController
from simsteer.core.learners.livecalib import CalStatus
from simsteer.core.learners.liveparams import LiveParams, TRUSTED_MIN_SAMPLES


def _flow(lp, **kwargs):
    camera = SimpleNamespace(cal_status=CalStatus.CALIBRATED,
                             blocks=3, _block_n=0, samples=300)
    flow = CalibFlow("ets2", lp.device_kind, live_calib=camera,
                     live_params=lp, calib=Calibration(fov_h_deg=75.0),
                     **kwargs)
    flow._finish_fov("kept")
    return flow


def _bends(flow, model_sign, axis_sign, frames=400):
    """An independent plant: raw axis -> delayed physical wheel.

    The model observes the road bend in its own coordinate system.
    Neither observation comes from the controller's command.
    """
    lag = flow.lp.lag_frames
    for i in range(frames):
        wheel = 0.018 * math.sin(2 * math.pi * max(0, i - lag) / 120)
        raw_axis = axis_sign * 4.0 * 0.018 * math.sin(2 * math.pi * i / 120) + 0.01
        flow.tick(Obs(now=i * 0.05, dt=0.05, v=22.0, wheel=wheel,
                      observed_axis=raw_axis, model_curvature=model_sign * wheel / 2.65,
                      engaged=False))
        if flow.direction.ready:
            return
    raise AssertionError("both bends did not establish steering direction")


def _plan(curvature, speed=22.0):
    plan = np.zeros((IDX_N, PLAN_WIDTH), dtype=np.float32)
    plan[:, 3] = speed
    plan[:, 11] = np.asarray(T_IDXS) * speed * curvature
    plan[:, 14] = speed * curvature
    return SimpleNamespace(plan=plan, desire_state=np.zeros(8),
                           pose_std=None, road_transform=None, road_transform_std=None)


def _probe_output(flow, sign):
    assert flow.output_direction.start(speed=0.0, observed=0.0)
    commands = deque([0.0] * (flow.lp.lag_frames + 1),
                     maxlen=flow.lp.lag_frames + 1)
    for _ in range(300):
        command = flow.measure_output(speed=0.0, observed=sign * commands[0], dt=0.05)
        commands.append(command)
        if flow.output_direction.ready:
            return
    raise AssertionError("stopped output check did not establish direction")


@pytest.mark.parametrize("model_sign", [1, -1])
@pytest.mark.parametrize("axis_sign", [1, -1], ids=["normal-input", "inverted-input"])
def test_manual_calibration_corrects_actual_output_and_persists(model_sign, axis_sign):
    lp = LiveParams(game="ets2", device_kind="gamepad",
                    seeds={"gamepad": (4.0, 0.0, 0.01)}, clock=lambda: 100.0)
    lp.samples = TRUSTED_MIN_SAMPLES  # Include an obsolete trusted positive seed.
    flow = _flow(lp)
    assert flow.step is Step.STEERING
    assert flow.engage_block() and not flow.probe_wanted

    _bends(flow, model_sign, axis_sign)

    assert flow.direction.model_sign == model_sign
    assert flow.direction.axis_sign == axis_sign
    assert flow.engage_block()
    assert not flow.ai_steering_ready
    _probe_output(flow, -1)
    assert flow.ai_steering_ready
    assert lp.a_linear * axis_sign > 0
    if axis_sign == -1:
        assert not lp.trusted()  # Wrong trusted gain was cleared, not cosmetically flipped.
    assert flow.calib.lateral_sign == -1  # Projection remains unchanged.

    controller = LateralController(ControllerConfig(wheelbase_m=2.65,
                                                   wheel_trim_enabled=False), lp)
    for bend in (-0.004, 0.004):
        controller.reset()
        for _ in range(30):
            command = controller.compute(_plan(bend), 22.0,
                                         wheel_sign=flow.steering_sign)
        physical_wheel = (command - lp.c_bias) / (axis_sign * 4.0)
        assert physical_wheel == pytest.approx(math.atan(model_sign * bend * 2.65),
                                               abs=1e-6)

    flow.save()
    again_lp = LiveParams(game="ets2", device_kind="gamepad",
                          seeds=lp.seeds, clock=lambda: 100.0)
    again = CalibFlow("ets2", "gamepad", live_calib=flow.lc,
                      live_params=again_lp, calib=flow.calib)
    assert again.direction.ready and again.steering_sign == model_sign
    assert again.output_direction.sign == -1
    assert again_lp.axis_sign == axis_sign
    again_lp.reset()
    assert again_lp.a_linear * axis_sign > 0

    again_lp.switch_device("vjoy")
    again.switch_device("vjoy")
    assert again.step is Step.FOV and not again.direction.ready
    assert again_lp.axis_sign is None
    assert CalibFlow("ac", "gamepad", live_calib=flow.lc,
                     live_params=again_lp, calib=flow.calib).step is Step.FOV

    again_lp.switch_device("gamepad")
    again.switch_device("gamepad")
    assert again.direction.ready and again.steering_sign == model_sign
    assert again_lp.axis_sign == axis_sign


@pytest.mark.parametrize("reason", ["single-turn", "straight", "noise", "engaged",
                                    "missing-wheel", "slow", "lane-change"])
def test_ambiguous_or_non_manual_evidence_cannot_unlock_output(reason):
    direction = SteeringDirection(lag_frames=1)
    rng = np.random.default_rng(7)
    for i in range(600):
        wheel = 0.015 * math.sin(i / 20)
        if reason == "single-turn":
            wheel = abs(wheel)
        if reason == "straight":
            wheel = 0.0
        curvature = wheel / 2.65 if reason != "noise" else float(rng.normal(0, 0.01))
        direction.update(curvature=curvature, axis=-4 * wheel,
                         wheel=None if reason == "missing-wheel" else wheel,
                         speed=3.0 if reason == "slow" else 22.0,
                         manual=reason != "engaged", lane_change=reason == "lane-change")
    assert not direction.ready


def test_partial_manual_direction_evidence_resumes():
    direction = SteeringDirection(lag_frames=1)
    for i in range(60):
        wheel = 0.005 + i * 0.0001
        direction.update(curvature=wheel / 2.65, axis=-4 * wheel, wheel=wheel,
                         speed=22.0, manual=True)
    saved = json.loads(json.dumps(direction.to_dict()))
    restored = SteeringDirection(lag_frames=1)
    restored.load_dict(saved)
    assert not restored.ready
    for i in range(60):
        wheel = -0.005 - i * 0.0001
        restored.update(curvature=wheel / 2.65, axis=-4 * wheel, wheel=wheel,
                        speed=22.0, manual=True)
    assert restored.model_sign == 1 and restored.axis_sign == -1


@pytest.mark.parametrize("axis_sign", [1, -1])
def test_trim_feedback_corrects_the_physical_wheel_for_either_input_sign(axis_sign):
    lp = LiveParams(game="ets2", device_kind="gamepad", clock=lambda: 100.0)
    lp.set_axis_sign(axis_sign)
    lp.samples = TRUSTED_MIN_SAMPLES
    # This test isolates trim direction using a qualified fit fixture;
    # sample count alone no longer establishes that qualification.
    lp.coverage = {"4": [25, 25], "8": [25, 25]}
    controller = LateralController(ControllerConfig(), lp)
    command = controller.compute(_plan(0.0), 22.0, actual_wheel_angle=-0.005)
    assert controller.last_trim_frozen_reason == ""
    # A negative measured wheel on a straight road needs positive physical correction.
    assert command * axis_sign > 0


def test_saved_check_without_direction_evidence_reenters_steering():
    lp = LiveParams(game="ets2", device_kind="gamepad")
    flow = _flow(lp)
    state = flow.to_dict()
    state["step"] = "done"
    state["result"] = {"passed": True}
    flow_path("ets2", "gamepad").write_text(json.dumps(state))
    restored = CalibFlow("ets2", "gamepad", live_calib=flow.lc,
                         live_params=lp, calib=flow.calib)
    assert restored.step is Step.STEERING and restored.engage_block()
    assert not restored.calibrated
