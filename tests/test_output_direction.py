"""The device binding and wheel telemetry have independent polarities."""

import json

import pytest

from simsteer.calib.flow import CalibFlow, Step, flow_path
from simsteer.calib.steering import OutputDirection
from test_steering_direction import _bends, _flow, _probe_output


def test_recalibrate_clears_the_saved_input_sign():
    from simsteer.core.calibration import Calibration
    from simsteer.core.learners.livecalib import LiveCalib
    from simsteer.core.learners.liveparams import LiveParams

    lp = LiveParams(game="ets2", device_kind="gamepad")
    lp.set_axis_sign(-1)
    flow = CalibFlow("ets2", "gamepad", live_calib=LiveCalib(game="ets2"),
                     live_params=lp, calib=Calibration())
    flow.restart("recalibrate")
    lp.save()
    assert lp.axis_sign is None
    assert json.loads(lp._resolve_path("ets2", "gamepad").read_text())["axis_sign"] is None


@pytest.mark.parametrize("gate", ["moving", "reversing", "missing", "player", "override"])
def test_stopped_check_rejects_unsafe_start(gate):
    direction = OutputDirection(lag_frames=5)
    assert not direction.start(speed=2.0 if gate == "moving" else -1.0 if gate == "reversing" else 0,
                               observed=None if gate == "missing" else 0.0,
                               user_axis=0.2 if gate == "player" else 0,
                               manual_override=gate == "override")
    assert not direction.active and not direction.ready and direction.reason


def test_driving_observations_cannot_verify_output_or_restore_old_evidence():
    direction = OutputDirection(lag_frames=5)
    direction.load_dict({"sign": -1, "pairs": [[-0.1, 0.1], [0.1, -0.1]] * 50})
    for i in range(400):
        command = 0.1 if i % 100 < 50 else -0.1
        assert not direction.update(command=command, observed=-command, speed=22,
                                    engaged=True, lane_change=False, dt=0.05)
    assert not direction.ready
    assert direction.to_dict()["pairs"] == []


def test_output_verification_is_per_device_and_cleared_on_switch_and_reset():
    from simsteer.core.learners.livecalib import LiveCalib
    from simsteer.core.learners.liveparams import LiveParams

    lp = LiveParams(game="ets2", device_kind="gamepad")
    flow = _flow(lp)
    _bends(flow, 1, -1)
    _probe_output(flow, -1)
    flow.save()
    saved = CalibFlow("ets2", "gamepad", live_calib=flow.lc, live_params=lp, calib=flow.calib)
    assert saved.ai_steering_ready and saved.output_direction.sign == -1
    lp.switch_device("wheel")
    saved.switch_device("wheel")
    assert not saved.ai_steering_ready and not saved.output_direction.ready
    lp.switch_device("gamepad")
    saved.switch_device("gamepad")
    assert saved.direction.ready and not saved.output_direction.ready
    assert not saved.ai_steering_ready
    assert not CalibFlow("ats", "gamepad", live_calib=flow.lc,
                         live_params=lp, calib=flow.calib).output_direction.ready
    _probe_output(saved, 1)
    saved.lc = LiveCalib(game="ets2")
    saved.restart("R pressed")
    assert saved.output_direction.sign is None and lp.axis_sign is None
    assert flow_path("ets2", "gamepad").exists()


def test_previous_version_done_requires_output_verification():
    from simsteer.core.learners.liveparams import LiveParams

    lp = LiveParams(game="ets2", device_kind="gamepad")
    flow = _flow(lp)
    _bends(flow, 1, 1)
    state = flow.to_dict()
    state["version"] = 2
    state["step"] = "done"
    state["result"] = {"passed": True}
    state["steering"].pop("output_direction")
    flow_path("ets2", "gamepad").write_text(json.dumps(state))
    restored = CalibFlow("ets2", "gamepad", live_calib=flow.lc,
                         live_params=lp, calib=flow.calib)
    assert restored.step is Step.STEERING and not restored.calibrated
    assert restored.device_axis(0.2) == 0.0
    assert not restored.output_direction.ready
