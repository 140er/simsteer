"""Public-API camera preservation regression, also runnable on older sources."""
from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from simsteer.calib.flow import CalibFlow, Step, flow_path
from simsteer.core.calibration import Calibration
from simsteer.core.learners.livecalib import LiveCalib, CalStatus
from simsteer.core.learners.liveparams import LiveParams
from simsteer.core.model_identity import pinned_model_identity
from simsteer.paths import state_path


def make_flow(device="gamepad", *, cal=None, lc=None, steering_fit=False):
    return CalibFlow("ets2", device, calib=cal if cal is not None else Calibration.load("ets2"),
                     live_calib=lc if lc is not None else LiveCalib(game="ets2"),
                     live_params=LiveParams(game="ets2", device_kind=device),
                     model_identity=pinned_model_identity(),
                     steering_fit=steering_fit)


def legacy_devices():
    seed = make_flow()
    for device in ("gamepad", "wheel"):
        data = seed.to_dict()
        data.update(device=device, step="done", attempts=1,
                    result={"passed": True, "score": 1.0},
                    legacy_result={"passed": True}, check_model=[4., .001, .01],
                    check_context={"camera": {"pitch_deg": 7.}}, check_requested=True,
                    redo_note="old device evidence")
        data["check"]["seconds"] = 120.
        data.pop("simple_warp_version")
        flow_path("ets2", device).write_text(json.dumps(data))
    calibration = asdict(seed.calib)
    calibration.update(pitch_deg=7., yaw_deg=1., simple_warp_version=1)
    state_path("calibration", "ets2").write_text(json.dumps(calibration))


def measure(current, count=300, **changes):
    for _ in range(count):
        values = dict(pose=np.array([10., .12, -.24]),
                      road_transform=np.array([0., 0., 1.4]), actual_yaw_rate=0.,
                      v_ego=10., pose_std=np.array([.01, .01, .01]),
                      road_transform_std=np.full(3, .01), game_steer=0.)
        current.lc.update(current.calib, **(values | changes))


def fresh_camera():
    legacy_devices()
    current = make_flow()
    assert current.lc.samples == current.lc.blocks == 0
    assert current.calib.pitch_deg == 0.
    measure(current)
    assert current.lc.cal_status == CalStatus.CALIBRATED
    assert abs(current.calib.pitch_deg) > .5
    assert abs(current.calib.yaw_deg) > .25
    current.save()
    return current


@pytest.mark.parametrize("route", ["constructor", "restart", "switch"])
@pytest.mark.parametrize("steering_fit", [False, True])
def test_second_legacy_device_preserves_fresh_camera(route, steering_fit):
    first = fresh_camera()
    pose = asdict(first.calib)
    history = first.lc.state_dict()
    generation = first.lc.reset_generation
    if route == "switch":
        second = first
        second.steering_fit = steering_fit
        second.switch_device("wheel")
    elif route == "restart":
        second = make_flow("wheel", steering_fit=steering_fit)
    else:
        second = make_flow("wheel", cal=first.calib, lc=first.lc, steering_fit=steering_fit)
    # On old code this assertion fails: the second migration zeros pose.
    assert asdict(second.calib) == pose
    assert second.lc.state_dict() == history
    assert second.lc.reset_generation == generation
    assert second.result is second.legacy_result is None
    assert second.attempts == second.check.seconds == 0
    assert second.redo_note == ""
    assert not second.calibrated
    assert second.step in (Step.STEERING, Step.CHECK)
    if second._check_context is not None:
        assert second._check_context["device"] == "wheel"
        assert second._check_context["camera"] == pose
    if steering_fit:
        assert not second.ai_steering_ready and not second.direction.ready
    else:
        assert not second._check_requested and not second.ai_steering_ready
    # Saving the migrated device must prevent a second migration on resume.
    second.save()
    resumed = make_flow("wheel", steering_fit=steering_fit)
    assert asdict(resumed.calib) == pose
    assert resumed.lc.reset_generation == generation
