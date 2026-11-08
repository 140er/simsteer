"""Driving after a saved calibration must never train the steering scale.

Uses only the pre-existing API so the same test exercises the unfixed base.
"""

import json
import math

import numpy as np
import pytest

from simsteer.app.probe import SteeringProbe
from simsteer.calib.flow import Step
from simsteer.core.learners.liveparams import LiveParams
from test_calib_flow import new_flow


@pytest.mark.parametrize("mode", ["engaged", "manual", "intervention", "probe"])
def test_driving_after_calibration_preserves_scale(mode, data_dir):
    path = data_dir / "liveparams_ets2_gamepad.json"
    path.write_text(json.dumps(dict(
        a=4.0, b=0.0, c=0.0, samples=300, axis_sign=1,
        coverage={"4": [75, 75], "8": [75, 75]}, coverage_version=1,
        coverage_speed_min=12.0, coverage_speed_max=24.0,
        scale_frozen=True, calibration_fit=True)))
    lp = LiveParams(game="ets2", device_kind="gamepad", lag_frames=0)
    flow = new_flow(lp=lp, persist=False)
    flow._enter(Step.CHECK)
    flow._set_step(Step.DONE)  # Leaving Check used to re-enable RLS.
    before, covariance, samples = lp.x.copy(), lp.P.copy(), lp.samples
    probe = SteeringProbe()
    for i in range(400):
        wheel = 0.03 * math.sin(i / 30)
        axis = 6.0 * wheel
        if mode == "probe":
            axis += probe.tick(0.05, axis, 20.0, False, True,
                               lp_trusted=lp.trusted(), calibrating=flow.probe_wanted)
        command = None if mode == "manual" else (axis - 0.3 if mode == "intervention" else axis)
        lp.update(axis, 20.0, wheel, commanded_axis=command)
    assert np.array_equal(lp.x, before), "driving changed the calibrated mapping"
    assert np.array_equal(lp.P, covariance)
    assert lp.samples == samples
    lp.save()
    loaded = LiveParams(game="ets2", device_kind="gamepad")
    assert np.array_equal(loaded.x, before)
