"""ETS2's vJoy wheel: measured steering scale and no gamepad input lag.

A live vJoy drive measured the wheel angle at 0.71 rad per unit of game
input, moving in the same frame as that input. With the gamepad's
5-frame lag and a 4.0 seed, the controller oversteered about 2.8 times
and the guided Steering fit never passed its residual check.
"""

from __future__ import annotations

import math

import pytest

from simsteer.core.learners.liveparams import LiveParams
from simsteer.games.base import get_profile


def test_ets2_wheel_profile_uses_the_measured_scale_and_lag():
    lp = get_profile("ets2").liveparams
    assert lp.seeds["wheel"] == (1.41, 0.0, 0.0)
    assert getattr(lp, "lags", {}).get("wheel") == 1
    assert "gamepad" not in getattr(lp, "lags", {})  # keeps the gate's 5 frames


def ets2_params(device: str) -> LiveParams:
    spec = get_profile("ets2").liveparams
    return LiveParams(game="ets2", device_kind=device, seeds=spec.seeds,
                      gates=spec.gates, lags=spec.lags)


def test_lag_follows_the_device_and_a_switch():
    lp = ets2_params("wheel")
    assert lp.lag_frames == 1
    lp.switch_device("gamepad")
    assert lp.lag_frames == 5 and lp._axis_buf.maxlen == 6
    lp.switch_device("wheel")
    assert lp.lag_frames == 1 and lp._axis_buf.maxlen == 2


def test_explicit_lag_wins_over_the_profile():
    spec = get_profile("ets2").liveparams
    lp = LiveParams(lag_frames=3, game="ets2", device_kind="wheel", seeds=spec.seeds,
                    gates=spec.gates, lags=spec.lags)
    lp.switch_device("gamepad")
    assert lp.lag_frames == 3


def test_steering_fit_passes_on_same_frame_wheel_data():
    lp = ets2_params("wheel")
    lp.set_axis_sign(1)
    lp.begin_steering_fit()
    for i in range(600):  # 30 s of gentle weaving at 20 Hz
        wheel = 0.04 * math.sin(i / 12) + 0.015 * math.sin(i / 3.1)
        lp.update(1.41 * wheel, 20.0, wheel, commanded_axis=1.41 * wheel)
    assert lp.finish_steering_fit(), lp.fit_reason
    assert lp.a_linear == pytest.approx(1.41, rel=0.03)
