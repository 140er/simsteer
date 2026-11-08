"""LiveParams: the measured fit of `axis = a*wheel + b*wheel*v^2 + c`."""
from __future__ import annotations

import numpy as np
import pytest

from simsteer.core.learners.liveparams import LiveParams

N = 6000
_t = np.arange(N) / 20.0
# Slow enough to pass the quiescence gate, with speed varying so the
# a and b regressors are not collinear.
STEER = 0.5 * np.sin(2 * np.pi * 0.05 * _t) + 0.3 * np.sin(2 * np.pi * 0.13 * _t + 1)
SPEED = 18.0 + 8.0 * np.sin(2 * np.pi * 0.011 * _t)


# The Forza and AC seed and gates from before game profiles.
SYNTH_WHEEL = dict(device_kind="gamepad", seeds={"gamepad": (3.0, 0.0010, 0.0)},
                   gates="loose")


def _fresh(game: str | None = None, **kwargs) -> LiveParams:
    lp = LiveParams(game=game, **kwargs)
    lp.reset()
    return lp


def _drive(lp: LiveParams, a: float, b: float, frames: range,
           steer: np.ndarray = STEER) -> None:
    lag = lp.lag_frames
    for i in frames:
        j = max(i - lag, 0)
        v = float(SPEED[j])
        wheel = float(steer[j]) / (a + b * v * v) if i >= lag else 0.0
        lp.update(float(steer[i]), float(SPEED[i]), wheel)


def test_steering_step_fits_then_freezes():
    lp = _fresh(lag_frames=0)
    lp.begin_steering_fit()
    _drive(lp, a=4.0, b=0.0, frames=range(1000))
    assert lp.a_linear == 10.0  # Collection cannot install a partial fit.
    assert lp.finish_steering_fit()
    assert lp.a_linear == pytest.approx(4.0, rel=0.01)
    assert lp.scale_frozen and lp.auto_frozen and lp.trusted()
    before = lp.x.copy()
    _drive(lp, a=8.0, b=0.0, frames=range(1000, 2000))
    assert np.array_equal(lp.x, before)


def test_fit_holds_speed_term_at_profile_seed():
    lp = _fresh(lag_frames=0, **SYNTH_WHEEL)
    lp.begin_steering_fit()
    _drive(lp, a=4.0, b=0.001, frames=range(1000))
    assert lp.finish_steering_fit()
    assert lp.a_linear == pytest.approx(4.0)
    assert lp.b_quad == 0.001


def test_fast_input_changes_are_rejected_as_transient():
    rng = np.random.default_rng(0)
    jittery = np.convolve(rng.standard_normal(N + 30), np.ones(20) / 20.0, mode="same") * 1.5
    lp = _fresh()
    lp.begin_steering_fit()
    _drive(lp, a=10.0, b=0.0, frames=range(500), steer=jittery)
    assert lp.rej_transient > 400
    assert lp.samples < 50


def test_low_speed_is_rejected_and_speed_collapse_is_not():
    lp = _fresh()
    lp.begin_steering_fit()
    for i in range(100):
        lp.update(float(STEER[i]), 3.0, float(STEER[max(i - 5, 0)]) / 10.0)
    assert lp.samples == 0
    assert lp.rej_slow == 100
    for i in range(100, 200):
        lp.update(float(STEER[i]), 25.0, float(STEER[i - 5]) / 10.0)
    before = lp.samples
    for i in range(200, 300):
        lp.update(float(STEER[i]), 6.0, float(STEER[i - 5]) / 10.0)
    assert lp.rej_trouble == 0
    assert lp.samples > before


def test_interventions_pause_trim_without_training():
    lp = _fresh(lag_frames=0, clock=lambda: 123.0)
    lp.begin_steering_fit()
    before = lp.x.copy()
    for _ in range(200):
        lp.update(0.5, 20.0, 0.05, commanded_axis=0.05)
    assert lp.last_intervened_ts == 123.0
    assert lp.rej_assist == 200 and lp.samples == 0
    assert np.array_equal(lp.x, before)


def test_manual_driving_outside_steering_does_not_collect():
    lp = _fresh()
    _drive(lp, a=4.0, b=0.0, frames=range(1000))
    assert lp.samples == 0
    assert lp.a_linear == 10.0
    assert not lp.finish_steering_fit()
