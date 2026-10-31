"""LiveParams: the online fit of `axis = a*wheel + b*wheel*v^2 + c`."""
from __future__ import annotations

import numpy as np
import pytest

from simsteer.core.learners.liveparams import (
    FORGET,
    TRUSTED_MIN_SAMPLES,
    LiveParams,
)

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


def test_learns_the_rack_from_its_seed():
    lp = _fresh("forza", **SYNTH_WHEEL)
    assert lp.a_linear == pytest.approx(3.0)
    assert lp.lag_frames == 1

    _drive(lp, a=4.0, b=0.0, frames=range(4000))

    assert lp.a_linear == pytest.approx(4.0, rel=0.02)
    assert 1.0 / lp.effective_scale_at(20.0) == pytest.approx(4.0, rel=0.02)


def test_ets2_profile_converges_toward_the_rack():
    lp = _fresh()
    assert lp.a_linear == pytest.approx(10.0)

    _drive(lp, a=8.0, b=0.0, frames=range(4000))

    assert 7.8 < lp.a_linear < 8.6


def test_cold_fit_forgets_faster_while_wrong_then_settles():
    lp = _fresh()
    _drive(lp, a=10.0 / 3.0, b=0.0, frames=range(40))
    assert not lp.trusted()
    assert lp.adaptive_forget() < FORGET

    _drive(lp, a=10.0 / 3.0, b=0.0, frames=range(40, 400))
    # The first short speed sweep has not covered both turns in separated
    # speed ranges. More samples alone used to claim trust here.
    assert not lp.trusted()
    _drive(lp, a=10.0 / 3.0, b=0.0, frames=range(400, 2000))
    assert lp.trusted()
    assert lp.adaptive_forget() == FORGET


def test_cold_watchdog_resets_covariance_on_persistent_misfit():
    lp = _fresh()
    _drive(lp, a=10.0 / 3.0, b=0.0, frames=range(400))
    assert lp.watchdog_resets >= 1


@pytest.mark.parametrize("new_a", [10.0 / 3.0, -10.0], ids=["new-car", "sign-flip"])
def test_trusted_fit_does_not_run_away(new_a):
    # Trusted fits deliberately skip fast forgetting and the watchdog,
    # after a highway fit ran away from a~4 to a~19 on one detour.
    lp = _fresh()
    _drive(lp, a=10.0, b=0.0, frames=range(500))
    assert lp.samples >= TRUSTED_MIN_SAMPLES
    resets = lp.watchdog_resets

    _drive(lp, a=new_a, b=0.0, frames=range(500, 2000))

    assert lp.a_linear == pytest.approx(10.0, rel=0.05)
    assert lp.watchdog_resets == resets
    assert lp.rej_innovation > 0


def test_fast_input_changes_are_rejected_as_transient():
    rng = np.random.default_rng(0)
    jittery = np.convolve(rng.standard_normal(N + 30), np.ones(20) / 20.0, mode="same") * 1.5
    lp = _fresh()
    _drive(lp, a=10.0, b=0.0, frames=range(500), steer=jittery)
    assert lp.rej_transient > 400
    assert lp.samples < 50


def test_low_speed_is_rejected_and_speed_collapse_is_not():
    lp = _fresh()
    for i in range(100):
        lp.update(float(STEER[i]), 3.0, float(STEER[max(i - 5, 0)]) / 10.0)
    assert lp.samples == 0
    assert lp.rej_slow == 100

    # The speed-collapse detector stays off: it latched after every pause.
    for i in range(100, 200):
        lp.update(float(STEER[i]), 25.0, float(STEER[i - 5]) / 10.0)
    before = lp.samples
    for i in range(200, 300):
        lp.update(float(STEER[i]), 6.0, float(STEER[i - 5]) / 10.0)
    assert lp.rej_trouble == 0
    assert lp.samples > before


class TestUserAssistGate:
    TRUE_SCALE = 0.10
    COMMANDED = 0.05

    def _step(self, lp: LiveParams, user_input: float, commanded=COMMANDED) -> None:
        game_steer = (commanded or 0.0) + user_input
        lp.update(game_steer, 20.0, self.TRUE_SCALE * game_steer, commanded_axis=commanded)

    def test_fit_skips_while_user_overrides_and_resumes_after(self):
        lp = _fresh()
        for _ in range(200):
            self._step(lp, 0.0)
        assert lp.session_samples == 200 - lp.lag_frames
        assert lp.rej_assist == 0

        pre, pre_rej = lp.session_samples, lp.rej_assist
        for _ in range(200):
            self._step(lp, 0.5)
        assert lp.session_samples == pre
        assert lp.rej_assist - pre_rej >= 200 - 2 * lp.lag_frames

        pre = lp.session_samples
        for _ in range(200):
            self._step(lp, 0.0)
        assert lp.session_samples - pre >= 200 - 2 * lp.lag_frames
        assert lp.scale == pytest.approx(self.TRUE_SCALE, rel=0.02)

    def test_disengaged_samples_bypass_the_gate(self):
        lp = _fresh()
        for _ in range(200):
            lp.update(0.3, 20.0, self.TRUE_SCALE * 0.3, commanded_axis=None)
        assert lp.session_samples == 200 - lp.lag_frames
        assert lp.rej_assist == 0
