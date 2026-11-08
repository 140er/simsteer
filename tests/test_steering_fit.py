"""A calibration measurement needs excitation, two turns, and a good residual."""

import math

import numpy as np
import pytest

from simsteer.calib.flow import Step
from simsteer.core.learners.liveparams import LiveParams
from test_calib_flow import Driver, FakeLiveCalib, new_flow
from test_calib_score import _drive as check_drive
from simsteer.calib.score import evaluate


def samples(lp, a=4.0, b=0.0, c=0.0, noise=0.0, wheel_scale=0.02,
            one_sided=False, count=300, speed=20.0):
    rng = np.random.default_rng(17)
    for i in range(count):
        wheel = wheel_scale * math.sin(i / 12)
        if one_sided:
            wheel = abs(wheel)
        lp.update((a + b * speed ** 2) * wheel + c + rng.normal(0, noise),
                  speed, wheel)


@pytest.mark.parametrize("a,noise", [(4.0, 0.0), (-4.0, 0.0), (4.0, 0.002)])
def test_through_origin_recovers_known_gain(a, noise):
    lp = LiveParams(lag_frames=0)
    lp.set_axis_sign(1 if a > 0 else -1)
    lp.begin_steering_fit()
    before = lp.x.copy()
    samples(lp, a=a, noise=noise)
    assert np.array_equal(lp.x, before)
    assert lp.finish_steering_fit()
    assert lp.a_linear == pytest.approx(a, rel=0.02)
    assert lp.b_quad == 0 and lp.c_bias == 0
    assert lp.axis_for_wheel_angle(0, 20) == 0
    assert lp.scale_frozen and lp.trusted()


def test_origin_solve_does_not_fit_an_intercept():
    lp = LiveParams(lag_frames=0)
    lp.begin_steering_fit()
    samples(lp, c=0.01)
    data = np.asarray(lp._fit_samples)
    expected = data[:, 0] @ data[:, 2] / (data[:, 0] @ data[:, 0])
    assert lp.finish_steering_fit()
    assert lp.a_linear == pytest.approx(expected)
    assert lp.c_bias == 0


def test_bias_is_separate_and_seed_speed_term_is_held():
    lp = LiveParams(lag_frames=0, device_kind="gamepad",
                    seeds={"gamepad": (3.0, 0.001, 0.01)})
    lp.begin_steering_fit()
    samples(lp, b=0.001, c=0.01)
    assert lp.finish_steering_fit()
    assert lp.x == pytest.approx([4.0, 0.001, 0.01])


@pytest.mark.parametrize("kwargs,reason", [
    ({"count": 30}, "more steering samples"),
    ({"one_sided": True}, "left and right"),
    ({"wheel_scale": 0.002}, "left and right"),
    ({"noise": 0.04}, "residual"),
    ({"a": 0.4}, "implausible"),
    ({"a": -4.0}, "direction"),
])
def test_rejected_fit_never_changes_coefficients(kwargs, reason):
    lp = LiveParams(lag_frames=0)
    lp.begin_steering_fit()
    before = lp.x.copy()
    samples(lp, **kwargs)
    assert not lp.finish_steering_fit()
    assert reason in lp.fit_reason
    assert np.array_equal(lp.x, before)
    assert not lp.scale_frozen


def test_poor_window_can_be_replaced_by_good_samples():
    lp = LiveParams(lag_frames=0)
    lp.begin_steering_fit()
    samples(lp, noise=0.04)
    assert not lp.finish_steering_fit()
    samples(lp, count=1300)
    assert lp.finish_steering_fit()
    assert lp.a_linear == pytest.approx(4.0)


def test_steering_installs_once_and_restart_refits():
    lc = FakeLiveCalib()
    lc.converge()
    lp = LiveParams(lag_frames=0)
    flow = new_flow(lc=lc, lp=lp, passive_fit=True, persist=False)
    flow._enter(Step.STEERING)
    driver = Driver(flow)
    driver.verify_direction()
    samples(lp)
    driver.frames(1)
    assert flow.step is Step.CHECK and lp.scale_frozen
    assert lp.a_linear == pytest.approx(4.0)
    flow.restart("test recalibration")
    assert not lp.scale_frozen and not lp.calibration_fit
    assert lp.samples == 0 and lp.axis_sign is None and not lp._fit_samples
    lc.converge()
    flow._enter(Step.STEERING)
    driver.verify_direction()
    samples(lp, a=6.0)
    driver.frames(1)
    assert flow.step is Step.CHECK
    assert lp.a_linear == pytest.approx(6.0)


def test_leaving_steering_closes_collection():
    lp = LiveParams(lag_frames=0)
    flow = new_flow(lp=lp, persist=False)
    flow._enter(Step.STEERING)
    samples(lp)
    flow._enter(Step.CAMERA)
    assert not lp.finish_steering_fit()
    before = lp.samples
    samples(lp)
    assert lp.samples == before


def test_check_blaming_steering_clears_freeze():
    lp = LiveParams(lag_frames=0)
    flow = new_flow(lp=lp, persist=False)
    flow._enter(Step.STEERING)
    samples(lp)
    assert lp.finish_steering_fit()
    flow._enter(Step.CHECK)
    flow._finish_check(evaluate(check_drive(fit_error=0.9)))
    assert flow.step is Step.STEERING
    assert not lp.scale_frozen and lp.samples == 0
    assert lp._steering_fit_active


def test_user_unlock_does_not_remove_calibration_freeze():
    lp = LiveParams(lag_frames=0)
    lp.begin_steering_fit()
    samples(lp)
    assert lp.finish_steering_fit()
    lp.frozen = False
    lp.begin_steering_fit()
    samples(lp, a=8.0)
    lp._intervention_learn(0.5, 0.02, 400)
    assert lp.a_linear == pytest.approx(4.0)
    assert not lp._steering_fit_active and lp.locked
    # Explicit tuner setters remain an intentional manual override.
    lp.a_linear = 5.0
    lp.b_quad = 0.002
    lp.save()
    assert lp.x[:2] == pytest.approx([5.0, 0.002])


def test_old_fit_loads_frozen_without_inventing_measurement(data_dir):
    import json
    (data_dir / "liveparams_ets2_gamepad.json").write_text(json.dumps(
        dict(a=4.0, b=0.001, c=0.01, samples=500)))
    lp = LiveParams(game="ets2", device_kind="gamepad", lag_frames=0)
    before = lp.x.copy()
    assert lp.scale_frozen and not lp.calibration_fit
    lp.begin_steering_fit()
    samples(lp, a=7.0)
    assert np.array_equal(lp.x, before)
    lp.save()
    assert LiveParams(game="ets2", device_kind="gamepad").scale_frozen


@pytest.mark.parametrize("extra", [dict(samples=50), dict(samples=500, coverage_version=1)])
def test_under_covered_old_fit_is_measured_not_frozen(data_dir, extra):
    import json
    (data_dir / "liveparams_ets2_gamepad.json").write_text(json.dumps(
        dict(a=4.0, b=0.001, c=0.0, **extra)))
    lc = FakeLiveCalib()
    lc.converge()
    lp = LiveParams(game="ets2", device_kind="gamepad", lag_frames=0)
    assert not lp.scale_frozen and not lp.legacy_fit
    flow = new_flow(lc=lc, lp=lp, passive_fit=True, persist=False)
    flow._enter(Step.STEERING)
    driver = Driver(flow)
    driver.verify_direction()
    driver.frames(1)
    assert flow.step is Step.STEERING
    samples(lp, a=6.0)
    driver.frames(1)
    assert flow.step is Step.CHECK and lp.calibration_fit
    assert lp.a_linear == pytest.approx(6.0)


def test_freeze_is_per_device_and_game():
    lp = LiveParams(game="ets2", device_kind="gamepad", lag_frames=0)
    lp.begin_steering_fit()
    samples(lp)
    assert lp.finish_steering_fit()
    lp.switch_device("wheel")
    assert not lp.scale_frozen and not lp.calibration_fit
    lp.switch_device("gamepad")
    assert lp.scale_frozen and lp.calibration_fit and lp.a_linear == pytest.approx(4.0)
    assert not LiveParams(game="ac", device_kind="gamepad").scale_frozen


def test_low_rms_excitation_is_rejected_even_with_both_turns():
    lp = LiveParams(lag_frames=0)
    lp.begin_steering_fit()
    for wheel in [-0.004] * 10 + [0.004] * 10 + [0.0001] * 280:
        lp.update(4.0 * wheel, 20.0, wheel, gate_override={"min_axis": 0})
    assert not lp.finish_steering_fit()
    assert "excitation" in lp.fit_reason
    assert lp.a_linear == 10.0


def test_bad_residual_does_not_stop_probe_by_claiming_trust():
    lp = LiveParams(lag_frames=0)
    lp.begin_steering_fit()
    for speed in (12.0, 24.0):
        samples(lp, noise=0.04, speed=speed)
    assert not lp.finish_steering_fit()
    assert not lp.trusted() and lp.trust_progress < 1


def test_restart_clears_session_lock_as_well_as_measurement():
    lp = LiveParams(lag_frames=0)
    flow = new_flow(lp=lp, persist=False)
    lp.frozen = True
    flow.restart("recalibrate")
    assert not lp.locked
    flow._enter(Step.STEERING)
    samples(lp)
    assert lp.finish_steering_fit()


def test_active_probe_measures_then_stops_at_freeze():
    from simsteer.app.probe import SteeringProbe
    lp = LiveParams(lag_frames=0)
    lp.begin_steering_fit()
    probe = SteeringProbe()
    for _ in range(300):
        axis = probe.tick(0.05, 0.0, 12.0, False, True,
                          lp_trusted=lp.trusted(), calibrating=True)
        lp.update(axis, 12.0, axis / 4.0, commanded_axis=axis)
    assert probe.active_frames == 300
    assert lp.finish_steering_fit()
    before = lp.x.copy()
    for _ in range(60):
        assert probe.tick(0.05, 0.0, 12.0, False, True,
                          lp_trusted=lp.trusted(), calibrating=False) == 0
        lp.update(0.2, 12.0, 0.02, commanded_axis=0.2)
    assert np.array_equal(lp.x, before)


def test_trim_corrects_residual_without_changing_measured_scale():
    from simsteer.core.control import ControllerConfig, LateralController
    from test_steering_direction import _plan
    lp = LiveParams(lag_frames=0, clock=lambda: 123.0)
    lp.begin_steering_fit()
    samples(lp)
    assert lp.finish_steering_fit()
    before = lp.x.copy()
    config = ControllerConfig()
    controller = LateralController(config, lp)
    for _ in range(200):
        controller.compute(_plan(0.0), 22.0, actual_wheel_angle=-0.005, dt=0.05)
    assert controller.last_trim_frozen_reason == ""
    assert 0 < controller.axis_trim_state <= config.wheel_trim_clip
    assert np.array_equal(lp.x, before)


def test_lag_matched_telemetry_recovers_gain():
    from test_liveparams import _drive
    lp = LiveParams(lag_frames=5)
    lp.begin_steering_fit()
    _drive(lp, a=4.0, b=0.0, frames=range(1000))
    assert lp.finish_steering_fit()
    assert lp.a_linear == pytest.approx(4.0)
