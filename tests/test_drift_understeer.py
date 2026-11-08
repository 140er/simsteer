"""Closed-loop lateral regressions with an independent bicycle plant.

Run this file with --report for the reproducible comparison table. Plans
come from a fixed road plus position/heading feedback, never from the
controller's desired curvature. The rack fit stays fixed in every run.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.core.constants import IDX_N, PLAN_WIDTH, T_IDXS
from simsteer.core.control import ControllerConfig, LateralController
from simsteer.core.learners.liveparams import LiveParams, TRUSTED_MIN_SAMPLES
from simsteer.core.steering import SteeringExecutor, SteeringTarget

T = np.asarray(T_IDXS)


def plan_for(curvature, speed=25., integral=None):
    ks = np.broadcast_to(curvature, T.shape)
    plan = np.zeros((IDX_N, PLAN_WIDTH), dtype=np.float64)
    plan[:, 3] = speed
    plan[:, 14] = speed * ks
    plan[:, 11] = speed * (ks * T if integral is None else integral)
    return SimpleNamespace(plan=plan, desire_state=np.zeros(8))


def controller(**config):
    lp = LiveParams(seeds={"gamepad": (4., 0., 0.)}, device_kind="gamepad",
                    clock=lambda: 100.)
    lp.samples = TRUSTED_MIN_SAMPLES
    lp.coverage = {"4": [25, 25], "8": [25, 25]}
    return LateralController(ControllerConfig(**config), lp)


@dataclass
class Result:
    lateral_m: float
    curvature_error: float
    inside_cut_m: float
    peak_lateral_m: float
    ff: float


def simulate(speed=25., road_curvature=0., bias=0., gradient=0., *, entry=False,
             duration=80., model_dt=.05, axis_sign=1, wheel_sign=1):
    """100 Hz plant/executor and 20 Hz synthetic model, with 0.30 s lag.

    wheel = delayed(axis / rack_gain) + bias*wheelbase
    curvature = tan(wheel) / (wheelbase * (1 + gradient*v²))

    The offset is an unmodelled rack zero error observed in wheel
    telemetry. The tire gradient is independent of the rack ratio.
    Positive lateral displacement is toward the inside of a left bend.
    """
    ctrl = controller()
    ctrl.live_params.x[0] = axis_sign * 4.
    executor = SteeringExecutor()
    lateral = heading = wheel = 0.
    samples = []
    next_model = 0.
    tick_dt = .01
    rack = axis_sign * 4.
    wb = ctrl.cfg.wheelbase_m
    yaw_supported = "actual_yaw_rate" in inspect.signature(ctrl.compute).parameters

    def road(t):
        t = np.asarray(t)
        if entry:
            return .002 + .008 * np.clip((t - 10.) / .6, 0., 1.)
        return road_curvature * np.clip((t - 5.) / 10., 0., 1.)

    def integral(t):
        t = np.asarray(t)
        if entry:
            return .002 * t + .008 / 1.2 * (np.maximum(t - 10., 0.) ** 2
                                            - np.maximum(t - 10.6, 0.) ** 2)
        return road_curvature / 20. * (np.maximum(t - 5., 0.) ** 2
                                       - np.maximum(t - 15., 0.) ** 2)

    for i in range(round(duration / tick_dt)):
        now = i * tick_dt
        actual_k = math.tan(wheel) / (wb * (1. + gradient * speed * speed))
        if now + 1e-9 >= next_model:
            # A fixed planner, shared by base and head. It replans to
            # recover lateral and heading error rather than prescribing
            # the controller's own output as a fake measured response.
            correction = -.001 * lateral - .03 * heading
            ks = road(now + T) + correction
            headings = integral(now + T) - integral(now) + correction * T
            if entry and 9.5 <= now <= 11.5:
                # Reproduce an anticipation mismatch: the heading head
                # predicts 35% more turn than the yaw-rate head at entry.
                headings *= 1.35
            decoded = plan_for(ks, speed, headings)
            kwargs = {"actual_yaw_rate": actual_k * speed} if yaw_supported else {}
            axis = ctrl.compute(decoded, speed, actual_wheel_angle=wheel_sign * wheel,
                                dt=model_dt, wheel_sign=wheel_sign, **kwargs)
            executor.submit(SteeringTarget(axis, speed, rack, 0., wb, 5.), now)
            next_model += model_dt
        axis = executor.tick(now)
        assert axis is not None
        wheel_target = wheel_sign * axis / rack + math.atan(bias * wb)
        wheel += -math.expm1(-tick_dt / .25) * (wheel_target - wheel)
        road_k = float(road(now))
        lateral += speed * math.sin(heading) * tick_dt
        heading += speed * (actual_k - road_k * math.cos(heading)
                            / (1. - road_k * lateral)) * tick_dt
        samples.append((now, lateral, ctrl.last_curvature - actual_k))
    tail = np.asarray(samples)[-round(10 / tick_dt):]
    entries = [max(0., e) for t, e, _ in samples if 9 <= t <= 16] if entry else [0.]
    return Result(float(tail[:, 1].mean()), float(np.abs(tail[:, 2]).mean()),
                  max(entries), max(abs(e) for _, e, _ in samples),
                  getattr(ctrl, "last_understeer_ff", 1.))


def test_closed_loop_straight_bias():
    result = simulate(bias=.0006)
    assert abs(result.lateral_m) < .08, result


@pytest.mark.parametrize("speed", [15., 25., 35.])
def test_closed_loop_understeer_curve(speed):
    result = simulate(speed, road_curvature=.004, gradient=.0006)
    assert abs(result.lateral_m) < .08, result
    assert result.curvature_error < .00012, result


def test_closed_loop_inside_cut():
    result = simulate(15., entry=True, duration=25.)
    assert result.inside_cut_m < .3, result


def report():
    print("scenario,speed_mps,lateral_m,want_actual_error_1pm,inside_cut_m,peak_offset_m,ff")
    cases = [("straight_bias", 25., {"bias": .0006})]
    cases += [("curve_R250", v, {"road_curvature": .004, "gradient": .0006})
              for v in (15., 25., 35.)]
    cases += [("curve_entry", 15., {"entry": True, "duration": 25.})]
    for name, speed, kw in cases:
        r = simulate(speed, **kw)
        print(f"{name},{speed:.0f},{r.lateral_m:.6f},{r.curvature_error:.9f},"
              f"{r.inside_cut_m:.6f},{r.peak_lateral_m:.6f},{r.ff:.6f}")


def teach(ctrl, *, speed=25., ratio=1.375, seconds=15., dt=.05, **kwargs):
    decoded = plan_for(.004, speed)
    wheel = math.atan(.004 * ctrl.cfg.wheelbase_m * ratio)
    for _ in range(round(seconds / dt)):
        ctrl.compute(decoded, speed, actual_wheel_angle=wheel,
                     actual_yaw_rate=.004 * speed, dt=dt, **kwargs)


def test_missing_evidence_starts_neutral_and_never_writes_rack():
    ctrl = controller()
    coefficients = ctrl.live_params.x.copy()
    for _ in range(100):
        ctrl.compute(plan_for(.004), 25., actual_wheel_angle=.016)
    assert ctrl.last_understeer_ff == 1.
    assert ctrl.last_understeer_gradient == 0.
    assert ctrl.last_understeer_frozen_reason == "no-telemetry"
    teach(ctrl)
    assert 1.2 < ctrl.last_understeer_ff < 1.4
    np.testing.assert_array_equal(ctrl.live_params.x, coefficients)
    ctrl.reset()
    assert ctrl.last_understeer_ff == 1.
    assert ctrl.last_understeer_gradient == 0.
    assert not ctrl._curve_history
    assert ctrl.axis_trim_state == 0.


@pytest.mark.parametrize("gate", ["lc", "cold", "user", "sat", "slow"])
def test_understeer_transient_gates_freeze_estimate(gate):
    ctrl = controller()
    teach(ctrl, seconds=5.)
    before = ctrl.last_understeer_gradient
    kwargs = {}
    speed = 25.
    if gate == "lc":
        kwargs["lane_change_command_active"] = True
    elif gate == "cold":
        ctrl.live_params.samples = 0
    elif gate == "user":
        ctrl.live_params.last_intervened_ts = 100.
    elif gate == "sat":
        ctrl.cfg.steer_max = .03
    else:
        speed = 5.
    teach(ctrl, speed=speed, seconds=.5, **kwargs)
    assert ctrl.last_understeer_gradient == before
    assert ctrl.last_understeer_frozen_reason == gate
    assert not ctrl._curve_history


def test_partial_speed_gate_can_remove_an_earlier_boost():
    ctrl = controller()
    teach(ctrl)
    before = ctrl.last_understeer_gradient
    teach(ctrl, speed=18., ratio=1., seconds=40.)
    assert ctrl.last_understeer_gradient < .02 * before
    assert ctrl.last_understeer_ff < 1.01
    teach(ctrl, speed=8., ratio=1.3)
    assert ctrl.last_understeer_ff == 1.


def test_learning_is_bounded_and_synthesized_wheel_is_neutral():
    ctrl = controller(understeer_gradient_max=.0001, understeer_cap=1.1)
    teach(ctrl, ratio=1.7, seconds=60.)
    assert 0. < ctrl.last_understeer_gradient <= .0001
    assert 1. < ctrl.last_understeer_ff <= 1.1
    ctrl.reset()
    teach(ctrl, ratio=4.)
    assert ctrl.last_understeer_gradient == 0.
    ctrl.reset()
    teach(ctrl, ratio=1.)
    assert ctrl.last_understeer_ff == 1.


@pytest.mark.parametrize("dt", [.01, .05, .1])
def test_learning_uses_elapsed_model_time(dt):
    ctrl = controller()
    teach(ctrl, dt=dt, seconds=40.)
    assert ctrl.last_understeer_ff == pytest.approx(1.372, abs=.003)


def test_bias_trim_freezes_in_gentle_curve_and_is_bounded():
    ctrl = controller()
    teach(ctrl, seconds=2.)
    assert ctrl.axis_trim_state == 0.
    assert ctrl.last_trim_frozen_reason == "corner"
    for _ in range(1000):
        ctrl.compute(plan_for(0.), 25., actual_wheel_angle=-.004)
    assert 0. < ctrl.axis_trim_state <= ctrl.cfg.wheel_trim_clip


@pytest.mark.parametrize("sign", [-1, 1])
def test_anticipation_cap_precedes_jerk_limit_and_includes_endpoint(sign):
    from simsteer.core.postprocess import desired_curvature_lag_adjusted
    ks = sign * (.002 + .02 * np.clip(T / .2, 0., 1.))
    # A heading prediction can disagree with the instantaneous yaw rate.
    decoded = plan_for(ks, integral=ks * T * 1.35)
    capped = desired_curvature_lag_adjusted(decoded.plan, 25., .3,
                                           extra_buffer_s=0., last_desired_curvature=sign*.022)
    assert 0. <= sign * capped <= .022 + 1e-10
    # A sudden shrinking plan must still respect the executor's jerk bound.
    limited = desired_curvature_lag_adjusted(plan_for(0.).plan, 25., .3,
                    last_desired_curvature=sign*.02, dt=.05, lat_jerk_max_mps3=5.)
    assert limited == pytest.approx(sign * (.02 - 5. * .05 / 625.))


@pytest.mark.parametrize("axis_sign,wheel_sign", [(-1, 1), (1, -1), (-1, -1)])
def test_closed_loop_inverted_directions(axis_sign, wheel_sign):
    r = simulate(25., road_curvature=.004, gradient=.0006,
                 axis_sign=axis_sign, wheel_sign=wheel_sign)
    assert abs(r.lateral_m) < .08, r


def test_straight_heading_recovery_remains_available():
    ctrl = controller()
    decoded = plan_for(0.)
    decoded.plan[:, 11] = .02
    assert ctrl.compute(decoded, 25.) > 0.
    clean = controller()
    for _ in range(100):
        assert clean.compute(plan_for(0.), 25., actual_wheel_angle=0., actual_yaw_rate=0.) == 0.
    assert clean.axis_trim_state == 0.
    assert clean.last_understeer_ff == 1.


def test_saturation_holds_boost_without_learning_or_chattering():
    ctrl = controller()
    teach(ctrl, ratio=1.7, seconds=30.)
    before = ctrl.last_understeer_gradient
    ff = ctrl.last_understeer_ff
    ctrl.cfg.steer_max = .1
    teach(ctrl, ratio=1.7, seconds=1.)
    assert ctrl.last_understeer_gradient == before
    assert ctrl.last_understeer_ff == ff
    assert ctrl.last_understeer_frozen_reason == "sat"


if __name__ == "__main__":
    report()
