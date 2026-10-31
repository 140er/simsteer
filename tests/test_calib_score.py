from __future__ import annotations

import math

import numpy as np

from simsteer.calib.score import CheckStats, evaluate

FRAMES = 2400  # two minutes at 20 Hz


def _drive(*, weave_m: float = 0.08, ratio: float = 1.01, axis_peak: float = 0.15,
           fit_error: float = 0.05, seed: int = 0) -> CheckStats:
    rng = np.random.default_rng(seed)
    s = CheckStats()
    for i in range(FRAMES):
        t = i / 20.0
        s.seconds += 0.05
        s.add_offset(weave_m * math.sin(2 * math.pi * t / 6.0) + rng.normal(0, 0.03))
        s.add_speed(22.0 * ratio * (1 + rng.normal(0, 0.02)), 22.0, 0.01)
        axis = axis_peak * math.sin(2 * math.pi * t / 15.0)
        s.add_axis(axis)
        if i % 4 == 0:
            # Independent check observations, not training innovations.
            s.add_fit(axis * fit_error * rng.choice([-1, 1]), axis, held_out=True)
    return s


def test_good_drive_passes():
    v = evaluate(_drive())
    assert v.passed
    assert v.redo is None
    assert v.score > 0.8
    assert {m.name for m in v.measurements} == {
        "lane_offset_spread", "speed_ratio_error", "steering_saturation",
        "steering_fit_error", "lane_offset_bias", "lane_excursions"}


def test_weaving_drive_fails_and_names_steering():
    v = evaluate(_drive(weave_m=0.7))
    assert not v.passed
    assert v.redo == "steering"
    assert "weaves" in v.reason


def test_wrong_fov_fails_and_names_fov():
    v = evaluate(_drive(ratio=0.75))
    assert not v.passed
    assert v.redo == "fov"


def test_saturating_steering_names_steering():
    v = evaluate(_drive(axis_peak=1.2))
    assert not v.passed
    assert v.redo == "steering"
    assert "full lock" in v.reason


def test_bad_steering_fit_names_steering():
    v = evaluate(_drive(fit_error=0.9))
    assert not v.passed
    assert v.redo == "steering"
    assert "mispredicts" in v.reason


def test_score_is_the_weakest_measurement():
    v = evaluate(_drive(weave_m=0.35))
    weakest = min(m.score for m in v.measurements)
    assert v.score == weakest


def test_measurement_without_enough_samples_is_insufficient():
    s = _drive()
    s.ratio_n, s.ratio_log_sum = 20, 20 * math.log(0.5)
    v = evaluate(s)
    # Omission used to certify a drive without measuring the required speed.
    assert not v.passed
    assert v.status == "INSUFFICIENT_EVIDENCE"
    assert "speed_ratio_error" in v.missing
    assert next(m for m in v.measurements if m.name == "speed_ratio_error").samples == 20


def test_nothing_measured_does_not_pass():
    v = evaluate(CheckStats(seconds=120.0))
    assert not v.passed
    assert v.redo == "check"


def test_large_lane_offsets_count_and_turns_do_not_measure_speed_ratio():
    s = CheckStats()
    s.add_offset(2.5)
    s.add_offset(None)
    s.add_speed(10.0, 22.0, 0.2)
    s.add_speed(10.0, 5.0, 0.0)
    # Lane changes are filtered by the flow; large lane-keeping errors must
    # remain in the excursion denominator rather than disappear.
    assert s.off_n == 1 and s.off_excursions == 1 and s.ratio_n == 0


def test_stats_roundtrip():
    s = _drive()
    back = CheckStats.from_dict(s.to_dict())
    assert back == s
