"""Turn-in lead learned from the measured command-to-response delay."""

from __future__ import annotations

import math

import numpy as np
import pytest

from simsteer.core.control import ControllerConfig
from simsteer.core.learners import anticipation as an
from simsteer.core.learners.anticipation import AnticipationLearner

DT = 0.05


def drive(learner, seconds, *, delay=0.6, gain=-1.2, speed=22.0, engaged=True,
          lane_change=False, amplitude=0.004, noise=0.0, seed=3):
    """Commanded curvature weaves; the vehicle follows `delay` seconds later."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds / DT))
    t0 = getattr(drive, "t", 0.0)
    cmd = lambda t: amplitude * (math.sin(t / 2.3) + 0.6 * math.sin(t / 0.9 + 1.0))
    moved = False
    for i in range(n):
        t = t0 + i * DT
        k_act = gain * cmd(t - delay) + rng.normal(0, noise)
        moved |= learner.update(dt=DT, commanded_k=cmd(t), yaw_rate=k_act * speed,
                                speed=speed, engaged=engaged, lane_change=lane_change)
    drive.t = t0 + n * DT
    return moved


@pytest.fixture(autouse=True)
def _clock():
    drive.t = 0.0


def test_learns_a_lead_that_matches_the_measured_delay():
    cfg = ControllerConfig(lookahead_s=0.3, curvature_anticipation_s=0.0)
    learner = AnticipationLearner(cfg)
    assert drive(learner, 120, delay=0.6, noise=0.0002)
    assert learner.last_delay_s == pytest.approx(0.6, abs=0.05)
    assert cfg.lookahead_s + cfg.curvature_anticipation_s == pytest.approx(0.6, abs=0.06)


def test_shorter_delay_lowers_the_lead():
    cfg = ControllerConfig(lookahead_s=0.3, curvature_anticipation_s=0.3)
    learner = AnticipationLearner(cfg)
    drive(learner, 120, delay=0.15)
    assert cfg.curvature_anticipation_s == pytest.approx(-0.15, abs=0.06)


@pytest.mark.parametrize("kw", [dict(engaged=False), dict(lane_change=True), dict(speed=5.0),
                                dict(amplitude=0.0001), dict(noise=0.05)])
def test_unusable_stretches_leave_the_lead_alone(kw):
    cfg = ControllerConfig(lookahead_s=0.3, curvature_anticipation_s=0.1)
    learner = AnticipationLearner(cfg)
    assert not drive(learner, 60, **kw)
    assert cfg.curvature_anticipation_s == 0.1


def test_switched_off_keeps_a_hand_set_lead():
    cfg = ControllerConfig(lookahead_s=0.3, curvature_anticipation_s=0.2, auto_anticipation=False)
    assert not drive(AnticipationLearner(cfg), 60)
    assert cfg.curvature_anticipation_s == 0.2


def test_lead_stays_within_the_knob_range():
    cfg = ControllerConfig(lookahead_s=0.0, curvature_anticipation_s=0.0)
    learner = AnticipationLearner(cfg)
    for _ in range(10):
        drive(learner, 30, delay=1.1)
    assert cfg.curvature_anticipation_s <= an.ANTICIPATION_MAX_S


def test_disengage_starts_a_fresh_window_and_learning_resumes():
    cfg = ControllerConfig(lookahead_s=0.3, curvature_anticipation_s=0.0)
    learner = AnticipationLearner(cfg)
    drive(learner, 8)
    drive(learner, 0.1, engaged=False)
    assert len(learner._samples) == 0
    estimates = learner.estimates
    assert drive(learner, 40, delay=0.6)
    assert learner.estimates > estimates and cfg.curvature_anticipation_s > 0


def test_loop_resets_the_learner_when_output_stops(monkeypatch):
    import simsteer.runtime.loop as loop
    from test_output_safety import drive as drive_loop

    resets = []

    class Recording(AnticipationLearner):
        def reset(self):
            resets.append(True)
            super().reset()

    monkeypatch.setattr(loop, "AnticipationLearner", Recording)
    drive_loop(monkeypatch, fault="vision")
    assert resets  # the fault's stop_output dropped the window
