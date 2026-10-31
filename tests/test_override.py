from dataclasses import replace
import math

import pytest

from simsteer.core.override import Commands, Output, Override, Physical
from fakes_wheel import FakeClock


class Drive:
    def __init__(self):
        self.clock, self.machine, self.commands = FakeClock(), Override(), Commands()

    def tick(self, t, degrees=0, *, request=None, fresh=True, throttle=0, brake=0, ok=True):
        self.clock.t = t
        if fresh:
            self.commands = replace(self.commands, ai_at=t)
        if request:
            self.commands = replace(self.commands, request=request, serial=self.commands.serial + 1)
        return self.machine.step(self.clock.now(), Physical(degrees / 450, throttle, brake, .3, ok), self.commands)

    def engage(self, degrees=0):
        self.commands = replace(self.commands, steer=.5, throttle=.2)
        self.tick(0, degrees, request="engage")
        self.tick(.25, degrees)
        return self.tick(.4, degrees)


@pytest.mark.parametrize("steer", [-1., -.123456789, -0., .1, .9, 1.])
def test_player_is_exact_identity(steer):
    p = Physical(steer, .123456789, .456, .789, True)
    out = Override().step(0, p, Commands())
    assert (out.steer, out.throttle, out.brake, out.clutch) == (steer, p.throttle, p.brake, p.clutch)
    assert out.source == "PLAYER" and not out.engaged


def test_engage_linear_ramp_and_pedals_immediate():
    d = Drive()
    d.commands = replace(d.commands, steer=.8, throttle=.4, brake=.05)
    first = d.tick(0, -90, request="engage")
    assert first.steer == -.2 and first.throttle == .4 and first.brake == .05
    assert d.tick(.125, -90).steer == pytest.approx(.3)
    out = d.tick(.25, -90)
    assert out.steer == .8 and out.source == "AI" and out.clutch == .3


@pytest.mark.parametrize("degrees,held,trips", [(12, .08, True), (11, .2, False), (12, .07, False)])
def test_travel_threshold_and_continuous_hold(degrees, held, trips):
    d = Drive()
    d.engage()
    d.tick(.42, degrees)
    for i in range(1, round(held / .002) + 1):
        out = d.tick(.42 + i * .002, degrees)
    assert out.engaged != trips
    if trips:
        assert out.reason == "steering"


def test_travel_timer_resets_when_released():
    d = Drive()
    d.engage()
    d.tick(.42, 12)
    d.tick(.48, 12)
    d.tick(.482, 0)
    d.tick(.49, 12)
    assert d.tick(.55, 12).engaged


@pytest.mark.parametrize("motion,seconds", [("noise", 60), ("drift", 10)])
def test_resting_hands_and_slow_drift_never_take_over(motion, seconds):
    d = Drive()
    d.engage()
    for i in range(1, seconds * 500 + 1):
        degrees = (2 if i % 2 else -2) if motion == "noise" else i * .002 * 5
        assert d.tick(.4 + i * .002, degrees).engaged, (motion, i)


def test_200_degree_per_second_flick_for_50ms_trips():
    d = Drive()
    d.engage()
    for i in range(1, 26):
        out = d.tick(.4 + i * .002, 200 * i * .002)
        if not out.engaged:
            break
    assert not out.engaged and out.reason == "steering"


def test_fast_noise_without_enough_travel_does_not_trip():
    d = Drive()
    d.engage()
    for i in range(1, 51):
        assert d.tick(.4 + i * .002, .5 * math.sin(i * 2)).engaged


@pytest.mark.parametrize("degrees", [-30, 30])
def test_hard_travel_trips_immediately(degrees):
    d = Drive()
    d.engage()
    assert d.tick(.402, degrees).reason == "steering"


def test_arm_grace_tracks_release_after_insert():
    d = Drive()
    d.tick(0, 90, request="engage")
    d.tick(.1, -20)
    for i in range(2, 51):
        assert d.tick(i * .01, -20).engaged


@pytest.mark.parametrize("brake,hold,trips", [(.10, .05, True), (.08, .2, False), (.10, .048, False)])
def test_brake_hold_including_during_grace(brake, hold, trips):
    d = Drive()
    d.tick(0, request="engage")
    d.tick(.002, brake=brake)
    out = d.tick(.002 + hold, brake=brake)
    assert out.engaged != trips
    if trips:
        assert out.reason == "brake" and out.brake == brake


def test_brake_blocks_engage_and_throttle_only_stays_engaged():
    d = Drive()
    assert not d.tick(0, request="engage", brake=.1).engaged
    d.engage()
    assert d.tick(.402, throttle=.9).throttle == .9
    assert d.tick(.404, throttle=.24).throttle == .2
    assert d.machine.engaged


@pytest.mark.parametrize("reason", ["insert", "loop", "stale"])
def test_handback_continuity_rate_monotonic_and_two_second_limit(reason):
    d = Drive()
    d.engage(-90)
    before = d.machine.output.steer
    t = .65 if reason == "stale" else .402
    out = d.tick(t, -90, request=None if reason == "stale" else reason, fresh=reason != "stale")
    assert not out.engaged and out.reason == reason and out.steer == pytest.approx(before)
    offset, duration = abs(before + .2), d.machine.duration
    assert .3 <= duration <= 2
    for i in range(1, 1002):
        out = d.tick(t + i * .002, -90)
        remaining = abs(out.steer + .2)
        assert remaining <= offset + 1e-9
        assert offset - remaining <= abs(before + .2) / duration * .002 + 1e-9
        offset = remaining
    assert out.steer == -.2 and out.source == "PLAYER"


def test_rim_deltas_pass_immediately_during_handback_and_insert_reengages():
    d = Drive()
    d.engage()
    out = d.tick(.402, request="insert")
    next_out = d.tick(.404, 4.5)
    assert next_out.steer - out.steer == pytest.approx(.01 - .5 / 2 * .002)
    again = d.tick(.406, 4.5, request="engage")
    assert again.engaged and again.source == "ENGAGING"
    assert again.steer == next_out.steer


def test_expired_steering_override_blends_even_with_pedal_override_alive():
    d = Drive()
    d.commands = replace(d.commands, forced={"steer": (.7, .2), "throttle": (.3, 1)})
    assert d.tick(0).steer == .7
    assert d.tick(.2).steer == .7
    assert d.tick(.3).steer < .7
    assert d.tick(.4).throttle == .3
    assert d.tick(2.21).steer == 0


@pytest.mark.parametrize("engaged", [False, True])
def test_loss_always_neutral_and_recovery_blends_without_auto_engage(engaged):
    d = Drive()
    if engaged:
        d.engage()
    else:
        d.tick(.4, 90, throttle=.8, brake=.5)
    out = d.tick(.402, 90, throttle=.8, brake=.5, ok=False)
    assert out == Output(reason="lost")
    assert not d.tick(.9, -90, ok=False).engaged
    assert d.tick(.902, -90).steer == 0
    assert d.tick(2.904, -90).steer == -.2


def test_lost_and_invalid_input_block_engagement():
    d = Drive()
    assert not d.tick(0, ok=False, request="engage").engaged
    assert not d.tick(.002).engaged
    assert not d.tick(.004, float("nan")).engaged
