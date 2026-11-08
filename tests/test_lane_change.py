import numpy as np
import pytest

from simsteer.core.control import ControllerConfig
from simsteer.core.lane_change import ManualLaneChange
from test_model_contract import model_inputs


def belief(left=0.0, right=0.0):
    state = np.zeros(8)
    state[3:5] = left, right
    return state


@pytest.mark.parametrize("index", [3, 4])
def test_no_cut_before_commitment(index):
    change = ManualLaneChange()
    change.request(index, now=0.0, hold_s=2.5)
    assert change.model_desire(None) == index
    for probability in (0.0, 0.09, 0.3, 0.1, 0.0):
        assert not change.observe(belief(probability), threshold=0.3)
        assert change.active


def test_completion_uses_sum_and_configured_commitment_threshold():
    change = ManualLaneChange()
    change.request(3, now=0.0, hold_s=2.5)
    change.model_desire(None)
    assert not change.observe(belief(0.2, 0.2), threshold=0.6)
    assert not change.observe(belief(), threshold=0.6)
    assert change.active
    assert not change.observe(belief(0.35, 0.35), threshold=0.6)
    assert not change.observe(belief(0.05, 0.05), threshold=0.6)
    assert change.observe(belief(0.04, 0.04), threshold=0.6)
    assert not change.active and change.index is None
    assert not change.observe(belief(), threshold=0.6)


@pytest.mark.parametrize("committed", [False, True])
def test_timer_is_an_upper_bound(committed):
    change = ManualLaneChange()
    change.request(4, now=10.0, hold_s=2.5)
    change.model_desire(None)
    change.observe(belief(right=0.8 if committed else 0.0), threshold=0.3)
    change.expire(12.499)
    assert change.active
    change.expire(12.5)
    assert not change.active
    assert change.model_desire(None) is None


def test_opposite_press_replaces_hold_and_resets_commitment():
    change = ManualLaneChange()
    change.request(3, now=0.0, hold_s=2.5)
    assert change.model_desire(None) == 3
    change.observe(belief(0.8), threshold=0.3)
    change.request(4, now=1.0, hold_s=2.5)
    assert change.model_desire(None) == 4
    assert not change.observe(belief(), threshold=0.3)
    change.expire(2.5)
    assert change.active
    change.observe(belief(right=0.8), threshold=0.3)
    assert change.observe(belief(), threshold=0.3)


@pytest.mark.parametrize("nav_index", [None, 1, 2, 3, 4])
@pytest.mark.parametrize("finish", ["completion", "expiry"])
def test_navigation_resumes_after_manual_hold(nav_index, finish):
    change = ManualLaneChange()
    change.request(3, now=0.0, hold_s=2.5)
    assert change.model_desire(nav_index) == 3
    change.observe(belief(0.8), threshold=0.3)
    if finish == "completion":
        assert change.observe(belief(), threshold=0.3)
    else:
        change.expire(2.5)
    assert change.model_desire(nav_index) == nav_index
    assert not change.observe(belief(), threshold=0.3)
    assert change.model_desire(nav_index) == nav_index


def test_release_step_does_not_commit_the_next_command():
    change = ManualLaneChange()
    change.request(3, now=0.0, hold_s=2.5)
    change.model_desire(None)
    change.request(3, now=0.1, hold_s=2.5)
    assert change.model_desire(3) is None
    assert not change.observe(belief(0.8), threshold=0.3)
    assert change.model_desire(3) == 3
    assert not change.observe(belief(), threshold=0.3)
    assert change.active


@pytest.mark.parametrize("neutral_step", [False, True])
def test_reset_clears_commitment_but_remembers_actual_model_input(neutral_step):
    change = ManualLaneChange()
    change.request(3, now=0.0, hold_s=2.5)
    change.model_desire(None)
    change.observe(belief(0.8), threshold=0.3)
    change.reset()
    assert not change.active
    if neutral_step:
        assert change.model_desire(None) is None
    change.request(3, now=0.1, hold_s=2.5)
    if not neutral_step:
        assert change.model_desire(None) is None
    assert change.model_desire(None) == 3
    assert not change.observe(belief(), threshold=0.3)
    assert change.active


@pytest.mark.parametrize("finish", ["completion", "expiry"])
@pytest.mark.parametrize("index", [3, 4])
def test_repress_sends_a_new_pulse_to_policy(model_inputs, finish, index):
    model, policy, image = model_inputs
    cfg = ControllerConfig()
    change = ManualLaneChange()
    change.request(index, now=0.0, hold_s=cfg.lane_change_hold_s)

    def step(probability):
        index = change.model_desire(None)
        desire = None if index is None else np.eye(8, dtype=np.float32)[index]
        model.step(image, desire=desire)
        return change.observe(belief(probability), cfg.lane_change_desire_threshold)

    for _ in range(3):
        assert not step(0.8)
    if finish == "completion":
        assert step(0.0)
    else:
        assert not step(0.8)
        change.expire(cfg.lane_change_hold_s)
    change.request(index, now=cfg.lane_change_hold_s, hold_s=cfg.lane_change_hold_s)
    step(0.0)  # Neutral model step separates the same-direction commands.
    assert policy.inputs["desire_pulse"][0, -1, index] == 0
    step(0.0)
    np.testing.assert_array_equal(policy.inputs["desire_pulse"][0, -2:, index], [1, 1])
    assert change.active
