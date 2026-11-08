"""Exercise manual requests through the live loop and split policy input."""

from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.core.control import ControllerConfig
from simsteer.app.global_keys import KEY_INSERT, KEY_NUMPAD4, KEY_NUMPAD6
from test_live_run_fixes import healthy_decoded
from test_model_contract import model_inputs


@pytest.fixture
def drive(monkeypatch, model_inputs):
    import simsteer.runtime.loop as loop

    model, policy, image = model_inputs
    state = SimpleNamespace(i=-1)
    clock = SimpleNamespace(now=lambda: max(0, state.i) * 0.05)
    cfg = ControllerConfig()
    monkeypatch.setattr(loop.ControllerConfig, "load", lambda *a, **kw: cfg)
    monkeypatch.setattr(loop, "FrameQueue", lambda: SimpleNamespace(
        push=lambda *args: (image, image)))
    manual_active, requests, pulses = [], [], []
    compute = loop.LateralController.compute

    def compute_lateral(self, *args, **kwargs):
        manual_active.append(kwargs["lane_change_command_active"])
        return compute(self, *args, **kwargs)

    monkeypatch.setattr(loop.LateralController, "compute", compute_lateral)
    model_step = model.step

    def step(*args, **kwargs):
        desire = kwargs["desire"]
        requests.append(None if desire is None else int(np.argmax(desire)))
        result = model_step(*args, **kwargs)
        pulses.append(policy.inputs["desire_pulse"].copy())
        return result

    monkeypatch.setattr(model, "step", step)

    def run(probabilities, keys, *, nav_index=None, hold_s=2.5, threshold=0.3,
            missing=(), engaged=False):
        cfg.lane_change_hold_s = hold_s
        cfg.lane_change_desire_threshold = threshold
        pad = SimpleNamespace(engaged=engaged, kind="gamepad", last_error=None,
                              center=lambda: None, set_steering=lambda *a, **kw: None,
                              set_throttle_brake=lambda *a, **kw: None)
        pad.disengage = lambda: setattr(pad, "engaged", False)
        monkeypatch.setattr(loop, "DeviceManager", lambda **kw: pad)
        if engaged:
            class ReadyFlow(loop.CalibFlow):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.step = loop.Step.STEERING
                    self.lc.cal_status = loop.CalStatus.CALIBRATED
                    self._check_requested = True
                    self.direction.model_sign = self.direction.axis_sign = 1
                    self.output_direction.sign = 1

            monkeypatch.setattr(loop, "CalibFlow", ReadyFlow)
        nav = loop.NavManager()
        monkeypatch.setattr(nav, "tick", lambda **kw: nav_index)
        monkeypatch.setattr(loop, "NavManager", lambda: nav)

        class Source:
            done = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def grab(self):
                state.i += 1
                if state.i == len(probabilities):
                    self.done = True
                    return None
                return None if state.i in missing else np.zeros((2, 2, 3), np.uint8)

        def decoded():
            result = healthy_decoded()
            probability = probabilities[state.i]
            result.desire_state[:] = 0
            result.desire_state[3:5] = probability / 2
            result.desire_state[0] = 1 - probability
            return result

        monkeypatch.setattr(loop, "decode", lambda *args: decoded())
        tel = SimpleNamespace(
            available=True, speed_mps=lambda: 10.0, yaw_rate_rad_s=lambda: 0.0,
            wheel_angle_rad=lambda wb: 0.0, game_steer=lambda: 0.0, close=lambda: None)
        assert loop.run(loop.Settings(no_tuner=True, no_gamepad=not engaged, no_probe=True),
                        "ets2", source=Source(), make_telemetry=lambda: tel, clock=clock,
                        model=model, keys=SimpleNamespace(poll=lambda: keys.get(state.i, [])),
                        headless=True) == 0
        return requests, pulses, manual_active

    return run


@pytest.mark.parametrize("key,index", [(KEY_NUMPAD4, 3), (KEY_NUMPAD6, 4)])
def test_loop_cuts_completed_hold_before_the_next_model_step(drive, key, index):
    requests, _, active = drive([0, 0.2, 0.4, 0.05, 0], {0: [key]})
    assert requests == [None, index, index, index, None]
    assert active == [False, True, True, False, False]


def test_loop_waits_for_commitment_and_uses_configured_threshold(drive):
    requests, _, active = drive([0, 0.4, 0, 0.8, 0.05, 0], {0: [KEY_NUMPAD4]}, threshold=0.6)
    assert requests == [None, 3, 3, 3, 3, None]
    assert active == [False, True, True, True, False, False]


def test_loop_does_not_cut_before_commitment(drive):
    requests, _, active = drive([0, 0.2, 0.05, 0, 0], {0: [KEY_NUMPAD4]})
    assert requests == [None, 3, 3, 3, 3]
    assert active == [False, True, True, True, True]


@pytest.mark.parametrize("key,index", [(KEY_NUMPAD4, 3), (KEY_NUMPAD6, 4)])
@pytest.mark.parametrize("press_frame", [3, 4])
def test_loop_repress_after_completion_reaches_policy(drive, key, index, press_frame):
    requests, pulses, _ = drive([0, 0.4, 0.4, 0.05, 0, 0],
                                {0: [key], press_frame: [key]})
    assert requests == [None, index, index, index, None, index]
    np.testing.assert_array_equal(pulses[-1][0, -2:, index], [1, 1])


def test_loop_timer_expiry_rearms_same_direction(drive):
    requests, pulses, active = drive([0, 0, 0, 0, 0, 0],
                                     {0: [KEY_NUMPAD4], 2: [KEY_NUMPAD4]}, hold_s=0.1)
    assert requests == [None, 3, None, 3, None, None]
    assert active == [False, True, False, True, False, False]
    assert pulses[3][0, -1, 3] == 1
    assert pulses[5][0, -1, 3] == 1  # The renewed pulse is still in the newest window.


def test_loop_opposite_press_replaces_committed_hold(drive):
    requests, pulses, active = drive([0, 0.4, 0.4, 0, 0.4, 0.05, 0],
                                     {0: [KEY_NUMPAD4], 2: [KEY_NUMPAD6]})
    assert requests == [None, 3, 3, 4, 4, 4, None]
    assert pulses[3][0, -1, 4] == 1
    assert active == [False, True, True, True, True, False, False]


@pytest.mark.parametrize("nav_index", [1, 2, 3, 4])
def test_loop_restores_navigation_after_completion(drive, nav_index):
    requests, _, active = drive([0, 0.4, 0.4, 0.05, 0],
                                {0: [KEY_NUMPAD4]}, nav_index=nav_index)
    # A same-direction nav request also requires a neutral step before manual input.
    if nav_index == 3:
        assert requests == [3, None, 3, 3, 3]
        assert active == [False, True, True, False, False]
    else:
        assert requests == [nav_index, 3, 3, 3, nav_index]
        assert active == [False, True, True, False, False]


def test_loop_repress_inserts_neutral_step_even_with_same_direction_navigation(drive):
    requests, pulses, _ = drive([0, 0, 0.4, 0.05, 0, 0],
                                {0: [KEY_NUMPAD4], 3: [KEY_NUMPAD4]}, nav_index=3)
    assert requests == [3, None, 3, 3, None, 3]
    np.testing.assert_array_equal(pulses[-1][0, -2:, 3], [1, 1])


def test_capture_fault_cancels_pending_manual_request(drive):
    requests, _, _ = drive([0, 0.4, 0.05, 0, 0, 0],
                           {0: [KEY_NUMPAD4], 2: [KEY_NUMPAD4]}, missing=(3,))
    assert requests == [None, 3, 3, None, None]


@pytest.mark.parametrize("key,engaged", [(ord("r"), False), (KEY_INSERT, True)])
def test_loop_stop_or_calibration_restart_clears_manual_hold(drive, key, engaged):
    requests, _, _ = drive([0, 0.4, 0.4, 0, 0],
                           {0: [KEY_NUMPAD4], 2: [key]}, engaged=engaged)
    expected = [None, 3, None, None, None] if engaged else [None, 3, 3, None, None]
    assert requests == expected
