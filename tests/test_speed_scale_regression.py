"""Behavioral regressions runnable unchanged against V0.12 and the fix."""

from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.calib.flow import Step
from simsteer.core.control import LongitudinalController
from simsteer.core.postprocess import decode


RATIO = 0.88


def plan(speed=20.0, fraction=1.0, accel=0.0):
    decoded = decode(np.zeros(1576, np.float32), np.zeros(1000, np.float32))
    decoded.pose[0] = RATIO * speed
    decoded.plan[:, 3] = RATIO * speed * fraction
    decoded.plan[:, 6] = RATIO * accel
    decoded.lead_prob[:] = 0.0
    return decoded


def session(monkeypatch, *, frames=600, vehicle=None, on_grab=None, decoded=None,
            capture=None, **kwargs):
    from test_output_safety import drive
    import simsteer.runtime.loop as loop

    controllers = []

    class Controller(LongitudinalController):
        def __init__(self, cfg):
            super().__init__(cfg)
            controllers.append(self)

        def compute(self, *args, **kwargs):
            result = super().compute(*args, **kwargs)
            if capture is not None:
                capture.append((self.last_v_target, self.last_a_target, result))
            return result

    telemetry = SimpleNamespace(
        available=True, speed_mps=lambda: 20., yaw_rate_rad_s=lambda: 0.,
        wheel_angle_rad=lambda wb: .01, game_steer=lambda: .2,
        user_steer=lambda: 0., close=lambda: None)
    if vehicle is not None:
        telemetry.vehicle_id = lambda: vehicle
    with monkeypatch.context() as patch:
        patch.setattr(loop, "LongitudinalController", Controller)
        patch.setattr(loop, "decode", lambda *args: decoded if decoded is not None else plan())
        drive(patch, frames=frames, on_grab=on_grab, telemetry_source=telemetry, **kwargs)
    return controllers[0]


def test_underreading_cruise_does_not_brake_toward_phantom_stop(monkeypatch):
    samples = []
    session(monkeypatch, capture=samples)
    assert samples[-1][0] == pytest.approx(20.0, abs=0.01)
    assert samples[-1][2][1] == 0.0


def test_converged_vehicle_warm_starts_next_session(monkeypatch):
    session(monkeypatch, vehicle="truck-a")
    samples = []
    session(monkeypatch, frames=1, vehicle="truck-a", capture=samples)
    assert samples[0][0] == pytest.approx(20.0, abs=0.01)
    # Another vehicle and a reader without identity cannot inherit truck-a.
    other, generic = [], []
    session(monkeypatch, frames=1, vehicle="truck-b", capture=other)
    session(monkeypatch, frames=1, capture=generic)
    assert other[0][0] < 18.0 and generic[0][0] < 18.0


def test_game_only_warm_start_without_vehicle_identity(monkeypatch):
    session(monkeypatch)
    samples = []
    session(monkeypatch, frames=1, capture=samples)
    assert samples[0][0] == pytest.approx(20.0, abs=0.01)


def test_disengage_reengage_keeps_learned_scale(monkeypatch):
    from simsteer.app.global_keys import KEY_INSERT

    samples = []
    session(monkeypatch, frames=650, capture=samples,
            keys_at={600: [KEY_INSERT], 602: [KEY_INSERT]})
    assert all(sample[0] == pytest.approx(20., abs=0.01) for sample in samples[599:])


def test_bad_session_cannot_poison_warm_start(monkeypatch):
    session(monkeypatch, vehicle="truck-a")
    bad = plan()
    bad.pose[0] = 4.0  # moving gate passes, but ratio=5 exceeds the ceiling
    session(monkeypatch, vehicle="truck-a", decoded=bad)
    samples = []
    session(monkeypatch, frames=1, vehicle="truck-a", capture=samples)
    assert samples[0][0] == pytest.approx(20.0, abs=0.01)


@pytest.mark.parametrize("step", [Step.FOV, Step.CAMERA])
def test_calibration_cannot_train_speed_scale(monkeypatch, step):
    def calibrating(st, manager):
        st.flow.step = step
        if step is Step.FOV:
            # Keep the solver in FOV without model-dependent extra runs.
            st.flow.fov_proposal = 70.0

    session(monkeypatch, on_grab=calibrating)
    samples = []
    session(monkeypatch, frames=1, capture=samples)
    # No saved calibration-stage ratio may appear in a new driving session.
    assert samples[0][0] < 18.0
    session(monkeypatch)  # only post-calibration evidence can warm-start
    samples = []
    session(monkeypatch, frames=1, capture=samples)
    assert samples[0][0] == pytest.approx(20.0, abs=0.01)
