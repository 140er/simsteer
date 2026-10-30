"""Independent device -> filtered game input -> physical wheel plant."""

from collections import deque
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.app.settings import Settings
from simsteer.calib.flow import CalibFlow, Step
from simsteer.calib.steering import OutputDirection
from simsteer.core.learners.liveparams import LiveParams
from test_steering_direction import _bends, _flow, _plan, _probe_output


def run_plant(monkeypatch, tmp_path, *, polarity=-1, convention=1, inversion=1,
              fps=20, lag_s=0.25, smoothing_s=0.0, moving=False,
              interrupt=None, no_probe=False, drive_s=0.0, flip=False,
              correction=False, no_response=False, raw_lag_frames=0,
              raw_gain=1.0, deadzone=0.0, player_axis=0.3):
    import simsteer.runtime.loop as loop

    dt = 1 / fps
    clock = SimpleNamespace(t=0.0)
    clock.now = lambda: clock.t
    holder = {}
    st = SimpleNamespace(t=0.0, speed=22.0, raw=0.0, game=0.0, phi=0.0,
                         pad=0.0, engaged=False, writes=[], throttle=[],
                         first_verified=None, first_engaged=None, tripped=None,
                         inserts=set(), smooth=0.0, frame=0)
    history = deque([0.0] * max(1, round(lag_s * fps)))
    raw_history = deque([0.0] * raw_lag_frames)

    class Flow(CalibFlow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lc.cal_status = loop.CalStatus.CALIBRATED
            self._finish_fov("kept")
            holder["flow"] = self

    class Pad:
        kind = "gamepad"
        last_error = None

        @property
        def engaged(self):
            return st.engaged

        def engage(self):
            st.engaged = True
            st.first_engaged = st.first_engaged or st.t

        def disengage(self):
            if st.engaged and st.first_engaged is not None:
                st.tripped = st.t
            st.engaged = False
            st.pad = 0.0

        def center(self):
            st.pad = 0.0

        def set_steering(self, value, **kwargs):
            if self.engaged or kwargs.get("force"):
                st.pad = value
                st.writes.append((st.t, st.speed, value, st.engaged,
                                  holder["flow"].output_direction.sign))

        def set_throttle_brake(self, throttle, brake, **kwargs):
            st.throttle.append((st.t, throttle, brake,
                                holder["flow"].output_direction.ready))

    class Source:
        done = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def grab(self):
            st.frame += 1
            st.t = clock.t = st.frame * dt
            flow = holder["flow"]
            if st.t < 16.0:
                # The player drives both bends. No fitted signs or gains
                # are injected; LiveParams uses its real seed and updates.
                st.game = inversion * 3.5 * 0.022 * math.sin(st.t * 0.65)
                history.append(st.game)
                st.phi = history.popleft() / (inversion * 3.5)
                st.raw = st.game
            else:
                st.speed = 22.0 if moving or st.engaged else 0.0
                if interrupt == "speed" and st.t >= 19:
                    st.speed = 2.0
                human = player_axis if interrupt == "player" and 19 <= st.t < 20 else 0.0
                if correction and st.first_engaged and 8 <= st.t - st.first_engaged < 10:
                    human = -inversion * convention * 0.3
                binding = polarity
                if flip and st.first_engaged and st.t - st.first_engaged >= 10:
                    binding = -polarity
                target = math.copysign(max(0.0, abs(st.pad) - deadzone)
                                       / (1 - deadzone), binding * st.pad)
                raw_history.append(target)
                st.raw = raw_gain * raw_history.popleft() + human
                history.append(target)
                delayed = history.popleft()
                alpha = 1 if smoothing_s == 0 else 1 - math.exp(-dt / smoothing_s)
                st.smooth += alpha * (delayed - st.smooth)
                st.game = 0.0 if no_response else st.smooth + human
                st.phi = st.game / (inversion * 3.5)
            if flow.output_direction.ready and st.first_verified is None:
                st.first_verified = st.t
            return np.zeros((2, 2, 3), np.uint8)

    def decoded():
        curvature = st.phi / 2.65 if st.t < 16 else 0.004
        d = _plan(curvature, st.speed)
        d.pose = np.array([st.speed, 0.0, 0.0])
        return d

    class Keys:
        def poll(self):
            # First INSERT requests verification, not AI engagement.
            if st.t >= 16.75 and "check" not in st.inserts:
                st.inserts.add("check")
                return [loop.KEY_INSERT]
            if (drive_s and holder["flow"].output_direction.ready
                    and "drive" not in st.inserts):
                st.first_verified = st.first_verified or st.t
                st.inserts.add("drive")
                return [loop.KEY_INSERT]
            return []

    monkeypatch.setattr(loop, "CalibFlow", Flow)
    monkeypatch.setattr(loop, "DeviceManager", lambda **kwargs: Pad())
    monkeypatch.setattr(loop, "decode", lambda *args: decoded())
    monkeypatch.setattr(loop, "_lane_offset", lambda decoded: 0.0)
    monkeypatch.setattr(loop, "LongitudinalController", lambda cfg: SimpleNamespace(
        compute=lambda *args: (0.3, 0.0), reset=lambda: None))
    monkeypatch.setattr(loop, "FrameQueue", lambda: SimpleNamespace(push=lambda *args: (None, None)))
    tel = SimpleNamespace(
        available=True, speed_mps=lambda: st.speed, yaw_rate_rad_s=lambda: 0.0,
        wheel_angle_rad=lambda wb: convention * st.phi,
        game_steer=lambda: st.game, user_steer=lambda: st.raw, close=lambda: None)
    log = tmp_path / "plant.jsonl"
    assert loop.run(Settings(no_tuner=True, no_probe=no_probe), "ets2",
                    source=Source(), make_telemetry=lambda: tel, clock=clock,
                    keys=Keys(), model=SimpleNamespace(step=lambda *a, **kw: (None, None)),
                    headless=True, max_frames=round((33 + drive_s) * fps),
                    log_path=str(log)) == 0
    return holder["flow"], st, [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize("polarity", [1, -1])
@pytest.mark.parametrize("convention", [1, -1])
@pytest.mark.parametrize("inversion", [1, -1])
def test_stopped_measurement_precedes_any_engagement(monkeypatch, tmp_path,
                                                     polarity, convention, inversion):
    flow, st, rows = run_plant(monkeypatch, tmp_path, polarity=polarity,
                              convention=convention, inversion=inversion,
                              drive_s=5)
    assert st.first_verified is not None, "stopped pad pulses must verify output direction"
    assert st.first_verified < 31
    assert st.first_engaged >= st.first_verified
    pulses = [w for w in st.writes if w[4] is None and abs(w[2]) > 0]
    assert pulses and all(speed < 0.5 and not engaged for _, speed, _, engaged, _ in pulses)
    assert min(w[2] for w in pulses) < 0 < max(w[2] for w in pulses)
    assert all(t == b == 0 for _, t, b, ready in st.throttle if not ready)
    assert all(not r["engaged"] for r in rows if r["steering_output_sign"] is None)
    driving = [r for r in rows if r["engaged"] and r["v_ego"] > 10]
    assert driving and all(polarity * r["ai_steer_axis"] / inversion > 0
                           for r in driving)
    assert flow.direction.ready and flow.lp.axis_sign == inversion * convention


def test_driver_correction_does_not_trip_output_guard():
    flow = _flow(LiveParams(game="ets2", device_kind="gamepad"))
    _bends(flow, 1, 1)
    _probe_output(flow, -1)
    for _ in range(30):
        assert not flow.observe_output(command=-0.06, observed=-0.24, speed=22,
                                       engaged=True,
                                       lane_change=False, dt=0.05)
    assert flow.ai_steering_ready


def test_binding_guard_clears_only_output_measurement():
    lp = LiveParams(game="ets2", device_kind="gamepad")
    flow = _flow(lp)
    _bends(flow, -1, -1)
    _probe_output(flow, -1)
    lp.save()
    disk = lp._resolve_path("ets2", "gamepad")
    saved = disk.read_bytes()
    rack = lp.x.copy()
    direction = flow.direction
    tripped = False
    for i in range(20):
        if flow.observe_output(command=0.1, observed=0.1, speed=22,
                               engaged=True, lane_change=False, dt=0.05):
            tripped = True
            break
    assert tripped and (i + 1) * 0.05 <= 1.0
    assert flow.direction is direction and flow.direction.ready
    assert np.array_equal(lp.x, rack) and lp.axis_sign == -1
    assert disk.read_bytes() == saved
    assert not flow.output_direction.ready and not flow.ai_steering_ready
    assert flow.engage_block() and flow.step is Step.STEERING
    _probe_output(flow, 1)
    assert flow.ai_steering_ready and not flow.redo_note


@pytest.mark.parametrize("fps", [10, 20, 60])
@pytest.mark.parametrize("polarity,convention,inversion", [
    (p, w, i) for p in (1, -1) for w in (1, -1) for i in (1, -1)])
def test_stopped_check_handles_lag_and_heavy_smoothing(monkeypatch, tmp_path, fps,
                                                       polarity, convention, inversion):
    flow, st, rows = run_plant(monkeypatch, tmp_path, fps=fps, polarity=polarity,
                              convention=convention, inversion=inversion,
                              lag_s=0.5, smoothing_s=0.8)
    assert flow.output_direction.sign == polarity
    assert st.first_verified < 31 and not any(r["engaged"] for r in rows)


@pytest.mark.parametrize("fps", [10, 20, 60])
@pytest.mark.parametrize("raw_lag_frames", [1, 2])
@pytest.mark.parametrize("polarity", [-1, 1])
def test_stopped_check_tolerates_delayed_raw_input(monkeypatch, tmp_path, fps,
                                                 raw_lag_frames, polarity):
    flow, st, rows = run_plant(monkeypatch, tmp_path, fps=fps, polarity=polarity,
                              raw_lag_frames=raw_lag_frames, drive_s=5)
    assert flow.output_direction.sign == polarity, flow.output_direction.reason
    assert st.first_verified < 31 and st.first_engaged >= st.first_verified
    assert not any(r["engaged"] for r in rows if r["steering_output_sign"] is None)


@pytest.mark.parametrize("raw_gain", [0.5, 0.55, 0.7, 1.3, 1.5, 1.6])
@pytest.mark.parametrize("polarity", [-1, 1])
def test_stopped_check_tolerates_scaled_raw_input(monkeypatch, tmp_path, raw_gain, polarity):
    flow, st, _ = run_plant(monkeypatch, tmp_path, polarity=polarity, raw_gain=raw_gain)
    assert flow.output_direction.sign == polarity, flow.output_direction.reason
    assert st.first_verified < 31


@pytest.mark.parametrize("fps", [10, 20, 60])
@pytest.mark.parametrize("raw_gain", [0.55, 1.5])
def test_stopped_check_tolerates_raw_lag_scale_and_small_deadzone(monkeypatch, tmp_path,
                                                                fps, raw_gain):
    flow, st, _ = run_plant(monkeypatch, tmp_path, fps=fps, raw_lag_frames=2,
                            raw_gain=raw_gain, deadzone=0.02,
                            lag_s=0.5, smoothing_s=0.8)
    assert flow.output_direction.sign == -1, flow.output_direction.reason
    assert st.first_verified < 31


@pytest.mark.parametrize("fps", [10, 20, 60])
@pytest.mark.parametrize("no_response", [False, True])
@pytest.mark.parametrize("raw_gain", [0.0, 0.02])
def test_zero_raw_response_gets_no_response_reason(monkeypatch, tmp_path, fps,
                                                  no_response, raw_gain):
    flow, st, _ = run_plant(monkeypatch, tmp_path, fps=fps, raw_gain=raw_gain,
                            no_response=no_response)
    output = flow.output_direction
    assert not output.ready and not output.active
    assert "No steering response" in flow.hint(0)
    assert "\n" not in flow.hint(0) and "Release steering" not in flow.hint(0)
    pulses = [t for t, _, value, _, _ in st.writes if value]
    assert max(pulses) - min(pulses) == pytest.approx(3 - 1 / fps, abs=1e-8)
    assert st.pad == 0 and st.first_engaged is None


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("takeover", ["opposite", "excessive"])
def test_stopped_check_still_aborts_for_player_takeover(monkeypatch, tmp_path,
                                                        polarity, takeover):
    human = -2 * polarity * OutputDirection.AMPLITUDE if takeover == "opposite" else 0.4
    flow, st, rows = run_plant(monkeypatch, tmp_path, polarity=polarity,
                              raw_lag_frames=2, interrupt="player", player_axis=human)
    assert not flow.output_direction.ready and not flow.output_direction.active
    assert "Release steering" in flow.hint(0)
    assert max(t for t, _, value, _, _ in st.writes if value) >= 18.9
    assert not any(value for t, _, value, _, _ in st.writes if t >= 19)
    assert st.pad == 0 and not any(r["engaged"] for r in rows)


@pytest.mark.parametrize("raw", [-1.0, 1.0])
def test_player_takeover_aborts_even_during_pulse_settle(raw):
    output = OutputDirection(lag_frames=5)
    assert output.start(speed=0, observed=0, user_axis=0)
    assert output.tick_stopped(speed=0, observed=0, user_axis=0, dt=0.05)
    assert output.tick_stopped(speed=0, observed=0, user_axis=raw, dt=0.05) == 0
    assert not output.active and not output.ready
    assert "Release steering" in output.reason


def test_stopped_check_progress_moves_during_first_pulse():
    flow = _flow(LiveParams(game="ets2", device_kind="gamepad"))
    _bends(flow, 1, 1)
    output = flow.output_direction
    assert output.start(speed=0, observed=0, user_axis=0)
    assert flow.progress == 0
    command = 0.0
    progress = []
    for _ in range(21):
        command = flow.measure_output(speed=0, observed=-command,
                                      user_axis=-command, dt=0.05)
        progress.append(flow.progress)
    assert not output.evidence.pairs, "the first pulse is still running"
    assert 0 < progress[1] < progress[-1] < 0.25
    assert progress == sorted(progress)
    assert 0 < flow.panel(0).progress < 0.25


def test_moving_vehicle_never_receives_measurement_pulses(monkeypatch, tmp_path):
    flow, st, rows = run_plant(monkeypatch, tmp_path, moving=True)
    assert not flow.output_direction.ready and not flow.output_direction.active
    assert not st.writes and not any(r["engaged"] for r in rows)
    assert flow.hint(22) == "Stop to check steering direction"


@pytest.mark.parametrize("interrupt", ["speed", "player"])
def test_measurement_aborts_and_centres_on_movement_or_input(monkeypatch, tmp_path, interrupt):
    flow, st, rows = run_plant(monkeypatch, tmp_path, interrupt=interrupt)
    assert any(abs(w[2]) > 0 for w in st.writes)
    assert not any(abs(w[2]) > 0 for w in st.writes if w[0] >= 19)
    assert not flow.output_direction.ready and not flow.output_direction.active
    assert not any(r["engaged"] for r in rows)
    expected = "Stop" if interrupt == "speed" else "Release steering"
    assert expected in flow.hint(2 if interrupt == "speed" else 0)


def test_no_probe_still_verifies_output(monkeypatch, tmp_path):
    flow, st, _ = run_plant(monkeypatch, tmp_path, no_probe=True)
    assert flow.output_direction.sign == -1 and st.first_verified < 31


def test_missing_game_response_finishes_with_one_line_reason(monkeypatch, tmp_path):
    flow, st, _ = run_plant(monkeypatch, tmp_path, no_response=True)
    assert not flow.output_direction.ready and not flow.output_direction.active
    assert "No steering response" in flow.hint(0)
    assert max(t for t, _, value, _, _ in st.writes if value) < 21


def test_manual_override_cannot_trip_guard():
    flow = _flow(LiveParams(game="ets2", device_kind="gamepad"))
    _bends(flow, 1, 1)
    _probe_output(flow, -1)
    for _ in range(40):
        assert not flow.observe_output(command=-0.1, observed=-0.1, speed=22,
                                       engaged=True, lane_change=False, dt=0.05,
                                       manual_override=True)
    assert flow.output_direction.ready


def test_runtime_driver_correction_keeps_verification(monkeypatch, tmp_path):
    flow, st, rows = run_plant(monkeypatch, tmp_path, drive_s=15, correction=True)
    assert flow.output_direction.ready and flow.direction.ready
    assert all(r["engaged"] for r in rows if r["t"] > st.first_engaged + 1)


@pytest.mark.parametrize("fps", [10, 20, 60])
def test_runtime_mirrored_binding_disengages_quickly(monkeypatch, tmp_path, fps):
    flow, st, rows = run_plant(monkeypatch, tmp_path, drive_s=15, flip=True, fps=fps)
    changed = st.first_engaged + 10
    stopped = next(r["t"] for r in rows if r["t"] >= changed and not r["engaged"])
    assert stopped - changed <= 1.0 + 1 / fps
    assert flow.direction.ready and flow.lp.axis_sign == 1
    assert flow.lp.trusted() and not flow.output_direction.ready


def test_real_liveparams_reaches_check_and_done(monkeypatch, tmp_path):
    flow, st, rows = run_plant(monkeypatch, tmp_path, drive_s=150)
    assert any(r["calib"] == "check" for r in rows)
    assert flow.step is Step.DONE and flow.result["passed"]
    assert flow.lp.trusted()


@pytest.mark.parametrize("fps", [2, 8, 120, 240])
def test_measurement_uses_seconds_at_slow_and_fast_loop_rates(fps):
    direction = OutputDirection(lag_frames=5)
    assert direction.start(speed=0, observed=0)
    dt = 1 / fps
    command = smooth = 0.0
    history = deque([0.0] * max(1, round(0.5 * fps)))
    for _ in range(math.ceil(14 * fps)):
        history.append(command)
        target = history.popleft()
        smooth += (1 - math.exp(-dt / 0.8)) * (target - smooth)
        command = direction.tick_stopped(speed=0, observed=-smooth,
                                         user_axis=-command, dt=dt)
        if not direction.active:
            break
    assert direction.sign == -1
    assert direction.elapsed <= 14


def test_stalled_loop_clears_pulse_with_a_reason():
    direction = OutputDirection(lag_frames=5)
    assert direction.start(speed=0, observed=0)
    assert direction.tick_stopped(speed=0, observed=0, dt=0.05) == 0.1
    assert direction.tick_stopped(speed=0, observed=0, dt=0.6) == 0
    assert not direction.active and "frame rate" in direction.reason


@pytest.mark.parametrize("response", ["single-direction", "noise"])
def test_ambiguous_stopped_response_finishes_without_authority(response):
    direction = OutputDirection(lag_frames=5)
    assert direction.start(speed=0, observed=0)
    command = 0.0
    for i in range(280):
        observed = abs(command) if response == "single-direction" else 0.1 * (-1) ** i
        command = direction.tick_stopped(speed=0, observed=observed,
                                         user_axis=command, dt=0.05)
        if not direction.active:
            break
    assert not direction.ready and not direction.active and direction.reason
    assert command == 0 and direction.elapsed <= 14
