"""Deterministic live-loop inputs shared by diagnostics tests and benchmarks."""

import contextlib
import io
import json
import time
from types import SimpleNamespace

import numpy as np


def drive(monkeypatch, root, *, frames=80, period=.05, gap_at=None,
          overrides=(), log=True, log_timing=False, render=False, on_frame=None,
          instrument=False):
    from simsteer.app.settings import Settings
    from simsteer.calib.flow import CalibFlow
    from simsteer.core.learners.livecalib import CalStatus
    from simsteer.runtime.faults import neutral_decoded
    from simsteer.runtime import loop

    monkeypatch.setenv("SIMSTEER_DATA_DIR", str(root))
    clock = SimpleNamespace(t=0., now=lambda: clock.t)
    state = SimpleNamespace(frame=0, writes=[], durations=[], cpu_durations=[], flow=None)
    decoded = neutral_decoded()
    decoded.plan[:, 3] = 22.
    decoded.plan[:, 14] = .022
    decoded.pose[0] = 22.
    decoded.road_transform[2] = 1.2
    decoded.lane_lines_prob[:] = .95
    decoded.lane_lines[1, :, 0] = -1.8
    decoded.lane_lines[2, :, 0] = 1.8
    state.decoded = decoded

    class Flow(CalibFlow):
        @property
        def ai_steering_ready(self):
            return True

        def tick(self, obs):
            pass

        def observe_output(self, **kwargs):
            return False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lc.cal_status = CalStatus.CALIBRATED
            self._finish_fov("kept")
            self.output_direction.sign = -1
            state.flow = self

    class Pad:
        kind = "gamepad"
        engaged = False

        def engage(self):
            self.engaged = True

        def disengage(self):
            self.engaged = False

        def center(self):
            pass

        def set_steering(self, value, **kwargs):
            state.writes.append(value)

        def set_throttle_brake(self, *args, **kwargs):
            pass

    pad = Pad()
    state.pad = pad
    frame = np.zeros((2, 2, 3), np.uint8)
    manual = loop.ManualInputs()

    class Source:
        done = False
        previous = None
        previous_cpu = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def grab(self):
            now = time.perf_counter_ns()
            cpu = time.thread_time_ns()
            if self.previous is not None:
                state.durations.append(now - self.previous)
                state.cpu_durations.append(cpu - self.previous_cpu)
            self.previous = now
            self.previous_cpu = cpu
            state.frame += 1
            clock.t += .2 if state.frame == gap_at else period
            manual.steer_override = state.frame in overrides
            if on_frame is not None:
                on_frame(state)
            return frame

    monkeypatch.setattr(loop, "CalibFlow", Flow)
    monkeypatch.setattr(loop, "DeviceManager", lambda **kwargs: pad)
    monkeypatch.setattr(loop, "ManualInputs", lambda: manual)
    monkeypatch.setattr(loop, "decode", lambda *args: decoded)
    if hasattr(loop, "RunDiagnostics"):
        make_diagnostics = loop.RunDiagnostics

        def make(*args, **kwargs):
            state.diagnostics = make_diagnostics(*args, **kwargs)
            if instrument:
                state.diagnostic_costs = []
                state.diagnostic_ns = 0
                for name in ("begin_tick", "control", "observe", "finish"):
                    original = getattr(state.diagnostics, name)

                    def measured(*args, method=original, stage=name, **kwargs):
                        started = time.perf_counter_ns()
                        result = method(*args, **kwargs)
                        state.diagnostic_ns += time.perf_counter_ns() - started
                        if stage == "finish":
                            state.diagnostic_costs.append(state.diagnostic_ns)
                            state.diagnostic_ns = 0
                        return result

                    setattr(state.diagnostics, name, measured)
            return state.diagnostics

        monkeypatch.setattr(loop, "RunDiagnostics", make)
    monkeypatch.setattr(loop, "FrameQueue", lambda: SimpleNamespace(
        push=lambda *args: (None, None), last_yuv=None,
        last_yuv_narrow=None, last_yuv_wide=None))
    if render:
        state.hud = []
        monkeypatch.setattr(loop, "setup_window", lambda *args: None)
        monkeypatch.setattr(loop.cv2, "setWindowTitle", lambda *args: None)
        monkeypatch.setattr(loop.cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(loop, "draw_overlay", lambda frame, *args, **kwargs: frame)
        monkeypatch.setattr(loop, "show_scaled", lambda *args: None)
        monkeypatch.setattr(loop, "draw_developer_panel", lambda *args: None)
        monkeypatch.setattr(loop, "HudRenderer", lambda: SimpleNamespace(
            draw=lambda frame, hud, **kwargs: state.hud.append(hud)))
    tel = SimpleNamespace(
        available=True, speed_mps=lambda: 22., yaw_rate_rad_s=lambda: 0.,
        wheel_angle_rad=lambda wb: 0., game_steer=lambda: 0., user_steer=lambda: 0.,
        close=lambda: None)
    path = root / "run.jsonl"
    with contextlib.redirect_stdout(io.StringIO()):
        state.rc = loop.run(
            Settings(no_tuner=True, no_probe=True, force_engage=True, hud_mode="dev"),
            "ets2", source=Source(), make_telemetry=lambda: tel, clock=clock,
            keys=loop.NoKeys(), model=SimpleNamespace(step=lambda *a, **kw: (None, None)),
            headless=not render, max_frames=frames, log_path=str(path) if log else None,
            log_timing=log_timing)
    state.rows = [json.loads(line) for line in path.read_text().splitlines()] if log else []
    return state
