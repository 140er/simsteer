"""Regressions for binding, capture cadence and fault presentation."""

import ctypes
import json
import math
import struct
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.app.global_keys import KEY_INSERT
from simsteer.core.calibration import Calibration
from simsteer.runtime.capture import Capture
from simsteer.runtime.telemetry import scs_shm as scs
from test_output_safety import BAD, drive, gamepad, telemetry


@pytest.mark.parametrize("paused,static", [(True, False), (False, True), (True, True)])
def test_binding_works_in_paused_and_static_menus(monkeypatch, paused, static):
    tel, buf, _ = telemetry(monkeypatch)
    struct.pack_into("<f", buf, scs.OFF_TRUCK_SPEED, 0.)
    buf[scs.OFF_PAUSED] = int(paused)

    def request(st, manager):
        if st.i == 0:
            manager.request_bind("left_stick_x")
    _, pad, manager = drive(monkeypatch, force=False, speed=0., frames=145,
                            telemetry_source=tel, no_frames=static, on_grab=request)
    assert any(abs(report[0]) == 1. for report in pad.raw.sent)
    assert pad.raw.visible == (0., 0., 0.)
    assert manager.binding_status == "Done"


def test_binding_reports_cancellation_when_engaged(monkeypatch):
    def request(st, manager):
        if st.i == 0:
            manager.request_bind("left_stick_x")
        if st.i == 50:
            manager.engage()
            st.status = manager.binding_status
            assert not manager.request_bind("left_stick_x")
    st, pad, _ = drive(monkeypatch, force=False, speed=0., frames=60, on_grab=request)
    assert st.status == "Cancelled"
    assert not any(abs(report[0]) == 1. for report in pad.raw.sent)


def test_tuner_receives_binding_completion_without_changing_controls():
    from simsteer.ui.tuner import Tuner
    labels = []
    tuner = Tuner(Calibration(), SimpleNamespace(), SimpleNamespace(),
                  device=SimpleNamespace(binding_status="Done"), start_thread=False)
    tuner._bind_status_label = SimpleNamespace(configure=lambda **kw: labels.append(kw["text"]))
    tuner._refresh_bind_status()
    assert labels == ["Done"]


@pytest.mark.parametrize("cost", [.003, .009])
def test_capture_keeps_deadline_cadence_with_coarse_waits(monkeypatch, cost):
    import simsteer.runtime.capture as capture
    clock = SimpleNamespace(t=0., n=0)
    times = []
    monkeypatch.setattr(capture.time, "monotonic", lambda: clock.t)

    class Stop:
        def is_set(self):
            return False

        def wait(self, seconds):
            assert seconds >= 0.
            clock.t += math.ceil(seconds / .015625) * .015625
            clock.n += 1
            return clock.n >= 200

    def read():
        clock.t += cost
        times.append(clock.t)
        return np.zeros((2, 2, 3), np.uint8)

    Capture()._read(SimpleNamespace(get_latest_frame=read), Queue(maxsize=1), Stop())
    period = (times[-1] - times[0]) / (len(times) - 1)
    assert period == pytest.approx(.05, abs=.0002)


@pytest.mark.parametrize("startup_error", [False, True])
def test_windows_timer_resolution_is_balanced(monkeypatch, startup_error):
    import simsteer.runtime.capture as capture
    import sys
    calls = []
    monkeypatch.setattr(capture, "sys", SimpleNamespace(platform="win32"), raising=False)
    timer = SimpleNamespace(timeBeginPeriod=lambda n: calls.append(("begin", n)) or 0,
                            timeEndPeriod=lambda n: calls.append(("end", n)) or 0)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: timer, raising=False)
    monkeypatch.setattr(capture, "Thread", lambda *a, **kw: SimpleNamespace(
        start=lambda: None, join=lambda **kw: None))

    def start(**kwargs):
        if startup_error:
            raise RuntimeError("camera unavailable")
    cam = SimpleNamespace(start=start, stop=lambda: None)
    monkeypatch.setitem(sys.modules, "dxcam", SimpleNamespace(create=lambda **kw: cam))
    cap = Capture()
    if startup_error:
        with pytest.raises(RuntimeError):
            cap.start()
    else:
        cap.start()
        cap.stop()
        cap.stop()
    assert calls == [("begin", 1), ("end", 1)]


def test_capture_avoids_extra_full_frame_copy():
    copies = []

    class Frame(np.ndarray):
        def copy(self, *a, **kw):
            copies.append(self.shape)
            return super().copy(*a, **kw)
    cap = Capture()
    cap._cam = object()
    frame = np.zeros((720, 1280, 3), np.uint8).view(Frame)
    import time
    cap._frames.put((time.monotonic(), frame))
    assert cap.grab() is frame
    assert frame.shape not in copies
    assert cap._previous.size <= 32 * 64 * 3


@pytest.mark.parametrize("fault,key", [(None, KEY_INSERT), ("speed", None), ("lost", None)])
def test_stops_play_one_sound_and_clean_banner(monkeypatch, fault, key):
    states = []
    factory = lambda st, pad: SimpleNamespace(take_restart_request=lambda: None,
                                               update_state=states.append)
    st, _, _ = drive(monkeypatch, fault=fault, key=key, tuner_factory=factory)
    assert st.sounds.count("disengage") == 1
    assert states[1].banner_text == ("Disengaged" if key else
                                     "Stopped: invalid data" if fault == "speed" else
                                     "Stopped: Telemetry unavailable")


@pytest.mark.parametrize("no_frames", [False, True])
def test_persistent_fault_keeps_hud_tuner_and_keys_alive(monkeypatch, no_frames):
    states = []
    factory = lambda st, pad: SimpleNamespace(take_restart_request=lambda: None,
                                               update_state=states.append)
    st, _, _ = drive(monkeypatch, frames=21, fault="persistent" if not no_frames else None,
                     no_frames=no_frames, tuner_factory=factory, keys_at={3: [ord("h")]})
    assert len(states) == 21
    assert states[1].banner_text.startswith("Stopped:")
    assert all(not state.engaged for state in states[1:])
    assert st.settings.hud_mode == "dev"
    assert all(report == (0., 0., 0.) for engaged, report in st.checkpoints[2:])


@pytest.mark.parametrize("no_frames", [False, True])
def test_source_loss_preserves_manual_selection_but_gates_output(monkeypatch, no_frames):
    def select(st, manager):
        if st.i == 1:
            st.manual.steer_override = st.manual.long_override = True
            st.manual.steer, st.manual.throttle, st.manual.brake = .4, .3, .2
    st, _, _ = drive(monkeypatch, force=False, no_frames=no_frames,
                     fault=None if no_frames else "lost", on_grab=select)
    assert st.manual.steer_override and st.manual.long_override
    assert (st.manual.steer, st.manual.throttle, st.manual.brake) == (.4, .3, .2)
    assert all(report == (0., 0., 0.) for engaged, report in st.checkpoints[2:])


def test_startup_loss_is_a_blocked_reason_not_a_disengage_banner(monkeypatch):
    states = []
    factory = lambda st, pad: SimpleNamespace(take_restart_request=lambda: None,
                                               update_state=states.append)
    st, _, _ = drive(monkeypatch, force=False, fault="lost", tuner_factory=factory)
    assert all(not state.banner_text for state in states)
    assert "Telemetry" in states[1].warnings[0] or "telemetry" in states[1].warnings[0]
    assert "disengage" not in st.sounds


def test_telemetry_flicker_debounces_banners_without_silencing_stops(monkeypatch):
    import simsteer.runtime.loop as loop
    deadlines = []
    original = loop._build_hud_state

    def build(**kwargs):
        deadlines.append(kwargs["banner_until"])
        return original(**kwargs)
    monkeypatch.setattr(loop, "_build_hud_state", build)
    tel = SimpleNamespace(available=True, speed_mps=lambda: 20., yaw_rate_rad_s=lambda: 0.,
                          wheel_angle_rad=lambda wb: 0., game_steer=lambda: 0.,
                          user_steer=lambda: 0., close=lambda: None)

    def flicker(st, manager):
        tel.available = st.i % 2 == 0
        if st.i in (0, 2, 4):
            manager.engage()
    factory = lambda st, pad: SimpleNamespace(take_restart_request=lambda: None,
                                               update_state=lambda state: None)
    st, _, _ = drive(monkeypatch, force=False, frames=6, on_grab=flicker,
                     telemetry_source=tel, tuner_factory=factory)
    assert deadlines[1] == deadlines[3] == deadlines[5]
    assert st.sounds.count("disengage") == 3


def test_insert_disengagement_uses_shared_handback_hook(monkeypatch):
    called = []

    def hook(st, manager):
        def disengage():
            called.append("insert")
            manager.disengage()
        manager.insert_disengage = disengage
    st, pad, _ = drive(monkeypatch, key=KEY_INSERT, on_grab=hook)
    assert called == ["insert"]
    assert st.sounds.count("disengage") == 1
    assert not pad.engaged


@pytest.mark.parametrize("field", ["pitch_deg", "yaw_deg", "height_m", "fov_h_deg", "crop_top_pct", "crop_bottom_pct"])
@pytest.mark.parametrize("bad", BAD)
def test_calibration_rejects_poison_once_and_never_saves_it(tmp_path, caplog, field, bad):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({field: bad, "pitch_deg": bad if field == "pitch_deg" else 2.}))
    loaded = Calibration.load(path=path, fov_h_deg=75.)
    assert loaded == Calibration(fov_h_deg=75.)
    assert len(caplog.records) == 1
    loaded.save(path=path)
    before = path.read_bytes()
    setattr(loaded, field, bad)
    loaded.save(path=path)
    assert path.read_bytes() == before


def test_shutdown_has_distinct_name_releases_backend_and_latches_writers(monkeypatch):
    import simsteer.runtime.output.base as devices
    pad = gamepad()
    released = []
    pad.close = lambda: released.append(True)
    monkeypatch.setattr(devices, "Gamepad", lambda: pad)
    manager = devices.DeviceManager()
    manager.engage()
    manager.set_steering(.4)
    manager.shutdown_output()
    manager.shutdown_output()
    manager.engage()
    manager.set_steering(.8, force=True)
    assert pad.raw.visible == (0., 0., 0.) and not pad.engaged
    assert released == [True]


def test_every_writer_uses_the_same_shutdown_lock(monkeypatch):
    import threading
    import simsteer.runtime.output.base as devices
    pad = gamepad()
    monkeypatch.setattr(devices, "Gamepad", lambda: pad)
    manager = devices.DeviceManager()
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()

    def write(backend):
        entered.set()
        assert release.wait(1.)
        backend.set_steering(.5, force=True)
    assert callable(manager.write_output)
    writer = threading.Thread(target=lambda: manager.write_output(write))
    closer = threading.Thread(target=lambda: (manager.shutdown_output(), stopped.set()))
    writer.start()
    try:
        assert entered.wait(1.)
        closer.start()
        assert not stopped.wait(.02)
    finally:
        release.set()
        writer.join(1.)
        if closer.ident is not None:
            closer.join(1.)
    assert stopped.is_set() and pad.raw.visible == (0., 0., 0.)
    manager.write_output(lambda backend: backend.set_steering(1., force=True))
    assert pad.raw.visible == (0., 0., 0.)
