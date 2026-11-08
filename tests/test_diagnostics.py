import io
import json
import threading

import pytest

from simsteer.runtime.diagnostics import (
    AUTO_DISENGAGE, EXECUTOR_FAULT, FAULT, INVALID_DATA, LANE_EXCURSION,
    OUTPUT_DIRECTION, USER_TAKEOVER, FlightRing, RunDiagnostics, TimingStats,
)


def test_ring_bounds_eviction_and_clock_reset():
    ring = FlightRing(seconds=2, max_samples=3)
    for t in (0., .1, .2, .3):
        ring.append({"t": t})
    assert [x["t"] for x in ring.samples] == [.1, .2, .3]
    ring.append({"t": 3.})
    assert list(ring.samples) == [{"t": 3.}]
    ring.append({"t": 0.})
    assert list(ring.samples) == [{"t": 0.}]


def test_dump_rate_limit_schema_and_no_tick_disk_io(tmp_path):
    diagnostics = RunDiagnostics(tmp_path)
    assert not diagnostics.root.exists()
    for i in range(1000):
        diagnostics.ring.append({"t": i / 100, "speed_mps": 22.})
    assert not diagnostics.root.exists()
    try:
        for t, reason in ((10., "frame rate dropped (5 fps)"),
                          (11., "Output direction changed"),
                          (20., "Steering executor failed")):
            diagnostics.incident(reason)
            diagnostics._submit(t)
            diagnostics._queue.join()
    finally:
        diagnostics.close()
    files = sorted(diagnostics.root.glob("*.json"))
    assert len(files) == 2
    first, second = [json.loads(p.read_text()) for p in files]
    assert first["events"] == [{"t": 10., "codes": [AUTO_DISENGAGE]}]
    assert second["events"][0]["codes"] == [EXECUTOR_FAULT]
    assert len(first["samples"]) == 600


def test_trigger_codes_discard_free_text(tmp_path):
    from simsteer.core.finite import NonFiniteError

    diagnostics = RunDiagnostics(tmp_path)
    for reason in ("arbitrary private fault text", "Output direction changed",
                   "Steering executor failed", NonFiniteError("test")):
        diagnostics.incident(reason)
    diagnostics.incident(user_stop=True)
    assert diagnostics._pending == {FAULT, OUTPUT_DIRECTION, EXECUTOR_FAULT, INVALID_DATA, USER_TAKEOVER}
    diagnostics.incident("Telemetry unavailable")
    assert 9 in diagnostics._pending


def test_queue_bound_and_write_failure_is_contained(tmp_path, monkeypatch, caplog):
    diagnostics = RunDiagnostics(tmp_path)
    diagnostics.root = tmp_path / "not-a-directory"
    diagnostics.root.write_text("occupied")
    diagnostics.ring.append({"t": 0.})
    diagnostics.incident("fault")
    diagnostics._submit(0.)
    diagnostics.close()
    assert diagnostics.disabled
    assert caplog.text.count("Session diagnostics unavailable") == 1
    out = io.StringIO()
    diagnostics.finish(1., {"frame": 1}, out)
    assert json.loads(out.getvalue()) == {"frame": 1}


def test_timing_stats_eviction_and_hysteresis():
    stats = TimingStats(window_s=1., max_samples=100, sustain_s=.5)
    for i in range(30):
        stats.update(i / 20, (1., 2., 3., 4., 5.), True)
    assert stats.hz == pytest.approx(20.) and not stats.slow
    assert stats.summary()["draw_mean_ms"] == 5.
    for i in range(1, 25):
        stats.update(1.45 + i / 10, (2.,) * 5, True)
    assert stats.slow and stats.hz == pytest.approx(10.)
    start = stats.samples[-1][0]
    for i in range(1, 20):
        stats.update(start + i / 18.5, (2.,) * 5, True)
    assert stats.slow  # Between the 18 Hz trip and 19 Hz recovery thresholds.
    start = stats.samples[-1][0]
    for i in range(1, 50):
        stats.update(start + i / 20, (2.,) * 5, True)
    assert not stats.slow and stats.summary()["model_mean_ms"] == 2.
    assert len(stats.samples) <= 21


def test_short_dip_does_not_warn_and_missing_frames_count():
    stats = TimingStats(window_s=1., sustain_s=1.5)
    for i in range(101):
        stats.update(i / 20, (0.,) * 5, i < 50 or i > 52)
        assert not stats.slow
    for i in range(101, 150):
        stats.update(i / 20, (0.,) * 5, False)
    assert stats.slow and stats.hz == 0.


def test_single_pending_dump_under_blocked_writer(tmp_path, monkeypatch):
    diagnostics = RunDiagnostics(tmp_path, dump_interval_s=0)
    started, release = threading.Event(), threading.Event()
    original = diagnostics._write

    def delayed():
        started.set()
        release.wait()
        original()

    monkeypatch.setattr(diagnostics, "_write", delayed)
    diagnostics.ring.append({"t": 0.})
    try:
        for i in range(5):
            diagnostics.incident("fault")
            diagnostics._submit(float(i))
        assert started.wait(2.)
        assert diagnostics._queue.qsize() == 1
    finally:
        release.set()
        diagnostics.close()


def test_manual_and_lane_triggers_include_current_tick(monkeypatch, tmp_path):
    from diagnostics_harness import drive

    state = drive(monkeypatch, tmp_path, overrides=(3,), frames=10)
    assert state.rc == 0
    files = list((tmp_path / "logs" / "incidents").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["events"][0]["codes"] == [USER_TAKEOVER]
    assert len(payload["samples"]) == 3


def test_lane_excursion_triggers_once_on_crossing(monkeypatch, tmp_path):
    from diagnostics_harness import drive

    def move(state):
        if state.frame >= 4:
            state.decoded.lane_lines[1, :, 0] = .2
            state.decoded.lane_lines[2, :, 0] = 3.8

    state = drive(monkeypatch, tmp_path, frames=10, on_frame=move)
    files = list((tmp_path / "logs" / "incidents").glob("*.json"))
    payload = json.loads(files[0].read_text())
    assert state.rc == 0 and len(files) == 1
    assert payload["events"][0]["codes"] == [LANE_EXCURSION]
    assert payload["samples"][-1]["lane_offset_m"] == -2.
    assert len(payload["samples"]) == 4


def test_timing_stage_accounting_preserves_fov_vision_time(tmp_path):
    diagnostics = RunDiagnostics(tmp_path, timing_log=True)
    diagnostics._sample = {"t": 0.}
    diagnostics._fresh = True
    diagnostics._counters = (0., .010, .010, .030, .060, .100, .005)
    diagnostics.finish(.150)
    assert diagnostics.fields["capture_wait_ms"] == 10.
    assert diagnostics.fields["warp_ms"] == 20.
    assert diagnostics.fields["model_ms"] == 35.
    assert diagnostics.fields["control_ms"] == 35.
    assert diagnostics.fields["draw_ms"] == 50.
    assert diagnostics.fields["timing_stats"]["draw_mean_ms"] == 50.


def test_initialization_failure_is_contained(tmp_path, monkeypatch, caplog):
    from simsteer.runtime import diagnostics as module

    def fail():
        raise OSError("unavailable")

    monkeypatch.setattr(module, "FlightRing", fail)
    diagnostics = RunDiagnostics(tmp_path)
    diagnostics.begin_tick()
    diagnostics.incident("fault")
    diagnostics.close()
    assert diagnostics.disabled
    assert caplog.text.count("Session diagnostics unavailable") == 1
