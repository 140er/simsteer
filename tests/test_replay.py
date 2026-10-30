from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simsteer.core.constants import IDX_N, PLAN_WIDTH, T_IDXS
from simsteer.runtime.replay import (
    ClipWriter, ReplayClock, ReplaySource, ReplayTelemetry,
)

N_FRAMES = 30
FPS = 20.0


class FakeModel:
    """Stands in for `DrivingModel`: a plan whose lateral offset follows
    the frame's brightness, so different frames steer differently."""

    active_provider = policy_provider = "fake"

    def step(self, img, big_img=None, desire=None):
        level = float(img[0, :4].mean()) / 255.0 - 0.5
        vision = np.zeros(1576, np.float32)
        vision[105 + 2] = 1.2  # road_transform height
        t = np.asarray(T_IDXS, np.float32)
        plan = np.zeros((IDX_N, PLAN_WIDTH), np.float32)
        plan[:, 0] = 20.0 * t
        plan[:, 1] = level * 0.02 * (20.0 * t) ** 2
        plan[:, 3] = 20.0
        plan[:, 14] = level * 0.05
        policy = np.zeros(1000, np.float32)
        policy[:IDX_N * PLAN_WIDTH] = plan.reshape(-1)
        return vision, policy


def _make_clip(root: Path, frames: int = N_FRAMES, gap: tuple[float, float] | None = None) -> Path:
    writer = ClipWriter(root, game="ets2")
    yy, xx = np.mgrid[0:180, 0:320]
    for i in range(frames):
        t = i / FPS
        img = np.zeros((180, 320, 3), np.uint8)
        img[..., 0] = (xx + 7 * i) % 256
        img[..., 1] = (yy * 2 + 3 * i) % 256
        img[..., 2] = (i * 8) % 256
        writer.write_frame(t, img)
        if gap and gap[0] <= t < gap[1]:
            continue
        writer.write_telemetry(t, {"speed_mps": 20.0 + i * 0.1, "yaw_rate_rad_s": 0.01,
                                   "wheel_angle_rad": 0.002, "game_steer": 0.0,
                                   "user_steer": 0.0})
    writer.close("test")
    return root


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return _make_clip(tmp_path / "clip")


@pytest.fixture
def fake_model(monkeypatch):
    import simsteer.runtime.loop as loop
    monkeypatch.setattr(loop, "DrivingModel", FakeModel)


def _replay(clip: Path, log: Path, *extra: str) -> int:
    from simsteer.app.main import main
    return main(["--source", str(clip), "--headless", "--log-jsonl", str(log), *extra])


def test_loop_replays_clip_deterministically(clip, tmp_path, fake_model, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert _replay(clip, a) == 0
    assert _replay(clip, b) == 0
    assert capsys.readouterr().out.rstrip().endswith(f"frames: {N_FRAMES}")

    lines = a.read_text().splitlines()
    assert len(lines) == N_FRAMES
    assert a.read_bytes() == b.read_bytes()
    rows = [json.loads(line) for line in lines]
    assert [r["t"] for r in rows] == [round(i / FPS, 4) for i in range(N_FRAMES)]
    assert {"v_ego", "steer_axis", "curvature", "telemetry"} <= rows[0].keys()
    assert len({r["steer_axis"] for r in rows}) > 1


def test_run_direct_with_injected_parts(clip, tmp_path):
    from simsteer.app.settings import Settings
    from simsteer.runtime.loop import run

    clock = ReplayClock()
    log = tmp_path / "direct.jsonl"
    rc = run(Settings(no_tuner=True, no_gamepad=True), "ets2",
             source=ReplaySource(clip, clock),
             make_telemetry=lambda: ReplayTelemetry(clip, clock),
             clock=clock, model=FakeModel(), headless=True, max_frames=12,
             log_path=str(log), log_timing=True)
    assert rc == 0
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 12
    assert {"cap_ms", "warp_ms", "model_ms", "ctrl_ms", "frame_ms"} <= rows[0].keys()


def test_replay_does_not_touch_user_state(clip, tmp_path, fake_model, data_dir):
    import os
    assert _replay(clip, tmp_path / "log.jsonl") == 0
    assert sorted(p.name for p in data_dir.iterdir()) == ["clip", "log.jsonl"]
    assert os.environ["SIMSTEER_DATA_DIR"] == str(data_dir)


def test_telemetry_gap_reported_and_run_continues(tmp_path, fake_model):
    clip = _make_clip(tmp_path / "gap", gap=(0.5, 0.95))
    log = tmp_path / "gap.jsonl"
    assert _replay(clip, log) == 0
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == N_FRAMES
    states = {r["t"]: r["telemetry"] for r in rows}
    assert states[0.0] == "ok"
    assert states[0.85] == "lost"
    assert states[1.0] == "ok"


def test_telemetry_unavailable_past_last_sample(clip):
    clock = ReplayClock()
    tel = ReplayTelemetry(clip, clock)
    clock.advance_to((N_FRAMES - 1) / FPS)
    assert tel.available
    assert tel.speed_mps() == pytest.approx(20.0 + (N_FRAMES - 1) * 0.1)
    clock.advance_to((N_FRAMES - 1) / FPS + 1.0)
    assert not tel.available
    assert tel.speed_mps() is None
    clock.advance_to(-0.5)
    assert not tel.available


def test_source_moves_clock_and_ends(clip):
    clock = ReplayClock()
    with ReplaySource(clip, clock) as src:
        times = []
        while (frame := src.grab()) is not None:
            assert frame.shape == (180, 320, 3)
            times.append(clock.now())
        assert src.done
        assert src.grab() is None
    assert times == [round(i / FPS, 4) for i in range(N_FRAMES)]


def test_source_reads_video(tmp_path, clip):
    import cv2

    video = tmp_path / "vid" / "video.avi"
    video.parent.mkdir()
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 180))
    if not out.isOpened():
        pytest.skip("no MJPG writer in this OpenCV build")
    for p in sorted((clip / "frames").glob("*.jpg")):
        out.write(cv2.imread(str(p)))
    out.release()
    clock = ReplayClock()
    with ReplaySource(video.parent, clock) as src:
        n = 0
        while src.grab() is not None:
            n += 1
    assert n == N_FRAMES
    assert clock.now() == pytest.approx((N_FRAMES - 1) / FPS)


def test_writer_refuses_to_overwrite_a_clip(clip):
    with pytest.raises(FileExistsError):
        ClipWriter(clip, game="ets2")


def test_writer_failure_raises_instead_of_blocking(tmp_path, monkeypatch):
    import threading

    w = ClipWriter(tmp_path / "rec", game="ets2", threaded=True)

    def disk_full(i, img):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(w, "_encode", disk_full)
    outcome = {}

    def record():
        frame = np.zeros((90, 160, 3), np.uint8)
        try:
            for i in range(500):
                w.write_frame(i * 0.05, frame)
        except RuntimeError as e:
            outcome["raised"] = str(e)
        outcome["meta"] = w.close(source="test")

    t = threading.Thread(target=record, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "recorder blocked after a write failure"
    assert "No space left" in outcome["raised"]
    assert "No space left" in outcome["meta"]["error"]


def test_record_writes_a_replayable_clip(tmp_path, monkeypatch, fake_model, data_dir):
    import time

    import simsteer.games.base as games_base
    import simsteer.runtime.capture as capture
    from simsteer.runtime.replay import clip_meta

    class FakeCapture:
        def __init__(self, cfg=None):
            self.i = 0
            self.done = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def grab(self):
            time.sleep(0.01)
            self.i += 1
            img = np.full((1080, 1920, 3), self.i % 256, np.uint8)
            img[:, : self.i % 1920] = 255
            return img

    class FakeTelemetry:
        def __init__(self):
            self.n = 0

        @property
        def available(self):
            self.n += 1
            return self.n % 10 != 0

        def speed_mps(self):
            return 15.0

        def yaw_rate_rad_s(self):
            return 0.02

        def wheel_angle_rad(self, wheelbase_m=1.0):
            return 0.01

        def game_steer(self):
            return 0.1

        def user_steer(self):
            return None

        def close(self):
            pass

    profile = games_base.get_profile("ets2")
    monkeypatch.setattr(profile, "make_telemetry", FakeTelemetry)
    monkeypatch.setattr(games_base, "get_profile", lambda game_id: profile)
    monkeypatch.setattr(capture, "Capture", FakeCapture)
    (data_dir / "calibration_ets2.json").write_text("{}")

    from simsteer.app.main import main
    out = tmp_path / "rec"
    assert main(["--record", str(out), "--seconds", "0.5"]) == 0

    meta = clip_meta(out)
    frames = sorted((out / "frames").glob("*.jpg"))
    assert meta["frames"] == len(frames) > 10
    assert (meta["width"], meta["height"]) == (960, 540)
    assert (out / "state" / "calibration_ets2.json").exists()
    tel_rows = (out / "telemetry.jsonl").read_text().splitlines()
    assert 0 < len(tel_rows) < len(frames)
    assert json.loads(tel_rows[0])["user_steer"] is None

    log = tmp_path / "rec.jsonl"
    assert _replay(out, log) == 0
    assert len(log.read_text().splitlines()) == len(frames)
