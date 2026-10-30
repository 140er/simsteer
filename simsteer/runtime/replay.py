"""Record a drive to a clip folder and play it back through the loop.

A clip is a folder:

    clip.json         format, game, fps, frame size, frame count, model hashes
    frames/000000.jpg one JPEG per frame (or one video file, see ReplaySource)
    frames.jsonl      {"i": 0, "t": 0.0} per frame, seconds from clip start
    telemetry.jsonl   {"t": 0.0, "speed_mps": ..., ...} per sample
    state/            optional per-game state (calibration etc.) copied at
                      record time; replay starts from it

Playback is driven by the recorded timestamps, not the wall clock:
`ReplaySource.grab()` moves a `ReplayClock` to each frame's time and
`ReplayTelemetry` answers for that time. Two runs over one clip therefore
see identical inputs and produce identical decisions.
"""

from __future__ import annotations

import hashlib
import json
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from simsteer.games.profile import GameProfile

CLIP_FORMAT = 1
RECORD_WIDTH = 960
JPEG_QUALITY = 90
VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".mov")
TELEMETRY_FIELDS = ("speed_mps", "yaw_rate_rad_s", "wheel_angle_rad",
                    "game_steer", "user_steer")
# Telemetry older than this at the current frame counts as lost. Live
# telemetry arrives once per frame at 20 Hz, so this allows a few drops.
TELEMETRY_MAX_AGE_S = 0.25


class LiveClock:
    """Seconds from a monotonic wall clock."""

    def now(self) -> float:
        return time.perf_counter()


class ReplayClock:
    """Seconds from the clip's recorded timestamps. `ReplaySource` moves it."""

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def advance_to(self, t: float) -> None:
        self._t = t


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clip_meta(clip: Path) -> dict:
    p = clip / "clip.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


class ReplaySource:
    """Frames from a recorded clip, with `Capture`'s start/stop/grab.

    `path` is a clip folder holding `frames/` or one video file, or a video
    file whose folder holds the jsonl files. Frames are decoded on a
    background thread, in order, so decode time overlaps the model like
    the live capture thread does. `grab()` blocks for the next frame and
    returns None once the clip is exhausted, after which `done` is True.
    """

    def __init__(self, path: str | Path, clock: ReplayClock,
                 prefetch: int = 8) -> None:
        path = Path(path).expanduser()
        self.clock = clock
        self.clip = path if path.is_dir() else path.parent
        meta = clip_meta(self.clip)
        self.fps = float(meta.get("fps", 20.0))
        self._video: Path | None = None
        self._frames: list[Path] = []
        if path.is_file():
            self._video = path
        elif (path / "frames").is_dir():
            self._frames = sorted((path / "frames").glob("*.jpg"))
        else:
            videos = [p for p in sorted(path.iterdir())
                      if p.suffix.lower() in VIDEO_SUFFIXES]
            if not videos:
                raise FileNotFoundError(f"{path}: no frames/ folder or video file")
            self._video = videos[0]
        self._times = [float(r["t"]) for r in _read_jsonl(self.clip / "frames.jsonl")]
        self._prefetch = prefetch
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._index = 0
        self.done = False

    def __enter__(self) -> "ReplaySource":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _time_of(self, i: int) -> float:
        return self._times[i] if i < len(self._times) else i / self.fps

    def _iter_frames(self):
        if self._video is None:
            for p in self._frames:
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"{p}: not a readable image")
                yield img
            return
        cap = cv2.VideoCapture(str(self._video))
        if not cap.isOpened():
            raise ValueError(f"{self._video}: cannot open video")
        if not self._times:
            self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    return
                yield img
        finally:
            cap.release()

    def _reader(self) -> None:
        q = self._queue
        assert q is not None
        try:
            for img in self._iter_frames():
                while not self._stop.is_set():
                    try:
                        q.put(img, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
        except Exception as e:
            q.put(e)
            return
        q.put(None)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._queue = queue.Queue(maxsize=self._prefetch)
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True,
                                        name="replay-reader")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def grab(self) -> np.ndarray | None:
        assert self._queue is not None, "call start() first"
        if self.done:
            return None
        item = self._queue.get()
        if item is None:
            self.done = True
            return None
        if isinstance(item, Exception):
            self.done = True
            raise item
        self.clock.advance_to(self._time_of(self._index))
        self._index += 1
        return item


class ReplayTelemetry:
    """Recorded telemetry, answered at the replay clock's time.

    Satisfies the `Telemetry` protocol. A value is the latest sample at or
    before `clock.now()`. `available` is False when that sample is older
    than `max_age_s`, which covers both gaps in the recording and running
    past its last sample."""

    def __init__(self, path: str | Path, clock: ReplayClock,
                 max_age_s: float = TELEMETRY_MAX_AGE_S) -> None:
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / "telemetry.jsonl"
        rows = sorted(_read_jsonl(path), key=lambda r: float(r["t"]))
        self._t = np.asarray([float(r["t"]) for r in rows], dtype=np.float64)
        self._rows = rows
        self.clock = clock
        self.max_age_s = max_age_s

    def _sample(self) -> dict | None:
        now = self.clock.now()
        i = int(np.searchsorted(self._t, now, side="right")) - 1
        if i < 0 or now - self._t[i] > self.max_age_s:
            return None
        return self._rows[i]

    def _value(self, key: str) -> float | None:
        s = self._sample()
        if s is None:
            return None
        v = s.get(key)
        return None if v is None else float(v)

    @property
    def available(self) -> bool:
        return self._sample() is not None

    def speed_mps(self) -> float | None:
        return self._value("speed_mps")

    def game_steer(self) -> float | None:
        return self._value("game_steer")

    def user_steer(self) -> float | None:
        return self._value("user_steer")

    def wheel_angle_rad(self, wheelbase_m: float = 1.0) -> float | None:
        return self._value("wheel_angle_rad")

    def yaw_rate_rad_s(self) -> float | None:
        return self._value("yaw_rate_rad_s")

    def close(self) -> None:
        pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class ClipWriter:
    """Writes the clip layout. Frames wider than `width` are downscaled.
    With `threaded`, JPEG encoding runs on a writer thread so a live
    recording keeps the capture cadence."""

    def __init__(self, out: str | Path, game: str, width: int = RECORD_WIDTH,
                 threaded: bool = False, overwrite: bool = False) -> None:
        self.out = Path(out).expanduser()
        frames = self.out / "frames"
        if frames.is_dir() and any(frames.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.out} already holds a clip")
            shutil.rmtree(frames)
        frames.mkdir(parents=True, exist_ok=True)
        self.game = game
        self.width = width
        self.frames = 0
        self.size: tuple[int, int] | None = None
        self._last_t = 0.0
        self._frames_f = (self.out / "frames.jsonl").open("w", encoding="utf-8")
        self._tel_f = (self.out / "telemetry.jsonl").open("w", encoding="utf-8")
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        if threaded:
            self._q = queue.Queue(maxsize=64)
            self._thread = threading.Thread(target=self._drain, daemon=True,
                                            name="clip-writer")
            self._thread.start()

    def _encode(self, i: int, img: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise ValueError("JPEG encode failed")
        buf.tofile(str(self.out / "frames" / f"{i:06d}.jpg"))

    def _drain(self) -> None:
        # After a failed write (disk full) keep emptying the queue, so the
        # capture thread never blocks forever in put().
        assert self._q is not None
        while (item := self._q.get()) is not None:
            if self._error is None:
                try:
                    self._encode(*item)
                except Exception as e:
                    self._error = e

    def write_frame(self, t: float, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        if w > self.width:
            bgr = cv2.resize(bgr, (self.width, round(h * self.width / w)),
                             interpolation=cv2.INTER_AREA)
        self.size = (bgr.shape[1], bgr.shape[0])
        if self._error is not None:
            raise RuntimeError(f"recording stopped, frame write failed: {self._error}")
        if self._q is not None:
            self._q.put((self.frames, bgr))
        else:
            self._encode(self.frames, bgr)
        self._frames_f.write(json.dumps({"i": self.frames, "t": round(t, 4)}) + "\n")
        self.frames += 1
        self._last_t = t

    def write_telemetry(self, t: float, sample: dict) -> None:
        row = {"t": round(t, 4)}
        row.update({k: (None if sample.get(k) is None else round(float(sample[k]), 6))
                    for k in TELEMETRY_FIELDS})
        self._tel_f.write(json.dumps(row) + "\n")

    def copy_state(self, data_dir: Path) -> None:
        """Snapshot the per-game state so replay starts where the drive did."""
        state = self.out / "state"
        state.mkdir(exist_ok=True)
        for p in [*data_dir.glob("*.json"), *data_dir.glob("*.flag")]:
            shutil.copy2(p, state / p.name)

    def close(self, source: str, models_dir: Path | None = None) -> dict:
        if self._q is not None and self._thread is not None:
            self._q.put(None)
            self._thread.join()
        self._frames_f.close()
        self._tel_f.close()
        duration = self._last_t
        meta = {
            "format": CLIP_FORMAT,
            "source": source,
            "game": self.game,
            "frames": self.frames,
            "duration_s": round(duration, 3),
            "fps": round((self.frames - 1) / duration, 3) if duration > 0 else 20.0,
            "width": self.size[0] if self.size else None,
            "height": self.size[1] if self.size else None,
        }
        if self._error is not None:
            meta["error"] = f"frame write failed: {self._error}"
        if models_dir is not None:
            meta["models"] = {p.name: _sha256(p)
                              for p in sorted(models_dir.glob("*.onnx"))}
        (self.out / "clip.json").write_text(json.dumps(meta, indent=2) + "\n",
                                            encoding="utf-8")
        return meta


def telemetry_sample(tel, wheelbase_m: float) -> dict | None:
    if not tel.available:
        return None
    return {
        "speed_mps": tel.speed_mps(),
        "yaw_rate_rad_s": tel.yaw_rate_rad_s(),
        "wheel_angle_rad": tel.wheel_angle_rad(wheelbase_m),
        "game_steer": tel.game_steer(),
        "user_steer": tel.user_steer(),
    }


def record(out: str | Path, seconds: float, profile: GameProfile) -> int:
    """Record screen frames and game telemetry while the user drives."""
    from simsteer.core.control import ControllerConfig
    from simsteer.paths import bundle_dir, data_dir
    from simsteer.runtime.capture import Capture, CaptureConfig

    wheelbase = ControllerConfig.load(profile.id, seed=profile.controller).wheelbase_m
    writer = ClipWriter(out, profile.id, threaded=True)
    writer.copy_state(data_dir())
    tel = profile.make_telemetry()
    clock = LiveClock()
    print(f"recording {profile.display_name} to {writer.out} for {seconds:.0f} s "
          f"(Ctrl+C stops early); telemetry "
          f"{'available' if tel.available else 'NOT detected yet'}")
    lost = 0
    try:
        with Capture(CaptureConfig(target_fps=20)) as cap:
            t0 = clock.now()
            next_report = 5.0
            while (t := clock.now() - t0) < seconds:
                frame = cap.grab()
                if frame is None:
                    time.sleep(0.002)
                    continue
                t = clock.now() - t0
                writer.write_frame(t, frame)
                sample = telemetry_sample(tel, wheelbase)
                if sample is None:
                    lost += 1
                    if lost % 40 == 0:
                        tel.close()
                        tel = profile.make_telemetry()
                else:
                    writer.write_telemetry(t, sample)
                if t >= next_report:
                    print(f"  {t:5.0f} s  {writer.frames} frames  "
                          f"telemetry lost on {lost}")
                    next_report += 5.0
    except KeyboardInterrupt:
        print("stopped")
    finally:
        tel.close()
        meta = writer.close("record", models_dir=bundle_dir() / "models")
    print(json.dumps(meta, indent=2))
    print(f"frames: {meta['frames']}")
    return 0
