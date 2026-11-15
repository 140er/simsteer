"""Capture crops to the game window instead of the whole primary screen."""

from __future__ import annotations

import time
from queue import Queue

import numpy as np

from simsteer.runtime.capture import Capture, CaptureConfig
from simsteer.runtime.window import MIN_HEIGHT, MIN_WIDTH, WindowRect, crop_to_window


def screen(w=2560, h=1440):
    # A distinct value per column and row, so a crop's origin is checkable.
    cols = np.arange(w, dtype=np.uint32) % 251
    rows = np.arange(h, dtype=np.uint32)[:, None] % 241
    return np.dstack([np.broadcast_to(cols, (h, w)), np.broadcast_to(rows, (h, w)),
                      np.zeros((h, w))]).astype(np.uint8)


def started(capture: Capture, frame) -> Capture:
    # Skip dxcam: feed one fresh frame through the reader queue.
    capture._cam = object()
    capture._frames = Queue(maxsize=1)
    capture._frames.put_nowait((time.monotonic(), frame))
    return capture


def test_window_crop_matches_the_client_area():
    frame = screen()
    rect = WindowRect(320, 180, 1920, 1080, on_primary=True)
    out = crop_to_window(frame, rect)
    assert out.shape[:2] == (900, 1600)
    np.testing.assert_array_equal(out[0, 0], frame[180, 320])
    np.testing.assert_array_equal(out[-1, -1], frame[1079, 1919])


def test_no_window_other_monitor_or_tiny_window_keep_the_full_screen():
    frame = screen()
    assert crop_to_window(frame, None) is frame
    assert crop_to_window(frame, WindowRect(0, 0, 1600, 900, on_primary=False)) is frame
    tiny = WindowRect(10, 10, 10 + MIN_WIDTH - 1, 10 + MIN_HEIGHT, on_primary=True)
    assert crop_to_window(frame, tiny) is frame


def test_window_partly_off_screen_is_clamped():
    out = crop_to_window(screen(), WindowRect(-40, -30, 1240, 690, on_primary=True))
    assert out.shape[:2] == (690, 1240)


def test_grab_crops_to_the_game_window_and_reports_it_once(capsys):
    calls = []

    def finder(names):
        calls.append(names)
        return WindowRect(320, 180, 1920, 1080, on_primary=True)

    capture = started(Capture(CaptureConfig(window_processes=("eurotrucks2.exe",)),
                              find_window=finder), screen())
    frame = capture.grab()
    assert frame.shape[:2] == (900, 1600)
    assert calls == [("eurotrucks2.exe",)]
    assert capsys.readouterr().out.strip() == "capture: game window 1600x900 at (320,180)"


def test_grab_without_window_processes_keeps_full_screen():
    capture = started(Capture(CaptureConfig(), find_window=lambda names: 1 / 0), screen())
    assert capture.grab().shape[:2] == (1440, 2560)


def test_missing_window_falls_back_to_full_screen_with_a_notice(capsys):
    capture = started(Capture(CaptureConfig(window_processes=("eurotrucks2.exe",)),
                              find_window=lambda names: None), screen())
    assert capture.grab().shape[:2] == (1440, 2560)
    assert "game window not found" in capsys.readouterr().out
