"""The live loop captures the profile's game window, not the whole screen.

Kept free of new imports so it runs, and fails by assertion, on code
without window cropping.
"""

from __future__ import annotations

import pytest


class Captured(Exception):
    pass


def test_live_loop_crops_capture_to_the_profile_game_window(monkeypatch):
    import simsteer.runtime.loop as loop
    from simsteer.app.settings import Settings

    seen = []

    def fake_capture(cfg=None, *a, **kw):
        seen.append(cfg)
        raise Captured

    monkeypatch.setattr(loop, "Capture", fake_capture)
    with pytest.raises(Captured):
        loop.run(Settings(no_gamepad=True, no_tuner=True), "ets2", headless=True)
    (cfg,) = seen
    assert getattr(cfg, "window_processes", ()) == ("eurotrucks2.exe",)
