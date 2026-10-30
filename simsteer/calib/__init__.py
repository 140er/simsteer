"""Calibration status per game and device, for the setup window (3B).

    status("ets2", "gamepad")   -> CalibStatus
    restart("ets2", "gamepad")  -> wipe it; the next drive starts at FOV

Both read and write the saved files only, so they work while the game is
not running. A running session restarts through `CalibFlow.request_restart`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from simsteer.calib.flow import CALIB_VERSION, TITLES, Step, flow_path
from simsteer.paths import state_path


@dataclass(frozen=True)
class CalibStatus:
    game: str
    device: str
    calibrated: bool
    step: str | None
    score: float | None
    date: str | None
    redo_step: str | None
    reason: str | None

    @property
    def label(self) -> str:
        if self.calibrated:
            return f"calibrated, score {self.score:.2f}"
        if self.redo_step:
            return f"redo {self.redo_step}: {self.reason}"
        if self.step in TITLES:
            return f"calibrating: {self.step}"
        return "not calibrated"


def status(game: str, device: str) -> CalibStatus:
    try:
        d = json.loads(flow_path(game, device).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    if d.get("version") != CALIB_VERSION:
        return CalibStatus(game, device, False, None, None, None, None, None)
    step = d.get("step")
    result = d.get("result") or {}
    passed = bool(result.get("passed"))
    calibrated = step == Step.DONE.value and passed
    redo = None if passed else result.get("redo")
    return CalibStatus(
        game=game, device=device, calibrated=calibrated, step=step,
        score=result.get("score"), date=result.get("date"),
        redo_step=redo, reason=result.get("reason") if redo else None)


def restart(game: str, device: str) -> None:
    """Delete the flow state, camera pose history and steering fit."""
    for p in (flow_path(game, device),
              state_path("livecalib_state", game),
              state_path("liveparams", f"{game}_{device}")):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = ["CalibStatus", "status", "restart"]
