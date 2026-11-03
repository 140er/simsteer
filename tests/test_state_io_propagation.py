"""Dedicated synchronization failures must escape existing recovery handlers.

This file intentionally uses only APIs present before the fix, so the exact
same selection can be exercised against an exported older source tree.
"""
from types import SimpleNamespace

import pytest

from simsteer import state
from simsteer.core import calibration
from simsteer.core.learners import livecalib
from simsteer.calib import flow
from simsteer.ui import tuner
from simsteer.core.learners.liveparams import LiveParams


class _InjectedStateIOUnavailable(OSError):
    pass


@pytest.mark.parametrize("case", ["learner-save", "learner-load", "learner-reset",
                                  "calibration-load", "flow-load", "tuner-persist"])
def test_unavailable_escapes_recovery(case, monkeypatch, tmp_path):
    error_type = getattr(state, "StateIOUnavailable", _InjectedStateIOUnavailable)
    error = error_type("unavailable test worker")
    for module in (calibration, livecalib, flow, tuner):
        monkeypatch.setattr(module, "StateIOUnavailable", error_type, raising=False)

    def fail(*args, **kwargs):
        raise error

    lc = livecalib.LiveCalib()
    lc._game = "ets2"
    path = tmp_path / "state.json"
    path.write_text("{}")
    if case == "learner-save":
        monkeypatch.setattr(livecalib, "atomic_save_json", fail)
        action = lambda: lc.save_state("ets2")
    elif case == "learner-load":
        monkeypatch.setattr(livecalib, "read_json_object", fail)
        action = lambda: lc.load_state("ets2", path=path)
    elif case == "learner-reset":
        monkeypatch.setattr(livecalib, "delete_state", fail)
        action = lc.reset
    elif case == "calibration-load":
        monkeypatch.setattr(calibration, "read_json_object", fail)
        action = lambda: calibration.Calibration.load(path=path)
    elif case == "flow-load":
        current = flow.CalibFlow("ets2", "gamepad", calib=calibration.Calibration(),
                                 live_calib=lc, live_params=LiveParams(), persist=False)
        current._persist = True
        monkeypatch.setattr(flow, "read_json_object", fail)
        action = current._load
    else:
        ui = object.__new__(tuner.Tuner)
        target = SimpleNamespace(save=fail)
        ui.manual = ui.calib = ui.ctrl_cfg = None
        ui._saved = lambda *args: None
        action = lambda: ui._persist(target)
    with pytest.raises(error_type) as caught:
        action()
    assert caught.value is error
