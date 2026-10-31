"""Sign-changing controls must leave engaged device commands intact."""

import importlib
import math
from types import SimpleNamespace

import pytest

from simsteer.app.global_keys import KEY_INSERT
from simsteer.core.calibration import Calibration
from test_output_safety import drive


@pytest.mark.parametrize("key,initial_sign", [(ord("m"), -1.), (ord("0"), 1.)])
def test_engaged_sign_controls_preserve_sign_and_engagement(monkeypatch, key, initial_sign):
    def observe(st, manager):
        if st.i == 1 and initial_sign == 1.:
            st.flow.calib.lateral_sign = initial_sign
            st.flow._check_context = st.flow._context()
            st.flow.result["context"] = st.flow._check_context
        if st.i == 2:
            st.before = len(manager._device.raw.sent)
            st.start_sign = st.flow.steering_sign
        if st.i == 4:
            st.end = len(manager._device.raw.sent)
            st.still_engaged = manager.engaged

    st, pad, _ = drive(monkeypatch, mode="dev", frames=5,
                       keys_at={2: [key]}, on_grab=observe)
    before = [r[0] for r in pad.raw.sent[:st.before] if abs(r[0]) > 1e-6]
    window = pad.raw.sent[st.before:st.end]
    assert before and window
    assert st.start_sign == initial_sign
    zeros = sum(abs(r[0]) <= 1e-6 for r in window)
    assert (st.flow.calib.lateral_sign, st.still_engaged, zeros) == (initial_sign, True, 0)
    assert all(c[0] and abs(c[1][0]) > 1e-6 for c in st.checkpoints[1:5])
    assert all(math.copysign(1., r[0]) == math.copysign(1., before[-1]) for r in window)


@pytest.mark.parametrize("module", ["simsteer.ui.tuner", "pilot.tuner"])
@pytest.mark.parametrize("engaged", [False, True])
def test_tuner_mirror_respects_engagement(monkeypatch, module, engaged):
    tuner_module = importlib.import_module(module)
    options = {"start_thread": False}
    if module == "pilot.tuner":
        monkeypatch.setattr(tuner_module, "threading", SimpleNamespace(
            Thread=lambda **kw: SimpleNamespace(start=lambda: None)))
        options = {}
    calib = Calibration(lateral_sign=-1.)
    tuner = tuner_module.Tuner(calib, SimpleNamespace(), SimpleNamespace(),
                               device=SimpleNamespace(engaged=engaged), **options)
    tuner._set_mirror(False)
    assert calib.lateral_sign == (-1. if engaged else 1.)


def test_legacy_mirror_switch_resyncs_rejected_click(monkeypatch):
    from pilot import tuner as tuner_module

    monkeypatch.setattr(tuner_module, "threading", SimpleNamespace(
        Thread=lambda **kw: SimpleNamespace(start=lambda: None)))
    calib = Calibration(lateral_sign=-1.)
    tuner = tuner_module.Tuner(calib, SimpleNamespace(), SimpleNamespace(),
                               device=SimpleNamespace(engaged=True))
    switch = SimpleNamespace(selected=False)
    switch.select = lambda: setattr(switch, "selected", True)
    switch.deselect = lambda: setattr(switch, "selected", False)
    tuner._switches["mirror_lateral_sign"] = switch
    tuner._set_mirror(False)
    assert switch.selected
    assert calib.lateral_sign == -1.


def test_disabled_mirror_hotkey_is_ignored(monkeypatch):
    def disable(st, manager):
        st.settings.disabled_hotkeys = ["mirror_sign"]

    st, _, _ = drive(monkeypatch, mode="dev", force=False,
                     keys_at={2: [ord("m")]}, on_grab=disable)
    assert not st.checkpoints[2][0] and not st.checkpoints[3][0]
    assert st.flow.calib.lateral_sign == -1.


def test_disengaged_mirror_still_flips(monkeypatch):
    st, _, _ = drive(monkeypatch, mode="dev", force=False, keys_at={2: [ord("m")]})
    assert not st.checkpoints[2][0]
    assert st.flow.calib.lateral_sign == 1.


def test_insert_then_mirror_reads_current_engagement(monkeypatch):
    st, _, _ = drive(monkeypatch, mode="dev", force=False,
                     keys_at={2: [KEY_INSERT, ord("m")]})
    assert st.checkpoints[3][0]
    assert st.flow.calib.lateral_sign == -1.


def test_mirror_then_insert_while_engaged_hands_back_first(monkeypatch):
    st, _, _ = drive(monkeypatch, mode="dev", keys_at={2: [ord("m"), KEY_INSERT]})
    assert st.checkpoints[2][0]
    assert st.checkpoints[3] == (False, (0., 0., 0.))
    assert st.flow.calib.lateral_sign == 1.


def test_disengaged_mirror_then_insert_starts_fresh_check(monkeypatch):
    from simsteer.calib.flow import Step

    st, _, _ = drive(monkeypatch, mode="dev", force=False,
                     keys_at={2: [ord("m"), KEY_INSERT]})
    assert st.flow.calib.lateral_sign == 1.
    assert st.flow.step is Step.CHECK
    assert st.checkpoints[3][0]
