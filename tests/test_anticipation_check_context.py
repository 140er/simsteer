"""Learning the turn-in lead must not revoke a passed steering Check.

Kept free of new imports so it runs, and fails by assertion, on code
without the anticipation learner.
"""

from __future__ import annotations

from simsteer.calib.flow import contexts_match
from simsteer.core.control import ControllerConfig


def context(**controller):
    base = {"steer_authority": 1.0, "curvature_anticipation_s": 0.0}
    return {"camera": {"simple_warp": True}, "controls": {"controller": {**base, **controller}}}


def test_a_learned_lead_does_not_revoke_a_passed_check():
    """While the lead is learned, a Check stays valid as it moves, including
    certificates saved before the lead could be learned."""
    assert getattr(ControllerConfig(), "auto_anticipation", False) is True
    old_certificate = context()  # saved before the learning switch existed
    assert contexts_match(old_certificate, context(curvature_anticipation_s=0.25, auto_anticipation=True))
    assert not contexts_match(old_certificate, context(steer_authority=1.2, auto_anticipation=True))


def test_a_hand_set_lead_stays_certified():
    saved = context(auto_anticipation=False)
    assert contexts_match(saved, context(auto_anticipation=False))
    assert not contexts_match(saved, context(curvature_anticipation_s=0.1, auto_anticipation=False))
