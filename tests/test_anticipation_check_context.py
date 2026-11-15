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
    """The learned offset moves freely under a passed Check, including
    certificates saved before it existed."""
    assert getattr(ControllerConfig(), "learned_anticipation_s", None) == 0.0
    old_certificate = context()  # saved before learning existed
    assert contexts_match(old_certificate,
                          context(auto_anticipation=True, learned_anticipation_s=0.25))
    assert not contexts_match(old_certificate, context(steer_authority=1.2))


def test_a_hand_set_lead_stays_certified_while_learning_is_on():
    saved = context(auto_anticipation=True, learned_anticipation_s=0.1)
    assert not contexts_match(saved, context(curvature_anticipation_s=0.1, auto_anticipation=True,
                                             learned_anticipation_s=0.1))
