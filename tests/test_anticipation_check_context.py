"""Learning the turn-in lead must not revoke a passed steering Check.

Kept free of new imports so it runs, and fails by assertion, on code
without the anticipation learner.
"""

from __future__ import annotations

from simsteer.calib.flow import contexts_match
from simsteer.core.control import ControllerConfig


def test_a_learned_lead_does_not_revoke_a_passed_check():
    """A Check certified with one lead stays valid as the lead is learned,
    including certificates saved before the lead was exempt."""
    def context(**controller):
        base = {"steer_authority": 1.0, "curvature_anticipation_s": 0.0}
        return {"camera": {"simple_warp": True}, "controls": {"controller": {**base, **controller}}}

    assert getattr(ControllerConfig(), "auto_anticipation", False) is True
    assert contexts_match(context(), context(curvature_anticipation_s=0.25))
    assert not contexts_match(context(), context(steer_authority=1.2))
