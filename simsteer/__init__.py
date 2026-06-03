"""SimSteer v2 — drive sim-racing games with comma.ai's openpilot model.

A plug-and-play reimagining of v1: continuous game detection with no-restart
hot-swap, auto-FOV, auto-setup, and one-file game profiles. The proven driving
math (warp, model, control, LiveParams, LiveCalib) is ported faithfully from v1
with the same static-knob discipline — see docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from simsteer.version import __version__

__all__ = ["__version__"]
