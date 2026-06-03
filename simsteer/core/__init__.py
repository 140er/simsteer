"""Game-agnostic driving math: warp, model, postprocess, control, learners.

Ported faithfully from v1 `pilot/`. Keeps the static-knob discipline — only
the steering-rack fit (LiveParams) and the camera pose (LiveCalib) learn
online; FOV / lookahead / authority / anticipation are static. See
docs/ARCHITECTURE.md "removed auto-tuners (and why)".
"""
