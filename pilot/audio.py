"""Tiny audio helper for engagement chimes and the wizard-complete sound.

Files live in `bundle_dir()/assets/<name>.wav`. They're NOT shipped in v1 —
the user can drop in their own later. This module no-ops gracefully when a
file is missing, so callers never need to check.

Use short (under ~1 s), low-amplitude tones. Loud or long sounds while
driving are dangerous.
"""

from __future__ import annotations

import sys

from pilot.paths import asset_path


def play(name: str) -> None:
    """Play `<bundle>/assets/<name>.wav` asynchronously. Silent no-op if
    the file is missing, if winsound is unavailable, or on play error."""
    p = asset_path(f"{name}.wav")
    if not p.exists():
        return
    if sys.platform != "win32":
        return
    try:
        import winsound
        winsound.PlaySound(str(p),
                           winsound.SND_FILENAME
                           | winsound.SND_ASYNC
                           | winsound.SND_NODEFAULT)
    except Exception:
        # Truly best-effort; never let an audio failure interrupt driving.
        pass
