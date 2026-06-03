"""In-loop, one-shot auto-FOV (the safe plug-and-play FOV fix).

The model is fed a warp that crops the captured frame to the model's
expected HFOV using `calib.fov_h_deg`. A wrong FOV makes the crop the
wrong width, so the world looks zoomed in/out and the model's perceived
forward speed `pose[0]` diverges from real speed by exactly the zoom
factor:

    vx_model / v_ego = tan(FOV_set / 2) / tan(FOV_true / 2)
    =>  FOV_true = 2 * atan( tan(FOV_set / 2) / ratio )

This is NOT the removed LiveFov (a *continuous* closed loop that fed
itself, because the model's vx is a function of the very FOV it was
tuning, so it could converge to nonsense). `FovResolver` collects the
ratio over one straight-and-fast stretch, solves once, applies, and
**freezes**. A single open-loop solve cannot run away.

The loop feeds it `(calib, vx_model, v_ego, yaw_rate)` each frame; it
self-gates on straight (low yaw) and fast (above a floor) samples, and
returns a one-line message when it commits a correction (so the loop can
log + persist + banner it).
"""
from __future__ import annotations

import math
from statistics import median


class FovResolver:
    def __init__(self, enabled: bool = True, need_samples: int = 60,
                 min_speed_mps: float = 8.0, max_yaw_rad_s: float = 0.06,
                 clamp_deg: tuple[float, float] = (40.0, 150.0)) -> None:
        self.enabled = enabled
        self.need = need_samples
        self.min_speed = min_speed_mps
        self.max_yaw = max_yaw_rad_s
        self.clamp = clamp_deg
        self.done = False
        self._ratios: list[float] = []
        self.last_ratio: float | None = None
        self.applied_fov: float | None = None

    @property
    def progress(self) -> float:
        """0..1 toward a confident solve (for the HUD)."""
        if self.done:
            return 1.0
        return min(1.0, len(self._ratios) / float(self.need))

    @property
    def status(self) -> str:
        if not self.enabled:
            return "off"
        if self.done:
            return f"FOV {self.applied_fov:.0f}deg (auto)"
        return f"measuring {len(self._ratios)}/{self.need}"

    def update(self, calib, vx_model: float | None,
               v_ego: float | None, yaw_rate: float | None) -> str | None:
        """Feed one frame. Returns a message string when it applies a
        correction (once), else None. Mutates `calib.fov_h_deg` on solve;
        the caller persists + banners it."""
        if self.done or not self.enabled:
            return None
        if v_ego is None or yaw_rate is None or vx_model is None:
            return None
        if v_ego < self.min_speed or abs(yaw_rate) > self.max_yaw:
            return None
        if vx_model <= 0.1:
            return None
        ratio = vx_model / v_ego
        self._ratios.append(ratio)
        self.last_ratio = median(self._ratios)
        if len(self._ratios) < self.need:
            return None
        # Enough straight+fast samples — solve once and freeze.
        r = float(median(self._ratios))
        fov_old = float(calib.fov_h_deg)
        fov_new = math.degrees(2.0 * math.atan(
            math.tan(math.radians(fov_old) / 2.0) / r))
        fov_new = max(self.clamp[0], min(self.clamp[1], fov_new))
        calib.fov_h_deg = fov_new
        self.applied_fov = fov_new
        self.done = True
        return (f"auto-FOV: {fov_old:.0f} -> {fov_new:.0f}deg "
                f"(vx/v_ego={r:.2f})")
