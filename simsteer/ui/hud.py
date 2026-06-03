"""On-overlay heads-up display.

Two modes:

  USER (default) — clean panels with what the driver actually needs:
    big engagement chip top-center, warning chips top-left, wizard banner
    above the chip (when active), speed dial bottom-left, calibration
    progress dashboard bottom-right, fps/version footer. Nothing else.

  DEV — the old stack-of-text-lines dump for debugging. Toggled with H.

The renderer takes a `HudState` snapshot built each frame from whatever
the main loop has computed — it does NOT reach back into LiveCalib /
LiveParams / etc. itself. Keeps the rendering pure / easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np


Color = tuple[int, int, int]    # BGR

# Palette. Keep the count small and intentional — the overlay reads
# best when only a couple of accent colors compete for attention.
COL_BG_DARK = (28, 30, 36)            # chip background, dark slate
COL_FG_LIGHT = (240, 240, 245)        # default light text
COL_FG_MUTED = (170, 175, 185)        # secondary text
COL_ACCENT_GREEN = (96, 220, 110)     # engaged, ready, healthy
COL_ACCENT_YELLOW = (95, 220, 240)    # warming, warnings
COL_ACCENT_RED = (95, 95, 245)        # blocked, fatal, emergency
COL_ACCENT_BLUE = (245, 195, 95)      # disengaged, info
COL_ACCENT_VIOLET = (210, 130, 230)   # wizard


Mode = Literal["user", "dev"]


@dataclass
class HudState:
    """Per-frame snapshot the renderer consumes. None-able fields are
    drawn only when the value is meaningful."""

    # ----- engagement -----
    engaged: bool = False
    engaged_label: str = "DISENGAGED"   # what the chip text says
    engaged_color: Color = COL_ACCENT_BLUE
    # Transient banner shown above the chip for a few seconds — engage
    # blip, cannot-engage reason, auto-disengage reason, etc. Empty
    # string = nothing.
    banner_text: str = ""
    banner_color: Color = COL_FG_MUTED

    # ----- wizard -----
    wizard_active: bool = False
    wizard_text: str = ""               # main wizard banner
    wizard_color: Color = COL_ACCENT_VIOLET
    wizard_progress: float = 0.0        # 0..100
    wizard_hint: str = ""               # one-line guidance below

    # ----- warnings -----
    warnings: list[str] = field(default_factory=list)

    # ----- speed -----
    v_ego_mps: float | None = None      # for the bottom-left dial

    # ----- calibration progress dashboard -----
    cal_camera_frac: float | None = None       # 0..1 or None to hide
    cal_camera_status: str = ""                # "WARMING" / "READY" / "RESET" / ""
    cal_steering_frac: float | None = None
    cal_steering_status: str = ""
    cal_fov_frac: float | None = None
    cal_fov_status: str = ""

    # ----- chips along the top -----
    probe_active: bool = False          # show "PROBE active" chip
    manual_override: bool = False       # show "MANUAL" chip
    lane_change: str | None = None      # "L"/"R" if active, else None
    lead_following: bool = False        # "ACC" chip
    aeb: bool = False                   # red "AEB!" chip

    # ----- footer -----
    fps: float = 0.0
    version: str = ""
    mode_hint: str = "H: dev view"      # tip in the corner

    # ----- DEV-mode payload: passthrough list -----
    dev_lines: list[tuple[str, Color]] = field(default_factory=list)


class HudRenderer:
    """Stateless drawing surface. Call `draw(overlay, state, mode)` per
    frame. The renderer creates a transparent layer for the panels so
    they don't smear the underlying lane visualization."""

    # ----- public -----

    def draw(self, img: np.ndarray, state: HudState, mode: Mode = "user") -> None:
        if mode == "dev":
            self._draw_dev(img, state.dev_lines)
            return
        self._draw_user(img, state)

    # ----- user mode -----

    def _draw_user(self, img: np.ndarray, s: HudState) -> None:
        h, w = img.shape[:2]
        margin = 16

        # Top center: wizard banner (when active) sits above the engage
        # chip — wizard is the primary attention-getter during first
        # drive.
        cursor_y = margin
        if s.wizard_active and s.wizard_text:
            cursor_y = self._wizard_panel(img, w, cursor_y, s)
            cursor_y += 8

        # The engage chip — fixed width, color-coded.
        self._engage_chip(img, w, cursor_y, s)

        # Top-left: warning chips, stacked.
        if s.warnings:
            wy = margin
            for warn in s.warnings:
                wy = self._warning_chip(img, margin, wy, warn) + 6

        # Mode chips along the very top, after warnings — pile right of
        # the centered engage chip.
        cy = margin + 8
        cx = w - margin
        for chip in self._top_right_chips(s):
            cx = self._right_chip(img, cx, cy, chip[0], chip[1])
            cx -= 6

        # Optional transient banner — sits BELOW the engage chip when
        # the cv2 window is tall enough to spare a slot. Otherwise we
        # overlay it on the engage chip (red text on red bg looks bad,
        # so use a separate row).
        if s.banner_text:
            self._transient_banner(img, w, cursor_y + 70, s)

        # Bottom-left: speed dial.
        if s.v_ego_mps is not None:
            self._speed_panel(img, margin, h - margin, s.v_ego_mps)

        # Bottom-right: calibration progress dashboard. Hidden when
        # everything is ready / no progress to show.
        if (s.cal_camera_frac is not None
                or s.cal_steering_frac is not None
                or s.cal_fov_frac is not None):
            self._calibration_panel(img, w - margin, h - margin, s)

        # Footer: fps + version + mode hint in muted text. Stick to
        # ASCII separators — cv2's Hershey font has no glyph for U+00B7
        # and substitutes "??" which looks unprofessional.
        footer = f"{s.fps:.0f} fps"
        if s.version:
            footer += f"   v{s.version}"
        if s.mode_hint:
            footer += f"   |   {s.mode_hint}"
        self._footer_text(img, w // 2, h - 8, footer)

    # ----- dev mode -----

    def _draw_dev(self, img: np.ndarray, lines: list[tuple[str, Color]]) -> None:
        y = 30
        for text, color in lines:
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 1, cv2.LINE_AA)
            y += 28

    # ----- chip primitives -----

    def _fill_rounded_rect(self, img: np.ndarray, x: int, y: int,
                            w: int, h: int, color: Color,
                            alpha: float = 0.78) -> None:
        """Filled rounded rect with alpha. Cheap radius approximation."""
        # Build a same-shape overlay, paint, then alpha-blend.
        overlay = img.copy()
        radius = 8
        cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h),
                      color, -1, cv2.LINE_AA)
        cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius),
                      color, -1, cv2.LINE_AA)
        for cx, cy in [(x + radius, y + radius), (x + w - radius, y + radius),
                        (x + radius, y + h - radius),
                        (x + w - radius, y + h - radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    def _text_size(self, text: str, font_scale: float, thickness: int) -> tuple[int, int]:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        return tw, th

    def _draw_text(self, img: np.ndarray, text: str, x: int, y: int,
                    color: Color, scale: float = 0.7, thickness: int = 1,
                    outline: bool = False) -> None:
        if outline:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)

    # ----- panel drawers -----

    def _engage_chip(self, img: np.ndarray, view_w: int, y: int,
                      s: HudState) -> None:
        # Center the chip horizontally. Engaged uses a bigger badge so
        # it dominates; disengaged is the same shape, just blue.
        text = s.engaged_label
        scale = 1.1
        thickness = 2
        tw, th = self._text_size(text, scale, thickness)
        pad_x, pad_y = 20, 12
        chip_w = tw + pad_x * 2
        chip_h = th + pad_y * 2
        x = (view_w - chip_w) // 2
        # Background is dark slate; the accent shows as a 4-px stripe
        # along the bottom edge.
        self._fill_rounded_rect(img, x, y, chip_w, chip_h, COL_BG_DARK,
                                  alpha=0.82)
        cv2.rectangle(img, (x, y + chip_h - 4),
                       (x + chip_w, y + chip_h),
                       s.engaged_color, -1, cv2.LINE_AA)
        self._draw_text(img, text, x + pad_x, y + pad_y + th - 2,
                        s.engaged_color, scale=scale, thickness=thickness)

    def _wizard_panel(self, img: np.ndarray, view_w: int, y: int,
                       s: HudState) -> int:
        """Wizard banner + progress bar. Returns y of the bottom edge."""
        title = s.wizard_text
        title_scale = 0.8
        tw_title, th_title = self._text_size(title, title_scale, 2)
        bar_w = max(360, tw_title + 80)
        bar_h = 12
        hint = s.wizard_hint or ""
        hint_h = 22 if hint else 0
        panel_w = bar_w + 32
        panel_h = th_title + bar_h + hint_h + 28
        x = (view_w - panel_w) // 2

        self._fill_rounded_rect(img, x, y, panel_w, panel_h, COL_BG_DARK,
                                  alpha=0.82)
        # Title text, centered.
        tx = x + (panel_w - tw_title) // 2
        ty = y + 8 + th_title
        self._draw_text(img, title, tx, ty, s.wizard_color,
                        scale=title_scale, thickness=2)
        # Progress bar.
        bar_x = x + 16
        bar_y = ty + 8
        frac = max(0.0, min(1.0, s.wizard_progress / 100.0))
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                       (60, 60, 70), -1, cv2.LINE_AA)
        if frac > 0:
            cv2.rectangle(img, (bar_x, bar_y),
                           (bar_x + int(bar_w * frac), bar_y + bar_h),
                           s.wizard_color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                       COL_FG_MUTED, 1, cv2.LINE_AA)
        # Hint line.
        if hint:
            hint_scale = 0.55
            tw_hint, th_hint = self._text_size(hint, hint_scale, 1)
            self._draw_text(img, hint,
                             x + (panel_w - tw_hint) // 2,
                             bar_y + bar_h + th_hint + 6,
                             COL_FG_MUTED, scale=hint_scale)
        return y + panel_h

    def _warning_chip(self, img: np.ndarray, x: int, y: int,
                       text: str) -> int:
        """Compact warning chip on the top-left. Returns y of bottom."""
        scale = 0.55
        thickness = 1
        prefix = "! "
        full = prefix + text
        tw, th = self._text_size(full, scale, thickness)
        pad_x, pad_y = 10, 6
        chip_w = tw + pad_x * 2
        chip_h = th + pad_y * 2
        self._fill_rounded_rect(img, x, y, chip_w, chip_h, COL_BG_DARK,
                                  alpha=0.82)
        cv2.rectangle(img, (x, y), (x + 3, y + chip_h),
                       COL_ACCENT_YELLOW, -1, cv2.LINE_AA)
        self._draw_text(img, full, x + pad_x, y + pad_y + th - 2,
                        COL_ACCENT_YELLOW, scale=scale, thickness=thickness)
        return y + chip_h

    def _right_chip(self, img: np.ndarray, x_right: int, y: int,
                     text: str, color: Color) -> int:
        """Chip aligned to a right edge. Returns the chip's left x."""
        scale = 0.55
        thickness = 1
        tw, th = self._text_size(text, scale, thickness)
        pad_x, pad_y = 10, 6
        chip_w = tw + pad_x * 2
        chip_h = th + pad_y * 2
        x = x_right - chip_w
        self._fill_rounded_rect(img, x, y, chip_w, chip_h, COL_BG_DARK,
                                  alpha=0.82)
        cv2.rectangle(img, (x + chip_w - 3, y),
                       (x + chip_w, y + chip_h),
                       color, -1, cv2.LINE_AA)
        self._draw_text(img, text, x + pad_x, y + pad_y + th - 2,
                        color, scale=scale, thickness=thickness)
        return x

    def _top_right_chips(self, s: HudState) -> list[tuple[str, Color]]:
        out: list[tuple[str, Color]] = []
        if s.aeb:
            out.append(("AEB!", COL_ACCENT_RED))
        if s.lead_following:
            out.append(("ACC", COL_ACCENT_BLUE))
        if s.lane_change == "L":
            out.append(("← LANE-L", COL_ACCENT_YELLOW))
        elif s.lane_change == "R":
            out.append(("LANE-R →", COL_ACCENT_YELLOW))
        if s.manual_override:
            out.append(("MANUAL", COL_ACCENT_BLUE))
        if s.probe_active:
            out.append(("PROBE", COL_ACCENT_VIOLET))
        return out

    def _transient_banner(self, img: np.ndarray, view_w: int, y: int,
                           s: HudState) -> None:
        text = s.banner_text
        scale = 0.7
        thickness = 1
        tw, th = self._text_size(text, scale, thickness)
        pad_x, pad_y = 14, 8
        chip_w = tw + pad_x * 2
        chip_h = th + pad_y * 2
        x = (view_w - chip_w) // 2
        self._fill_rounded_rect(img, x, y, chip_w, chip_h, COL_BG_DARK,
                                  alpha=0.85)
        cv2.rectangle(img, (x, y + chip_h - 3),
                       (x + chip_w, y + chip_h),
                       s.banner_color, -1, cv2.LINE_AA)
        self._draw_text(img, text, x + pad_x, y + pad_y + th - 2,
                        s.banner_color, scale=scale, thickness=thickness)

    def _speed_panel(self, img: np.ndarray, x: int, y_bottom: int,
                      v_mps: float) -> None:
        """Big km/h number bottom-left."""
        kph = v_mps * 3.6
        big = f"{kph:.0f}"
        unit = "km/h"
        big_scale = 1.8
        big_thick = 3
        unit_scale = 0.55
        unit_thick = 1
        bw, bh = self._text_size(big, big_scale, big_thick)
        uw, uh = self._text_size(unit, unit_scale, unit_thick)
        pad_x, pad_y = 16, 10
        panel_w = max(bw, uw) + pad_x * 2
        panel_h = bh + uh + pad_y * 2 + 6
        py = y_bottom - panel_h
        self._fill_rounded_rect(img, x, py, panel_w, panel_h, COL_BG_DARK,
                                  alpha=0.82)
        # Big number, top-aligned in the panel.
        self._draw_text(img, big, x + pad_x, py + pad_y + bh,
                        COL_FG_LIGHT, scale=big_scale, thickness=big_thick)
        # Unit, dimmer, below.
        self._draw_text(img, unit, x + pad_x, py + pad_y + bh + 6 + uh,
                        COL_FG_MUTED, scale=unit_scale, thickness=unit_thick)

    def _calibration_panel(self, img: np.ndarray, x_right: int,
                            y_bottom: int, s: HudState) -> None:
        """Bottom-right dashboard with up to 3 small progress bars."""
        rows: list[tuple[str, float, str]] = []
        if s.cal_camera_frac is not None:
            rows.append(("Camera", s.cal_camera_frac, s.cal_camera_status))
        if s.cal_steering_frac is not None:
            rows.append(("Steering", s.cal_steering_frac, s.cal_steering_status))
        if s.cal_fov_frac is not None:
            rows.append(("FOV", s.cal_fov_frac, s.cal_fov_status))
        if not rows:
            return
        scale = 0.55
        thickness = 1
        label_max = max(self._text_size(name, scale, thickness)[0]
                        for name, _, _ in rows)
        bar_w = 160
        bar_h = 10
        row_h = 22
        pad_x, pad_y = 14, 12
        panel_w = label_max + 12 + bar_w + 12 + 50 + pad_x * 2
        panel_h = row_h * len(rows) + pad_y * 2 - 4
        x = x_right - panel_w
        y = y_bottom - panel_h
        self._fill_rounded_rect(img, x, y, panel_w, panel_h, COL_BG_DARK,
                                  alpha=0.82)
        rx = x + pad_x
        ry = y + pad_y
        for name, frac, status in rows:
            frac_c = max(0.0, min(1.0, frac))
            # Label.
            self._draw_text(img, name, rx,
                              ry + bar_h + 1,
                              COL_FG_LIGHT, scale=scale, thickness=thickness)
            # Bar.
            bx = rx + label_max + 12
            cv2.rectangle(img, (bx, ry), (bx + bar_w, ry + bar_h),
                           (60, 60, 70), -1, cv2.LINE_AA)
            color = (COL_ACCENT_GREEN if frac_c >= 1.0 - 1e-3
                     else COL_ACCENT_YELLOW)
            if status == "RESET":
                color = COL_ACCENT_RED
            if frac_c > 0:
                cv2.rectangle(img, (bx, ry),
                               (bx + int(bar_w * frac_c), ry + bar_h),
                               color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (bx, ry), (bx + bar_w, ry + bar_h),
                           COL_FG_MUTED, 1, cv2.LINE_AA)
            # Pct value, monospace alignment.
            pct = f"{frac_c * 100:3.0f}%"
            self._draw_text(img, pct, bx + bar_w + 10,
                              ry + bar_h + 1,
                              COL_FG_LIGHT, scale=scale, thickness=thickness)
            ry += row_h

    def _footer_text(self, img: np.ndarray, x_center: int, y_bottom: int,
                       text: str) -> None:
        scale = 0.45
        thickness = 1
        tw, th = self._text_size(text, scale, thickness)
        # No background — pure text with outline for legibility.
        self._draw_text(img, text, x_center - tw // 2, y_bottom,
                        COL_FG_MUTED, scale=scale, thickness=thickness,
                        outline=True)
