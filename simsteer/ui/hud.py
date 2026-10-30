"""The status bar shared by the normal and developer HUDs.

The loop supplies a snapshot; drawing never reads the controller or the
calibration engine. The road and developer column live in overlay.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from simsteer.ui.state import CalibrationView, status_label
from simsteer.ui.text import draw_text, text_width

Color = tuple[int, int, int]  # BGR
COL_BG_DARK = (38, 37, 35)
COL_FG_LIGHT = (242, 242, 242)
COL_FG_MUTED = (164, 164, 164)
COL_ACCENT_GREEN = (113, 224, 99)
COL_ACCENT_YELLOW = (95, 220, 240)
COL_ACCENT_RED = (85, 85, 255)
COL_ACCENT_BLUE = (221, 190, 64)
COL_ACCENT_VIOLET = COL_ACCENT_BLUE
Mode = Literal["user", "dev"]
BAR_H = 42
CALIBRATION_H = 42
READOUT_H = 26


@dataclass
class HudState:
    engaged: bool = False
    engaged_label: str = "Disengaged"
    engaged_color: Color = COL_FG_LIGHT
    banner_text: str = ""
    banner_color: Color = COL_FG_MUTED
    calibration: CalibrationView | None = None
    warnings: list[str] = field(default_factory=list)
    v_ego_mps: float | None = None
    max_speed_mps: float = 25.0
    acc_enabled: bool = True
    probe_active: bool = False
    manual_override: bool = False
    lane_change: str | None = None
    lead_following: bool = False
    aeb: bool = False
    fps: float = 0.0
    version: str = ""
    lookahead_s: float = 0.0
    fit_a: float = 0.0
    fit_b: float = 0.0
    fit_c: float = 0.0
    fit_samples: int = 0
    fit_locked: bool = False
    telemetry_status: str = "—"
    dev_lines: list[tuple[str, Color]] = field(default_factory=list)


def bar_height(state: HudState, mode: str = "user") -> int:
    return BAR_H + (CALIBRATION_H if state.calibration else 0) + (READOUT_H if mode == "dev" else 0)


class HudRenderer:
    def draw(self, img: np.ndarray, state: HudState, mode: Mode = "user") -> None:
        h, w = img.shape[:2]
        height = min(h, bar_height(state, mode))
        img[:height] = COL_BG_DARK
        if w < 260 or h < 42:
            return
        s = state
        cal = s.calibration
        label = status_label(s)
        color = s.engaged_color
        if cal and label in ("Calibrating", cal.step_name):
            color = {"passed": COL_ACCENT_GREEN, "failed": COL_ACCENT_RED}.get(
                cal.tone, COL_ACCENT_BLUE
            )
        if label in ("No gamepad", "No telemetry", "Low fps"):
            color = COL_ACCENT_RED
        draw_text(img, label, 16, 10, color, 16, True, max_width=max(80, w // 4))
        left = 16 + min(text_width(label, 16, True), max(80, w // 4)) + 24

        speed = "—" if s.v_ego_mps is None else f"{s.v_ego_mps * 3.6:.0f}"
        suffix = f"km/h   Max {s.max_speed_mps * 3.6:.0f}"
        suffix_w = text_width(suffix, 12)
        speed_w = text_width(speed, 24, True)
        right = w - 18 - suffix_w - speed_w - 8
        draw_text(img, speed, right, 4, COL_FG_LIGHT, 24, True)
        draw_text(img, suffix, right + speed_w + 8, 14, COL_FG_MUTED, 12)
        middle_end = right - 20

        marks = self._assist_marks(s, mode)
        message = "  ·  ".join([m for m in (s.banner_text, *s.warnings) if m])
        if cal and cal.instruction:
            message = "  ·  ".join(m for m in (cal.instruction, message) if m)
        mark_width = sum(text_width(text, 12, True) + 20 for text, _ in marks)
        available = max(0, middle_end - left)
        # Centre the marks when there is no message, as in the design board.
        x = left if message else left + max(0, (available - mark_width) // 2)
        for text, mark_color in marks:
            tw = text_width(text, 12, True)
            if x + tw + 16 > middle_end:
                break
            cv2.rectangle(img, (x, 10), (x + tw + 14, 31), (51, 50, 48), -1)
            draw_text(img, text, x + 6, 12, mark_color, 12, True)
            x += tw + 22
        if message:
            message_color = (
                s.banner_color
                if s.banner_text
                else (COL_ACCENT_YELLOW if s.warnings else COL_FG_LIGHT)
            )
            draw_text(img, message, x, 12, message_color, 14, max_width=max(0, middle_end - x))
        if cal:
            self._calibration_line(img, cal)
        if mode == "dev":
            y = BAR_H + (CALIBRATION_H if cal else 0)
            readout = (
                f"{s.fps:.0f} fps    Look {s.lookahead_s:.2f} s    "
                f"a {s.fit_a:+.2f}  b {s.fit_b:+.4f}  c {s.fit_c:+.3f}    "
                f"n {s.fit_samples:,}"
                f"{'  LOCKED' if s.fit_locked else ''}    v{s.version}"
            )
            x = max(16, (w - text_width(readout, 12)) // 2)
            draw_text(img, readout, x, y + 3, COL_FG_MUTED, 12, max_width=w - x - 16)

    @staticmethod
    def _assist_marks(s: HudState, mode: str) -> list[tuple[str, Color]]:
        out = []
        if s.aeb:
            out.append(("AEB", COL_ACCENT_RED))
        if s.manual_override:
            out.append(("MANUAL", COL_ACCENT_BLUE))
        if s.lane_change:
            out.append(("← Lane" if s.lane_change == "L" else "Lane →", COL_ACCENT_YELLOW))
        out.append(
            (
                "ACC On" if s.acc_enabled else "ACC Off",
                COL_ACCENT_GREEN if s.acc_enabled and s.lead_following else COL_FG_MUTED,
            )
        )
        if s.probe_active and mode == "dev":
            out.append(("PROBE", COL_ACCENT_BLUE))
        return out

    @staticmethod
    def _calibration_line(img, cal: CalibrationView) -> None:
        w = img.shape[1]
        track_w = min(480, max(100, w // 2 - 24))
        count = max(1, len(cal.steps))
        gap = 6
        segment = (track_w - (count - 1) * gap) // count
        for i, name in enumerate(cal.steps):
            x = 16 + i * (segment + gap)
            y = BAR_H + 4
            cv2.rectangle(img, (x, y), (x + segment, y + 3), (62, 61, 59), -1)
            frac = 1.0 if i < cal.step_index else cal.progress if i == cal.step_index else 0.0
            if frac:
                color = COL_FG_MUTED if i < cal.step_index else COL_ACCENT_BLUE
                cv2.rectangle(img, (x, y), (x + int(segment * frac), y + 3), color, -1)
            name = f"{i + 1} {name}"
            if i == cal.step_index:
                name += f" {cal.progress:.0%}"
            draw_text(
                img,
                name,
                x,
                y + 8,
                COL_FG_LIGHT if i == cal.step_index else COL_FG_MUTED,
                11,
                i == cal.step_index,
                max_width=segment,
            )
        eta = ""
        if cal.time_left_s is not None:
            eta = (
                f"~{cal.time_left_s:.0f} s"
                if cal.time_left_s < 60
                else f"~{cal.time_left_s / 60:.0f} min"
            )
        eta_width = text_width(eta, 12) + 20 if eta else 0
        prompt_x = 16 + track_w + 24
        draw_text(
            img,
            cal.key_prompt,
            prompt_x,
            BAR_H + 13,
            COL_FG_LIGHT,
            12,
            max_width=max(0, w - prompt_x - eta_width - 20),
        )
        if eta:
            draw_text(img, eta, w - eta_width, BAR_H + 13, COL_FG_MUTED, 12)
