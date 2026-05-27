"""First-drive wizard.

Guides the user through ~10-15 minutes of highway driving on first launch
to converge LiveCalib (camera pose) and LiveParams (steering rack). Three
phases:

  CAMERA   — LiveCalib not yet CALIBRATED. INSERT is BLOCKED. The user
             drives manually; AC/Forza already passive-fit LiveParams in
             this state, so progress on both happens in parallel. ETS2's
             gamepad-rack-assist makes passive-fit risky, so for ETS2
             this phase is essentially camera-only.
  STEERING — LiveCalib CALIBRATED, LiveParams not yet trusted. INSERT
             allowed; user engages and drives gently while the AI runs.
             A persistent overlay warns that steering may be bumpy.
  DONE     — both converged. A green READY banner shows for 10 s, then
             collapses. The `first_drive_done_<game>.flag` file is
             written so future launches skip the wizard entirely.

Detection: the flag file at `<data>/first_drive_done_<game>.flag`. If it
exists at startup, the wizard is permanently disabled for that game.
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import TYPE_CHECKING

from pilot import audio
from pilot.livecalib import (
    BLOCK_SIZE, CalStatus, INPUTS_NEEDED, MIN_SPEED_FILTER_MPS,
)
from pilot.liveparams import TRUSTED_MIN_SAMPLES
from pilot.paths import data_dir

if TYPE_CHECKING:
    from pilot.livecalib import LiveCalib
    from pilot.liveparams import LiveParams


class WizardPhase(IntEnum):
    DONE = 0
    CAMERA = 1
    STEERING = 2


# How long to show the bright READY banner after both converge.
READY_BANNER_SECONDS = 10.0


def _fmt_eta(seconds: float) -> str:
    """Format an ETA as `<N>s` / `<N>min` / `<N>h<M>m`. Used in the
    wizard banner so the user has a real "are we there yet" answer."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}min"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


class Wizard:
    """Frame-by-frame observer over LiveCalib + LiveParams. Reports the
    current phase, a 0-100 progress percentage, a one-line hint, and
    whether engagement should be allowed at all."""

    def __init__(self, game: str | None,
                 live_calib: "LiveCalib",
                 live_params: "LiveParams") -> None:
        self._lc = live_calib
        self._lp = live_params
        self._game = game
        # If we don't know the game (no telemetry yet), we can't write
        # a per-game flag — disable the wizard.
        if game is None:
            self._flag_path = None
        else:
            self._flag_path = data_dir() / f"first_drive_done_{game}.flag"
        # If the user has already completed the wizard for this game,
        # disable it permanently. We re-check existence at construct time
        # only — deleting the flag at runtime won't re-enable.
        self._active = (self._flag_path is not None
                        and not self._flag_path.exists())
        self._chimed = False
        self._ready_banner_until = 0.0

    # ----- state ------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    @property
    def phase(self) -> WizardPhase:
        """Returns the current wizard phase. DONE when not active or
        when both learners have converged."""
        if not self._active:
            return WizardPhase.DONE
        if self._lc.cal_status != CalStatus.CALIBRATED:
            return WizardPhase.CAMERA
        if not self._lp.trusted():
            return WizardPhase.STEERING
        return WizardPhase.DONE

    @property
    def in_camera_phase(self) -> bool:
        """True only while the user must drive manually (camera not
        converged). Wired into the LiveParams passive-fit gate so the
        wizard can still build steering data on ETS2 during phase A,
        even though normal passive-fit is gated for ETS2."""
        return self.phase == WizardPhase.CAMERA

    def allows_engage(self) -> tuple[bool, str]:
        """Wizard's contribution to the engagement gate. Returns
        `(allowed, reason)`. The main engagement gate ANDs this with
        the FPS / telemetry / pad checks."""
        if not self._active:
            return True, ""
        if self.phase == WizardPhase.CAMERA:
            return False, "first drive: camera calibration not done — drive manually"
        return True, ""

    # ----- progress ---------------------------------------------------

    @staticmethod
    def _calib_frac(lc: "LiveCalib") -> float:
        return min(1.0,
                   (lc.blocks * BLOCK_SIZE + lc._block_n)
                   / float(INPUTS_NEEDED * BLOCK_SIZE))

    @staticmethod
    def _lp_frac(lp: "LiveParams") -> float:
        return min(1.0,
                   max(lp.samples, lp.session_samples)
                   / float(TRUSTED_MIN_SAMPLES))

    def progress_pct(self) -> float:
        """0-100 unified progress: min(camera%, steering%).

        We use the minimum, not the average, so the bar reflects the
        slowest learner — the one that's actually blocking. A weighted
        average can show 70% while camera (the gate) is still at 40%,
        which misleads the user into thinking they're almost done."""
        return 100.0 * min(self._calib_frac(self._lc),
                           self._lp_frac(self._lp))

    def eta_seconds(self) -> float | None:
        """Estimated seconds until both learners are ready. None if we
        don't have a rate signal yet. Bounded by the slower of the two
        — that's what gates engagement."""
        calib_eta = self._lc.eta_to_calibrated_s()
        # LiveParams session_samples grows at ~5-15 Hz when engaged.
        # Use the same acceptance-rate trick: time since first
        # session sample / current count = avg rate.
        lp_eta: float | None = None
        if not self._lp.trusted():
            need = TRUSTED_MIN_SAMPLES - max(self._lp.samples,
                                              self._lp.session_samples)
            if need > 0:
                # Treat LiveParams' rate as ~10 Hz when actively fitting.
                # If we have a session, refine from session_samples.
                # (LiveParams doesn't expose its own rate; this is a
                # conservative constant — fine for ETA display.)
                lp_eta = need / 10.0
        if calib_eta is None and lp_eta is None:
            return None
        if calib_eta is None:
            return lp_eta
        if lp_eta is None:
            return calib_eta
        return max(calib_eta, lp_eta)

    def hint(self, v_ego: float | None) -> str:
        """One-line, plain-English next-action for the user."""
        # Sticky LiveCalib reset notice — surface for one bar render so
        # the user knows why the bar suddenly returned to 0%. (The next
        # call clears it after consuming.)
        if self._lc.last_reset_reason:
            msg = self._lc.last_reset_reason
            self._lc.last_reset_reason = None
            return f"calibration reset: {msg}"
        # Camera phase guidance dominates until LiveCalib is done.
        if self.phase == WizardPhase.CAMERA:
            if v_ego is not None and v_ego < MIN_SPEED_FILTER_MPS:
                mph = v_ego * 2.23694
                return (f"DRIVE FASTER — currently {mph:.0f} mph, "
                        f"need 15+ for calibration")
            if (self._lc.samples > 50
                    and self._lc.rej_human > self._lc.samples * 0.3):
                # If we've rejected lots of samples as human-input,
                # the user is probably steering hard.
                return "DRIVE STRAIGHT — small/no steering inputs while calibrating"
            eta = self._lc.eta_to_calibrated_s()
            if eta is not None and eta > 0:
                return (f"keep driving on highway — camera ~{_fmt_eta(eta)} "
                        f"@ {self._lc.acceptance_rate_hz:.0f} samples/s")
            return "keep driving on highway — camera calibrating"
        if self.phase == WizardPhase.STEERING:
            return "ENGAGE and drive gently — steering fit warming up"
        return "calibration complete"

    # ----- banner -----------------------------------------------------

    @staticmethod
    def _banner_with_eta(prefix: str, eta_s: float | None) -> str:
        if eta_s is None or eta_s <= 0:
            return prefix
        return f"{prefix} (~{_fmt_eta(eta_s)} left)"

    def banner(self) -> tuple[str, tuple[int, int, int]] | None:
        """Big top banner for the overlay. None when wizard inactive."""
        if not self._active:
            return None
        ph = self.phase
        pct = self.progress_pct()
        eta = self.eta_seconds()
        if ph == WizardPhase.CAMERA:
            prefix = f"FIRST DRIVE — {pct:.0f}% — drive manually on highway"
            return (self._banner_with_eta(prefix, eta), (80, 200, 255))
        if ph == WizardPhase.STEERING:
            prefix = f"FIRST DRIVE — {pct:.0f}% — press INSERT, drive gently"
            return (self._banner_with_eta(prefix, eta), (80, 220, 200))
        # DONE: show bright READY banner for a window, then collapse.
        if time.time() < self._ready_banner_until:
            return ("READY TO ENGAGE — press INSERT", (80, 255, 80))
        return ("READY", (80, 255, 80))

    def progress_lines(self) -> list[tuple[str, tuple[int, int, int]]]:
        """Two-line progress detail under the banner."""
        if not self._active:
            return []
        calib_pct = 100.0 * min(
            1.0, (self._lc.blocks * BLOCK_SIZE + self._lc._block_n)
            / float(INPUTS_NEEDED * BLOCK_SIZE))
        lp_pct = 100.0 * min(
            1.0, max(self._lp.samples, self._lp.session_samples)
            / float(TRUSTED_MIN_SAMPLES))
        camera_color = ((80, 255, 80)
                        if self._lc.cal_status == CalStatus.CALIBRATED
                        else (200, 200, 80))
        steer_color = ((80, 255, 80) if self._lp.trusted()
                       else (200, 200, 80))
        return [
            (f"  Camera: {calib_pct:.0f}% "
             f"({self._lc.blocks}/{INPUTS_NEEDED} blocks, "
             f"{self._lc.samples} samples)",
             camera_color),
            (f"  Steering: {lp_pct:.0f}% "
             f"({max(self._lp.samples, self._lp.session_samples)}/"
             f"{TRUSTED_MIN_SAMPLES} samples)",
             steer_color),
        ]

    # ----- reset ------------------------------------------------------

    def reset(self) -> None:
        """Re-enable the wizard from scratch. Called by the overlay `R`
        hotkey and the `--reset-calib` flag. Deletes the per-game flag
        file so a fresh launch would also see the wizard.

        Doesn't touch LiveCalib or LiveParams — the caller is responsible
        for resetting those (typically alongside this call)."""
        if self._flag_path is not None:
            try:
                if self._flag_path.exists():
                    self._flag_path.unlink()
            except OSError:
                pass
        self._active = self._flag_path is not None
        self._chimed = False
        self._ready_banner_until = 0.0

    # ----- tick -------------------------------------------------------

    def tick(self) -> None:
        """Called once per frame. Fires the READY chime + writes the
        flag file when the wizard transitions to DONE."""
        if not self._active:
            return
        if self.phase != WizardPhase.DONE:
            return
        if self._chimed:
            return
        # First time we hit DONE.
        self._chimed = True
        self._ready_banner_until = time.time() + READY_BANNER_SECONDS
        audio.play("ready")
        if self._flag_path is not None:
            try:
                self._flag_path.write_text(
                    f"completed at {time.time()}\n")
            except OSError:
                pass
        # Stay nominally active so the READY banner can render until
        # _ready_banner_until elapses; banner() handles the time check.
