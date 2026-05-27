"""Preflight checks — run before the cv2 window opens.

Surfaces missing dependencies (ViGEm, model files), broken installs, and
per-game misconfigurations (SCS plugin not installed, ETS2 deadzone not 0,
Forza Data Out off, AC shared memory missing) so the user knows what to fix
instead of getting silent failures.

Two phases:
- `run_global_preflight()` — game-agnostic checks (drivers, models, DML).
   Runs BEFORE telemetry opens, so fatal checks can abort cleanly.
- `run_game_preflight(game)` — per-game checks for the detected game.
   Runs AFTER telemetry detects which game is producing data.

Display helper `show_preflight_dialog` uses Tk `messagebox` and MUST be
called before any cv2 window is created (Tk+cv2 deadlock on Windows
otherwise).
"""

from __future__ import annotations

import mmap
import os
import re
import shutil
import socket
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from pilot.paths import model_path, prereq_path

Severity = Literal["fatal", "warn", "info"]


@dataclass
class Check:
    id: str
    severity: Severity
    title: str               # one-line HUD banner / dialog header
    detail: str              # multi-line dialog body
    fix_url: str | None = None
    # When True, the launcher offers an [Install] action that runs
    # `install_action`. Used for the bundled SCS plugin DLL.
    can_install: bool = False
    install_action: Callable[[], tuple[bool, str]] | None = None


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def fatals(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "fatal"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "warn"]

    def extend(self, more: list[Check]) -> None:
        self.checks.extend(more)


def run_global_preflight(device: str = "gamepad") -> PreflightReport:
    """Game-agnostic checks. Run BEFORE telemetry opens so the launcher
    can abort cleanly on fatals (ViGEm missing, no model files)."""
    rpt = PreflightReport()
    rpt.extend(_check_models())
    rpt.extend(_check_vigem(required=(device == "gamepad")))
    rpt.extend(_check_vjoy(required=(device == "wheel")))
    rpt.extend(_check_directml())
    return rpt


def run_game_preflight(game: str | None) -> PreflightReport:
    """Per-game checks. Runs after telemetry auto-detect knows which
    game is producing data."""
    rpt = PreflightReport()
    if game == "ets2":
        rpt.extend(_check_ets2())
    elif game == "ac":
        rpt.extend(_check_ac())
    elif game == "forza":
        # If telemetry already detected Forza, we already know Data Out
        # works — no warning needed here. Reserved for future per-game
        # tips (e.g. recommended HFOV).
        pass
    return rpt


# ----- always-run checks -----

def _check_models() -> list[Check]:
    out: list[Check] = []
    for name in ("driving_vision.onnx", "driving_policy.onnx"):
        p = model_path(name)
        if not p.exists():
            out.append(Check(
                id=f"model_{name}",
                severity="fatal",
                title=f"Model file missing: {name}",
                detail=(f"Expected at {p}.\n\n"
                        "For dev installs: run `python tools\\fetch_model.py`.\n"
                        "For shipped bundles: the bundle is incomplete — "
                        "redownload and reinstall."),
            ))
    return out


def _check_vigem(required: bool) -> list[Check]:
    try:
        import vgamepad
        # Open + close a virtual pad as the actual driver probe.
        pad = vgamepad.VX360Gamepad()
        del pad
        return []
    except Exception as e:
        severity: Severity = "fatal" if required else "warn"
        return [Check(
            id="vigem",
            severity=severity,
            title="ViGEm Bus Driver not detected",
            detail=("The virtual gamepad cannot open. Install ViGEm:\n"
                    "  https://github.com/nefarius/ViGEmBus/releases\n"
                    "Run the .msi, reboot, relaunch.\n\n"
                    f"Error: {e}"),
            fix_url="https://github.com/nefarius/ViGEmBus/releases",
        )]


def _check_vjoy(required: bool) -> list[Check]:
    try:
        import pyvjoy  # noqa: F401
        return []
    except ImportError:
        if not required:
            return []
        return [Check(
            id="vjoy",
            severity="fatal",
            title="vJoy driver / pyvjoy not detected",
            detail=("--device=wheel requires the vJoy driver and pyvjoy.\n"
                    "Install vJoy: https://github.com/njz3/vJoy/releases\n"
                    "Then: pip install pyvjoy"),
            fix_url="https://github.com/njz3/vJoy/releases",
        )]


def _check_directml() -> list[Check]:
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if "DmlExecutionProvider" not in avail:
            return [Check(
                id="dml",
                severity="warn",
                title="DirectML unavailable — vision will run on CPU (slow)",
                detail=("Expect 4-8 FPS instead of 20+. The engagement gate "
                        "will refuse to engage below ~8 FPS.\n\n"
                        "Install onnxruntime-directml:\n"
                        "  pip install onnxruntime-directml\n"
                        f"Available providers: {avail}"),
            )]
        return []
    except Exception as e:
        return [Check(
            id="dml",
            severity="warn",
            title=f"DirectML probe failed: {e.__class__.__name__}",
            detail=f"Vision will fall back to CPU.\n\n{e}",
        )]


# ----- ETS2 -----

def _ets2_install_dir() -> Path | None:
    """Find the ETS2 install dir via Steam's registry entry, or None
    if neither Steam nor ETS2 is installed."""
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None
    candidates: list[Path] = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for subkey in (r"SOFTWARE\Wow6432Node\Valve\Steam",
                       r"SOFTWARE\Valve\Steam"):
            try:
                with winreg.OpenKey(hive, subkey) as h:
                    install_path, _ = winreg.QueryValueEx(h, "InstallPath")
                candidate = (Path(install_path) / "steamapps" / "common"
                             / "Euro Truck Simulator 2")
                if candidate.exists():
                    candidates.append(candidate)
            except OSError:
                continue
    return candidates[0] if candidates else None


def _ets2_config_path() -> Path | None:
    """ETS2 user config.cfg location, or None if not found."""
    home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    p = home / "Documents" / "Euro Truck Simulator 2" / "config.cfg"
    return p if p.exists() else None


def _parse_ets2_deadzone(cfg_path: Path) -> float | None:
    """Parse `uset g_steer_dead_zone "X"` from config.cfg. None if absent."""
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(r'uset\s+g_steer_dead_zone\s+"([0-9.+\-eE]+)"', text)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _install_scs_plugin(target: Path, bundled: Path) -> tuple[bool, str]:
    """Copy the bundled SCS plugin DLL into ETS2's plugins folder.
    Needs UAC if ETS2 is under Program Files."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundled, target)
        return True, f"Installed to {target}"
    except PermissionError:
        return False, (
            f"Permission denied writing to {target.parent}.\n"
            f"Run SimSteer as Administrator, OR copy the file manually:\n"
            f"  FROM: {bundled}\n"
            f"  TO:   {target}")
    except OSError as e:
        return False, f"Install failed: {e}"


def _check_ets2() -> list[Check]:
    out: list[Check] = []
    ets2_dir = _ets2_install_dir()
    if ets2_dir is None:
        return out  # Not installed via Steam — nothing to warn about.

    plugin_path = (ets2_dir / "bin" / "win_x64" / "plugins"
                   / "scs-telemetry.dll")
    if not plugin_path.exists():
        bundled = prereq_path("scs-telemetry.dll")
        has_bundle = bundled.exists()
        out.append(Check(
            id="ets2_scs_plugin",
            severity="warn",
            title="ETS2: SCS Telemetry plugin not installed",
            detail=(f"ETS2 is installed at {ets2_dir}\n"
                    f"but the SCS plugin is missing from\n"
                    f"  {plugin_path.parent}\\\n\n"
                    + ("We bundled the plugin — click [Install] to copy "
                       "it into your ETS2 plugins folder.\n"
                       f"(Bundled at {bundled})"
                       if has_bundle else
                       "Download the plugin manually from:\n"
                       "  https://github.com/RenCloud/scs-sdk-plugin/releases\n"
                       "and drop scs-telemetry.dll into:\n"
                       f"  {plugin_path.parent}\\")),
            fix_url="https://github.com/RenCloud/scs-sdk-plugin/releases",
            can_install=has_bundle,
            install_action=(lambda: _install_scs_plugin(plugin_path, bundled))
                           if has_bundle else None,
        ))

    cfg = _ets2_config_path()
    if cfg is None:
        out.append(Check(
            id="ets2_config",
            severity="info",
            title="ETS2: config.cfg not yet generated",
            detail=("Couldn't find ETS2 user config at\n"
                    "  %USERPROFILE%\\Documents\\Euro Truck Simulator 2\\config.cfg\n"
                    "Launch ETS2 once (to the main menu) to generate it. "
                    "Then the deadzone check can run."),
        ))
    else:
        dz = _parse_ets2_deadzone(cfg)
        if dz is not None and dz > 0.001:
            out.append(Check(
                id="ets2_deadzone",
                severity="warn",
                title=f"ETS2: steering deadzone is {dz * 100:.0f}% — must be 0",
                detail=(f"ETS2's `g_steer_dead_zone` is {dz:.3f}. With any "
                        "deadzone, the AI's small steering inputs are "
                        "silenced and the truck won't track lanes.\n\n"
                        "Fix it manually in-game:\n"
                        "  ETS2 -> Options -> Controls\n"
                        "  Find the Steering deadzone slider; drag to 0%.\n\n"
                        f"Config file: {cfg}"),
            ))
    return out


# ----- AC -----

def _check_ac() -> list[Check]:
    try:
        mm = mmap.mmap(-1, 4096, "Local\\acpmf_physics",
                       access=mmap.ACCESS_READ)
    except OSError:
        # No mapping — telemetry auto-detect already failed AC. If we got
        # here, the user explicitly asked for AC. Tell them what to do.
        return [Check(
            id="ac_no_shmem",
            severity="warn",
            title="AC: shared memory not available",
            detail=("Couldn't open `Local\\acpmf_physics`. Either AC isn't "
                    "running yet, or shared memory output is disabled.\n\n"
                    "Launching AC through Content Manager is the most "
                    "reliable way to expose telemetry:\n"
                    "  https://acstuff.ru/app/\n\n"
                    "If AC is in a session and you still see this, check\n"
                    "  Documents\\Assetto Corsa\\cfg\\acos.ini\n"
                    "for shared-memory options."),
            fix_url="https://acstuff.ru/app/",
        )]
    try:
        packet_id = struct.unpack_from("<i", mm, 0)[0]
    except struct.error:
        packet_id = 0
    finally:
        mm.close()
    if packet_id == 0:
        return [Check(
            id="ac_inactive",
            severity="info",
            title="AC: shared memory open but no data",
            detail=("AC's shared memory is mapped but `packet_id` is 0 — "
                    "either you're at the main menu, or AC is paused.\n"
                    "Load a track and start driving."),
        )]
    return []


# ----- preflight UI -----

def show_preflight_dialog(rpt: PreflightReport) -> bool:
    """Show fatal errors and warning install offers in Tk dialogs BEFORE
    cv2 opens its window. Returns True if it's safe to proceed (no fatals
    OR user dismissed them), False if launch should abort.

    Tk+cv2 deadlock on Windows if Tk dialogs run after a cv2 window is
    created — this function must be called first."""
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        # No Tk — fall back to printing.
        print("preflight checks (no Tk available, printing instead):")
        for c in rpt.checks:
            print(f"  [{c.severity.upper()}] {c.title}")
            for line in c.detail.splitlines():
                print(f"      {line}")
        return not rpt.fatals

    # Hidden root so messagebox doesn't open an empty window.
    root = tk.Tk()
    root.withdraw()
    try:
        if rpt.fatals:
            body = "\n\n".join(
                f"[FATAL] {c.title}\n{c.detail}" for c in rpt.fatals)
            messagebox.showerror(
                "SimSteer — cannot start",
                body + "\n\nFix the errors above and relaunch.")
            return False

        for c in rpt.warnings:
            if c.can_install and c.install_action is not None:
                answer = messagebox.askyesno(
                    f"SimSteer — {c.title}",
                    f"{c.detail}\n\nInstall now?")
                if answer:
                    ok, msg = c.install_action()
                    if ok:
                        messagebox.showinfo("SimSteer — Install OK", msg)
                    else:
                        messagebox.showwarning(
                            "SimSteer — Install failed", msg)
        return True
    finally:
        root.destroy()


def warnings_for_hud(rpt: PreflightReport) -> list[str]:
    """One-line strings for each warning, in display order. Drop info-
    level checks from the HUD — they're shown in the Tk dialog only."""
    return [c.title for c in rpt.warnings]
