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
    rpt.extend(_check_vjoy(required=(device in ("wheel", "fanatec"))))
    rpt.extend(_check_fanatec(required=(device == "fanatec")))
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
    # Add a universal FOV reminder for all games
    rpt.extend(_check_fov_reminder(game))
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
                        "ACTION REQUIRED:\n"
                        "  1. Open a terminal in the SimSteer directory\n"
                        "  2. Run: python tools\\fetch_model.py\n"
                        "  3. Wait for the models to download (~60 MB total)\n"
                        "  4. Relaunch SimSteer\n\n"
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
            detail=("--device=wheel or --device=fanatec requires the vJoy driver and pyvjoy.\n"
                    "Install vJoy: https://github.com/njz3/vJoy/releases\n"
                    "Then: pip install pyvjoy\n\n"
                    "IMPORTANT: After installing vJoy, run 'Configure vJoy' from the Start menu\n"
                    "and enable device #1 with at least 3 axes (X, Y, Z)."),
            fix_url="https://github.com/njz3/vJoy/releases",
        )]


def _check_fanatec(required: bool) -> list[Check]:
    """Check for Fanatec wheel and provide setup guidance."""
    if not required:
        return []
    out: list[Check] = []
    # Try to detect Fanatec wheels
    try:
        import pygame
        pygame.init()
        count = pygame.joystick.get_count()
        found = False
        for i in range(count):
            joy = pygame.joystick.Joystick(i)
            name = joy.get_name().lower()
            if "fanatec" in name or "csl" in name or "clubsport" in name or "podium" in name:
                found = True
                out.append(Check(
                    id="fanatec_detected",
                    severity="info",
                    title=f"Fanatec wheel detected: {joy.get_name()}",
                    detail=(f"Found Fanatec wheel: {joy.get_name()}\n\n"
                            "SimSteer will use vJoy as a second virtual wheel alongside your "
                            "Fanatec. In your game, bind SimSteer's vJoy axes for AI steering "
                            "and keep your Fanatec bound for manual control.\n\n"
                            "Make sure:\n"
                            "  - Fanatec driver installed from https://fanatec.com/en-us/technology/firmware-update\n"
                            "  - Wheel is in PC mode (check Fanatec Control Panel)\n"
                            "  - vJoy driver installed and device #1 enabled\n"
                            "  - Game supports multiple steering wheels"),
                ))
                break
        pygame.quit()
        if not found:
            out.append(Check(
                id="fanatec_not_detected",
                severity="warn",
                title="No Fanatec wheel detected",
                detail=("SimSteer is set to Fanatec mode but no Fanatec wheel was detected.\n\n"
                        "Check that:\n"
                        "  - Fanatec wheel is powered on and connected via USB\n"
                        "  - Fanatec driver is installed: https://fanatec.com/en-us/technology/firmware-update\n"
                        "  - Wheel is in PC mode (not compatibility mode)\n"
                        "  - Check Fanatec Control Panel to verify the wheel is recognized\n\n"
                        "SimSteer will fall back to vJoy output, which will work but won't "
                        "coexist with your real wheel."),
                fix_url="https://fanatec.com/en-us/technology/firmware-update",
            ))
    except ImportError:
        out.append(Check(
            id="fanatec_no_pygame",
            severity="info",
            title="pygame not available for Fanatec detection",
            detail=("Could not detect Fanatec wheels (pygame not installed).\n"
                    "SimSteer will use vJoy fallback mode.\n\n"
                    "To enable detection: pip install pygame"),
        ))
    except Exception:
        pass
    return out


def _check_directml() -> list[Check]:
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if "DmlExecutionProvider" not in avail:
            return [Check(
                id="dml",
                severity="warn",
                title="DirectML unavailable — vision will run on CPU (VERY SLOW)",
                detail=("⚠️ PERFORMANCE WARNING ⚠️\n\n"
                        "Vision model will run on CPU at ~4-8 FPS instead of 20+ FPS.\n"
                        "The engagement gate will REFUSE to engage below ~8 FPS.\n\n"
                        "FIX THIS NOW:\n"
                        "1. Install onnxruntime-directml:\n"
                        "   pip install onnxruntime-directml\n\n"
                        "2. Restart SimSteer after installation\n\n"
                        "DirectML works with AMD, NVIDIA, and Intel GPUs on Windows 10+.\n"
                        "If you have a GPU but DirectML still fails, update your GPU drivers.\n\n"
                        f"Available providers: {avail}"),
            )]
        return []
    except Exception as e:
        return [Check(
            id="dml",
            severity="warn",
            title=f"DirectML probe failed: {e.__class__.__name__}",
            detail=(f"Vision will fall back to CPU (very slow).\n\n"
                    f"Error: {e}\n\n"
                    "Try: pip install --upgrade onnxruntime-directml"),
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
            detail=(f"⚠️ ETS2 WILL NOT WORK without the telemetry plugin ⚠️\n\n"
                    f"ETS2 is installed at {ets2_dir}\n"
                    f"but the SCS plugin is missing from:\n"
                    f"  {plugin_path.parent}\n\n"
                    + ("CLICK [Install] to copy the bundled plugin into ETS2,\n"
                       "or copy it manually:\n"
                       f"  FROM: {bundled}\n"
                       f"  TO:   {plugin_path}"
                       if has_bundle else
                       "Download from:\n"
                       "  https://github.com/RenCloud/scs-sdk-plugin/releases\n"
                       "Extract scs-telemetry.dll and copy to:\n"
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
            detail=("Couldn't find ETS2 user config at:\n"
                    "  %USERPROFILE%\\Documents\\Euro Truck Simulator 2\\config.cfg\n\n"
                    "Launch ETS2 once (to the main menu) to generate it.\n"
                    "Then SimSteer can check the steering deadzone setting."),
        ))
    else:
        dz = _parse_ets2_deadzone(cfg)
        if dz is not None and dz > 0.001:
            out.append(Check(
                id="ets2_deadzone",
                severity="warn",
                title=f"ETS2: steering deadzone is {dz * 100:.0f}% — MUST BE 0",
                detail=(f"⚠️ TRUCK WILL NOT STEER with deadzone > 0 ⚠️\n\n"
                        f"ETS2's steering deadzone is currently {dz * 100:.0f}%.\n"
                        "Any deadzone silences the AI's small steering inputs.\n\n"
                        "FIX IN GAME:\n"
                        "1. Launch ETS2\n"
                        "2. Options → Controls\n"
                        "3. Find 'Steering deadzone' slider\n"
                        "4. Drag to 0%\n"
                        "5. Apply and restart SimSteer\n\n"
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
            detail=("⚠️ ASSETTO CORSA TELEMETRY NOT DETECTED ⚠️\n\n"
                    "Couldn't open shared memory `Local\\acpmf_physics`.\n\n"
                    "POSSIBLE CAUSES:\n"
                    "1. AC isn't running yet — launch AC and load a track\n"
                    "2. AC launched as Administrator but SimSteer did not\n"
                    "   → Relaunch SimSteer as Administrator\n"
                    "3. Shared memory disabled in AC settings\n\n"
                    "RECOMMENDED:\n"
                    "Launch AC through Content Manager for reliable telemetry:\n"
                    "  https://acstuff.ru/app/\n\n"
                    "If AC is running and you still see this, check:\n"
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
            title="AC: shared memory open but no data yet",
            detail=("AC's shared memory is mapped but `packet_id` is 0.\n\n"
                    "This means you're at the main menu or AC is paused.\n"
                    "Load a track and start driving to begin telemetry."),
        )]
    return []


# ----- FOV reminder -----

def _check_fov_reminder(game: str | None) -> list[Check]:
    """Prominent FOV setup reminder. Wrong FOV is the #1 cause of the
    plan veering off the road. This surfaces every time until the user
    has verified FOV at least once."""
    if game is None:
        return []
    return [Check(
        id="fov_reminder",
        severity="warn",
        title="⚠️ CRITICAL: Set camera FOV to match your in-game setting",
        detail=(
            "Wrong FOV is the #1 cause of the AI veering off the road.\n\n"
            "STEPS TO SET FOV:\n"
            "1. Find your in-game FOV:\n"
            "   - ETS2: Options → Gameplay → Camera → Field of view\n"
            "   - AC: Options → Video → Camera FOV\n"
            "   - Forza: Settings → Difficulty → Camera FOV\n\n"
            "2. In SimSteer tuner → Camera & Calibration → set Capture VFOV to match\n\n"
            "3. Verify while driving:\n"
            "   - Look at HUD 'FOV' line: 'ratio vx_model/v_ego'\n"
            "   - Drive straight at highway speed\n"
            "   - Ratio should be ~1.00 (±0.05)\n"
            "   - If >1.05: FOV too high, narrow it\n"
            "   - If <0.95: FOV too low, widen it\n\n"
            "SimSteer now includes AUTOMATIC FOV detection — it will measure and\n"
            "correct FOV after 60 straight+fast samples (~30-60 seconds of highway).\n"
            "Watch the HUD 'auto-FOV' line for progress."
        ),
    )]


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
