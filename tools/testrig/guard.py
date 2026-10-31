"""Pure, fail-closed evaluations of Windows inventory."""

from datetime import datetime, timezone
import ntpath
import math
import time

from . import RigError


BUSY_NAMES = {
    "acs.exe", "acs_x86.exe", "assettocorsa.exe", "eurotrucks2.exe", "amtrucks.exe",
    "forzahorizon5.exe", "forza_horizon_5.exe", "le mans ultimate.exe", "lemansultimate.exe",
    "acc.exe", "ac2-win64-shipping.exe", "iracingui.exe", "iracingsim64dx11.exe",
    "rfactor2.exe", "automobilista2.exe", "f1_24.exe", "f1_25.exe",
    "vrserver.exe", "vrmonitor.exe", "vrcompositor.exe", "vrdashboard.exe",
    "oculusclient.exe", "ovrserver_x64.exe", "virtualdesktop.streamer.exe",
    "fortniteclient-win64-shipping.exe", "cyberpunk2077.exe", "rdr2.exe", "gta5.exe",
    "gta5_enhanced.exe", "fc26.exe", "bf6.exe", "overwatch.exe", "valorant-win64-shipping.exe",
    "flightsimulator.exe", "flightsimulator2024.exe",
}
BUSY_NAMES -= {"ovrserver_x64.exe", "virtualdesktop.streamer.exe"}
IDLE_HELPERS = {"nvidia share.exe", "nvidia overlay.exe", "gamebar.exe", "ovrserver_x64.exe",
                "sunshine.exe", "parsecd.exe", "virtualdesktop.streamer.exe", "wallpaper32.exe",
                "wallpaper64.exe", "losslessscaling.exe", "obs.exe", "obs32.exe", "obs64.exe",
                "streamlabs obs.exe", "streamlabs desktop.exe", "bcastdvr.exe", "gamebarftserver.exe"}


def owned_descendants(processes, owned):
    trusted = {row["pid"]: row for row in processes if any(same_process(row, item) for item in owned)}
    children = {}
    for row in processes:
        children.setdefault(row.get("parent"), []).append(row)
    pending = list(trusted.values())
    while pending:
        parent = pending.pop()
        for child in children.get(parent["pid"], []):
            if (child["pid"] not in trusted and child.get("created") and child.get("path") and
                    child.get("born", 0) >= parent.get("born", 0)):
                trusted[child["pid"]] = child
                pending.append(child)
    return trusted


def same_process(a, b):
    return (bool(a.get("created")) and bool(a.get("path")) and
            a.get("pid") == b.get("pid") and a["created"] == b.get("created") and
            ntpath.normcase(a["path"]) == ntpath.normcase(b.get("path") or ""))


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.timestamp()


def restart_reasons(row, now=None):
    """Missing, stale, malformed or denied inventory is a refusal, never clear."""
    now = time.time() if now is None else now
    try:
        for field in ("signals", "tasks", "errors"):
            if not isinstance(row[field], list):
                raise ValueError(f"invalid {field}")
        if any(not isinstance(value, str) for value in row["signals"] + row["errors"]):
            raise ValueError("invalid restart signals/errors")
        checked = timestamp(row["checked_at"])
        if not -5 <= now - checked <= 60:
            raise ValueError("stale restart inventory")
        reasons = [signal for signal in row["signals"] if signal != "PendingFileRenameOperations"]
        for task in row["tasks"]:
            state = task["state"].lower()
            if state not in ("disabled", "ready", "running", "queued", "unknown"):
                raise ValueError("invalid reboot task state")
            if state == "disabled":
                continue
            when = task["next_run"]
            if state in ("running", "queued", "unknown") or (when and checked - 300 <= timestamp(when) <= checked + 900):
                reasons.append("UpdateOrchestrator/" + task["name"])
        if row["errors"]:
            raise ValueError("; ".join(row["errors"]))
        return reasons
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise RigError(f"Windows restart check incomplete ({exc}); not starting") from exc


def require_free(snapshot, owned=(), idle_seconds=120, *, check_idle=True, now=None):
    reasons = restart_reasons(snapshot.get("restart", {}), now)
    if reasons:
        raise RigError("Windows restart pending (" + "; ".join(reasons) + "); not starting")
    if not isinstance(snapshot.get("errors"), list) or snapshot["errors"] or not isinstance(snapshot.get("processes"), list):
        raise RigError("Owner activity/process check incomplete; not starting")
    notes = list(snapshot.get("warnings", [])) + list(snapshot["restart"].get("warnings", []))
    if "PendingFileRenameOperations" in snapshot["restart"]["signals"]:
        notes.append("PendingFileRenameOperations is set; restart may be needed after installation")
    for task in snapshot["restart"]["tasks"]:
        if task["state"].lower() == "ready" and task.get("next_run") and timestamp(task["next_run"]) > timestamp(snapshot["restart"]["checked_at"]) + 900:
            notes.append("UpdateOrchestrator reboot is scheduled beyond this session's immediate startup window")
    if any(not isinstance(note, str) for note in notes):
        raise RigError("Invalid guard warnings; not starting")
    trusted = owned_descendants(snapshot["processes"], owned)
    for activity in snapshot.get("activity", []):
        if (activity.get("active") or (not activity.get("observable", True) and activity.get("signals"))) and activity.get("pid") not in trusted:
            raise RigError("Active recording/streaming detected (" + activity["kind"] + "); not starting")
    for process in snapshot["processes"]:
        if process["pid"] in trusted:
            continue
        name = process.get("name", "").lower()
        if name in BUSY_NAMES:
            raise RigError(f"Owner may be gaming, using VR or streaming ({name}); not starting")
    foreground = snapshot.get("foreground") or {}
    foreground_pid = foreground.get("pid")
    foreground_process = next((row for row in snapshot["processes"] if row["pid"] == foreground_pid), {})
    if ((foreground.get("exclusive") or (foreground.get("fullscreen") and
             not foreground.get("maximized") and not foreground.get("caption"))) and foreground_pid not in trusted and
            foreground_process.get("name", "").lower() not in IDLE_HELPERS and
            foreground.get("class") not in ("Progman", "WorkerW", "Shell_TrayWnd")):
        raise RigError("Owner may be gaming (unowned fullscreen foreground window); not starting")
    if check_idle:
        idle = snapshot.get("idle_seconds")
        if not isinstance(idle, (int, float)) or not math.isfinite(idle) or idle < 0:
            raise RigError("User activity check unavailable; not starting")
        if idle < idle_seconds:
            raise RigError(f"User is active (idle {idle:.0f}s; need {idle_seconds}s); not starting")
    return sorted(set(notes))


def blocking_prompt(windows):
    groups = (
        (("allow game launch", "allow this game to launch", "launch game with these options", "launch confirmation"),
         "Steam launch confirmation detected; approve or cancel the launch manually outside the test session"),
        (("sign in", "sign-in", "log in", "login", "world of trucks", "steam guard"),
         "Sign-in screen detected; test sessions never sign in to accounts"),
        (("license agreement", "licence agreement", "eula", "terms of service"),
         "EULA detected; accept it manually outside the test session"),
        (("windows security alert", "windows defender firewall", "allow access", "firewall"),
         "Firewall prompt detected; resolve it manually outside the test session"),
        (("purchase", "no licenses", "no licences", "not owned", "buy now"),
         "Game ownership/license prompt detected; verify ownership in Steam manually"),
    )
    for window in windows:
        text = (window.get("title", "") + " " + window.get("text", "") + " " + window.get("class", "")).lower()
        for phrases, message in groups:
            if any(phrase in text for phrase in phrases):
                return message
        if window.get("class", "").lower() == "vguipopupwindow" and window.get("owner") and window.get("title", "").lower() == "steam":
            return "Steam launch confirmation/modal detected; resolve it manually outside the test session"
        if (window.get("class", "").lower() in ("sdl_app", "chrome_widgetwin_1") and
                window.get("steam") and (window.get("owner") or "launch options" in text or "launch game" in text)):
            return "Steam launch confirmation/modal detected; resolve it manually outside the test session"
        if window.get("class", "").lower() == "sdl_app" and window.get("title", "").lower() == "launch options":
            return "Steam launch confirmation detected; resolve launch options manually outside the test session"
    return None


def utc_now():
    return datetime.now(timezone.utc).isoformat()
