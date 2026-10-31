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
    "obs.exe", "obs32.exe", "obs64.exe", "streamlabs obs.exe", "streamlabs desktop.exe",
    "xsplit.core.exe", "sunshine.exe", "parsecd.exe", "bcastdvr.exe", "gamebar.exe",
    "nvidia share.exe", "nvidia overlay.exe",
}


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
        reasons = list(row["signals"])
        for task in row["tasks"]:
            state = task["state"].lower()
            if state not in ("disabled", "ready", "running", "queued", "unknown"):
                raise ValueError("invalid reboot task state")
            if state == "disabled":
                continue
            when = task["next_run"]
            if state in ("running", "queued", "unknown") or (when and timestamp(when) >= checked - 300):
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
    for process in snapshot["processes"]:
        if any(same_process(process, item) for item in owned):
            continue
        name = process.get("name", "").lower()
        path = (process.get("path") or "").replace("/", "\\").lower()
        if name in BUSY_NAMES or "\\steamapps\\common\\" in path:
            raise RigError(f"Owner may be gaming, using VR or streaming ({name}); not starting")
    if check_idle:
        idle = snapshot.get("idle_seconds")
        if not isinstance(idle, (int, float)) or not math.isfinite(idle) or idle < 0:
            raise RigError("User activity check unavailable; not starting")
        if idle < idle_seconds:
            raise RigError(f"User is active (idle {idle:.0f}s; need {idle_seconds}s); not starting")


def blocking_prompt(windows):
    groups = (
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
        text = (window.get("title", "") + " " + window.get("text", "")).lower()
        for phrases, message in groups:
            if any(phrase in text for phrase in phrases):
                return message
    return None


def utc_now():
    return datetime.now(timezone.utc).isoformat()
