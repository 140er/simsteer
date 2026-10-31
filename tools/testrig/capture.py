"""Read-only capture signals. Residency alone is never proof of capture."""

import configparser
import json
from itertools import islice
from datetime import datetime
from io import StringIO
import math
import os
from pathlib import Path
import re
import time


OBS_NAMES = {"obs.exe", "obs32.exe", "obs64.exe"}
STREAMLABS_NAMES = {"streamlabs obs.exe", "streamlabs desktop.exe"}
CAPTURE_NAMES = OBS_NAMES | STREAMLABS_NAMES | {
    "nvidia share.exe", "nvidia overlay.exe", "nvsphelper64.exe", "gamebar.exe",
    "gamebarftserver.exe", "bcastdvr.exe", "discord.exe", "teams.exe", "ms-teams.exe"}
OUTPUT_KINDS = ("recording", "streaming", "replay buffer", "virtual camera")
MEDIA_SUFFIXES = {".mkv", ".mp4", ".flv", ".mov", ".ts", ".m4a", ".webm"}


def output_log_active(text, kind):
    active = False
    for line in text.lower().splitlines():
        if re.search(rf"(?:{kind}.*(?:stop|end)|(?:stop|end).*{kind})", line):
            active = False
        elif re.search(rf"(?:{kind}.*(?:start|begin)|(?:start|begin).*{kind})", line):
            active = True
    return active


def steam_stream_active(lines, born=None, *, corroborated=False):
    """Attribute Remote Play events to this Steam client, not the log's mtime."""
    process_cutoff = None
    if isinstance(born, (int, float)) and not isinstance(born, bool) and math.isfinite(born) and born > 0:
        process_cutoff = math.floor(born)  # the log timestamps have second precision
    cutoff = process_cutoff
    boundary = False
    active = None
    if isinstance(lines, str):
        lines = StringIO(lines)
    for line in lines:
        stamp = None
        match = re.match(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\]", line)
        if match:
            try:
                # Steam writes local wall time; CIM process birth is epoch time.
                # The repeated DST fall-back hour is ambiguous (fold=0), so a
                # second-pass timestamp can appear older than process birth.
                stamp = datetime.fromisoformat(match[1]).timestamp()
            except (ValueError, OSError, OverflowError):
                pass
        message = re.sub(r"^(?:\[[^\]]*\]\s*)*", "", line).strip().lower()
        if message.startswith("client version:"):
            # Reset as we pass each client start; only the last session survives.
            boundary, active = True, None
            cutoff = process_cutoff
            if stamp is not None:
                cutoff = stamp if cutoff is None else max(cutoff, stamp)
            continue
        if stamp is None:
            if not corroborated:
                continue
        elif (cutoff is None and not boundary and not corroborated) or (cutoff is not None and stamp < cutoff):
            continue
        if re.match(r"^(?:stream(?:ing)?(?: session)? (?:stopped|ended|closed|finished)|"
                    r"(?:stopping|ending|stopped|ended) (?:the )?stream(?:ing)?(?: session)?|"
                    r"(?:session|remote play session) (?:ended|stopped|closed|finished))\b", message):
            active = False
        elif re.match(r"^(?:(?:stream(?:ing)?|remote play)(?: session)? started|"
                      r"(?:starting|started) (?:the )?stream(?:ing)?(?: session)?)\b", message):
            active = True
        elif active is None and re.match(r"^(?:audio source \[.*\]:.*\b(?:init|through|last_start)=\d|"
                                         r"(?:audio|video|stream(?:ing)?) (?:stats|statistics)\b)", message):
            # Activity proves an otherwise unknown stream. Trailing diagnostics
            # after a confirmed end do not reopen it without a new start marker.
            active = True
    return active is True


def steam_log_active(path, born=None, *, corroborated=False):
    """Read all session events line by line, retaining no log-sized buffer."""
    try:
        with path.open(encoding="utf-8", errors="replace") as lines:
            return steam_stream_active(lines, born, corroborated=corroborated)
    except OSError:
        return False


def config_roots(process):
    name = process["name"].lower()
    roots = []
    if name in OBS_NAMES:
        if value := os.environ.get("APPDATA"):
            roots.append(Path(value) / "obs-studio")
        if process.get("path"):
            executable = Path(process["path"])
            portable = [executable.parent / "config/obs-studio"]
            if len(executable.parents) >= 3:
                portable.append(executable.parents[2] / "config/obs-studio")
            if "--portable" in (process.get("command") or "").lower() or any(root.exists() for root in portable):
                roots = portable
    elif name in STREAMLABS_NAMES and (value := os.environ.get("APPDATA")):
        roots.append(Path(value) / "slobs-client")
    return roots


def configured_outputs(roots):
    """Read only output-path keys, never change capture configuration."""
    folders = set()
    for root in roots:
        try:
            profiles = list(islice((root / "basic/profiles").glob("*/basic.ini"), 32))
            for path in profiles:
                config = configparser.ConfigParser(interpolation=None, strict=False)
                config.read(path, encoding="utf-8-sig")
                for section, key in (("SimpleOutput", "FilePath"), ("AdvOut", "RecFilePath"), ("AdvOut", "FFFilePath")):
                    value = config.get(section, key, fallback="")
                    if value and Path(value).is_absolute():
                        folders.add(Path(value))
            settings = root / "settings.json"
            if settings.exists():
                def paths(value):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if key.casefold() in ("recfilepath", "recordingpath", "recordingdirectory", "filepath"):
                                if isinstance(child, str) and Path(child).is_absolute():
                                    folders.add(Path(child))
                            elif isinstance(child, (dict, list)):
                                paths(child)
                    elif isinstance(value, list):
                        for child in value:
                            paths(child)
                paths(json.loads(settings.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError, configparser.Error, RecursionError):
            continue
    return folders


def output_sizes(folders, born):
    sizes = {}
    for folder in folders:
        try:
            for path in islice(folder.iterdir(), 512):
                if path.suffix.casefold() in MEDIA_SUFFIXES and path.is_file():
                    info = path.stat()
                    if info.st_mtime >= born:
                        sizes[path] = (info.st_ino, info.st_size)
        except OSError:
            continue
    return sizes


def growing_output(folders, born, *, pause=time.sleep):
    before = output_sizes(folders, born)
    if not before:
        return False
    pause(0.2)
    after = output_sizes(folders, born)
    return any(path in after and after[path][0] == info[0] and after[path][1] > info[1]
               for path, info in before.items())


def capture_activity(processes, windows, read_log):
    records, warnings = [], []
    for process in processes:
        name, pid = process["name"].lower(), process["pid"]
        if name not in CAPTURE_NAMES and not process.get("capture_tool") and not process.get("encoder_active"):
            continue
        text = " ".join(row.get("title", "") + " " + row.get("text", "") for row in windows if row["pid"] == pid).lower()
        active = any(phrase in text for phrase in (
            "stop recording", "pause recording", "stop streaming", "stop broadcasting", "stop replay buffer",
            "stop virtual camera", "stop sharing", "you are sharing", "you're sharing", "go live — streaming",
            "recording in progress", "(recording)", "discord — live"))
        observable = active or any(phrase in text for phrase in (
            "start recording", "start streaming", "start replay buffer", "start virtual camera"))
        roots = config_roots(process)
        for root in roots:
            logs = []
            for folder in (root / "logs", root / "node-obs/logs"):
                try:
                    logs.extend(path for path in islice(folder.iterdir(), 256) if path.suffix.lower() in (".txt", ".log"))
                except OSError:
                    pass
            try:
                log = max(logs, key=lambda path: path.stat().st_mtime) if logs else None
                if log and log.stat().st_mtime >= process.get("born", 0):
                    contents = read_log(log)
                    # Generic Electron logs, or a truncated tail with no known
                    # output state, must not turn an unread state into "idle".
                    known = any(kind in contents.lower() for kind in OUTPUT_KINDS) or bool(
                        re.search(r"\bobs(?:\s+\d|\s+started|.*version)", contents, re.I))
                    observable |= known and log.stat().st_size <= 256 * 1024
                    active |= any(output_log_active(contents, kind) for kind in OUTPUT_KINDS)
            except OSError:
                pass
        signals = []
        if growing_output(configured_outputs(roots), process.get("born", 0)):
            signals.append("growing_output")
            active = True
        if process.get("encoder_active"):
            signals.append("encoder")
            active = True
        cpu = process.get("cpu_percent")
        if isinstance(cpu, (int, float)) and cpu >= 25:
            signals.append("high_cpu")
        command = (process.get("command") or "").lower()
        if re.search(r"(?:^|\s)(?:--?(?:startrecording|startstreaming|startreplaybuffer|record(?:ing)?)|/record)(?:\s|=|$)", command):
            signals.append("recording_indicator")
        if not observable:
            warnings.append(f"{process.get('capture_tool') or process['name']} output state is not observable; active capture signals veto startup, otherwise verify capture is stopped")
        records.append({"pid": pid, "kind": (process.get("capture_tool") or process["name"]) + " capture/recording/streaming",
                        "active": bool(active), "observable": bool(observable), "signals": signals})
    return records, warnings
