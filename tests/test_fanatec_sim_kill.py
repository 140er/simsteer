"""Kill the FFB bridge while a force plays: the base must be quiet within
250 ms from the renewal windows alone, assuming the worst (the driver keeps
playing every effect until its duration ends). A clean stop sends STOPALL."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from simsteer.core import ffb_model as fm
from simsteer.testing.fanatec import FfbLog
from simsteer.testing.fanatec.__main__ import report
import ffb_fixtures as fx

ROOT = Path(__file__).resolve().parents[1]


def endless_constant(path: Path) -> None:
    steps = [(0.0, fx.gain(255)), (0.001, fx.new_effect(1, 1)), (0.002, fx.constant(1, 4000)),
             (0.003, fx.effect_report(1, 1)), (0.004, fx.effect_op(1, fm.OP_START))]
    path.write_text(json.dumps({"packets": [{"t": t, "cmd": c, "data": d.hex()} for t, (c, d) in steps]}))


def start_bridge(tmp_path):
    packets, events = tmp_path / "game.json", tmp_path / "events.jsonl"
    endless_constant(packets)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        [sys.executable, "-m", "simsteer.testing.fanatec", "bridge", "--cap", "5", "--packets", str(packets),
         "--events", str(events), "--seconds", "30"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=flags,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
    seen = []
    deadline = time.monotonic() + 20
    while True:
        try:
            line = lines.get(timeout=max(0.1, deadline - time.monotonic()))
        except queue.Empty:
            proc.kill()
            pytest.fail(f"bridge never played a force: {seen}")
        seen.append(line)
        if line.startswith("playing"):
            return proc, events, lines


def test_killed_bridge_leaves_no_force_after_250_ms(tmp_path):
    proc, events, _ = start_bridge(tmp_path)
    time.sleep(0.35)  # a few renewals
    proc.kill()
    killed = time.monotonic()
    proc.wait(10)
    log = FfbLog.load(events)
    assert log.force_at(killed - 0.01) == pytest.approx(0.4 * 0.05)  # it was playing
    assert "STOPALL" not in log.commands()  # no clean-up ran
    log.assert_all_stopped(after=killed, within_ms=250)
    log.assert_force_never_above(0.05)
    assert report(log, cap=5, deadline=killed + 0.25)["failures"] == []


def test_interrupted_bridge_stops_every_force_at_once(tmp_path):
    proc, events, lines = start_bridge(tmp_path)
    time.sleep(0.2)
    asked = time.monotonic()
    proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
    assert proc.wait(10) == 0
    log = FfbLog.load(events)
    stopall = [e["t"] for e in log.calls("SendForceFeedbackCommand") if e["detail"] == "STOPALL"]
    assert stopall and stopall[-1] - asked < 0.25
    log.assert_all_stopped(by=stopall[-1])
    log.assert_quiet_after(stopall[-1])
    rest = []
    while True:
        try:
            rest.append(lines.get(timeout=5))
        except queue.Empty:
            break
        if rest[-1].startswith("{"):
            break
    summary = json.loads(rest[-1])
    assert summary["failure"] is None and summary["unknown"] == 0 and summary["cap_percent"] == 5


def test_report_cli_judges_an_event_log(tmp_path, capsys):
    from simsteer.testing.fanatec.__main__ import main

    path = tmp_path / "ev.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in [
        {"t": 0.0, "ev": "gain", "value": 500},
        {"t": 1.0, "ev": "play", "effect": 1, "kind": "ConstantForce", "level": 1.0, "start": 1.0, "until": 1.2}]))
    assert main(["report", str(path), "--cap", "5", "--killed-at", "1.0"]) == 0
    assert json.loads(capsys.readouterr().out)["peak_force"] == 0.05
    assert main(["report", str(path), "--cap", "4", "--killed-at", "0.9"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert [f.split(":")[0] for f in out["failures"]] == ["cap", "gain first", "stopped"]
