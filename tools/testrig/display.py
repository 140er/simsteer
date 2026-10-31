"""Independent display worker, with a persisted topology receipt for recovery."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import ntpath

from . import RigError
from .display_api import BlackOverlay, NativeDisplay
from .guard import same_process
from .state import atomic_json, read_json
from .state import Journal, lock, reject_links, monitor_covers, valid_time
from .discovery import GAMES
from .windows import ProcessHandle, Windows


class DisplayLifecycle:
    """Every mutation has a saved intent; cleanup steps cannot skip one another."""
    def __init__(self, api, overlay, save, options, session, receipt=None):
        self.api, self.overlay, self.save, self.options = api, overlay, save, options
        self.row = receipt or {"session": session, "ready": False, "errors": [],
                               "original": None, "enable_intent": False, "topology_intent": False,
                               "overlay_closed": False, "vdd_disabled": False, "topology_restored": False}
        self.row.setdefault("warnings", [])

    def start(self):
        try:
            if self.options["display"] or self.options["overlay"]:
                self.row["original"] = self.api.capture()
            self.save(self.row)  # original paths/modes/primary BEFORE enabling VDD
            if self.options["overlay"]:
                self.overlay.keep_awake()
            if self.options["display"] == "vdd":
                instance = self.options["vdd_device"]
                if self.api.device_state(instance) != 22:
                    raise RigError("VDD must be disabled before the session; an already enabled device is refused")
                # Enabling alone can auto-extend/change topology. Both intents
                # must commit before Enable-PnpDevice, including partial failure.
                self.row.update(enable_intent=True, topology_intent=True)
                self.save(self.row)
                self.api.enable(instance)
                self.row["virtual"] = self.api.extend_primary(instance, self.row["original"],
                                                              self.options["width"], self.options["height"])
                self.save(self.row)
            if self.options["overlay"]:
                rectangles = self.api.physical_rectangles(self.row["original"])
                self.overlay.start(rectangles)
            self.row["ready"] = True
            self.save(self.row)
        except BaseException as exc:
            self.row["errors"].append(str(exc))
            self.close()
            raise

    def close(self):
        self.row["ready"] = False
        # Restore the physical primary first. Even a topology failure must not
        # leave an enabled VDD or live cover behind. Retry exact topology once
        # after disabling in case a failing virtual target blocked restoration.
        operations = []
        if self.row["topology_intent"] and self.row["original"] and not self.row["topology_restored"]:
            operations.append(("topology_restored", lambda: self.api.restore(self.row["original"])))
        elif not self.row["topology_intent"]:
            self.row["topology_restored"] = True
        if self.row["enable_intent"] and not self.row["vdd_disabled"]:
            operations.append(("vdd_disabled", lambda: self.api.disable(self.options["vdd_device"])))
        elif not self.row["enable_intent"]:
            self.row["vdd_disabled"] = True
        if not self.row["overlay_closed"]:
            operations.append(("overlay_closed", self.overlay.close))
        for label, operation in operations:
            try:
                operation()
                self.row[label] = True
            except BaseException as exc:
                if label == "topology_restored":
                    self.row["warnings"].append(label + ": " + str(exc))
                    try:
                        self.api.physical_primary(self.row["original"])
                    except BaseException as fallback:
                        self.row["warnings"].append("physical primary fallback: " + str(fallback))
                else:
                    self.row["errors"].append(label + ": " + str(exc))
        if not self.row["topology_restored"] and self.row["original"]:
            try:
                self.api.restore_physical(self.row["original"])
                self.row["topology_restored"] = True
                self.row["topology_fallback"] = True
                self.row["warnings"].append("Saved display IDs unavailable; verified physical modes/primary restored; " +
                                            ("VDD disabled" if self.row["vdd_disabled"] else "VDD disable still requires recovery"))
            except BaseException as exc:
                self.row["errors"].append("physical topology recovery: " + str(exc))
        self.save(self.row)


def start_worker(journal, options, backend):
    directory = journal.directory
    receipt = directory / "display-state.json"
    for name in ("display-stop", "display-go", "display-state.json"):
        (directory / name).unlink(missing_ok=True)
    parent = next((row for row in backend.snapshot()["processes"] if row["pid"] == os.getpid()), None)
    if parent is None or not parent.get("path") or not parent.get("created"):
        raise RigError("Cannot establish display worker supervisor identity")
    journal.data["display_parent"] = parent
    journal.save()
    process = subprocess.Popen([sys.executable, "-m", "tools.testrig.display", "--directory", str(directory)],
                               cwd=Path(__file__).resolve().parents[2],
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    deadline = backend.monotonic() + 15
    while backend.monotonic() < deadline:
        child = next((row for row in backend.snapshot()["processes"] if row["pid"] == process.pid), None)
        if child:
            journal.data["display_worker"] = child
            journal.save()
            (directory / "display-go").touch()
            break
        if process.poll() is not None:
            raise RigError("Display worker exited before initialization")
        backend.sleep(0.1)
    else:
        (directory / "display-stop").touch()
        raise RigError("Display worker identity timed out; no display changes were authorized")
    deadline = backend.monotonic() + 150
    while backend.monotonic() < deadline:
        if receipt.exists():
            row = read_json(receipt)
            if row.get("session") == journal.data["id"] and row.get("ready"):
                return
        if process.poll() is not None:
            detail = read_json(receipt).get("errors", []) if receipt.exists() else ["no receipt"]
            raise RigError("Display worker failed: " + "; ".join(detail))
        backend.sleep(0.1)
    raise RigError("Display worker readiness timed out")


def stop_worker(journal, backend):
    directory, data = journal.directory, journal.data
    (directory / "display-stop").touch()
    worker = data.get("display_worker")
    deadline = backend.monotonic() + 150
    while worker and worker["pid"] != os.getpid() and backend.monotonic() < deadline:
        if not any(same_process(worker, row) for row in backend.snapshot()["processes"]):
            break
        backend.sleep(0.2)
    else:
        if worker and worker["pid"] != os.getpid():
            raise RigError("Display worker is still cleaning up; keep its state and retry restore")
    # If the worker was forcibly killed, its PowerShell PnP call may still be
    # running. Wait for those children before restoring; never race a late enable.
    while worker and backend.monotonic() < deadline:
        children = [row for row in backend.snapshot()["processes"]
                    if row.get("parent") == worker["pid"] and row.get("name", "").lower() == "powershell.exe"
                    and row.get("born", 0) >= worker.get("born", 0)]
        if not children:
            break
        backend.sleep(0.2)
    else:
        if worker:
            raise RigError("Display PowerShell operation is still running; retain state and retry restore")
    path = directory / "display-state.json"
    if not path.exists():
        # The go-file handshake prevents any mutation without a recoverable receipt.
        return
    receipt = read_json(path)
    if receipt.get("session") != data["id"]:
        raise RigError("Display receipt belongs to another session; cleanup refused")
    if not (worker and worker["pid"] == os.getpid() and getattr(backend, "owner_monitor", None)):
        if not monitor_covers(receipt, receipt.get("monitor_covered_until")):
            receipt["owner_input_seen"] = True
            receipt.setdefault("warnings", []).append("Owner monitor ended without proven coverage; game rewrites require conflict review")
            atomic_json(path, receipt)
    if not all(receipt.get(key) for key in ("topology_restored", "vdd_disabled", "overlay_closed")):
        # A killed worker loses its windows automatically. Recover the topology
        # and only the explicitly selected virtual device from the saved intent.
        lifecycle = DisplayLifecycle(NativeDisplay(), BlackOverlay(), lambda row: atomic_json(path, row),
                                     data["options"], data["id"], receipt)
        lifecycle.close()
        receipt = lifecycle.row
    if not all(receipt.get(key) for key in ("topology_restored", "vdd_disabled", "overlay_closed")):
        raise RigError("Display cleanup could not be verified: " + "; ".join(receipt["errors"]))


def watch_session(directory, state, parent, lifecycle, activity, *, open_process=ProcessHandle,
                  clock=time.monotonic, pause=time.sleep, wall=time.time):
    """GUI loop uses lightweight handles, never PowerShell/CIM inventory."""
    handles = {}
    start = clock()
    try:
        while clock() - start < state["options"]["max_seconds"]:
            if state["options"]["overlay"]:
                lifecycle.overlay.pump()
            if activity.active():
                lifecycle.row["owner_input_seen"] = True
                if getattr(activity, "failure", None):
                    lifecycle.row["warnings"].append(activity.failure)
                lifecycle.save(lifecycle.row)
                return "owner-input", True
            current = read_json(directory / "state.json")
            if current["id"] != state["id"]:
                return "session-changed", False
            if (directory / "display-stop").exists():
                return "explicit-stop", True
            if current["phase"] in ("stopped", "restoring", "restored", "restored-with-conflicts", "cleanup-required"):
                return "cleanup", False
            if current["phase"] == "running":
                executable = ntpath.normcase(str(Path(current["installation"]["game"]) / GAMES[current["game"]].runner))
                runners = [row for row in current["owned"] if ntpath.normcase(row["path"]) == executable]
                active = False
                for row in runners:
                    key = (row["pid"], row["created"], row["path"])
                    if key not in handles:
                        handles[key] = open_process(row)
                    active |= handles[key].alive()
                if not active:
                    lifecycle.row["game_exited_at"] = wall()
                    lifecycle.save(lifecycle.row)
                    return "game-exited", True
            elif not parent.alive():
                return "startup-caller-exited", True
            pause(0.1)
        return "deadline", True
    finally:
        for handle in handles.values():
            handle.close()


def worker(directory):
    from .activity import OwnerInput
    state = Journal(directory).data
    backend = Windows()
    parent = ProcessHandle(state["display_parent"])
    receipt = directory / "display-state.json"
    lifecycle = DisplayLifecycle(NativeDisplay(), BlackOverlay(), lambda row: atomic_json(receipt, row),
                                 state["options"], state["id"])
    start = time.monotonic()
    recover = False
    supervisor_alive = True
    activity = None
    try:
        while not (directory / "display-go").exists():
            if not parent.alive() or (directory / "display-stop").exists() or time.monotonic() - start > 30:
                raise RigError("Display startup was cancelled before the mutation handshake")
            time.sleep(0.1)
        activity = OwnerInput()
        backend.owner_monitor = activity
        lifecycle.row["monitor_started_at"] = backend.wall_time()
        lifecycle.save(lifecycle.row)
        lifecycle.start()
        reason, recover = watch_session(directory, state, parent, lifecycle, activity)
        lifecycle.row["stop_reason"] = reason
        lifecycle.row["monitor_covered_until"] = backend.wall_time()
        lifecycle.save(lifecycle.row)
        if reason == "deadline":
            lifecycle.row["warnings"].append("Session deadline expired")
    except BaseException as exc:
        lifecycle.row["errors"].append(str(exc))
        # An unreadable/dead input monitor cannot authorize rolling back a
        # rewrite whose author is unknown.
        lifecycle.row["owner_input_seen"] = True
        recover = True
    finally:
        try:
            supervisor_alive = parent.alive()
            lifecycle.close()
        finally:
            parent.close()
    try:
        if activity:
            try:
                lifecycle.row["owner_input_seen"] = bool(lifecycle.row.get("owner_input_seen") or activity.active())
            except RigError:
                lifecycle.row["owner_input_seen"] = True
            lifecycle.row["monitor_covered_until"] = backend.wall_time()
            lifecycle.save(lifecycle.row)
        if recover and not supervisor_alive:
            recover_session(directory, backend, lifecycle)
    finally:
        if activity:
            try:
                lifecycle.row["owner_input_seen"] = bool(lifecycle.row.get("owner_input_seen") or activity.active())
                lifecycle.row["monitor_covered_until"] = backend.wall_time()
                lifecycle.row["monitor_stopped_at"] = backend.wall_time()
                lifecycle.save(lifecycle.row)
            finally:
                activity.close()
    return int(bool(lifecycle.row["errors"]))


def recover_session(directory, backend, lifecycle):
    """A worker failure after start returned must also recover game/config state."""
    from .session import Session
    try:
        with lock(directory.parents[1]):
            state = Journal(directory).data
            if state["id"] != lifecycle.row["session"]:
                raise RigError("Session changed before autonomous display-failure recovery")
            session = Session(state["game"], directory, backend)
            session.restore()
    except Exception as exc:
        if "Another test-rig command is running" in str(exc):
            # start/ready/stop holds the command lock and observes our closed
            # receipt; that caller performs recovery once this worker exits.
            lifecycle.row["warnings"].append("Cleanup is being handled by the active test-rig command")
            lifecycle.save(lifecycle.row)
            return
        lifecycle.row["errors"].append("Game/config recovery requires restore: " + str(exc))
        lifecycle.save(lifecycle.row)


def session_alive(data, snapshot):
    recorded = data.get("owned", []) + ([data["display_worker"]] if data.get("display_worker") else [])
    return any(same_process(owned, row) for owned in recorded for row in snapshot["processes"])


def previous_boot(data, snapshot):
    """Prefer the boot counter; time-based fallback requires uptime discontinuity."""
    saved, current = data.get("boot_id"), snapshot.get("boot_id")
    if not saved or not current or saved == current:
        return False
    if all(isinstance(value, str) and value.startswith("counter:") and value[8:].isdigit() for value in (saved, current)):
        return int(saved[8:]) != int(current[8:])
    try:
        before = int(data.get("boot_time") or str(saved).removeprefix("time:"))
        after = int(snapshot.get("boot_time") or str(current).removeprefix("time:"))
    except (TypeError, ValueError):
        return False
    if abs(after - before) <= 300 * 10_000_000:
        return False  # LastBootUpTime can shift after time synchronization
    old_up, new_up = data.get("boot_uptime"), snapshot.get("boot_uptime")
    old_at, new_at = data.get("boot_observed_at"), snapshot.get("boot_observed_at")
    if not all(valid_time(value) for value in (old_up, new_up, old_at, new_at)):
        return False  # insufficient evidence: leave the display for explicit restore
    return new_up < old_up - 1 or abs((new_at - old_at) - (new_up - old_up)) > 300


def recover_leftover_displays(root, backend, snapshot):
    """Before a new command, recover incomplete display intents from older boots."""
    if not root or not root.exists() or not snapshot.get("boot_id"):
        return []
    pending = []
    for game in ("ac", "ets2"):
        parent = root / game
        reject_links(parent)
        if not parent.exists():
            continue
        for directory in parent.iterdir():
            reject_links(directory)
            if not directory.is_dir() or not (directory / "display-state.json").exists():
                continue
            journal = Journal(directory)
            if not journal.data or session_alive(journal.data, snapshot) or not previous_boot(journal.data, snapshot):
                continue
            receipt = read_json(directory / "display-state.json")
            if receipt.get("session") != journal.data["id"]:
                raise RigError("Previous-boot display receipt belongs to another session; restore manually")
            if receipt.get("enable_intent") and not all(receipt.get(key) for key in
                    ("topology_restored", "vdd_disabled", "overlay_closed")):
                pending.append(directory)
    recovered = []
    if pending:
        with lock(root):
            for directory in pending:
                journal = Journal(directory)
                current = backend.snapshot()
                if session_alive(journal.data, current) or not previous_boot(journal.data, current):
                    continue  # recheck identities under the command lock, before any mutation
                backend.stop_display(journal)
                journal.data.setdefault("warnings", []).append("Recovered an incomplete display layout from a previous boot")
                journal.save()
                recovered.append(str(directory))
    return recovered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args(argv)
    return worker(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
