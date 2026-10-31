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
from .state import Journal
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
                  clock=time.monotonic, pause=time.sleep):
    """GUI loop uses lightweight handles, never PowerShell/CIM inventory."""
    handles = {}
    start = clock()
    try:
        while clock() - start < state["options"]["max_seconds"]:
            if state["options"]["overlay"]:
                lifecycle.overlay.pump()
            if activity.active():
                return "owner-input", True
            current = read_json(directory / "state.json")
            if current["id"] != state["id"]:
                return "session-changed", False
            if (directory / "display-stop").exists():
                return "explicit-stop", True
            if current["phase"] in ("stopped", "restoring", "restored", "cleanup-required"):
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
        lifecycle.start()
        reason, recover = watch_session(directory, state, parent, lifecycle, activity)
        lifecycle.row["stop_reason"] = reason
        if reason == "deadline":
            lifecycle.row["warnings"].append("Session deadline expired")
    except BaseException as exc:
        lifecycle.row["errors"].append(str(exc))
        recover = True
    finally:
        try:
            supervisor_alive = parent.alive()
            lifecycle.close()
        finally:
            parent.close()
            if activity:
                activity.close()
    if recover and not supervisor_alive:
        recover_session(directory, backend, lifecycle)
    return int(bool(lifecycle.row["errors"]))


def recover_session(directory, backend, lifecycle):
    """A worker failure after start returned must also recover game/config state."""
    from .session import Session
    from .state import lock
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args(argv)
    return worker(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
