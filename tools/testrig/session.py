"""Durable session orchestration, with a fakeable Windows boundary."""

import ntpath
import os
from pathlib import Path
import uuid
import sys

from . import RigError
from .discovery import GAMES, Installation, discover
from .games import ADAPTERS, expected_paths
from .guard import blocking_prompt, require_free, same_process, utc_now
from .state import Journal, read_json


DEFAULTS = {"track": "magione", "car": "lotus_elise_sc", "layout": "", "hfov": 120.0,
            "width": 1920, "height": 1080, "xinput_slot": 0, "idle_seconds": 120,
            "display": None, "vdd_device": None, "overlay": False, "max_seconds": 1800,
            "profile_source": None, "telemetry_plugin": None}


def norm(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


class Session:
    def __init__(self, game, directory, backend):
        self.game, self.backend = game, backend
        self.journal = Journal(Path(directory))
        self.adapter = ADAPTERS[game]
        self.warnings = set()
        if self.data and self.data["game"] != game:
            raise RigError("This session belongs to a different game; choose another --session")

    @property
    def data(self):
        return self.journal.data

    @property
    def installed(self):
        return Installation.from_dict(self.data["installation"])

    def save(self):
        self.journal.save()

    def status(self, *, live=False):
        if not self.data:
            return {"game": self.game, "phase": "absent"}
        row = {key: self.data.get(key) for key in ("id", "game", "phase", "error", "owned")}
        row["state_file"] = str(self.journal.path)
        row["backups"] = len(self.data.get("backups", []))
        row["restoration"] = self.journal.report()
        row["warnings"] = self.data.get("warnings", [])
        if self.journal.recovery:
            row["recovery"] = self.journal.recovery
        receipt = self.journal.directory / "display-state.json"
        if receipt.exists():
            row["display"] = read_json(receipt)
        if live:
            processes = self.backend.snapshot()["processes"]
            row["observed_owned_pids"] = [item["pid"] for item in processes
                                          if any(same_process(item, owned) for owned in self.data["owned"])]
            row["telemetry_sample"] = self.backend.telemetry(self.game)
        return row

    def ensure_steam(self, snapshot):
        executable = norm(self.installed.steam / "steam.exe")
        if not any(norm(item.get("path") or "") == executable for item in snapshot["processes"]):
            raise RigError("Steam is not running; start it and sign in manually before a test")

    def check_options(self, supplied):
        for key, value in supplied.items():
            if self.data["options"].get(key) != value:
                raise RigError(f"Session option {key} differs from its saved value; restore and use a new --session")

    def guard(self, snapshot, *, check_idle=True):
        notes = require_free(snapshot, self.data["owned"] if self.data else (),
                             self.data["options"]["idle_seconds"] if self.data else self.pending_idle,
                             check_idle=check_idle)
        for note in notes:
            if note not in self.warnings:
                print("testrig: warning: " + note, file=sys.stderr)
                self.warnings.add(note)
        if self.data:
            self.data["warnings"] = sorted(set(self.data.get("warnings", [])) | self.warnings)

    def prepare(self, supplied):
        if self.data:
            self.check_options(supplied)
            if self.data["phase"] in ("stopped", "restored"):
                raise RigError("Session finished; use a new --session")
            if self.data["phase"] in ("prepared", "starting", "running"):
                return self.status()
            raise RigError("Interrupted preparation; run restore before creating another session")
        options = DEFAULTS | supplied
        if self.journal.directory.exists() and any(self.journal.directory.iterdir()):
            raise RigError("Session directory contains files without a journal; retain them for recovery "
                           "and choose a new --session")
        if options["display"] == "vdd":
            if not options["vdd_device"]:
                raise RigError("--display vdd requires the exact --vdd-device virtual adapter instance ID")
            if any(char in options["vdd_device"] for char in "*?[]"):
                raise RigError("--vdd-device must select one exact instance ID without wildcards")
            options["overlay"] = True
        snapshot = self.backend.snapshot()
        self.pending_idle = options["idle_seconds"]
        self.guard(snapshot)
        installed = discover(self.game, self.backend)
        if options.get("profile_source"):
            source = Path(options["profile_source"]).absolute()
            if self.journal.directory == source or self.journal.directory in source.parents:
                raise RigError("The ETS2 source profile must be outside this session directory")
        self.adapter.validate(installed, options)
        self.journal.data = {"version": Journal.VERSION, "id": uuid.uuid4().hex, "game": self.game,
                             "phase": "backing-up", "created_at": utc_now(), "options": options,
                             "installation": installed.as_dict(), "owned": [], "backups": []}
        self.ensure_steam(snapshot)
        self.save()
        self.journal.backup(self.adapter.targets(installed, options))
        # Backups can take time. Repeat the owner/restart gate before any edits.
        snapshot = self.backend.snapshot()
        self.guard(snapshot)
        self.ensure_steam(snapshot)
        self.data["phase"] = "preparing"
        self.save()
        self.adapter.prepare(installed, self.journal.directory, options, self.journal)
        self.data["phase"] = "prepared"
        self.save()
        return self.status()

    def adopt(self, snapshot):
        launch = self.data.get("launch")
        if not launch:
            return
        expected = {norm(path) for path in expected_paths(self.game, self.installed)}
        rows = snapshot["processes"]
        changed = False
        # Repeat to attach a launcher and its child even if CIM returned children first.
        for _ in range(3):
            owned_parents = {item["pid"]: item for item in self.data["owned"]
                             if any(same_process(item, row) for row in rows)}
            parents = set(owned_parents)
            parents.update(item["pid"] for item in launch["steam"]
                           if any(same_process(item, row) for row in rows))
            if launch.get("direct_parent"):
                parents.add(launch["direct_parent"])
            for process in rows:
                parent = owned_parents.get(process.get("parent"))
                helper = bool(parent and process.get("born", 0) >= parent.get("born", 0))
                if not helper and norm(process.get("path") or "") not in expected:
                    continue
                if any(same_process(process, item) for item in self.data["owned"] + launch["baseline"]):
                    continue
                if (not process.get("created") or not process.get("path") or process.get("parent") not in parents or
                        not (helper or launch["began"] <= process.get("born", 0) <= launch["until"])):
                    continue
                self.data["owned"].append(process.copy())
                changed = True
        if changed:
            self.save()  # ownership commits BEFORE argument validation/cleanup

    def start(self, supplied, timeout):
        if not self.data:
            self.prepare(supplied)
        else:
            self.check_options(supplied)
        if self.data["phase"] == "running":
            return self.ready(timeout)
        if self.data["phase"] != "prepared":
            raise RigError("Session is not prepared; restore it and use a new --session")
        options = self.data["options"]
        snapshot = self.backend.snapshot()
        self.guard(snapshot)
        self.ensure_steam(snapshot)
        self.data["phase"] = "starting"
        self.data["display_requested"] = bool(options["display"] or options["overlay"])
        self.data["worker_requested"] = True
        self.save()
        self.backend.start_display(self.journal, options)
        # The topology/overlay operation may have taken seconds. Check directly
        # before asking Steam to launch, with no delayed task or automatic retry.
        snapshot = self.backend.snapshot()
        self.guard(snapshot)
        self.ensure_steam(snapshot)
        began = self.backend.wall_time()
        self.data["launch"] = {
            "began": began - 1, "until": began + timeout,
            "baseline": [{key: item.get(key) for key in ("pid", "created", "path")}
                         for item in snapshot["processes"]],
            "steam": [item for item in snapshot["processes"]
                      if norm(item.get("path") or "") == norm(self.installed.steam / "steam.exe")],
        }
        self.save()  # durable launch intent also supports recovery after a crash
        self.backend.launch_uri(self.adapter.launch_uri(self.installed, self.journal.directory))
        return self.ready(timeout)

    def check_display(self):
        if self.data.get("worker_requested"):
            receipt = read_json(self.journal.directory / "display-state.json")
            if receipt.get("session") != self.data["id"] or not receipt.get("ready"):
                detail = "; ".join(receipt.get("errors", []) + receipt.get("warnings", []))
                raise RigError("Session worker is not ready: " + (detail or receipt.get("stop_reason") or "worker exited"))

    def ready(self, timeout):
        if not self.data or self.data["phase"] not in ("starting", "running"):
            raise RigError("No started session; run start first")
        deadline = self.backend.monotonic() + timeout
        previous, advances = None, 0
        waiting = "game process"
        while self.backend.monotonic() < deadline:
            snapshot = self.backend.snapshot()
            self.adopt(snapshot)
            owned = [row for row in snapshot["processes"]
                     if any(same_process(row, item) for item in self.data["owned"])]
            self.guard(snapshot, check_idle=False)
            windows = self.backend.windows()
            prompt = blocking_prompt(windows)
            if prompt:
                raise RigError(prompt)
            self.check_display()
            for process in owned:
                if norm(process["path"]) in {norm(path) for path in expected_paths(self.game, self.installed)}:
                    self.adapter.verify_process(process, self.installed, self.journal.directory, self.backend)
            runner = norm(self.installed.game / GAMES[self.game].runner)
            running = [row for row in owned if norm(row["path"]) == runner]
            visible = any(window["pid"] in {row["pid"] for row in running} for window in windows)
            if self.data["options"]["display"] == "vdd":
                placed = self.data.setdefault("placed_windows", [])
                for window in windows:
                    process = next((row for row in running if row["pid"] == window["pid"]), None)
                    identity = [window["pid"], window["hwnd"]]
                    if process and identity not in placed:
                        self.backend.place_window(window, process, self.data["options"]["width"],
                                                  self.data["options"]["height"])
                        placed.append(identity)
                        self.save()
            if self.game == "ac" and not running and not self.data["launch"].get("direct_requested"):
                launcher_ids = {row["pid"] for row in owned}
                if any(window["pid"] in launcher_ids for window in windows):
                    # Direct simulation start is the established AC practice path.
                    self.guard(snapshot)
                    self.data["launch"].update(direct_requested=True, direct_parent=os.getpid())
                    self.save()
                    self.backend.launch_ac(self.installed.game / "acs.exe")
            token = self.backend.telemetry(self.game) if running and visible else None
            if token and previous and all(a != b for a, b in zip(token, previous)):
                advances += 1
            elif not token or token == previous:
                advances = 0
            previous = token
            scenario = self.adapter.scenario_ready(self.installed, self.journal.directory)
            if running and visible and advances >= 2 and scenario:
                self.data["phase"] = "running"
                self.data["ready_at"] = utc_now()
                self.save()
                return self.status()
            waiting = ("game process" if not running else "visible game window" if not visible else
                       "live changing telemetry (unpaused practice/truck required)" if advances < 2 else
                       "local test profile and repository camera mod in the ETS2 game log")
            self.backend.sleep(0.5)
        raise RigError(f"Ready timed out after {timeout}s waiting for {waiting}; "
                       "check for a sign-in screen, EULA, firewall prompt or incompatible test seed")

    def _stop(self):
        if not self.data or self.data["phase"] == "restored":
            return self.status()
        errors = []
        if self.data.get("worker_requested"):
            (self.journal.directory / "display-stop").touch()
        try:
            snapshot = self.backend.snapshot()
            self.adopt(snapshot)
            for process in reversed(self.data["owned"]):
                if not any(same_process(process, row) for row in snapshot["processes"]):
                    continue
                try:
                    self.backend.stop_process(process)
                except Exception as exc:
                    errors.append(str(exc))
            # An intent may still be in flight in Steam. Do not restore into its
            # eventual writer; cancel/clear the Steam launch manually if necessary.
            snapshot = self.backend.snapshot()
            expected = {norm(path) for path in expected_paths(self.game, self.installed)}
            if any(norm(row.get("path") or "") in expected for row in snapshot["processes"]):
                errors.append("Game/launcher is still running; settings restoration refused")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            if self.data.get("worker_requested"):
                try:
                    self.backend.stop_display(self.journal)
                except Exception as exc:
                    errors.append("Display cleanup: " + str(exc))
        if errors:
            self.data["phase"] = "cleanup-required"
            self.data["error"] = "; ".join(errors)
            self.save()
            raise RigError(self.data["error"])
        self.data["phase"] = "stopped"
        self.save()
        return self.status()

    def stop(self):
        # Ending a session also restores its config and display arrangement.
        # restore remains an explicit recovery alias after a crash.
        return self.restore()

    def _restore(self):
        if not self.data or self.data["phase"] == "restored":
            return self.status()
        stop_error = None
        try:
            self._stop()
        except Exception as exc:
            stop_error = exc
        # A display failure must not skip config recovery; remaining game writers
        # still veto it. A PID reuse never grants permission to kill or restore.
        snapshot = self.backend.snapshot()
        names = {Path(item).name.lower() for item in GAMES[self.game].executables}
        if any(row["name"].lower() in names for row in snapshot["processes"]):
            raise RigError("A game/launcher is still running; restore is refused until it exits")
        if self.data.get("launch") and not self.data["owned"]:
            raise RigError("Steam launch ownership is unresolved; clear any pending Steam launch, then "
                           "use restore --launch-cancelled after confirming no launch can still start")
        self.data["phase"] = "restoring"
        self.save()
        if self.data.get("backups"):
            self.backend.sleep(2)
            self.journal.restore()
            self.backend.sleep(2)
            self.journal.verify_restored()
        if stop_error:
            self.data["phase"] = "cleanup-required"
            self.save()
            raise stop_error
        self.journal.purge_backups()
        self.data["phase"] = "restored"
        self.data.pop("error", None)
        self.save()
        return self.status()

    def restore(self):
        try:
            return self._restore()
        except (OSError, RigError) as exc:
            if self.data:
                self.data.update(phase="cleanup-required", error=str(exc))
                try:
                    self.save()
                except OSError:
                    pass
            raise RigError(f"Restore requires cleanup: {exc}") from exc

    def failure_cleanup(self, error):
        if not self.data or self.data["phase"] == "restored":
            return
        self.data["error"] = str(error)
        try:
            self.save()
        except OSError as state_error:
            error.add_note("Could not save failure state: " + str(state_error))
        try:
            self.restore()
        except Exception as cleanup:
            self.data["phase"] = "cleanup-required"
            self.data["error"] = f"{error}; cleanup: {cleanup}"
            try:
                self.save()
            except OSError as state_error:
                error.add_note("Could not save cleanup state: " + str(state_error))
            error.add_note("Cleanup requires restore: " + str(cleanup))
