"""Write-ahead backups and crash-recoverable, hash-verified restoration."""

from contextlib import contextmanager
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import uuid

from . import RigError


def retry_sharing(operation):
    """Bound Windows sharing/delete-pending races to roughly two seconds."""
    for attempt in range(40):
        try:
            return operation()
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)) or attempt == 39:
                raise
            time.sleep(0.05)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        retry_sharing(lambda: os.replace(temporary, path))
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    try:
        return json.loads(retry_sharing(lambda: Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise RigError(f"Cannot read session state {path}: {exc}") from exc


@contextmanager
def lock(root: Path):
    """An OS lock releases on crash; no stale PID/lock-file guessing."""
    reject_links(root)
    root.mkdir(parents=True, exist_ok=True)
    reject_links(root / ".lock")
    fd = os.open(root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+b") as stream:
        stream.seek(0)
        stream.write(b"\0")
        stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RigError("Another test-rig command is running; retry after it finishes") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def reject_links(path: Path):
    # Python 3.11 has no Path.is_junction(). Windows junctions and other reparse
    # points must be rejected too, including ancestors of a backup/restore target.
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise RigError(f"Refusing a symlink/junction/reparse point: {part}")


def digest(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fingerprint(path: Path):
    reject_links(path)
    if not path.exists():
        return {"kind": "absent"}
    if path.is_file():
        return {"kind": "file", "sha256": digest(path)}
    if not path.is_dir():
        raise RigError(f"Unsupported backup entry: {path}")
    files, directories = {}, []
    for folder, dirs, names in os.walk(path, followlinks=False):
        for name in sorted(dirs + names):
            item = Path(folder) / name
            reject_links(item)
            relative = item.relative_to(path).as_posix()
            if item.is_dir():
                directories.append(relative)
            elif item.is_file():
                files[relative] = digest(item)
            else:
                raise RigError(f"Unsupported backup entry: {item}")
    return {"kind": "directory", "files": files, "directories": sorted(directories)}


def remove(path):
    reject_links(path)
    if path.is_dir():
        raise RigError(f"Directory deletion is forbidden: {path}")
    retry_sharing(lambda: path.unlink(missing_ok=True))


def file_fingerprint(path):
    reject_links(path)
    return {"kind": "directory"} if path.is_dir() else fingerprint(path)


def creation_time(path, info=None):
    info = info or Path(path).stat()
    # On supported Python/Windows, st_ctime is the file creation time.
    return getattr(info, "st_birthtime", info.st_ctime)


def entry_metadata(path):
    reject_links(path)
    info = path.stat()
    row = {"created_at": creation_time(path, info), "identity": [info.st_dev, info.st_ino],
           "mode": stat.S_IMODE(info.st_mode), "attributes": getattr(info, "st_file_attributes", 0)}
    if path.is_file():
        row.update(size=info.st_size, mtime_ns=info.st_mtime_ns)
    return row


def tree_listing(root):
    listing = fingerprint(root)  # rejects reparse points at every entry
    if listing["kind"] != "directory":
        raise OwnerChanged("Session-created Documents root is no longer a directory")
    entries = {name: entry_metadata(root / name) for name in
               [".", *listing["directories"], *listing["files"]]}
    if fingerprint(root) != listing:
        raise OwnerChanged("Documents tree changed while listing; left as is")
    return {"fingerprint": listing, "entries": entries}


class OwnerChanged(RigError):
    pass


class RestoreConflicts(RigError):
    """Recovery completed, but owner changes need an explicit review."""


def valid_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def monitor_covers(receipt, ended_at):
    """A recovered display receipt alone is not proof of owner monitoring."""
    started, covered = receipt.get("monitor_started_at"), receipt.get("monitor_covered_until")
    stopped = receipt.get("monitor_stopped_at")
    return (valid_time(started) and valid_time(covered) and valid_time(ended_at) and
            started <= ended_at <= covered and receipt.get("errors") == [] and
            receipt.get("stop_reason") in ("explicit-stop", "game-exited", "deadline", "owner-input",
                                           "startup-caller-exited", "cleanup") and
            (receipt.get("monitor_live") is True or (valid_time(stopped) and ended_at <= stopped)))


class Journal:
    VERSION = 2

    def __init__(self, directory: Path):
        reject_links(directory)
        self.directory = directory
        self.path = directory / "state.json"
        self.manifest_path = directory / "manifest.json"
        self.recovery = None
        self.data = None
        candidates = []
        for path in (self.path, self.manifest_path):
            if not path.exists():
                continue
            try:
                data = read_json(path)
                if not isinstance(data, dict) or data.get("version") != self.VERSION:
                    raise RigError("Unsupported/legacy journal; do not restore folder backups automatically")
                if (not isinstance(data.get("revision", 0), int) or data.get("revision", 0) < 0 or
                        not isinstance(data.get("backups", []), list) or
                        any(not isinstance(row, dict) for row in data.get("backups", []))):
                    raise RigError("Invalid journal revision or per-file records")
                if any(not isinstance(row.get("baseline"), dict) or row["baseline"].get("kind") not in ("file", "absent") for row in data.get("backups", [])):
                    raise RigError("Folder backups cannot be restored automatically")
                candidates.append((data.get("revision", 0), path, data))
            except RigError as exc:
                self.recovery = {"source": str(path.name), "message": str(exc)}
        if candidates:
            _, path, self.data = max(candidates, key=lambda item: (item[0], item[1] == self.path))
            self.recovery = ({"source": "manifest.json", "message": "State unavailable or older; recovered from the per-file manifest"}
                             if path == self.manifest_path else None)
        if self.data is None and self.recovery:
            raise RigError("State and recovery manifest are unreadable or legacy. Preserve backup/ and displaced/; "
                           "recover individual files using the README manual recovery recipe. " + self.recovery["message"])
        if self.data:
            self.validate_paths()

    def save(self):
        # The independent manifest also carries launch/display identities, so a
        # corrupt state file never requires guessing which processes we own.
        self.data["revision"] = self.data.get("revision", 0) + 1
        atomic_json(self.manifest_path, self.data)
        atomic_json(self.path, self.data)

    def backup(self, targets, *, generated=False):
        rows = self.data.setdefault("backups", [])
        for target in targets:
            target = Path(target).absolute()
            inside = self.directory in target.parents
            if target == self.directory or target in self.directory.parents or (inside and not generated):
                raise RigError("Backup targets must be outside the session directory")
            if generated and (self.directory / "ets2-home") not in target.parents:
                raise RigError("Generated files must be inside this session's ETS2 home")
            if any(row["path"] == str(target) for row in rows):
                continue
            baseline = file_fingerprint(target)
            if baseline["kind"] not in ("file", "absent"):
                raise RigError(f"Only individual edited files may be backed up: {target}")
            if generated and baseline["kind"] != "absent":
                raise RigError(f"Untracked generated file already exists; left as is: {target}")
            stored = None
            mode = None
            if baseline["kind"] == "file":
                stored = self.directory / "backup" / f"{len(rows)}-{uuid.uuid4().hex}.bin"
                stored.parent.mkdir(parents=True, exist_ok=True)
                mode = stat.S_IMODE(target.stat().st_mode)
                retry_sharing(lambda: shutil.copyfile(target, stored))
            if fingerprint(target) != baseline or (stored and fingerprint(stored) != baseline):
                raise RigError(f"Backup changed while copying: {target}; no settings were edited")
            rows.append({"path": str(target), "stored": str(stored) if stored else None,
                         "baseline": baseline, "mode": mode, "written": None})
            self.save()
        self.data["backups_complete"] = True
        self.save()

    def storage_path(self, row, key):
        value = row.get(key)
        if not value:
            return None
        folder = "backup" if key == "stored" else "displaced"
        path = Path(value)
        reject_links(path)
        if not path.is_absolute() or path.parent.resolve() != (self.directory / folder).resolve():
            raise RigError(f"Manifest {key} path must be inside this session's {folder}/: {path}")
        return path

    def validate_paths(self):
        for row in self.data.get("backups", []):
            self.storage_path(row, "stored")
            self.storage_path(row, "archive")
            if row.get("temporary"):
                self.temporary_path(row)
        for row in self.data.get("created_roots", []):
            if (self.data.get("game") != "ac" or row.get("created_by_session") is not True or
                    row.get("path") != self.data.get("installation", {}).get("documents")):
                raise RigError("Created-root cleanup is restricted to the recorded AC Documents root")

    def create_documents_root(self, root, recorded_at):
        """Only an absent AC root is eligible; exclusive mkdir proves our directories."""
        root = Path(root).absolute()
        if fingerprint(root)["kind"] != "absent":
            return
        row = {"path": str(root), "created_by_session": True, "recorded_at": recorded_at,
               "created_directories": {}, "monitor_verified": False}
        self.data.setdefault("created_roots", []).append(row)
        self.validate_paths()
        self.save()  # absence and ownership intent BEFORE either mkdir
        for directory in (root, root / "cfg"):
            reject_links(directory)
            os.mkdir(directory)  # never adopt an existing owner directory
            row["created_directories"][directory.relative_to(root).as_posix()] = entry_metadata(directory)
            self.save()

    def _root_coverage(self, receipt):
        return (not self.data.get("owner_input_seen") and not self.data.get("error") and not self.data.get("failure_seen") and
                not self.data.get("attribution_disabled") and receipt.get("owner_input_seen") is False and
                isinstance(receipt.get("warnings"), list) and
                receipt.get("errors") == [] and not receipt.get("heartbeat_warning") and
                not (receipt.get("input_diagnostics") or {}).get("failure") and
                not any("heartbeat" in str(warning).lower() for warning in receipt.get("warnings", [])) and
                receipt.get("stop_reason") in ("explicit-stop", "game-exited", "deadline") and
                receipt.get("monitor_started_at") == self.data.get("monitor_started_at") and
                monitor_covers(receipt, self.data.get("session_end")))

    def _check_root_window(self, row, listing):
        started, covered = row["coverage"]["monitor_started_at"], row["coverage"]["monitor_covered_until"]
        for name, entry in listing["entries"].items():
            created = entry["created_at"]
            if not valid_time(created) or created > covered:
                raise OwnerChanged(f"Documents entry created outside covered session: {name}")
            if created < started:
                old = (row.get("begin_listing") or {}).get("entries", {}).get(name)
                if (not old or any(entry[key] != old[key] for key in ("created_at", "identity")) or
                        (name not in row["created_directories"] and name not in row["begin_listing"]["fingerprint"]["files"])):
                    raise OwnerChanged(f"Documents entry predates owner-monitor coverage: {name}")

    def _seal_created_roots(self, receipt):
        for row in self.data.get("created_roots", []):
            if row.get("result") or row.get("post_stop"):
                continue
            try:
                if not row.get("monitor_verified") or not self._root_coverage(receipt):
                    raise OwnerChanged("Incomplete clean owner-monitor coverage; Documents tree retained")
                row["coverage"] = {key: receipt.get(key) for key in (
                    "monitor_started_at", "monitor_covered_until", "monitor_stopped_at", "monitor_live",
                    "stop_reason", "owner_input_seen", "errors", "warnings")}
                row["post_stop"] = tree_listing(Path(row["path"]))
                self._check_root_window(row, row["post_stop"])
            except (OSError, RigError) as exc:
                row.update(result="owner_changed", message="Created Documents tree conflict: " + str(exc))

    def _recheck_root(self, row, expected):
        current = tree_listing(Path(row["path"]))
        if current["fingerprint"] != expected or any(
                entry != row["post_stop"]["entries"].get(name) for name, entry in current["entries"].items()):
            raise OwnerChanged("Documents tree changed after its post-stop listing; remaining entries retained")

    def _archive_remove_root(self, row, index):
        root = Path(row["path"])
        if not row.get("monitor_verified") or not row.get("post_stop") or not self._root_coverage(row.get("coverage", {})):
            raise OwnerChanged("Incomplete clean owner-monitor coverage; Documents tree retained")
        expected = row["post_stop"]["fingerprint"]
        self._check_root_window(row, row["post_stop"])
        self._recheck_root(row, expected)
        archive = self.directory / "created-root-archive" / str(index)
        if row.get("archive") and row["archive"] != str(archive):
            raise RigError("Invalid created-root archive path")
        reject_links(archive)
        row["archive"] = str(archive)
        self.save()
        archive.mkdir(parents=True, exist_ok=True)
        for name in expected["directories"]:
            destination = archive / name
            reject_links(destination)
            destination.mkdir(parents=True, exist_ok=True)
        for name, sha in expected["files"].items():
            source, destination = root / name, archive / name
            reject_links(destination)
            wanted = {"kind": "file", "sha256": sha}
            if not destination.exists():
                if fingerprint(source) != wanted:
                    raise OwnerChanged(f"Documents entry changed before archive: {name}")
                retry_sharing(lambda: shutil.copyfile(source, destination))
            if fingerprint(destination) != wanted:
                raise RigError(f"Created-root archive hash mismatch: {name}")
        if fingerprint(archive) != expected:
            raise RigError("Created-root archive hash/listing mismatch")
        self._recheck_root(row, expected)  # ALL copies verified before ANY unlink
        remaining = {**expected, "files": expected["files"].copy(), "directories": expected["directories"].copy()}
        for name, sha in expected["files"].items():
            row["remove_intent"] = name
            self.save()
            # Recheck this exact entry after persisting intent, just before unlink.
            self._recheck_root(row, remaining)
            if fingerprint(archive / name) != {"kind": "file", "sha256": sha}:
                raise RigError(f"Created-root archive hash mismatch before unlink: {name}")
            if (fingerprint(root / name) != {"kind": "file", "sha256": sha} or
                    entry_metadata(root / name) != row["post_stop"]["entries"][name]):
                raise OwnerChanged(f"Documents entry changed before unlink: {name}")
            target = root / name
            reject_links(target)
            mode = stat.S_IMODE(target.stat().st_mode)
            if not mode & stat.S_IWUSR:
                os.chmod(target, mode | stat.S_IWUSR)
            remove(target)
            del remaining["files"][name]
            row.setdefault("removed_entries", []).append(name)
            self.save()
        for name in sorted(expected["directories"], key=lambda value: len(Path(value).parts), reverse=True) + ["."]:
            self._recheck_root(row, remaining)
            directory = root / name
            if fingerprint(directory) != {"kind": "directory", "files": {}, "directories": []}:
                raise OwnerChanged(f"Documents directory is no longer empty: {name}")
            row["remove_intent"] = name
            self.save()
            if entry_metadata(directory) != row["post_stop"]["entries"][name]:
                raise OwnerChanged(f"Documents directory changed before rmdir: {name}")
            reject_links(directory)
            os.rmdir(directory)
            if name != ".":
                remaining["directories"].remove(name)
            row.setdefault("removed_entries", []).append(name)
            self.save()
        row.update(result="removed", remove_intent=None)

    def restore_created_roots(self):
        for index, row in enumerate(self.data.get("created_roots", [])):
            if row.get("result") in ("removed", "owner_changed"):
                continue
            try:
                self._archive_remove_root(row, index)
            except (OSError, RigError) as exc:
                row.update(result="owner_changed", message="Created Documents tree conflict: " + str(exc))
            self.save()

    def _verify_backup(self, row):
        stored = self.storage_path(row, "stored")
        archive = self.storage_path(row, "archive")
        if row.get("archive_pending") or (stored and not stored.exists() and
                (self.directory / "displaced" / stored.name).exists()):
            # Also recover a move made by the earlier journal implementation.
            archive = archive or self.directory / "displaced" / stored.name
            row.update(archive=str(archive), archive_pending=True)
            self._owner_changed(row)
        if row.get("result") in ("restored", "owner_changed", "left_session_file") or row["baseline"]["kind"] == "absent":
            return
        if not stored or fingerprint(stored) != row["baseline"]:
            raise RigError(f"Backup hash mismatch: {stored}; this file requires recovery")

    def verify_backups(self):
        self.validate_paths()
        for row in self.data["backups"]:
            self._verify_backup(row)

    def temporary_path(self, row):
        identifier = self.data.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier):
            raise RigError("Invalid session identity for temporary-file recovery")
        index = next(index for index, item in enumerate(self.data["backups"]) if item is row)
        expected = Path(row["path"]).parent / f".testrig-{identifier}-{index}.tmp"
        path = Path(row.get("temporary") or expected)
        reject_links(path)
        if path != expected:
            raise RigError("Temporary path is not the exact journaled session filename")
        return path

    def cleanup_temporary(self, row):
        if not row.get("temporary"):
            return
        path = self.temporary_path(row)
        if not path.exists():
            return
        allowed = row.get("temporary_hashes", [])
        def unlink():
            if file_fingerprint(path) not in allowed:
                try:
                    content = base64.b64decode(row.get("temporary_content", ""), validate=True)
                except (ValueError, TypeError) as exc:
                    raise RigError("Temporary-file intent is invalid; preserve it for recovery") from exc
                expected = {"kind": "file", "sha256": hashlib.sha256(content).hexdigest()}
                with path.open("rb") as stream:
                    partial = stream.read(len(content) + 1)
                if (expected not in allowed or len(partial) > len(content) or
                        not (content.startswith(partial) or not any(partial))):
                    raise RigError(f"Journaled temporary file changed; preserved for inspection: {path}")
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            path.unlink()
        retry_sharing(unlink)

    def begin_monitor(self, started_at):
        """Validate the unmonitored prepare/start interval before game launch."""
        if not valid_time(started_at):
            self.data["owner_input_seen"] = True
        else:
            self.data["monitor_started_at"] = started_at
            for row in self.data["backups"]:
                if row.get("written") and self.directory not in Path(row["path"]).parents:
                    row["monitor_verified"] = file_fingerprint(Path(row["path"])) == row["written"]
            for root in self.data.get("created_roots", []):
                try:
                    path = Path(root["path"])
                    listing = tree_listing(path)
                    written = {Path(row["path"]).relative_to(path).as_posix(): row["written"]["sha256"]
                               for row in self.data["backups"] if row.get("written") and path in Path(row["path"]).parents}
                    directories = {parent.as_posix() for name in written for parent in Path(name).parents}
                    if (listing["fingerprint"]["files"] != written or
                            set(listing["fingerprint"]["directories"]) | {"."} != directories or
                            set(root["created_directories"]) != directories or
                            any(any(listing["entries"][name][key] != saved[key] for key in ("created_at", "identity"))
                                for name, saved in root["created_directories"].items())):
                        raise OwnerChanged("Unmonitored entries or changed files in the prepare/start gap")
                    root.update(monitor_verified=True, begin_listing=listing)
                except (OSError, RigError) as exc:
                    root.update(result="owner_changed", message="Created Documents tree conflict: " + str(exc))
        self.save()

    def record_session_end(self, ended_at, *, owner_input_seen=False, monitor=None):
        """Seal hashes once, while the owned writers' lifetime is still known."""
        monitor = monitor or {}
        proven = monitor_covers(monitor, ended_at) and self.data.get("monitor_started_at") == monitor.get("monitor_started_at")
        self.data["owner_input_seen"] = bool(self.data.get("owner_input_seen") or owner_input_seen or not proven)
        if self.data.get("session_end"):
            self.save()
            return
        self.data["session_end"] = ended_at
        for row in self.data["backups"]:
            if (self.data["owner_input_seen"] or self.data.get("attribution_disabled") or
                    not row.get("monitor_verified") or row.get("result") or self.directory in Path(row["path"]).parents):
                continue
            path = Path(row["path"])
            try:
                live = file_fingerprint(path)
                if (row.get("written_at_ns") and live["kind"] == "file" and
                        max(row["written_at_ns"], int(monitor["monitor_started_at"] * 1_000_000_000)) <=
                        path.stat().st_mtime_ns <= int(ended_at * 1_000_000_000)):
                    row["left_by_session"] = live
            except (OSError, RigError) as exc:
                self.data.setdefault("warnings", []).append(f"Cannot attribute session write for {path}: {exc}")
        self._seal_created_roots(monitor)
        self.save()

    def _expect(self, path, allowed):
        if file_fingerprint(path) not in allowed:
            raise OwnerChanged(f"owner changed {path}; left as is")

    def _replace(self, path, content, allowed, mode):
        reject_links(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = next(row for row in self.data["backups"] if row["path"] == str(path))
        self.cleanup_temporary(row)
        self.data.setdefault("id", uuid.uuid4().hex)
        temporary = self.temporary_path(row)
        row.update(temporary=str(temporary), temporary_content=base64.b64encode(content).decode("ascii"), temporary_hashes=[
            {"kind": "file", "sha256": hashlib.sha256(value).hexdigest()} for value in (b"", content)])
        self.save()  # deterministic name and expected content BEFORE creation
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, mode)
            def replace():
                self._expect(path, allowed)
                if path.is_file() and not path.stat().st_mode & stat.S_IWUSR:
                    os.chmod(path, stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
                os.replace(temporary, path)
            retry_sharing(replace)
        finally:
            self.cleanup_temporary(row)

    def write(self, path, content):
        """Record the exact bytes BEFORE editing any game file."""
        path = Path(path).absolute()
        row = next((row for row in self.data["backups"] if row["path"] == str(path)), None)
        if row is None:
            if self.directory not in path.parents:
                raise RigError(f"Unjournaled external edit refused: {path}")
            self.backup([path], generated=True)
            row = self.data["backups"][-1]
        previous = row.get("written") or row["baseline"]
        self._expect(path, [previous])
        if row["baseline"]["kind"] == "file" and fingerprint(Path(row["stored"])) != row["baseline"]:
            raise RigError(f"Backup hash mismatch: {row['stored']}")
        self.validate_paths()
        row.update(previous=previous, written={"kind": "file", "sha256": hashlib.sha256(content).hexdigest()},
                   written_at_ns=time.time_ns())
        self.save()
        self._replace(path, content, [previous], (row["mode"] or 0o600) | stat.S_IWUSR)
        row["previous"] = None
        self.save()

    def _owner_changed(self, row):
        if self.directory in Path(row["path"]).parents:
            row.update(result="left_session_file")
            row.pop("message", None)
            row.pop("error", None)
            return
        stored = self.storage_path(row, "stored")
        archive = self.storage_path(row, "archive")
        if stored or row.get("archive_pending"):
            archive = archive or self.directory / "displaced" / stored.name
            row.update(archive=str(archive), archive_pending=True)
            self.save()  # intent commits BEFORE the only copy is moved
            archive.parent.mkdir(parents=True, exist_ok=True)
            if stored and stored.exists():
                if fingerprint(stored) != row["baseline"] or (archive.exists() and fingerprint(archive) != row["baseline"]):
                    raise RigError("Displaced original hash mismatch; preserve both files for recovery")
                retry_sharing(lambda: os.replace(stored, archive))
            if fingerprint(archive) != row["baseline"]:
                raise RigError(f"Displaced original is unavailable: {archive}")
            row.update(stored=None, archive_pending=False)
        row.update(result="owner_changed", message=f"owner changed {row['path']}; left as is")
        row.pop("error", None)

    def report(self):
        return [{key: row[key] for key in ("path", "result", "message", "error", "archive") if key in row}
                for row in self.data.get("backups", []) + self.data.get("created_roots", []) if row.get("result") or row.get("error")]

    def restore(self):
        self.validate_paths()
        self.restore_created_roots()
        errors = []
        for row in self.data["backups"]:
            target = Path(row["path"])
            try:
                if any(root.get("result") == "owner_changed" and Path(root["path"]) in target.parents
                       for root in self.data.get("created_roots", [])):
                    self._owner_changed(row)
                    continue  # retain the entire conflicted tree, including tracked files
                self.cleanup_temporary(row)
                self._verify_backup(row)
                if row.get("result") in ("owner_changed", "left_session_file"):
                    continue
                live = file_fingerprint(target)
                allowed = [value for value in (row.get("written"), row.get("previous")) if value]
                if not self.data.get("owner_input_seen") and row.get("left_by_session"):
                    allowed.append(row["left_by_session"])
                if live == row["baseline"]:
                    row["result"] = "restored"
                elif row.get("result") == "restored" or live not in allowed:
                    self._owner_changed(row)
                else:
                    if row["baseline"]["kind"] == "file":
                        self._replace(target, Path(row["stored"]).read_bytes(), allowed, row["mode"])
                    else:
                        def unlink():
                            self._expect(target, allowed)
                            if not target.stat().st_mode & stat.S_IWUSR:
                                os.chmod(target, stat.S_IMODE(target.stat().st_mode) | stat.S_IWUSR)
                            target.unlink()
                        retry_sharing(unlink)
                    row["result"] = "restored"
                row.pop("error", None)
            except (OSError, RigError) as exc:
                try:
                    if isinstance(exc, OwnerChanged):
                        self._owner_changed(row)
                    else:
                        raise exc
                except (OSError, RigError) as deferred:
                    row["error"] = f"Restore deferred for {target}: {deferred}"
                    errors.append(row["error"])
            try:
                self.save()
            except OSError as exc:
                errors.append(f"Cannot save restore progress: {exc}")
        if errors:
            self.data["phase"] = "cleanup-required"
            self.save()
            raise RigError("; ".join(errors))

    def verify_restored(self):
        for row in self.data["backups"]:
            if row.get("result") == "restored" and file_fingerprint(Path(row["path"])) != row["baseline"]:
                self._owner_changed(row)
        self.save()

    def purge_backups(self):
        """Only known backup files are removed, after verified restoration."""
        self.validate_paths()
        for row in self.data["backups"]:
            if row.get("result") in ("restored", "owner_changed", "left_session_file"):
                row.pop("temporary_content", None)
            if row.get("result") != "restored":
                continue
            if row.get("stored"):
                path = self.storage_path(row, "stored")
                if path.exists() and fingerprint(path) != row["baseline"]:
                    raise RigError(f"Backup hash mismatch during cleanup: {path}")
                remove(path)
                row["stored"] = None
        backup = self.directory / "backup"
        if backup.exists() and not any(backup.iterdir()):
            backup.rmdir()  # our empty backup directory, never a game/owner folder
        self.save()
