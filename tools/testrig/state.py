"""Write-ahead backups and crash-recoverable, hash-verified restoration."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
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


class OwnerChanged(RigError):
    pass


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

    def verify_backups(self):
        for row in self.data["backups"]:
            if row.get("result") in ("restored", "owner_changed") or row["baseline"]["kind"] == "absent":
                continue
            if not row.get("stored") or fingerprint(Path(row["stored"])) != row["baseline"]:
                raise RigError(f"Backup hash mismatch: {row['stored']}; restoration refused")

    def _expect(self, path, allowed):
        if file_fingerprint(path) not in allowed:
            raise OwnerChanged(f"owner changed {path}; left as is")

    def _replace(self, path, content, allowed, mode):
        reject_links(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".testrig-", dir=path.parent)
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
            temporary = Path(temporary)
            if temporary.exists():
                os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
                retry_sharing(lambda: temporary.unlink(missing_ok=True))

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
        row.update(previous=previous, written={"kind": "file", "sha256": hashlib.sha256(content).hexdigest()})
        self.save()
        self._replace(path, content, [previous], (row["mode"] or 0o600) | stat.S_IWUSR)
        row["previous"] = None
        self.save()

    def _owner_changed(self, row):
        if row.get("stored"):
            archive = self.directory / "displaced" / Path(row["stored"]).name
            archive.parent.mkdir(parents=True, exist_ok=True)
            if Path(row["stored"]).exists():
                retry_sharing(lambda: os.replace(row["stored"], archive))
            if archive.exists():
                row["archive"] = str(archive)
                row["stored"] = None
        row.update(result="owner_changed", message=f"owner changed {row['path']}; left as is")
        row.pop("error", None)

    def report(self):
        return [{key: row[key] for key in ("path", "result", "message", "error", "archive") if key in row}
                for row in self.data.get("backups", []) if row.get("result") or row.get("error")]

    def restore(self):
        self.verify_backups()
        errors = []
        for row in self.data["backups"]:
            target = Path(row["path"])
            try:
                if row.get("result") == "owner_changed":
                    continue
                live = file_fingerprint(target)
                allowed = [value for value in (row.get("written"), row.get("previous")) if value]
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
            except OwnerChanged:
                self._owner_changed(row)
            except (OSError, RigError) as exc:
                row["error"] = f"Restore deferred for {target}: {exc}"
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
        for row in self.data["backups"]:
            if row.get("result") != "restored":
                continue
            if row.get("stored"):
                remove(Path(row["stored"]))
                row["stored"] = None
        backup = self.directory / "backup"
        if backup.exists() and not any(backup.iterdir()):
            backup.rmdir()  # our empty backup directory, never a game/owner folder
        self.save()
