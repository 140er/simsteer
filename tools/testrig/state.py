"""Write-ahead backups and crash-recoverable, hash-verified restoration."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from . import RigError


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RigError(f"Cannot read session state {path}: {exc}") from exc


@contextmanager
def lock(root: Path):
    """An OS lock releases on crash; no stale PID/lock-file guessing."""
    reject_links(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+b") as stream:
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
        # Inspect descendants before rmtree, including Windows junctions.
        fingerprint(path)
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def copy(source, destination, baseline):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if baseline["kind"] == "directory":
        shutil.copytree(source, destination, copy_function=shutil.copy2)
    elif baseline["kind"] == "file":
        shutil.copy2(source, destination)


class Journal:
    def __init__(self, directory: Path):
        reject_links(directory)
        self.directory = directory
        self.path = directory / "state.json"
        self.data = read_json(self.path) if self.path.exists() else None
        if self.data is not None and (not isinstance(self.data, dict) or self.data.get("version") != 1):
            raise RigError("Unsupported session state version; retain backups for recovery")

    def save(self):
        atomic_json(self.path, self.data)

    def backup(self, targets):
        rows = self.data.setdefault("backups", [])
        for target in targets:
            target = Path(target).absolute()
            if target == self.directory or target in self.directory.parents or self.directory in target.parents:
                raise RigError("Backup targets must be outside the session directory")
            if any(row["path"] == str(target) for row in rows):
                continue
            baseline = fingerprint(target)
            stored = self.directory / "backup" / str(len(rows))
            if stored.exists():
                # An interrupted COPY is disposable; a committed backup is never
                # overwritten. No settings may change before all rows commit.
                remove(stored)
            copy(target, stored, baseline)
            if fingerprint(target) != baseline or fingerprint(stored) != baseline:
                raise RigError(f"Backup changed while copying: {target}; no settings were edited")
            rows.append({"path": str(target), "stored": str(stored), "baseline": baseline})
            self.save()
        self.data["backups_complete"] = True
        self.save()

    def verify_backups(self):
        # Validate every backup before deleting any live file.
        for row in self.data["backups"]:
            if fingerprint(Path(row["stored"])) != row["baseline"]:
                raise RigError(f"Backup hash mismatch: {row['stored']}; restoration refused")

    def restore(self):
        self.verify_backups()
        for row in self.data["backups"]:
            target = Path(row["path"])
            if fingerprint(target) != row["baseline"]:
                remove(target)
                copy(Path(row["stored"]), target, row["baseline"])
            if fingerprint(target) != row["baseline"]:
                raise RigError(f"Restoration hash mismatch: {target}")
            row["restored"] = True
            self.save()

    def verify_restored(self):
        if any(fingerprint(Path(row["path"])) != row["baseline"] for row in self.data["backups"]):
            raise RigError("Files changed after restoration; retain state and retry restore after writers exit")
