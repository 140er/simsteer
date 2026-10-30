"""Fetch openpilot's driving models, pinned and hash-checked.

Downloads `driving_vision.onnx` and `driving_policy.onnx` over HTTPS from
GitHub's LFS media endpoint at one pinned openpilot commit, verifies each
SHA256 into a cache dir, then installs them into `models/`.

The pin is the newest openpilot commit that still ships the two-model
layout whose inputs, outputs and output slices match
`simsteer/core/constants.py` (`simsteer.core.model.check_contract`).
Later commits rename the policy to `driving_on_policy.onnx` (same bytes),
then merge both into one `driving_supercombo.onnx` with a different I/O,
which the runtime does not speak.

    python tools/fetch_model.py
    python tools/fetch_model.py --cache ~/.cache/simsteer/models --check
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# openpilot 0f17a98 (2026-06-02, "modeld: fix capnp memory leak"): the
# last commit before the policy file was renamed. Both blobs date from
# 4988a62 (2026-04-21, "Revert POP model"), i.e. the CD210 model.
OPENPILOT_COMMIT = "0f17a9879364c6f410df7ddf990380ad623a4c92"
MODELS: dict[str, tuple[str, str, int]] = {
    "driving_vision.onnx": (
        "selfdrive/modeld/models/driving_vision.onnx",
        "ee29ee5bce84d1ce23e9ff381280de9b4e4d96d2934cd751740354884e112c66",
        46877473,
    ),
    "driving_policy.onnx": (
        "selfdrive/modeld/models/driving_policy.onnx",
        "78477124cbf3ffe30fa951ebada8410b43c4242c6054584d656f1d329b067e15",
        14060847,
    ),
}
URL = "https://media.githubusercontent.com/media/commaai/openpilot/{commit}/{path}"
DEFAULT_CACHE = Path.home() / ".cache" / "simsteer" / "models"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, want_sha: str, want_size: int,
             attempts: int = 3) -> None:
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            h = hashlib.sha256()
            req = urllib.request.Request(url, headers={"User-Agent": "simsteer-fetch"})
            with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as f:
                while chunk := r.read(CHUNK):
                    h.update(chunk)
                    f.write(chunk)
            size = part.stat().st_size
            if size != want_size or h.hexdigest() != want_sha:
                raise ValueError(f"got {size} bytes sha256 {h.hexdigest()}, "
                                 f"want {want_size} bytes sha256 {want_sha}")
            os.replace(part, dest)
            return
        except Exception as e:
            part.unlink(missing_ok=True)
            print(f"  download attempt {attempt}/{attempts} failed: {e}",
                  file=sys.stderr)
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def ensure_cached(name: str, cache: Path) -> Path:
    path, want_sha, want_size = MODELS[name]
    cached = cache / name
    if cached.exists():
        got = sha256_of(cached)
        if got == want_sha:
            print(f"  {name}: cache ok")
            return cached
        print(f"  {name}: cached file sha256 {got} != pinned {want_sha}, "
              "refusing it and downloading again")
        cached.unlink()
    url = URL.format(commit=OPENPILOT_COMMIT, path=path)
    print(f"  {name}: downloading {url}")
    download(url, cached, want_sha, want_size)
    return cached


def install(name: str, cached: Path, dest_dir: Path) -> None:
    _, want_sha, _ = MODELS[name]
    dest = dest_dir / name
    if dest.exists() and sha256_of(dest) == want_sha:
        return
    if dest.exists():
        print(f"  {name}: installed copy does not match the pin, replacing it")
    tmp = dest.with_name(dest.name + ".part")
    shutil.copyfile(cached, tmp)
    os.replace(tmp, dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help=f"download cache (default {DEFAULT_CACHE})")
    ap.add_argument("--dest", type=Path, default=ROOT / "models",
                    help="where the runtime loads models from (default models/)")
    ap.add_argument("--check", action="store_true",
                    help="also load both models and check their I/O against "
                         "simsteer/core/constants.py")
    args = ap.parse_args(argv)

    cache = args.cache.expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"openpilot {OPENPILOT_COMMIT}")
    try:
        for name in MODELS:
            install(name, ensure_cached(name, cache), args.dest)
    except Exception as e:
        print(f"fetch_model: {e}", file=sys.stderr)
        return 1
    for name, (_, want_sha, _) in MODELS.items():
        got = sha256_of(args.dest / name)
        status = "OK" if got == want_sha else "MISMATCH"
        print(f"  {args.dest / name}  sha256 {got}  {status}")
        if got != want_sha:
            return 1

    if args.check:
        sys.path.insert(0, str(ROOT))
        from simsteer.core.model import check_contract
        problems = check_contract(args.dest / "driving_vision.onnx",
                                  args.dest / "driving_policy.onnx")
        for p in problems:
            print(f"  contract: {p}", file=sys.stderr)
        if problems:
            return 1
        print("  contract: inputs, outputs and output slices match constants.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
