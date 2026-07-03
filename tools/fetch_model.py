"""Fetch openpilot's driving models via sparse + LFS clone.

Comma split the old monolithic `supercombo.onnx` into a vision model and
a policy model:

    driving_vision.onnx    (image encoder; takes both narrow + wide cameras)
    driving_policy.onnx    (temporal planner / policy head)

The repo also contains `big_driving_vision.onnx` and `big_driving_policy.onnx`
but on inspection those are just symlinks back to the two files above —
historical naming, same binaries — so we don't fetch them separately.

Run from the project root:
    python tools/fetch_model.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OPENPILOT_REPO = "https://github.com/commaai/openpilot.git"
MODELS_SUBPATH = "selfdrive/modeld/models"
# Pinned to the last openpilot revision that ships the two-stage split model.
# On 2026-06-13 openpilot #38173 merged driving_vision + driving_policy into a
# single driving_supercombo.onnx and deleted the split files, so master no
# longer has what SimSteer loads. This commit's models are byte-for-byte the
# ones bundled in the v0.11 release.
OPENPILOT_REF = "4988a62b310b536fb5336c35151eb2df56e37d97"

WANTED = [
    "driving_vision.onnx",
    "driving_policy.onnx",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    missing = [n for n in WANTED if not (MODELS_DIR / n).exists()
               or (MODELS_DIR / n).stat().st_size < 1000]
    if not missing:
        print("all model files already present")
        return 0

    tmp = ROOT / ".openpilot-sparse"
    if tmp.exists():
        shutil.rmtree(tmp)

    try:
        # A depth-1 clone only fetches master's tip, which no longer has the
        # split models — so init + shallow-fetch the pinned commit directly.
        tmp.mkdir(parents=True)
        run(["git", "init", "-q"], cwd=tmp)
        run(["git", "remote", "add", "origin", OPENPILOT_REPO], cwd=tmp)
        run(["git", "fetch", "--depth=1", "--filter=blob:none",
             "origin", OPENPILOT_REF], cwd=tmp)
        run(["git", "sparse-checkout", "set", MODELS_SUBPATH], cwd=tmp)
        run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=tmp)

        includes = ",".join(f"{MODELS_SUBPATH}/{n}" for n in WANTED)
        run(["git", "lfs", "pull", "--include", includes], cwd=tmp)

        ok = True
        for name in WANTED:
            src = tmp / MODELS_SUBPATH / name
            if not src.exists():
                print(f"WARN: {name} not found in clone", file=sys.stderr)
                ok = False
                continue
            size_mb = src.stat().st_size / 1e6
            if size_mb < 0.5:
                print(f"WARN: {name} is only {size_mb:.3f} MB — LFS likely failed",
                      file=sys.stderr)
                ok = False
                continue
            shutil.copy2(src, MODELS_DIR / name)
            print(f"  {name:30s} {size_mb:7.1f} MB")
        return 0 if ok else 1
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
