"""Benchmark the loop's non-model time per frame on a replay clip.

Checks out each revision into a temporary git worktree, then replays the
clip headless with `--log-timing` `--runs` times per revision,
interleaving the revisions so drift on a shared host hits both equally.
Prints p50 and p95 of non-model time (frame time minus model time) and
of each stage. A revision without replay mode is reported and skipped.

    python tools/replay_bench.py --head HEAD --runs 3
    python tools/replay_bench.py --head .          # the working tree as is
    python tools/replay_bench.py --trunk origin/main --head HEAD --max-p95 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP = Path.home() / ".cache" / "simsteer" / "clips" / "synthetic"
STAGES = ("cap_ms", "warp_ms", "model_ms", "ctrl_ms", "frame_ms")
# The first frames include ORT warm-up and allocator growth.
WARMUP = 20


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def checkout(rev: str, tmp: Path) -> Path | None:
    if rev == ".":
        return ROOT
    sha = git("rev-parse", "--verify", f"{rev}^{{commit}}")
    tree = tmp / sha[:12]
    if not tree.exists():
        git("worktree", "add", "--detach", str(tree), sha)
    if not (tree / "simsteer" / "runtime" / "replay.py").exists():
        return None
    models = tree / "models"
    for onnx in (ROOT / "models").glob("*.onnx"):
        target = models / onnx.name
        if not target.exists():
            models.mkdir(exist_ok=True)
            target.symlink_to(onnx)
    return tree


def replay(tree: Path, clip: Path, frames: int, log: Path) -> list[dict]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, "-m", "simsteer", "--source", str(clip), "--headless",
           "--max-frames", str(frames), "--log-jsonl", str(log), "--log-timing"]
    subprocess.run(cmd, cwd=tree, env=env, check=True, stdout=subprocess.DEVNULL)
    with log.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f][WARMUP:]


def summarize(rows: list[dict]) -> dict[str, tuple[float, float]]:
    cols = {k: np.array([r[k] for r in rows]) for k in STAGES}
    cols["non_model_ms"] = cols["frame_ms"] - cols["model_ms"]
    return {k: (float(np.percentile(v, 50)), float(np.percentile(v, 95)))
            for k, v in cols.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--trunk", default=None)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--max-p95", type=float, default=None,
                    help="exit 1 when head's non-model p95 exceeds this (ms)")
    args = ap.parse_args(argv)

    if not any((ROOT / "models").glob("*.onnx")):
        print("no models; run python tools/fetch_model.py", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="simsteer-bench-"))
    try:
        revs = {"head": args.head}
        if args.trunk:
            revs = {"trunk": args.trunk, **revs}
        trees: dict[str, Path] = {}
        for label, rev in revs.items():
            tree = checkout(rev, tmp)
            if tree is None:
                print(f"{label} {rev}: no replay mode at this revision, skipped")
            else:
                trees[label] = tree
        if "head" not in trees:
            return 2

        rows: dict[str, list[dict]] = {label: [] for label in trees}
        for run in range(args.runs):
            for label, tree in trees.items():
                log = tmp / f"{label}-{run}.jsonl"
                rows[label] += replay(tree, args.clip.expanduser(), args.frames, log)
                print(f"run {run + 1}/{args.runs} {label}: done", flush=True)

        print(f"clip {args.clip}  frames {args.frames} x {args.runs} runs "
              f"(first {WARMUP} of each run dropped)")
        result = 0
        for label, tree in trees.items():
            sha = git("rev-parse", "--short=12", "HEAD", cwd=tree)
            if tree == ROOT:
                sha += " (working tree)"
            print(f"\n{label} {sha}  n={len(rows[label])}")
            stats = summarize(rows[label])
            for k, (p50, p95) in stats.items():
                print(f"  {k:13s} p50 {p50:7.2f}  p95 {p95:7.2f}")
            if (label == "head" and args.max_p95 is not None
                    and stats["non_model_ms"][1] > args.max_p95):
                print(f"  FAIL: non-model p95 {stats['non_model_ms'][1]:.2f} ms "
                      f"> {args.max_p95} ms")
                result = 1
        return result
    finally:
        for tree in tmp.iterdir():
            if tree.is_dir() and (tree / ".git").exists():
                subprocess.run(["git", "worktree", "remove", "--force", str(tree)],
                               cwd=ROOT, capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
