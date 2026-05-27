"""Build an ETS2 mod that widens the bumper cam (and optionally other
camera) FOVs and/or raises the camera mount.

ETS2's camera definitions live in `def/camera/units/*.sui`. We patch
two fields on every targeted unit:

  - `camera_fov`        — HFOV in degrees. Set via `--fov`.
  - `camera_position`   — `(x, y, z)` in vehicle-local coords with
                          Y as the vertical axis. We add
                          `--height-boost-m` to the Y component,
                          leaving X (lateral) and Z (forward) alone.

`--height-boost-m 0` (the default) leaves the position untouched, so
the mod is fov-only — backwards-compatible with prior generations.

Defaults:
    bumper_basic.sui          camera_fov: 65 -> --fov,
                              camera_position.y += --height-boost-m
    interior_*.sui            UNTOUCHED (use --include interior)
    cabin                     UNTOUCHED (use --include cabin)

Workflow:
    1. Use a HashFS-v2-capable extractor (sk-zk/Extractor) to extract
       def.scs to a folder, e.g. F:\\extracted.
    2. python -m tools.make_fov_mod --extracted-dir F:\\extracted \\
           --fov 120 --height-boost-m 1.0    # 1m is the default
    3. The .scs mod is written here and auto-copied into your ETS2
       mod folder if it exists. Activate in your profile's Mod Manager.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
EXTRACTOR_PATH = TOOLS / "scs_extractor.exe"
EXTRACTOR_URL = "https://download.eurotrucksimulator2.com/scs_extractor.zip"

CANDIDATE_DEF_SCS = [
    Path(r"D:\games\steamapps\common\Euro Truck Simulator 2\def.scs"),
    Path(r"D:\SteamLibrary\steamapps\common\Euro Truck Simulator 2\def.scs"),
    Path(r"C:\Program Files (x86)\Steam\steamapps\common\Euro Truck Simulator 2\def.scs"),
]

DEFAULT_MOD_DIR = Path.home() / "Documents" / "Euro Truck Simulator 2" / "mod"

# In `def/camera/units/*.sui`, each file declares one or more `camera_*`
# units like `vehicle_bumper_camera`, `vehicle_interior_camera`, etc.
# The HFOV value is `camera_fov: <degrees>` (single number, integer or
# float). We just regex-replace the value — no need to parse the whole
# unit, since `camera_fov` only appears once per file in practice.
CAMERA_FOV_RE = re.compile(r'(\bcamera_fov\s*:\s*)([\d.]+)')

# Matches `camera_offset: (x, y, z)` where each component is a
# signed float. We capture the three numbers separately so we can
# add the height boost to the middle (Y, vertical) component while
# leaving X (lateral) and Z (forward) intact. ETS2 uses an LH coord
# system with Y up. (Field is named `camera_offset` in the .sui
# files — it's the offset from the vehicle origin.)
CAMERA_POSITION_RE = re.compile(
    r'(\bcamera_offset\s*:\s*\(\s*)'
    r'(-?\d+(?:\.\d+)?)'        # x
    r'(\s*,\s*)'
    r'(-?\d+(?:\.\d+)?)'        # y (we modify this)
    r'(\s*,\s*)'
    r'(-?\d+(?:\.\d+)?)'        # z
    r'(\s*\))'
)

# Filenames in def/camera/units/ that we care about, by category.
TARGET_GROUPS = {
    # The user is on bumper cam; this is the default and the only one
    # widened by default.
    "bumper":   {"bumper_basic.sui"},
    # Driver-cab interior cam ("F1 view"). One file per truck variant.
    "interior": {"interior_*.sui"},
    # Misc others — generally not what you want to touch.
    "cabin":    {"cabin_basic.sui"},
}


def ensure_extractor() -> Path:
    """Download scs_extractor.exe if not present. Returns its path."""
    if EXTRACTOR_PATH.exists():
        return EXTRACTOR_PATH
    print(f"downloading SCS Extractor from {EXTRACTOR_URL}...")
    with urllib.request.urlopen(EXTRACTOR_URL) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.lower().endswith("scs_extractor.exe"):
                with zf.open(name) as src, open(EXTRACTOR_PATH, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  extracted to {EXTRACTOR_PATH}")
                return EXTRACTOR_PATH
    raise RuntimeError("scs_extractor.exe not found inside downloaded zip")


def find_def_scs(user_path: Path | None) -> Path:
    if user_path is not None:
        if not user_path.exists():
            raise FileNotFoundError(f"--def-scs path does not exist: {user_path}")
        return user_path
    for cand in CANDIDATE_DEF_SCS:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "couldn't find def.scs in any default Steam location — pass --def-scs")


def extract_def_scs(extractor: Path, def_scs: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {def_scs.name} -> {out_dir}")
    proc = subprocess.run(
        [str(extractor), str(def_scs), str(out_dir)],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError(f"scs_extractor failed (exit {proc.returncode})")


def collect_camera_files(extracted_root: Path, targets: set[str]) -> list[Path]:
    """Glob `def/camera/units/` for every file matching any pattern in
    `targets`. `targets` are filename globs (with or without wildcards),
    not paths."""
    units_dir = extracted_root / "def" / "camera" / "units"
    found: list[Path] = []
    for pat in targets:
        # Allow plain filenames (no wildcards) or globs like "interior_*.sui"
        for p in sorted(units_dir.glob(pat)):
            if p.is_file():
                found.append(p)
    # de-dupe while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _fmt_float(x: float) -> str:
    """Render a float the way SCS's parser wants — integer when whole,
    decimal otherwise. Avoids accidentally introducing scientific
    notation for tiny values."""
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def modify_sui(text: str, new_fov: float | None,
               height_boost_m: float = 0.0) -> tuple[str, int, int]:
    """Patch a single .sui camera definition file.

    - Every `camera_fov: <num>` becomes `camera_fov: <new_fov>` (if
      `new_fov` is not None).
    - Every `camera_position: (x, y, z)` becomes
      `camera_position: (x, y + height_boost_m, z)` (if height_boost_m
      is non-zero).

    Returns `(modified_text, n_fov_replacements, n_pos_replacements)`.
    """
    n_fov = 0
    if new_fov is not None:
        text, n_fov = CAMERA_FOV_RE.subn(
            rf"\g<1>{_fmt_float(new_fov)}", text)

    n_pos = 0
    if abs(height_boost_m) > 1e-6:
        def _bump_y(m: re.Match) -> str:
            x, y, z = float(m.group(2)), float(m.group(4)), float(m.group(6))
            y_new = y + height_boost_m
            return (f"{m.group(1)}{_fmt_float(x)}{m.group(3)}"
                    f"{_fmt_float(y_new)}{m.group(5)}{_fmt_float(z)}"
                    f"{m.group(7)}")
        text, n_pos = CAMERA_POSITION_RE.subn(_bump_y, text)

    return text, n_fov, n_pos


def write_manifest(zf: zipfile.ZipFile, fov: float, height_boost_m: float,
                   target_label: str) -> None:
    parts = []
    if fov is not None:
        parts.append(f"{fov:g}deg")
    if abs(height_boost_m) > 1e-6:
        sign = "+" if height_boost_m > 0 else ""
        parts.append(f"y{sign}{height_boost_m:g}m")
    desc_parts = "/".join(parts) if parts else "default"
    manifest = (
        "SiiNunit\n"
        "{\n"
        "mod_package : .comma_wide_cam\n"
        "{\n"
        f"    package_version: \"1.0\"\n"
        f"    display_name: \"Comma Wide Cam ({target_label} {desc_parts})\"\n"
        f"    author: \"comma-ets\"\n"
        f"    category[]: \"physics\"\n"
        f"    description_file: \"description.txt\"\n"
        "}\n"
        "}\n"
    )
    desc_lines = [f"Modifies {target_label} cameras:"]
    if fov is not None:
        desc_lines.append(f"  - camera_fov set to {fov:g} degrees")
    if abs(height_boost_m) > 1e-6:
        desc_lines.append(
            f"  - camera_position.y boosted by {height_boost_m:+g} m (vertical)")
    desc_lines.append("Generated by tools/make_fov_mod.py.")
    description = "\n".join(desc_lines) + "\n"
    zf.writestr("manifest.sii", manifest)
    zf.writestr("description.txt", description)


def _process(extracted_root: Path, output: Path, fov: float | None,
             height_boost_m: float,
             targets: set[str], target_label: str, mod_dir: Path) -> int:
    cams = collect_camera_files(extracted_root, targets)
    units_dir = extracted_root / "def" / "camera" / "units"
    print(f"\nfound {len(cams)} camera unit files under {units_dir}/")
    if not cams:
        print("nothing to do — is this the correct extraction root? It should "
              "contain a `def/camera/units/` subtree.")
        return 1

    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as out_zf:
        write_manifest(out_zf, fov, height_boost_m, target_label)
        modified_files = 0
        modified_fovs = 0
        modified_positions = 0
        for cam_path in cams:
            text = cam_path.read_text(encoding="utf-8", errors="replace")
            new_text, n_fov, n_pos = modify_sui(text, fov, height_boost_m)
            if n_fov == 0 and n_pos == 0:
                continue
            rel = cam_path.relative_to(extracted_root).as_posix()
            out_zf.writestr(rel, new_text)
            modified_files += 1
            modified_fovs += n_fov
            modified_positions += n_pos
            change_parts = []
            if n_fov:
                change_parts.append(f"{n_fov} fov(s)")
            if n_pos:
                change_parts.append(f"{n_pos} position(s)")
            print(f"  {rel}  ({', '.join(change_parts)})")

    if modified_files == 0:
        print("\nWARN: no matching fields found in target cameras. Mod is empty.")
        return 1

    print(f"\nwrote {output}")
    print(f"  files modified: {modified_files}, "
          f"fov fields modified: {modified_fovs}, "
          f"position fields modified: {modified_positions}")

    if mod_dir.exists():
        target = mod_dir / output.name
        shutil.copy2(output, target)
        print(f"\ninstalled to {target}")
        print("In ETS2: open your profile -> Mod Manager -> activate "
              "'Comma Wide Cam'.")
    else:
        print(f"\nETS2 mod folder not at {mod_dir}.")
        print(f"Copy {output} into your ETS2 mod folder manually.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--extracted-dir", type=Path, default=None,
                    help="Path to a folder where you've already extracted "
                         "def.scs (with sk-zk/Extractor or similar). Recommended "
                         "for current ETS2 (HashFS v2). Skips internal extraction.")
    ap.add_argument("--def-scs", type=Path, default=None,
                    help="Path to ETS2 def.scs. Used only if --extracted-dir is "
                         "NOT given; auto-runs SCS's own extractor (HashFS v1 "
                         "only — fails on current ETS2).")
    ap.add_argument("--fov", type=float, default=120.0,
                    help="New camera HFOV in degrees. Default 120 to match the "
                         "openpilot wide-camera training distribution. Pass 0 "
                         "to leave camera_fov untouched (height-boost-only mod).")
    # Backward-compat alias for the old --fov-y name.
    ap.add_argument("--fov-y", dest="fov", type=float,
                    help=argparse.SUPPRESS)
    ap.add_argument("--height-boost-m", type=float, default=1.0,
                    help="Meters to add to each targeted camera's Y "
                         "(vertical) position. Positive raises the mount; "
                         "negative lowers it. 0 = leave camera_position "
                         "untouched. ETS2's vehicle Y axis is up. Default "
                         "is 1.0 — lifts the bumper cam to roughly "
                         "windshield height, which matches the openpilot "
                         "training distribution.")
    ap.add_argument("--include", action="append", default=[],
                    choices=list(TARGET_GROUPS.keys()),
                    help="Which camera categories to widen. Repeatable. "
                         f"Default: bumper. Available: {list(TARGET_GROUPS)}.")
    ap.add_argument("--output", type=Path, default=ROOT / "comma_wide_cam.scs",
                    help="Where to write the .scs mod file.")
    ap.add_argument("--mod-dir", type=Path, default=DEFAULT_MOD_DIR,
                    help="ETS2 mod directory; mod is auto-copied here if it exists.")
    args = ap.parse_args()

    selected = args.include or ["bumper"]
    targets: set[str] = set()
    for group in selected:
        targets.update(TARGET_GROUPS[group])
    target_label = "+".join(selected)

    # --fov 0 = "don't touch FOV" so the user can build a height-only mod.
    fov_arg: float | None = None if args.fov <= 0 else args.fov

    if fov_arg is None and abs(args.height_boost_m) < 1e-6:
        print("error: --fov 0 AND --height-boost-m 0 leaves nothing to do.")
        return 1

    print(f"target groups: {target_label}  globs: {sorted(targets)}")
    print(f"  fov: "
          f"{f'{fov_arg:g} deg' if fov_arg is not None else 'untouched'}")
    print(f"  height boost: "
          f"{f'{args.height_boost_m:+g} m on Y' if abs(args.height_boost_m) > 1e-6 else 'untouched'}")

    if args.extracted_dir is not None:
        if not args.extracted_dir.exists():
            print(f"--extracted-dir doesn't exist: {args.extracted_dir}")
            return 1
        return _process(args.extracted_dir, args.output, fov_arg,
                        args.height_boost_m,
                        targets, target_label, args.mod_dir)

    # Fallback for HashFS v1 only (current ETS2 won't work — needs sk-zk).
    def_scs = find_def_scs(args.def_scs)
    print(f"def.scs: {def_scs}")
    print("(no --extracted-dir given; trying SCS's official extractor — only "
          "works for HashFS v1)")
    extractor = ensure_extractor()
    print(f"extractor: {extractor}")
    with tempfile.TemporaryDirectory(prefix="ets2_def_") as tmp:
        tmp_dir = Path(tmp)
        try:
            extract_def_scs(extractor, def_scs, tmp_dir)
        except RuntimeError as e:
            print(f"\nextraction failed: {e}")
            print("\nThis ETS2 install uses HashFS v2 (which SCS's extractor "
                  "doesn't support).")
            print("Get a community v2-capable extractor "
                  "(https://github.com/sk-zk/Extractor), extract def.scs, "
                  "then re-run with --extracted-dir <path>.")
            return 1
        return _process(tmp_dir, args.output, fov_arg,
                        args.height_boost_m,
                        targets, target_label, args.mod_dir)


if __name__ == "__main__":
    raise SystemExit(main())
