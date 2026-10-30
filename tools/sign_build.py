"""Code-sign the PyInstaller build, when a certificate is configured.

build.bat runs this after PyInstaller. With SIMSTEER_SIGN_CERT_SHA1 unset
it prints one line and exits 0, so unsigned builds keep working.

    SIMSTEER_SIGN_CERT_SHA1      thumbprint of the signing certificate in
                                 the Windows certificate store (required)
    SIMSTEER_SIGN_TOOL           path to signtool.exe (default: PATH, then
                                 the newest Windows SDK)
    SIMSTEER_SIGN_TIMESTAMP_URL  RFC 3161 timestamp server (default below)

The key stays in the certificate store or on the hardware token, so no
key file or password passes through here.

    python tools/sign_build.py [--dist dist/SimSteer]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TIMESTAMP_URL = "http://timestamp.digicert.com"
EXES = ("SimSteer.exe", "SimSteer-debug.exe")


class SignError(Exception):
    pass


class SignConfig(NamedTuple):
    thumbprint: str
    signtool: Path
    timestamp_url: str


def clean_thumbprint(raw: str) -> str:
    # The certificate dialog shows thumbprints with spaces and, when
    # copied, a leading invisible left-to-right mark.
    thumb = re.sub(r"[\s\u200e\u200f]", "", raw).upper()
    if not re.fullmatch(r"[0-9A-F]{40}", thumb):
        raise SignError("SIMSTEER_SIGN_CERT_SHA1 must be a 40-character hex SHA1 thumbprint.")
    return thumb


def find_signtool(env: dict[str, str]) -> Path:
    explicit = env.get("SIMSTEER_SIGN_TOOL", "").strip().strip('"')
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SignError(f"SIMSTEER_SIGN_TOOL points at {path}, which does not exist.")
        return path
    found = shutil.which("signtool")
    if found:
        return Path(found)
    kits = Path(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10" / "bin"
    candidates = sorted(kits.glob("*/x64/signtool.exe"), key=lambda p: _version_key(p.parent.parent.name))
    if candidates:
        return candidates[-1]
    raise SignError("signtool.exe not found. Install the Windows SDK or set SIMSTEER_SIGN_TOOL.")


def _version_key(name: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", name))


def load_config(env: dict[str, str]) -> SignConfig | None:
    raw = env.get("SIMSTEER_SIGN_CERT_SHA1", "")
    if not raw.strip():
        return None
    url = env.get("SIMSTEER_SIGN_TIMESTAMP_URL", "").strip() or DEFAULT_TIMESTAMP_URL
    if not re.match(r"https?://", url):
        raise SignError("SIMSTEER_SIGN_TIMESTAMP_URL must start with http:// or https://.")
    return SignConfig(clean_thumbprint(raw), find_signtool(env), url)


def targets(dist: Path) -> list[Path]:
    found = [dist / name for name in EXES if (dist / name).is_file()]
    if not found:
        raise SignError(f"no SimSteer exe found in {dist}. Run PyInstaller first.")
    return found


def sign_command(cfg: SignConfig, files: list[Path]) -> list[str]:
    return [str(cfg.signtool), "sign", "/sha1", cfg.thumbprint,
            "/fd", "SHA256", "/tr", cfg.timestamp_url, "/td", "SHA256",
            "/d", "SimSteer", *map(str, files)]


def verify_command(cfg: SignConfig, files: list[Path]) -> list[str]:
    return [str(cfg.signtool), "verify", "/pa", *map(str, files)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dist", type=Path, default=ROOT / "dist" / "SimSteer")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(dict(os.environ))
        if cfg is None:
            print("sign_build: SIMSTEER_SIGN_CERT_SHA1 not set, signing skipped.")
            return 0
        files = targets(args.dist)
    except SignError as e:
        print(f"sign_build: {e}", file=sys.stderr)
        return 2
    for step, cmd in (("sign", sign_command(cfg, files)), ("verify", verify_command(cfg, files))):
        print(f"sign_build: {step} {', '.join(f.name for f in files)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"sign_build: signtool {step} failed (exit code {rc}).", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
