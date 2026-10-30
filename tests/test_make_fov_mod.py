from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

from tools.make_fov_mod import write_manifest

ROOT = Path(__file__).resolve().parent.parent
CAMERA_PATH = "def/camera/units/bumper_basic.sui"
# Camera bytes shipped before the manifest fix; changing metadata must preserve them.
CAMERA_SHA256 = "71c81863c2a4fe65120960fff71c3a2edd966ca34d5571b59668a7487453e574"


def assert_package_manifest(manifest: bytes) -> None:
    unit = re.fullmatch(
        r"\s*SiiNunit\s*\{\s*mod_package\s*:\s*\.package_name\s*"
        r"\{(?P<fields>[^{}]*)\}\s*\}\s*",
        manifest.decode("utf-8"),
    )
    assert unit is not None, "Expected a single .package_name mod_package unit"
    fields = dict(re.findall(r'([\w\[\]]+):\s*"([^"\n]*)"', unit["fields"]))
    assert fields == {
        "package_version": "1.0",
        "display_name": "Comma Wide Cam (bumper 120deg/y+1m)",
        "author": "comma-ets",
        "category[]": "physics",
        "description_file": "description.txt",
    }


def test_generated_manifest_uses_package_name():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        write_manifest(zf, 120.0, 1.0, "bumper")
    with zipfile.ZipFile(archive) as zf:
        assert_package_manifest(zf.read("manifest.sii"))
        assert zf.read("description.txt").startswith(b"Modifies bumper cameras:\n")


def test_committed_mod_preserves_camera_and_correct_manifest():
    expected = io.BytesIO()
    with zipfile.ZipFile(expected, "w", zipfile.ZIP_STORED) as zf:
        write_manifest(zf, 120.0, 1.0, "bumper")
    with zipfile.ZipFile(ROOT / "comma_wide_cam.scs") as zf, zipfile.ZipFile(expected) as metadata:
        assert zf.namelist() == ["manifest.sii", "description.txt", CAMERA_PATH]
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())
        assert_package_manifest(zf.read("manifest.sii"))
        for name in metadata.namelist():
            assert zf.read(name) == metadata.read(name)
        assert hashlib.sha256(zf.read(CAMERA_PATH)).hexdigest() == CAMERA_SHA256
