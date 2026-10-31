"""Model files and adapter semantics certified by a calibration Check."""

import hashlib
from pathlib import Path

MODEL_SOURCE_REVISION = "0f17a9879364c6f410df7ddf990380ad623a4c92"
PINNED_SHA256 = {
    "vision_sha256": "ee29ee5bce84d1ce23e9ff381280de9b4e4d96d2934cd751740354884e112c66",
    "policy_sha256": "78477124cbf3ffe30fa951ebada8410b43c4242c6054584d656f1d329b067e15",
}
# Bump for semantic changes to model inputs, decoding, shapes or constants.
# Comments and formatting must not invalidate saved calibration Checks.
ADAPTER_VERSION = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_model_identity() -> dict:
    return {"source_revision": MODEL_SOURCE_REVISION,
            "adapter_version": ADAPTER_VERSION, **PINNED_SHA256}


def actual_model_identity(vision: Path, policy: Path) -> dict:
    """Hash once at model startup; a replaced file cannot reuse a pass."""
    return {**pinned_model_identity(), "vision_sha256": _sha256(vision),
            "policy_sha256": _sha256(policy)}
