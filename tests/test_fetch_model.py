from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fetch(monkeypatch):
    spec = importlib.util.spec_from_file_location("fetch_model", ROOT / "tools" / "fetch_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    good = b"model bytes"
    sha = hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(mod, "MODELS", {"m.onnx": ("x/m.onnx", sha, len(good))})
    calls = []

    def fake_download(url, dest, want_sha, want_size, attempts=3):
        calls.append(url)
        dest.write_bytes(good)

    monkeypatch.setattr(mod, "download", fake_download)
    mod.calls = calls
    mod.good = good
    return mod


def test_corrupt_cache_is_refused_and_fetched_again(fetch, tmp_path, capsys):
    cache, dest = tmp_path / "cache", tmp_path / "models"
    assert fetch.main(["--cache", str(cache), "--dest", str(dest)]) == 0
    assert len(fetch.calls) == 1

    assert fetch.main(["--cache", str(cache), "--dest", str(dest)]) == 0
    assert len(fetch.calls) == 1

    corrupt = bytearray(fetch.good)
    corrupt[0] ^= 0xFF
    (cache / "m.onnx").write_bytes(bytes(corrupt))
    (dest / "m.onnx").write_bytes(bytes(corrupt))
    assert fetch.main(["--cache", str(cache), "--dest", str(dest)]) == 0
    assert len(fetch.calls) == 2
    assert "refusing it" in capsys.readouterr().out
    assert (dest / "m.onnx").read_bytes() == fetch.good


def test_pin_is_a_full_commit_and_sha256s():
    spec = importlib.util.spec_from_file_location("fm", ROOT / "tools" / "fetch_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert len(mod.OPENPILOT_COMMIT) == 40
    assert set(mod.MODELS) == {"driving_vision.onnx", "driving_policy.onnx"}
    for _, sha, size in mod.MODELS.values():
        assert len(sha) == 64 and size > 1_000_000
    from simsteer.core.model_identity import pinned_model_identity
    identity = pinned_model_identity()
    assert identity["source_revision"] == mod.OPENPILOT_COMMIT
    assert identity["vision_sha256"] == mod.MODELS["driving_vision.onnx"][1]
    assert identity["policy_sha256"] == mod.MODELS["driving_policy.onnx"][1]
    assert identity["adapter_version"]
    assert "adapter_sha256" not in identity  # Comments are not model semantics.
