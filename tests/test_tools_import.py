from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_measure_fov_imports():
    path = ROOT / "tools" / "measure_fov.py"
    spec = importlib.util.spec_from_file_location("measure_fov", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
