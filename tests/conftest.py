from __future__ import annotations

import os
import shutil
import tempfile

import pytest

# Some modules resolve state paths at import time, which happens during
# collection, before any fixture runs.
_SESSION_DATA_DIR = tempfile.mkdtemp(prefix="simsteer-tests-")
os.environ["SIMSTEER_DATA_DIR"] = _SESSION_DATA_DIR


def pytest_unconfigure(config: pytest.Config) -> None:
    shutil.rmtree(_SESSION_DATA_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMSTEER_DATA_DIR", str(tmp_path))
    return tmp_path
