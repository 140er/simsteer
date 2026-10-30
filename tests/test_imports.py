from __future__ import annotations

import importlib
import importlib.abc
import pkgutil
import sys
import types

import pytest

import simsteer

# Windows-only or heavy native deps, stubbed only where they are not installed.
OPTIONAL_NATIVE = ("onnxruntime", "dxcam", "customtkinter")

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(simsteer.__path__, "simsteer.")
)


class _BlockPilot(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pilot" or name.startswith("pilot."):
            raise ImportError(f"simsteer must not import the legacy {name!r}")
        return None


@pytest.fixture
def isolated_imports(monkeypatch):
    for name in OPTIONAL_NATIVE:
        try:
            importlib.import_module(name)
        except Exception:
            # Missing, or present but unusable here: dxcam raises COMError
            # at import in a session without a desktop.
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    def ours():
        return [n for n in sys.modules if n.split(".")[0] in ("simsteer", "pilot")]

    saved = {n: sys.modules.pop(n) for n in ours()}
    blocker = _BlockPilot()
    sys.meta_path.insert(0, blocker)
    yield
    sys.meta_path.remove(blocker)
    for n in ours():
        del sys.modules[n]
    sys.modules.update(saved)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_pilot(module, isolated_imports):
    importlib.import_module(module)
    leaked = sorted(n for n in sys.modules if n.split(".")[0] == "pilot")
    assert not leaked, f"{module} pulled in {leaked}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only drivers")
def test_windows_native_imports():
    import vgamepad  # noqa: F401

    try:
        import dxcam  # noqa: F401
    except Exception as e:
        if type(e).__name__ != "COMError":
            raise
        pytest.skip(f"dxcam needs a desktop session: {e}")
