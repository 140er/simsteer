"""The vJoy wheel talks to the installed driver's DLL, not pyvjoy's copy."""

from __future__ import annotations

import ctypes
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def fake_pyvjoy(monkeypatch):
    mod = ModuleType("pyvjoy")
    mod._sdk = SimpleNamespace(_vj="bundled")
    axes = {}
    mod.HID_USAGE_X, mod.HID_USAGE_Y, mod.HID_USAGE_Z = 0x30, 0x31, 0x32

    class VJoyDevice:
        def __init__(self, device_id):
            self.dll_at_open = mod._sdk._vj

        def set_axis(self, axis, value):
            axes[axis] = value

        def set_button(self, idx, state):
            pass

    mod.VJoyDevice = VJoyDevice
    monkeypatch.setitem(sys.modules, "pyvjoy", mod)
    return mod


def test_wheel_uses_the_installed_driver_dll(fake_pyvjoy, monkeypatch, tmp_path):
    from simsteer.runtime.output.wheel import Wheel

    driver = tmp_path / "vJoyInterface.dll"
    driver.write_bytes(b"")
    monkeypatch.setenv("SIMSTEER_VJOY_DLL", str(driver))
    monkeypatch.setattr(ctypes, "CDLL", lambda path: ("driver", path))
    wheel = Wheel(1)
    assert fake_pyvjoy._sdk._vj == ("driver", str(driver))
    assert wheel._dev.dll_at_open == ("driver", str(driver))


def test_wheel_keeps_pyvjoys_dll_without_an_installed_driver(fake_pyvjoy, monkeypatch, tmp_path):
    from simsteer.runtime.output.wheel import Wheel

    monkeypatch.setenv("SIMSTEER_VJOY_DLL", str(tmp_path / "missing.dll"))
    monkeypatch.setattr(ctypes, "CDLL", lambda path: pytest.fail("must not load"))
    Wheel(1)
    assert fake_pyvjoy._sdk._vj == "bundled"
