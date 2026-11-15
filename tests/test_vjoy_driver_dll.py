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
            # pyvjoy's SetAxis calls through the module's current DLL.
            axes[axis] = (mod._sdk._vj, value)

        def set_button(self, idx, state):
            pass

    mod.VJoyDevice = VJoyDevice
    mod.axes = axes
    monkeypatch.setitem(sys.modules, "pyvjoy", mod)
    return mod


def test_wheel_uses_the_installed_driver_dll(fake_pyvjoy, monkeypatch, tmp_path):
    from simsteer.runtime.output.wheel import Wheel

    driver = tmp_path / "vJoyInterface.dll"
    driver.write_bytes(b"")
    monkeypatch.setenv("SIMSTEER_VJOY_DLL", str(driver))
    monkeypatch.setattr(ctypes, "CDLL", lambda path: ("driver", path))
    wheel = Wheel(1)
    dll = ("driver", str(driver))
    assert fake_pyvjoy._sdk._vj == dll and wheel._dev.dll_at_open == dll
    wheel.engage()
    wheel.set_steering(0.3)
    assert fake_pyvjoy.axes[fake_pyvjoy.HID_USAGE_X][0] == dll


def test_unloadable_driver_dll_is_a_wheel_error(fake_pyvjoy, monkeypatch, tmp_path):
    from simsteer.runtime.output.wheel import Wheel, WheelError

    driver = tmp_path / "vJoyInterface.dll"
    driver.write_bytes(b"")
    monkeypatch.setenv("SIMSTEER_VJOY_DLL", str(driver))

    def bad(path):
        raise OSError("not a valid Win32 application")

    monkeypatch.setattr(ctypes, "CDLL", bad)
    with pytest.raises(WheelError, match="could not load the vJoy driver DLL"):
        Wheel(1)


def test_wheel_keeps_pyvjoys_dll_without_an_installed_driver(fake_pyvjoy, monkeypatch, tmp_path):
    from simsteer.runtime.output import wheel as w
    from simsteer.runtime.ffb import vjoy_ffb

    monkeypatch.delenv("SIMSTEER_VJOY_DLL", raising=False)
    monkeypatch.setattr(vjoy_ffb, "DEFAULT_DLL", str(tmp_path / "missing.dll"))
    monkeypatch.setattr(ctypes, "CDLL", lambda path: pytest.fail("must not load"))
    w.Wheel(1)
    assert fake_pyvjoy._sdk._vj == "bundled"


def test_a_missing_explicit_override_is_a_wheel_error(fake_pyvjoy, monkeypatch, tmp_path):
    from simsteer.runtime.output.wheel import Wheel, WheelError

    monkeypatch.setenv("SIMSTEER_VJOY_DLL", str(tmp_path / "typo.dll"))
    with pytest.raises(WheelError, match="missing file"):
        Wheel(1)
