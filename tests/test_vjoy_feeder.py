import threading
from types import SimpleNamespace

import pytest

from simsteer.core.override import Output
from simsteer.runtime.output.vjoy_feeder import VJoyFeeder, axis_value
from simsteer.runtime.output.wheel import Wheel, WheelError
from fakes_wheel import FakeVJoyDll


@pytest.mark.parametrize("value", [-1, -.25, 0, .25, 1])
def test_value_map_matches_existing_wheel(value):
    writes = []
    old = Wheel.__new__(Wheel)
    old._engaged = True
    old._dev = SimpleNamespace(set_axis=lambda usage, value: writes.append((usage, value)))
    old._pyvjoy = SimpleNamespace(HID_USAGE_X=0x30, HID_USAGE_Y=0x31, HID_USAGE_Z=0x32)
    old.set_steering(value)
    old.set_throttle_brake(abs(value), abs(value))
    assert writes == [(0x30, axis_value("steer", value)),
                      (0x31, axis_value("throttle", abs(value))), (0x32, axis_value("brake", abs(value)))]


def test_missing_clutch_and_close_neutral_then_relinquish_without_reset():
    dll = FakeVJoyDll(clutch=False)
    feeder = VJoyFeeder(dll=dll)
    feeder.write(Output(.7, .8, .9, 1))
    feeder.close()
    assert dll.values == {0x30: 0x4000, 0x31: 1, 0x32: 1}
    assert dll.calls[-1][0] == "RelinquishVJD"
    assert [c[0] for c in dll.calls[-4:]] == ["SetAxis", "SetAxis", "SetAxis", "RelinquishVJD"]
    assert not any(c[0] == "ResetVJD" for c in dll.calls)


def test_acquire_failure_and_required_axes():
    with pytest.raises(WheelError, match="acquire vJoy"):
        VJoyFeeder(dll=FakeVJoyDll(acquire=False))
    dll = FakeVJoyDll()
    dll.GetVJDAxisExist = lambda device, usage: usage != 0x31
    with pytest.raises(WheelError, match="X, Y and Z"):
        VJoyFeeder(dll=dll)
    assert dll.calls[-1][0] == "RelinquishVJD"


def test_neutral_attempts_all_pedals_even_if_one_axis_fails():
    dll = FakeVJoyDll()
    feeder = VJoyFeeder(dll=dll)
    feeder.write(Output(.9, .9, .9, .9))
    dll.fail = 0x30
    with pytest.raises(WheelError):
        feeder.write(Output())
    assert [dll.values[i] for i in (0x31, 0x32, 0x33)] == [1, 1, 1]
    feeder.close()


def test_feeder_rejects_second_writer_thread():
    feeder = VJoyFeeder(dll=FakeVJoyDll())
    errors = []
    def wrong_thread():
        try:
            feeder.write(Output())
        except Exception as e:
            errors.append(e)
    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert isinstance(errors[0], RuntimeError)
    feeder.close()


def test_plain_wheel_fallback_and_bind_use_one_owner(monkeypatch):
    import sys
    from simsteer.runtime.output import vjoy_feeder as vf
    dll = FakeVJoyDll()
    monkeypatch.setitem(sys.modules, "pyvjoy", None)
    original = vf.VJoyFeeder
    monkeypatch.setattr(vf, "VJoyFeeder", lambda device_id: original(device_id, dll=dll))
    wheel = Wheel()
    try:
        wheel.engage()
        wheel.set_steering(.2)
        wheel.disengage()
        worker = threading.Thread(target=wheel.wiggle, args=("throttle", .01))
        worker.start()
        worker.join()
    finally:
        wheel.close()
    assert len({c[3] for c in dll.calls}) == 1
    assert dll.calls[-1][0] == "RelinquishVJD"
