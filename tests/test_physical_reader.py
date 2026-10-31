import ctypes as c

from simsteer.app.settings import Settings
from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.input.physical import PhysicalWheel
from fakes_wheel import FakeDi, descriptor, profile


def reader(monkeypatch, *profiles, settings=None):
    api = FakeDi(*profiles)
    monkeypatch.setattr(di, "_dev", api.call)
    wheel = PhysicalWheel(settings, di_factory=api)
    assert wheel.open()
    return wheel, api


def test_full_format_and_selection_never_write_force(monkeypatch):
    a, b, other, vjoy = profile(1), profile(2), profile(3), profile(4)
    other.inst, vjoy.inst = descriptor(3, vid=1), descriptor(4, vid=di.VJOY_VID, pid=di.VJOY_PID)
    wheel, api = reader(monkeypatch, a, b, other, vjoy)
    first = wheel.read(0)
    assert first.ok and first.throttle is None
    b.axes[0] += 1000
    wheel.read(.002)
    assert wheel.instance_proven == str(b.inst.guidInstance)
    assert len(wheel.records) == 1
    assert [v[1] for v in api.calls if v[0] == "CreateDevice"] == [1, 2]
    assert all(v[2] == di.DISCL_NONEXCLUSIVE | di.DISCL_BACKGROUND for v in api.calls if v[0] == "SetCooperativeLevel")
    wheel.close()
    assert not {v[0] for v in api.calls} & {"SetProperty", "CreateEffect", "SendForceFeedbackCommand"}
    fmt, keep = di.joystick_state_format()
    assert c.sizeof(di.JOYSTATE) == 176
    assert (fmt.dwDataSize, fmt.dwNumObjs) == (176, 140)
    assert fmt.rgodf[139].dwOfs == 175 and keep


def test_saved_instance_only_and_missing_falls_back(monkeypatch, caplog):
    a, b = profile(1), profile(2)
    settings = Settings(wheel_instance=str(b.inst.guidInstance))
    wheel, api = reader(monkeypatch, a, b, settings=settings)
    assert len(wheel.records) == 1 and api.calls[-1][0] == "Acquire"
    assert wheel.active["dev"].info.instance == settings.wheel_instance
    wheel.close()
    wheel, api = reader(monkeypatch, a, settings=settings)
    assert "Saved wheel instance missing" in caplog.text
    assert not wheel.saved and wheel.axes == {}
    wheel.close()


def test_lost_retry_interval_and_reenumeration(monkeypatch):
    p = profile()
    wheel, api = reader(monkeypatch, p)
    assert wheel.read(0).ok
    p.reads.append(di.DIERR_INPUTLOST)
    assert not wheel.read(.01).ok
    p.acquire_errors.extend([di.DIERR_INPUTLOST] * 2)
    acquire_count = lambda: sum(v[0] == "Acquire" for v in api.calls)
    assert not wheel.read(.49).ok and acquire_count() == 1
    assert not wheel.read(.51).ok and acquire_count() == 2
    assert not wheel.read(.99).ok and acquire_count() == 2
    assert not wheel.read(1.01).ok and acquire_count() == 3
    assert wheel.read(1.51).ok and api.enumerations == 2
    wheel.close()


def test_pedals_disabled_without_map_and_startup_throttle_latch(monkeypatch):
    p = profile()
    settings = Settings(wheel_instance=str(p.inst.guidInstance), wheel_axes={
        "throttle": {"slot": 1, "rest": 0, "full": 65535},
        "brake": {"slot": 2, "rest": 0, "full": 65535},
        "clutch": {"slot": 3, "rest": 32767, "full": 0}})
    p.axes[1:4] = [65535, 20000, 0]
    wheel, _ = reader(monkeypatch, p, settings=settings)
    sample = wheel.read(0)
    assert sample.throttle is sample.brake is sample.clutch is None
    assert wheel.status == "Check pedals"
    p.axes[1] = 0
    sample = wheel.read(.002)
    assert sample.throttle == 0 and sample.brake > 0 and sample.clutch == 1
    wheel.close()


def test_pedal_map_publishes_only_after_explicit_flow(monkeypatch):
    p = profile()
    wheel, _ = reader(monkeypatch, p)
    wheel.read(0)
    wheel.map_pedals(0)
    for index, slot in enumerate((1, 2, 3)):
        p.axes[slot] = 65535
        assert wheel.read(index * 3 + 1).throttle is None
        p.axes[slot] = 32767 if slot == 3 else 0
        wheel.read((index + 1) * 3)
    assert wheel.mapper.done
    assert set(wheel.axes) == {"steer", "throttle", "brake", "clutch"}
    wheel.close()
