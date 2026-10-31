"""The simulated Fanatec base under SimSteer's real DirectInput and vJoy
code: detection, the DD2's two instances, reading input, the rules a real
device enforces, tracks and the vJoy setup check."""

from __future__ import annotations

import ctypes
import json
from dataclasses import replace

import pytest

from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.ffb import vjoy_ffb as vj
from simsteer.testing.fanatec import DD2_A, FanatecSim, SimDevice, Track
from simsteer.testing.fanatec import dinput as sdi


class Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def sim():
    with FanatecSim() as s:
        yield s
    assert not s.installed


def instances(ff_only=True):
    found = []
    cb_type = di._FUNCTYPE(ctypes.c_int, ctypes.POINTER(di.DIDEVICEINSTANCEW), ctypes.c_void_p)

    def on_device(inst, _ref):
        found.append(di.info_of(inst.contents))
        return 1

    d = di.open_directinput()
    try:
        cb = cb_type(on_device)
        flags = di.DIEDFL_ATTACHEDONLY | (di.DIEDFL_FORCEFEEDBACK if ff_only else 0)
        assert di._com_call(d.ptr, "IDirectInput8W", "EnumDevices", [], di.DI8DEVCLASS_GAMECTRL, cb, None,
                            flags) == di.DI_OK
    finally:
        d.close()
    return found


class Reader:
    """Opens one instance read-only the way a wheel reader does (non-exclusive,
    background, the axis-only joystick format)."""

    def __init__(self, info: di.DeviceInfo, fmt=None, coop=0x2 | di.DISCL_BACKGROUND) -> None:
        self.api = di.open_directinput()
        inst = next(i for i in self.api.ff_devices() if di.info_of(i).instance == info.instance)
        self.ptr = self.api.create_device(inst)
        self.fmt = fmt or di.joystick_data_format()
        self.size = self.fmt[0].dwDataSize
        assert self.call("SetDataFormat", ctypes.byref(self.fmt[0])) == di.DI_OK
        assert self.call("SetCooperativeLevel", self.api.hwnd, coop) == di.DI_OK

    def call(self, method, *args):
        return di._dev(self.ptr, method, [], *args)

    def read(self):
        buf = (ctypes.c_uint8 * self.size)()
        hr = self.call("GetDeviceState", self.size, ctypes.byref(buf))
        return hr, bytes(buf)

    def axes(self):
        hr, raw = self.read()
        assert hr == di.DI_OK, di.hresult_name(hr)
        return list((di.LONG * 8).from_buffer_copy(raw[:32]))

    def name_at(self, offset):
        obj = di.DIDEVICEOBJECTINSTANCEW()
        obj.dwSize = ctypes.sizeof(obj)
        hr = self.call("GetObjectInfo", ctypes.byref(obj), offset, 1)
        return di.wide(obj.tszName) if hr == di.DI_OK else None

    def close(self):
        self.call("Unacquire")
        di._dev(self.ptr, "Release", [], restype=di.DWORD)
        self.api.close()


def fanatec(_sim):
    return [d for d in di.attached_ff_devices() if d.is_fanatec]


# --------------------------------------------------------------------------
# detection


def test_dd2_enumerates_as_two_same_name_instances_like_the_real_base(sim):
    devices = di.list_ff_devices()
    dd2 = [d for d in devices if d["fanatec"]]
    assert [(d["name"], d["vid"], d["pid"], d["axes"], d["buttons"]) for d in dd2] == [
        ("FANATEC Podium Wheel Base DD2", "0x0EB7", "0x0007", 8, 108),
        ("FANATEC Podium Wheel Base DD2", "0x0EB7", "0x0007", 12, 63),
    ]
    assert dd2[0]["instance"] != dd2[1]["instance"]
    for d in dd2:
        assert d["actuators"] == [{"name": "Wheel Axis", "id": "0x01000002", "max_force": 10}]
        assert d["effects"][0] == "Constant" and d["effects"][-1] == "CustomForce" and len(d["effects"]) == 12
        assert d["ff_sample_period"] == 1000
    vjoy = [d for d in devices if d["vjoy"]]
    assert len(vjoy) == 1 and not vjoy[0]["fanatec"] and len(vjoy[0]["actuators"]) == 2
    sim.assert_no_leaks()
    sim.assert_no_force_writes()


@pytest.mark.parametrize("profile,count", [("dd2", 2), ("single", 1), ("two_bases", 2), ("none", 0)])
def test_profiles(profile, count):
    with FanatecSim(profile) as sim:
        found = fanatec(sim)
        assert len(found) == count
        assert len({d.instance for d in found}) == count
        assert [d.is_vjoy for d in di.attached_ff_devices()].count(True) == 1


def test_unplugged_base_is_not_enumerated_until_replugged(sim):
    sim.wheel.unplug()
    assert fanatec(sim) == []
    sim.wheel.replug()
    assert len(fanatec(sim)) == 2


def test_force_feedback_filter_is_honoured(sim):
    sim.devices.append(SimDevice(sim, (replace(DD2_A, instance="5E1A0042-0000-11F1-8000-53494D534D30",
                                               effects=()),)))
    assert len(instances(ff_only=True)) == 3  # two DD2 views + vJoy
    assert len(instances(ff_only=False)) == 4


def test_sink_never_targets_vjoy():
    with FanatecSim("none") as sim:
        sink = di.DirectInputSink()
        assert not sink.open(500) and sink.last_error == "no matching force feedback device"
        sim.assert_no_leaks()


# --------------------------------------------------------------------------
# reading


@pytest.mark.parametrize("live", [0, 1])
def test_only_the_live_instance_moves(live):
    with FanatecSim(live_instance=live) as sim:
        a, b = fanatec(sim)
        readers = [Reader(a), Reader(b)]
        for r in readers:
            assert r.call("Acquire") == di.DI_OK
        sim.wheel.set(steer_deg=225, throttle=1.0)
        moving = [r.axes()[0] for r in readers]
        assert moving[live] == round(0.75 * 65535)
        assert moving[1 - live] == round(0.5 * 65535)
        for r in readers:
            r.close()
        sim.assert_no_leaks()
        sim.assert_no_force_writes()


def test_slot_names_follow_the_data_format_per_instance(sim):
    a, b = fanatec(sim)
    ra, rb = Reader(a), Reader(b)
    names = lambda r: {off: r.name_at(off) for off in range(0, 32, 4)}  # noqa: E731
    assert names(ra) == {0: "Wheel Axis", 4: "Accelerator", 8: None, 12: "X Rotation", 16: "Y Rotation",
                         20: "Brake", 24: "Combined Pedals", 28: "Clutch"}
    assert names(rb) == {0: "Wheel Axis", 4: "Z Axis", 8: None, 12: "X Rotation", 16: "Y Rotation",
                         20: "Brake", 24: "Accelerator", 28: "Clutch"}
    ra.close()
    rb.close()


@pytest.mark.parametrize("inverted", [False, True])
def test_pedals_and_their_direction(inverted):
    with FanatecSim("single", pedals_inverted=inverted) as sim:
        r = Reader(fanatec(sim)[0])
        r.call("Acquire")
        rest = r.axes()
        sim.wheel.set(throttle=1.0, brake=0.5)
        pressed = r.axes()
        full, half = (0, 32768) if inverted else (65535, 32768)
        assert rest[1] == (65535 if inverted else 0)
        assert (pressed[1], pressed[5]) == (full, half)
        assert pressed[6] == round((0.5 + (0.5 - 1.0) / 2) * 65535)  # combined pedals
        sim.wheel.unplug_pedals()
        assert r.axes()[1] == rest[1]
        r.close()


def test_buttons_and_hat_on_a_custom_format(sim):
    guid_pov = di.GUID.parse("A36D02F2-C9F3-11CF-BFC7-444553540000")
    rows = [(ctypes.pointer(guid_pov), 0, di.DIDFT_OPTIONAL | 0x10 | di.DIDFT_ANYINSTANCE, 0)]
    rows += [(None, 4 + i, di.DIDFT_OPTIONAL | 0x0C | di.DIDFT_ANYINSTANCE, 0) for i in range(4)]
    objects = (di.DIOBJECTDATAFORMAT * len(rows))(*rows)
    fmt = di.DIDATAFORMAT(ctypes.sizeof(di.DIDATAFORMAT), ctypes.sizeof(di.DIOBJECTDATAFORMAT), di.DIDF_ABSAXIS,
                          8, len(rows), ctypes.cast(objects, ctypes.POINTER(di.DIOBJECTDATAFORMAT)))
    r = Reader(fanatec(sim)[0], fmt=(fmt, [guid_pov, objects]))
    r.call("Acquire")
    hr, raw = r.read()
    assert hr == di.DI_OK and raw == b"\xff\xff\xff\xff\x00\x00\x00\x00"
    sim.wheel.press(2)
    sim.wheel.set(pov=9000)
    hr, raw = r.read()
    assert raw == (9000).to_bytes(4, "little") + b"\x00\x00\x80\x00"
    sim.wheel.release(2)
    assert r.read()[1][6] == 0
    r.close()


def test_unplug_loses_the_device_and_replug_needs_a_new_object(sim):
    r = Reader(fanatec(sim)[0])
    assert r.call("Acquire") == di.DI_OK
    assert r.read()[0] == di.DI_OK
    sim.wheel.unplug()
    assert r.call("Poll") == di.DIERR_INPUTLOST
    assert r.read()[0] == di.DIERR_NOTACQUIRED
    assert r.call("Acquire") == di.DIERR_UNPLUGGED
    sim.wheel.replug()
    assert r.call("Acquire") == di.DIERR_UNPLUGGED  # the old object stays dead
    r.close()
    fresh = Reader(fanatec(sim)[0])
    assert fresh.call("Acquire") == di.DI_OK and fresh.read()[0] == di.DI_OK
    fresh.close()
    sim.assert_no_leaks()


def test_device_rules(sim):
    info = fanatec(sim)[0]
    api = di.open_directinput()
    inst = next(i for i in api.ff_devices() if di.info_of(i).instance == info.instance)
    ptr = api.create_device(inst)
    dev = lambda method, *args: di._dev(ptr, method, [], *args)  # noqa: E731
    assert dev("Acquire") == di.DIERR_INVALIDPARAM  # no data format yet
    state = (ctypes.c_uint8 * 32)()
    fmt = di.joystick_data_format()
    assert dev("SetDataFormat", ctypes.byref(fmt[0])) == di.DI_OK
    assert dev("GetDeviceState", 32, ctypes.byref(state)) == di.DIERR_NOTACQUIRED
    assert dev("SetCooperativeLevel", 0x9999, 0x2 | di.DISCL_BACKGROUND) == di.DIERR_INVALIDPARAM  # unknown window
    assert dev("SetCooperativeLevel", api.hwnd, di.DISCL_EXCLUSIVE | 0x2 | di.DISCL_BACKGROUND) == di.DIERR_INVALIDPARAM
    assert dev("SetCooperativeLevel", api.hwnd, 0x2 | di.DISCL_BACKGROUND) == di.DI_OK
    assert dev("Acquire") == di.DI_OK
    assert dev("GetDeviceState", 16, ctypes.byref(state)) == di.DIERR_INVALIDPARAM  # wrong size
    assert dev("SetDataFormat", ctypes.byref(fmt[0])) == sdi.dv.DIERR_ACQUIRED
    assert dev("SendForceFeedbackCommand", 1) == di.DIERR_NOTEXCLUSIVEACQUIRED
    assert dev("Unacquire") == di.DI_OK and dev("Unacquire") == di.S_FALSE
    # A required (non-optional) object the device lacks fails the format.
    guid_z = di.GUID.parse(di._AXIS_GUIDS[2])
    rows = (di.DIOBJECTDATAFORMAT * 1)((ctypes.pointer(guid_z), 0, di.DIDFT_AXIS | di.DIDFT_ANYINSTANCE, 0))
    bad = di.DIDATAFORMAT(ctypes.sizeof(di.DIDATAFORMAT), ctypes.sizeof(di.DIOBJECTDATAFORMAT), di.DIDF_ABSAXIS,
                          4, 1, ctypes.cast(rows, ctypes.POINTER(di.DIOBJECTDATAFORMAT)))
    assert dev("SetDataFormat", ctypes.byref(bad)) == di.DIERR_INVALIDPARAM
    assert dev("Escape") == sdi.dv.DIERR_UNSUPPORTED
    di._dev(ptr, "Release", [], restype=di.DWORD)
    api.close()
    sim.assert_no_leaks()
    assert [c["op"] for c in sim.ffb.calls()] == ["SendForceFeedbackCommand"]  # refused, but attempted


def test_game_holding_the_base_blocks_exclusive_but_not_reading(sim):
    assert sim.wheel.hold_exclusive()
    sink = di.DirectInputSink()
    assert not sink.open(500)
    assert "DIERR_OTHERAPPHASPRIO" in sink.last_error
    r = Reader(fanatec(sim)[0])
    assert r.call("Acquire") == di.DI_OK and r.read()[0] == di.DI_OK
    r.close()
    sim.assert_no_leaks()


def test_injected_faults_and_thread_check(sim):
    import threading

    sim.wheel.fail("Acquire", di.DIERR_INPUTLOST)
    r = Reader(fanatec(sim)[0])
    assert r.call("Acquire") == di.DIERR_INPUTLOST
    assert r.call("Acquire") == di.DI_OK
    sim.assert_single_thread()
    t = threading.Thread(target=r.read)
    t.start()
    t.join()
    with pytest.raises(AssertionError, match="off the owner thread"):
        sim.assert_single_thread()
    r.close()


def test_leak_check_names_what_is_left_open(sim):
    r = Reader(fanatec(sim)[0])
    with pytest.raises(AssertionError, match="IDirectInputDevice8W"):
        sim.assert_no_leaks()
    r.close()
    sim.assert_no_leaks()


# --------------------------------------------------------------------------
# install


def test_install_is_explicit_and_undone():
    assert di._backend is None and vj._backend is None
    with FanatecSim() as sim:
        assert di.available() and vj.load_dll() is sim.vjoy
        with pytest.raises(RuntimeError, match="already installed"):
            FanatecSim().install()
    assert di._backend is None and vj._backend is None
    if not di.sys.platform == "win32":
        assert not di.available() and di.list_ff_devices() == []
        with pytest.raises(vj.VJoyError):
            vj.load_dll("/nonexistent/vJoyInterface.dll")


def test_vjoy_false_keeps_the_real_dll():
    with FanatecSim(vjoy=False):
        assert vj._backend is None and di._backend is not None


# --------------------------------------------------------------------------
# tracks


def test_track_builders_and_interpolation():
    c = (Track().hold(1).ramp(steer_deg=12, over=0.1).hold(0.5).flick(steer_deg=-6, rate_dps=200)
         .pedal("brake", 0.12, hold=0.06).press(3, hold=0.1).unplug(for_s=1.5)
         .compile({"steer_deg": 2.0}))
    assert c.value("steer_deg", 0.5) is None  # untouched yet: the wheel's own value holds
    assert c.value("steer_deg", 1.0) == 2.0
    assert c.value("steer_deg", 1.05) == pytest.approx(7.0)
    assert c.value("steer_deg", 1.6 + 0.015) == pytest.approx(9.0)
    assert c.value("steer_deg", 1.7) == pytest.approx(6.0)
    assert c.value("brake", 1.64) == 0.12 and c.value("brake", 1.70) == 0.0
    assert c.value("button:3", 1.65) is None
    assert c.value("button:3", 1.70) is True and c.value("button:3", 1.80) is False
    assert c.transitions("plugged", 0, 99) == [(pytest.approx(1.79), False), (pytest.approx(3.29), True)]
    assert c.value("throttle", 1.0) is None  # never touched: the wheel's own value holds
    assert c.duration == pytest.approx(3.29)


def test_noise_is_bounded_and_repeatable():
    a = Track().noise(seconds=60, seed=7, steer_deg=2).compile({"steer_deg": 10})
    b = Track().noise(seconds=60, seed=7, steer_deg=2).compile({"steer_deg": 10})
    values = [a.value("steer_deg", i * 0.0137) for i in range(4000)]
    assert values == [b.value("steer_deg", i * 0.0137) for i in range(4000)]
    assert all(8 <= v <= 12 for v in values) and max(values) - min(values) > 3
    assert a.value("steer_deg", 60) == 10


def test_track_files_round_trip(tmp_path):
    track = Track().set(steer_deg=0).hold(1).set(steer_deg=30).ramp(over=0.5, throttle=1).press(1)
    track.save(tmp_path / "t.jsonl")
    a, b = track.compile(), Track.load(tmp_path / "t.jsonl").compile()
    for i in range(200):
        t = i * 0.01
        assert a.values(t) == pytest.approx(b.values(t)), t
    (tmp_path / "k.json").write_text(json.dumps({"keyframes": [{"t": 0, "steer_deg": 0}, {"t": 2, "steer_deg": 10}]}))
    assert Track.load(tmp_path / "k.json").compile().value("steer_deg", 1) == 5
    with pytest.raises(ValueError, match="unknown track channel"):
        Track().set(gear=3)


def test_track_from_a_simsteer_run_log(tmp_path):
    rows = [{"t": 10.0, "rim_axis": 0.0, "phys_throttle": 0.0}, {"t": 10.5, "fps": 20},
            {"t": 11.0, "rim_axis": 0.1, "phys_throttle": 0.5, "phys_brake": None}]
    (tmp_path / "run.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    c = Track.from_simsteer_jsonl(tmp_path / "run.jsonl", rim_range_deg=900).compile()
    assert c.value("steer_deg", 0.5) == pytest.approx(22.5) and c.value("throttle", 1.0) == 0.5


def test_wheel_plays_a_track_on_the_simulator_clock():
    clock = Clock()
    with FanatecSim("single", clock=clock) as sim:
        r = Reader(fanatec(sim)[0])
        r.call("Acquire")
        sim.wheel.set(steer_deg=90)
        sim.wheel.play(Track().ramp(steer_deg=0, over=1.0).unplug(for_s=0.5))
        clock.t += 0.5
        assert sim.wheel.state()["steer_deg"] == pytest.approx(45)
        assert r.axes()[0] == round((0.5 + 45 / 900) * 65535)
        clock.t += 0.7
        assert r.read()[0] == di.DIERR_INPUTLOST
        assert sim.ffb.events[-1] == {"t": 101.0, "ev": "gate", "on": False, "why": "unplugged"}
        clock.t += 0.5
        assert sim.wheel.track_done and sim.wheel.state()["steer_deg"] == 0
        assert sim.wheel.plugged and sim.wheel.generation == 1
        r.close()
        sim.wheel.set(steer_deg=-10)  # set() after a finished track
        assert sim.wheel.state()["steer_deg"] == -10
        sim.wheel.play(Track().unplug())  # a plug event at the very start
        assert fanatec(sim) == []


# --------------------------------------------------------------------------
# vJoy


def test_vjoy_setup_rows_from_the_real_check():
    cases = [
        ({}, True, "vJoy device 1 force feedback on, 11 effects"),
        ({"ffb": False}, False, "vJoy device 1 has force feedback off"),
        ({"effects": [0x26]}, False, "vJoy device 1 lacks effects"),
        ({"devices": (2,)}, False, "vJoy device 1 does not exist"),
        ({"enabled": False}, False, "the vJoy driver is not running"),
    ]
    for options, ok, row in cases:
        with FanatecSim(vjoy_options=options):
            check = vj.vjoy_ffb_check(vj.probe_vjoy(device_id=1), 1)
            assert check.ok is ok and check.row.startswith(row), (options, check)


def test_vjoy_owned_by_another_program():
    with FanatecSim() as sim:
        sim.game.take_vjoy()
        check = vj.vjoy_ffb_check(vj.probe_vjoy(device_id=1), 1)
        assert not check.ok and check.row.endswith("owned by process 4242")
        rx = vj.VJoyFfbReceiver(1)
        assert not rx.open().ok


def test_game_sees_what_is_fed_to_vjoy_and_the_vjoy_di_view_echoes_it():
    with FanatecSim() as sim:
        dll = vj.load_dll()
        assert dll.AcquireVJD(1) == 1
        dll.SetAxis.argtypes = [ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32]  # settable like a CDLL export
        assert dll.SetAxis(0x6000, 1, 0x30) and dll.SetAxis(0x8000, 1, 0x31) and dll.SetBtn(1, 1, 2)
        assert sim.game.steer() == pytest.approx(0.5, abs=1e-3) and sim.game.pedal("y") == 1.0
        assert sim.game.buttons == {2}
        vjoy = next(d for d in di.attached_ff_devices() if d.is_vjoy)
        r = Reader(vjoy)
        r.call("Acquire")
        assert r.axes()[0] == round((0x6000 - 1) / (0x8000 - 1) * 65535)
        r.close()
        assert dll.ResetVJD(1) and sim.game.axes["x"] == 0x4000 and sim.game.buttons == set()
        dll.RelinquishVJD(1)
        assert not dll.SetAxis(0x1000, 1, 0x30)  # not owned any more
