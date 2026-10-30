import ctypes
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("wheel_probe", ROOT / "tools" / "wheel_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["wheel_probe"] = module
    spec.loader.exec_module(module)
    return module


wp = _load()
IS_64 = ctypes.sizeof(ctypes.c_void_p) == 8


def ramp(n, hz, axes, foreign=True, start=0.0):
    """n samples at hz; each axis is a function of the sample index."""
    return [(start + i / hz, tuple(f(i) for f in axes), (0,), foreign) for i in range(n)]


# --- import and CLI -------------------------------------------------------


def test_import_does_not_load_windows_or_pygame():
    assert "pygame" not in sys.modules
    assert not hasattr(wp, "windll")


def test_help_prints_usage(capsys):
    with pytest.raises(SystemExit) as exc:
        wp.main(["--help"])
    assert exc.value.code == 0
    assert "--ffb-test" in capsys.readouterr().out


# --- FFB level safety -----------------------------------------------------


@pytest.mark.parametrize("level", [5, 7.5, 10, 15])
def test_levels_up_to_15_are_accepted(level):
    assert wp.validate_ffb_level(level) == level


@pytest.mark.parametrize("level", [15.01, 16, 100, 0, -5, float("nan"), float("inf")])
def test_levels_outside_range_are_refused(level):
    with pytest.raises(ValueError):
        wp.validate_ffb_level(level)


def test_ffb_level_16_is_refused_before_any_hardware(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("hardware path reached")

    monkeypatch.setattr(wp, "run_ffb_test", boom)
    monkeypatch.setattr(wp, "probe_vjoy", boom)
    assert wp.main(["--ffb-test", "--ffb-level", "16"]) == wp.EXIT_USAGE
    err = capsys.readouterr().err
    assert "refusing --ffb-level 16" in err and "15 percent" in err


def test_ffb_level_without_ffb_test_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        wp.main(["--ffb-level", "5"])
    assert exc.value.code == 2


def test_default_level_is_5(monkeypatch):
    seen = {}

    def fake(level, only_vid, countdown_s, with_sdl=False):
        seen["level"] = level
        return {"attempted": False, "result": "no_device", "error": "none"}, []

    monkeypatch.setattr(wp, "run_ffb_test", fake)
    monkeypatch.setattr(wp, "probe_vjoy", lambda: {"available": False})
    wp.main(["--ffb-test"])
    assert seen["level"] == 5


def test_spring_is_capped_at_the_level():
    p = wp.spring_parameters(15)
    assert p["coefficient"] == p["saturation"] == 1500
    assert p["gain"] == wp.DI_FFNOMINALMAX
    assert p["offset"] == 0
    assert p["duration_us"] == 200_000
    assert wp.spring_parameters(5)["coefficient"] == 500
    with pytest.raises(ValueError):
        wp.spring_parameters(16)


# --- axis summary ---------------------------------------------------------


def test_summary_flags_frozen_axis_and_computes_read_rate():
    samples = ramp(1001, 1000.0, [lambda i: i, lambda i: 0])
    s = wp.summarize_samples(samples, ["steer", "throttle"])
    steer, throttle = s["axes"]
    assert not steer["frozen"] and steer["changed_in_background"]
    assert throttle["frozen"] and not throttle["changed_in_background"]
    assert s["read_hz"] == pytest.approx(1000.0, rel=1e-6)
    assert s["poll_hz"] == pytest.approx(1000.0, rel=1e-6)
    assert s["background_ok"] is True
    assert s["max_gap_ms"] == pytest.approx(1.0)


def test_read_rate_counts_device_updates_not_polls():
    # Polled at 1 kHz, the value only changes every 4th poll (250 Hz device).
    samples = ramp(2001, 1000.0, [lambda i: i // 4])
    s = wp.summarize_samples(samples)
    assert s["poll_hz"] == pytest.approx(1000.0, rel=1e-6)
    assert s["read_hz"] == pytest.approx(250.0, rel=1e-6)


def test_read_rate_ignores_pauses_between_movements():
    moving = ramp(200, 500.0, [lambda i: i])
    still = ramp(500, 500.0, [lambda i: 199], start=0.4)
    again = ramp(200, 500.0, [lambda i: 1000 + i], start=1.4)
    s = wp.summarize_samples(moving + still + again)
    assert s["read_hz"] == pytest.approx(500.0, rel=1e-3)


def test_changes_only_with_own_window_focused_are_not_background_ok():
    samples = ramp(500, 1000.0, [lambda i: i], foreign=False)
    s = wp.summarize_samples(samples, ["steer"])
    assert s["axes"][0]["changes"] == 499
    assert s["axes"][0]["changed_in_background"] is False
    assert s["background_ok"] is False
    assert s["background_read_hz"] is None
    assert s["foreign_focus_fraction"] == 0


def test_stall_shows_in_max_gap():
    samples = ramp(100, 1000.0, [lambda i: i]) + ramp(100, 1000.0, [lambda i: 500 + i], start=0.35)
    s = wp.summarize_samples(samples)
    assert s["max_gap_ms"] == pytest.approx(251.0, abs=0.01)


def test_buttons_changed_are_reported():
    samples = [(0.0, (0,), (0, 0), True), (0.001, (0,), (0, 1), True), (0.002, (0,), (0, 0), False)]
    s = wp.summarize_samples(samples)
    assert s["buttons_changed"] == [1]
    assert s["buttons_changed_in_background"] == [1]


def test_empty_samples_summary():
    s = wp.summarize_samples([], ["steer"])
    assert s["polls"] == 0 and s["background_ok"] is False
    assert s["axes"][0]["frozen"] is True


def test_backend_summary():
    moving = wp.summarize_samples(ramp(300, 1000.0, [lambda i: i]))
    still = wp.summarize_samples(ramp(300, 1000.0, [lambda i: 0]))
    out = wp.summarize_backend([{"fanatec": True, "summary": moving}, {"fanatec": True, "summary": still}])
    assert out["background_ok"] is True and out["summary_basis"] == "fanatec"
    assert out["read_hz"] == pytest.approx(1000.0, rel=1e-6)
    assert wp.summarize_backend([])["background_ok"] is None


def test_backend_summary_ignores_a_busy_vjoy_device():
    moving = wp.summarize_samples(ramp(300, 1000.0, [lambda i: i]))
    still = wp.summarize_samples(ramp(300, 1000.0, [lambda i: 0]))
    out = wp.summarize_backend([
        {"fanatec": False, "virtual": True, "summary": moving},
        {"fanatec": True, "virtual": False, "summary": still},
    ])
    assert out["summary_basis"] == "fanatec"
    assert out["background_ok"] is False and out["read_hz"] is None
    only_vjoy = wp.summarize_backend([{"fanatec": False, "virtual": True, "summary": moving}])
    assert only_vjoy["summary_basis"] == "all"


# --- ID parsing -----------------------------------------------------------


@pytest.mark.parametrize("text,value", [("0x0EB7", 0x0EB7), ("0eb7", 0x0EB7), ("FFFF", 0xFFFF)])
def test_parse_vid(text, value):
    assert wp.parse_vid(text) == value


@pytest.mark.parametrize("text", ["zz", "0x10000", ""])
def test_parse_vid_rejects(text):
    with pytest.raises(ValueError):
        wp.parse_vid(text)


def test_bad_only_vid_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        wp.main(["--only-vid", "nothex"])
    assert exc.value.code == 2


def test_sdl_guid_vid_pid():
    # bus 0x0003 (USB), crc 0, vendor 0x0EB7, product 0x0020, version 0x0111
    guid = "03000000" + "b70e0000" + "20000000" + "11010000"
    assert wp.sdl_vid_pid(guid) == (0x0EB7, 0x0020)
    assert wp.sdl_vid_pid("05000000" + "46616e61746563000000000000") == (None, None)
    assert wp.sdl_vid_pid("not hex") == (None, None)


def test_dinput_product_guid_vid_pid():
    assert wp.dinput_vid_pid(0x00200EB7) == (0x0EB7, 0x0020)


def test_fanatec_and_vjoy_flags():
    assert wp.is_fanatec(0x0EB7)
    assert not wp.is_fanatec(0x046D, "Fanatec lookalike")
    assert wp.is_fanatec(None, "FANATEC Podium")
    assert wp.is_vjoy(0x1234, 0xBEAD)


def test_hresult_names():
    assert wp.hresult_name(0x80070005) == "DIERR_OTHERAPPHASPRIO"
    assert wp.hresult_name(-2147024891) == "DIERR_OTHERAPPHASPRIO"
    assert wp.hresult_name(0x8007000C) == "DIERR_NOTACQUIRED"
    assert wp.hresult_name(0x8007001E) == "DIERR_INPUTLOST"
    assert wp.hresult_name(0x80040209) == "DIERR_UNPLUGGED"
    assert wp.hresult_name(0x12345678) == "0x12345678"
    assert wp.hresult_failed(0x80004005) and not wp.hresult_failed(1)


def test_vjoy_version_format():
    assert wp.format_vjoy_version(0x0219) == "2.1.9"
    assert wp.format_vjoy_version(0x0216) == "2.1.6"


def test_vjoy_off_windows_reports_unavailable():
    if sys.platform == "win32":
        pytest.skip("Linux-only expectation")
    assert wp.probe_vjoy()["available"] is False


# --- COM layout ------------------------------------------------------------

DINPUT_H_SLOTS = {
    ("IDirectInput8W", "Release"): 2,
    ("IDirectInput8W", "CreateDevice"): 3,
    ("IDirectInput8W", "EnumDevices"): 4,
    ("IDirectInputDevice8W", "Release"): 2,
    ("IDirectInputDevice8W", "GetCapabilities"): 3,
    ("IDirectInputDevice8W", "EnumObjects"): 4,
    ("IDirectInputDevice8W", "GetProperty"): 5,
    ("IDirectInputDevice8W", "SetProperty"): 6,
    ("IDirectInputDevice8W", "Acquire"): 7,
    ("IDirectInputDevice8W", "Unacquire"): 8,
    ("IDirectInputDevice8W", "GetDeviceState"): 9,
    ("IDirectInputDevice8W", "SetDataFormat"): 11,
    ("IDirectInputDevice8W", "SetCooperativeLevel"): 13,
    ("IDirectInputDevice8W", "GetObjectInfo"): 14,
    ("IDirectInputDevice8W", "CreateEffect"): 18,
    ("IDirectInputDevice8W", "EnumEffects"): 19,
    ("IDirectInputDevice8W", "GetForceFeedbackState"): 21,
    ("IDirectInputDevice8W", "SendForceFeedbackCommand"): 22,
    ("IDirectInputDevice8W", "Poll"): 25,
    ("IDirectInputEffect", "Release"): 2,
    ("IDirectInputEffect", "SetParameters"): 6,
    ("IDirectInputEffect", "Start"): 7,
    ("IDirectInputEffect", "Stop"): 8,
    ("IDirectInputEffect", "Unload"): 11,
}


def test_every_used_slot_matches_dinput_h():
    assert set(wp.USED_METHODS) == set(DINPUT_H_SLOTS)
    for (iface, method), slot in DINPUT_H_SLOTS.items():
        assert wp.vtable_slot(iface, method) == slot, (iface, method)
    assert len(wp.VTABLES["IDirectInputEffect"]) == 13
    assert len(wp.VTABLES["IDirectInputDevice8W"]) == 32
    assert len(wp.VTABLES["IDirectInput8W"]) == 11


def test_com_calls_only_use_listed_methods():
    source = (ROOT / "tools" / "wheel_probe.py").read_text()
    import re

    calls = set(re.findall(r'_com_call\(\s*[\w.]+,\s*"(\w+)",\s*"(\w+)"', source))
    device_calls = {("IDirectInputDevice8W", m) for m in re.findall(r'self\.call\(\s*"(\w+)"', source)}
    assert calls | device_calls <= set(wp.USED_METHODS)


def test_flag_values_from_dinput_h():
    assert wp.DIEP_START == 0x20000000
    assert wp.DIEP_TYPESPECIFICPARAMS == 0x100
    assert wp.DIEFF_OBJECTIDS == 0x1 and wp.DIEFF_CARTESIAN == 0x10
    assert wp.DIPH_BYOFFSET == 1
    assert wp.DISCL_EXCLUSIVE | wp.DISCL_BACKGROUND == 0x9
    assert wp.DISCL_NONEXCLUSIVE | wp.DISCL_BACKGROUND == 0xA


def test_guid_bytes_match_windows_layout():
    g = wp.GUID.parse(wp.GUID_Spring)
    assert (g.Data1, g.Data2, g.Data3) == (0x13541C27, 0x8E33, 0x11D0)
    assert bytes(g.Data4) == bytes([0x9A, 0xD0, 0x00, 0xA0, 0xC9, 0xA0, 0x6E, 0x35])
    assert str(g) == wp.GUID_Spring


def test_struct_sizes_match_dinput_h():
    assert ctypes.sizeof(wp.GUID) == 16
    assert ctypes.sizeof(wp.DIDEVICEINSTANCEW) == 1100
    assert ctypes.sizeof(wp.DIDEVICEOBJECTINSTANCEW) == 576
    assert ctypes.sizeof(wp.DIDEVCAPS) == 44
    assert ctypes.sizeof(wp.DICONDITION) == 24
    assert ctypes.sizeof(wp.DIEFFECTINFOW) == 552
    assert ctypes.sizeof(wp.DIPROPDWORD) == 20
    assert ctypes.sizeof(wp.JOYSTATE) == 176
    assert ctypes.sizeof(wp.DIOBJECTDATAFORMAT) == (24 if IS_64 else 16)
    assert ctypes.sizeof(wp.DIDATAFORMAT) == (32 if IS_64 else 24)
    assert ctypes.sizeof(wp.DIEFFECT) == (80 if IS_64 else 56)


def test_data_format_covers_axes_povs_buttons():
    fmt, _keep = wp.joystick_data_format()
    assert fmt.dwDataSize == 176 and fmt.dwDataSize % 4 == 0
    assert fmt.dwNumObjs == 8 + 4 + 128
    assert fmt.dwFlags == wp.DIDF_ABSAXIS
    rows = [fmt.rgodf[i] for i in range(fmt.dwNumObjs)]
    assert [r.dwOfs for r in rows[:8]] == [0, 4, 8, 12, 16, 20, 24, 28]
    assert str(rows[0].pguid.contents) == wp.GUID_XAxis
    assert rows[0].dwType == 0x80FFFF03
    assert rows[0].dwFlags == wp.DIDOI_ASPECTPOSITION
    assert rows[8].dwOfs == 32 and rows[8].dwType == 0x80FFFF10
    assert rows[12].dwOfs == 48 and not rows[12].pguid and rows[12].dwType == 0x80FFFF0C
    assert rows[-1].dwOfs == 48 + 127


def test_spring_effect_struct():
    params = wp.spring_parameters(5)
    # The DD2 reports its wheel actuator as object ID 0x01000002.
    eff, _keep = wp._spring_effect(params, 0x01000002)
    assert eff.dwFlags == wp.DIEFF_CARTESIAN | wp.DIEFF_OBJECTIDS
    assert eff.dwDuration == 200_000
    assert eff.cAxes == 1 and eff.rgdwAxes[0] == 0x01000002
    assert eff.cbTypeSpecificParams == ctypes.sizeof(wp.DICONDITION)
    cond = ctypes.cast(eff.lpvTypeSpecificParams, ctypes.POINTER(wp.DICONDITION)).contents
    assert (cond.lOffset, cond.lPositiveCoefficient, cond.lNegativeCoefficient) == (0, 500, 500)
    assert (cond.dwPositiveSaturation, cond.dwNegativeSaturation) == (500, 500)


# --- report assembly and runs with mocked devices ---------------------------

WHEEL = dict(backend="dinput", key="dinput:wheel", name="FANATEC Podium Wheel Base DD2", vid=0x0EB7, pid=0x0020,
             axis_names=["X", "Y", "Z"], buttons=12)
PEDALS = dict(backend="dinput", key="dinput:pedals", name="FANATEC ClubSport Pedals", vid=0x0EB7, pid=0x183B,
              axis_names=["X", "Y", "Z"], buttons=0)
VJOY = dict(backend="dinput", key="dinput:vjoy", name="vJoy Device", vid=0x1234, pid=0xBEAD,
            axis_names=["X"], buttons=8)


class FakeBackend:
    """Replays scripted polls. `script` is a list of (key, axes, buttons) per
    poll, or an exception to raise."""

    name = "dinput"

    def __init__(self, only_vid, devices=(), script=()):
        self.only_vid = only_vid
        self.all = [wp.device_entry(**d) for d in devices]
        self.script = list(script)
        self.errors = []
        self.closed = False

    def open(self):
        return wp.filter_devices(self.all, self.only_vid)

    def poll(self, t, events):
        if not self.script:
            return []
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        if step == "unplug":
            events.append({"t": t, "backend": self.name, "device": "pedals", "event": "lost"})
            return []
        return step

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self, dt=0.001):
        self.t = 0.0
        self.dt = dt

    def __call__(self):
        self.t += self.dt
        return self.t


def _install(monkeypatch, backend_factory, clock=None):
    monkeypatch.setattr(wp, "BACKENDS", {"sdl": backend_factory, "dinput": backend_factory})
    monkeypatch.setattr(wp, "probe_vjoy", lambda device_id=1: {"available": False, "error": "test"})
    monkeypatch.setattr(wp, "countdown", lambda *a, **k: None)
    monkeypatch.setattr(wp, "beep", lambda *a: None)
    clock = clock or FakeClock()
    real_log = wp.log_inputs
    monkeypatch.setattr(wp, "log_inputs", lambda b, s, f: real_log(b, s, f, clock=clock, sleep=lambda _: None))


def test_build_report_without_fanatec_notes_it():
    sections = {"dinput": {"available": True, "devices": [wp.device_entry(**VJOY)]}}
    report = wp.build_report(mode="log", args={}, backends=sections, vjoy={"available": False}, host={"platform": "test"})
    assert report["fanatec_found"] is False
    assert any("unplugged" in n for n in report["notes"])
    assert report["devices"][0]["virtual"] is True
    assert "devices" not in report["backends"]["dinput"]
    json.dumps(report)


def test_only_vid_matching_nothing_reports_zero_devices(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, lambda only_vid: FakeBackend(only_vid, [WHEEL, PEDALS, VJOY]))
    out = tmp_path / "probe-empty.json"
    assert wp.main(["--only-vid", "0xFFFF", "--backend", "dinput", "--report", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["devices"] == []
    assert report["fanatec_found"] is False
    assert "No Fanatec wheel found" in capsys.readouterr().err


def test_no_fanatec_plain_run_exits_0_with_a_plain_note(monkeypatch, tmp_path, capsys):
    script = [[("dinput:vjoy", (i,), (0,) * 8)] for i in range(50)]
    _install(monkeypatch, lambda only_vid: FakeBackend(only_vid, [VJOY], script))
    out = tmp_path / "probe.json"
    assert wp.main(["--backend", "dinput", "--seconds", "0.1", "--report", str(out)]) == 0
    err = capsys.readouterr().err
    assert "No Fanatec wheel found" in err and "Traceback" not in err
    report = json.loads(out.read_text())
    assert report["notes"] and report["devices"][0]["name"] == "vJoy Device"


def test_logging_run_summarises_wheel_and_pedals(monkeypatch, tmp_path):
    script = []
    for i in range(300):
        script.append([
            ("dinput:wheel", (i, 0, 0), (0,) * 12),
            ("dinput:pedals", (0, i // 2, 0), ()),
        ])
    _install(monkeypatch, lambda only_vid: FakeBackend(only_vid, [WHEEL, PEDALS], script))
    out = tmp_path / "probe.json"
    assert wp.main(["--backend", "dinput", "--seconds", "0.25", "--report", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["fanatec_found"] is True and report["notes"] == []
    wheel, pedals = report["devices"]
    assert wheel["summary"]["axes"][0]["frozen"] is False
    assert wheel["summary"]["axes"][1]["frozen"] is True
    assert pedals["summary"]["axes"][1]["changes"] > 0
    assert wheel["summary"]["read_hz"] == pytest.approx(1000.0, rel=0.01)
    assert report["backends"]["dinput"]["read_hz"] == pytest.approx(1000.0, rel=0.01)
    samples = (tmp_path / "probe.samples.jsonl").read_text().splitlines()
    assert json.loads(samples[0])["device"] == "dinput:wheel"


def test_ctrl_c_during_logging_writes_partial_report(monkeypatch, tmp_path):
    script = [[("dinput:wheel", (i, 0, 0), (0,) * 12)] for i in range(100)] + [KeyboardInterrupt()]
    _install(monkeypatch, lambda only_vid: FakeBackend(only_vid, [WHEEL], script))
    out = tmp_path / "interrupt.json"
    assert wp.main(["--backend", "dinput", "--seconds", "20", "--report", str(out)]) == wp.EXIT_INTERRUPTED
    report = json.loads(out.read_text())
    assert report["interrupted"] is True
    assert report["devices"][0]["summary"]["polls"] == 100


def test_unplug_event_is_logged_and_wheel_keeps_logging(monkeypatch, tmp_path):
    both = lambda i: [("dinput:wheel", (i, 0, 0), (0,) * 12), ("dinput:pedals", (0, i, 0), ())]
    wheel_only = lambda i: [("dinput:wheel", (i, 0, 0), (0,) * 12)]
    script = [both(i) for i in range(50)] + ["unplug"] + [wheel_only(i) for i in range(50, 150)]
    _install(monkeypatch, lambda only_vid: FakeBackend(only_vid, [WHEEL, PEDALS], script))
    out = tmp_path / "unplug.json"
    assert wp.main(["--backend", "dinput", "--seconds", "0.2", "--report", str(out)]) == 0
    report = json.loads(out.read_text())
    assert [e["event"] for e in report["events"]] == ["lost"]
    wheel, pedals = report["devices"]
    assert wheel["summary"]["polls"] > pedals["summary"]["polls"] == 50


def test_missing_pygame_gives_install_hint(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "pygame", None)
    monkeypatch.setattr(wp, "probe_vjoy", lambda device_id=1: {"available": False})
    monkeypatch.setattr(wp, "BACKENDS", {"sdl": wp.SdlBackend, "dinput": wp.DInputBackend})
    out = tmp_path / "p.json"
    assert wp.main(["--backend", "sdl", "--report", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["backends"]["sdl"]["available"] is False
    assert "pip install pygame-ce" in report["backends"]["sdl"]["error"]
    assert "pip install pygame-ce" in capsys.readouterr().err


def test_ffb_without_device_exits_nonzero_with_one_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(wp, "probe_vjoy", lambda device_id=1: {"available": False})
    out = tmp_path / "ffb-empty.json"
    code = wp.main(["--ffb-test", "--only-vid", "0xFFFF", "--report", str(out)])
    assert code != 0
    err = capsys.readouterr().err
    assert "Traceback" not in err
    report = json.loads(out.read_text())
    assert report["ffb"]["result"] in {"no_device", "unavailable"}
    assert report["ffb"]["attempted"] is False


def test_pick_ffb_device_prefers_fanatec_and_never_vjoy():
    wheel = wp.device_entry(**WHEEL)
    vjoy = wp.device_entry(**VJOY)
    assert wp.pick_ffb_device([vjoy, wheel], None) is wheel
    assert wp.pick_ffb_device([vjoy], None) is None
    assert wp.pick_ffb_device([vjoy], 0x1234) is None
    assert wp.pick_ffb_device([wheel], 0xFFFF) is None


def test_with_sdl_without_ffb_test_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        wp.main(["--with-sdl"])
    assert exc.value.code == 2


class FakeDevice:
    def __init__(self):
        self.commands = []

    def ff_command(self, flags):
        self.commands.append(flags)
        return 0


def test_spring_guard_repeats_stop_sends_stopall_when_stop_fails_and_never_touches_released(monkeypatch):
    calls = []

    def fake_com(ptr, iface, method, argtypes, *args, **kw):
        calls.append(method)
        raise OSError("access violation")

    monkeypatch.setattr(wp, "_com_call", fake_com)
    dev = FakeDevice()
    guard = wp.SpringGuard(effect=1, device=dev)
    assert guard._on_console_event(0) == 0
    guard.stop("done")
    assert calls == ["Stop", "Stop"]
    assert dev.commands == [wp.DISFFC_STOPALL] * 2
    assert guard.reason == "console event 0"
    guard.release()
    assert calls[-3:] == ["Stop", "Unload", "Release"] and guard.released
    guard.stop("late")
    guard.release()
    assert calls.count("Release") == 1 and len(dev.commands) == 3


def test_autocenter_off_rule():
    assert wp.autocenter_off_ok(0) and wp.autocenter_off_ok(1)
    assert wp.autocenter_off_ok(0x80004001)
    assert not wp.autocenter_off_ok(0x80070057)


def test_pick_ffb_device_prefers_driving_type():
    other = wp.device_entry(**dict(PEDALS, key="dinput:p"), type="supplemental")
    base = wp.device_entry(**WHEEL, type="driving")
    assert wp.pick_ffb_device([other, base], None) is base


def test_ff_state_flags():
    assert wp.ff_state_flags(0x00000212) == ["STOPPED", "ACTUATORSON", "SAFETYSWITCHOFF"]
    assert wp.ff_state_flags(0) == []


class FakePygame:
    """Joysticks appear only after a few pumps, as SDL's WGI driver does."""

    JOYDEVICEADDED, JOYDEVICEREMOVED = 1541, 1542
    error = RuntimeError

    def __init__(self, guids, appear_after=3):
        self.guids = list(guids)
        self.pumps = 0
        self.appear_after = appear_after
        self.queue = []
        pg = self

        class Joystick:
            def __init__(self, index):
                self.index = index

            def init(self):
                pass

            def quit(self):
                pass

            def get_instance_id(self):
                return 100 + self.index

            def get_guid(self):
                return pg.guids[self.index]

            def get_name(self):
                return f"pad{self.index}"

            def get_numaxes(self):
                return 2

            def get_numbuttons(self):
                return 1

            def get_numhats(self):
                return 0

            def get_axis(self, i):
                return 0.0

            def get_button(self, i):
                return 0

        class _Joy:
            Joystick = None

            @staticmethod
            def init():
                pass

            @staticmethod
            def get_count():
                return len(pg.guids) if pg.pumps >= pg.appear_after else 0

        class _Event:
            @staticmethod
            def pump():
                pg.pumps += 1

            @staticmethod
            def clear():
                pg.queue.clear()

            @staticmethod
            def get():
                out, pg.queue[:] = list(pg.queue), []
                return out

        class _Display:
            @staticmethod
            def init():
                pass

        _Joy.Joystick = Joystick
        self.joystick, self.event, self.display = _Joy, _Event, _Display
        self.version = type("v", (), {"ver": "fake"})
        self.IS_CE = True

    @staticmethod
    def get_sdl_version():
        return (2, 32, 10)

    def quit(self):
        pass


FANATEC_GUID = "03000000" + "b70e0000" + "07000000" + "11010000"
OTHER_GUID = "03000000" + "6d040000" + "1bc20000" + "11010000"


def test_sdl_waits_for_late_devices_and_opens_hotplugged_ones(monkeypatch):
    fake = FakePygame([FANATEC_GUID, OTHER_GUID])
    monkeypatch.setitem(sys.modules, "pygame", fake)
    monkeypatch.setattr(wp.time, "sleep", lambda s: None)
    backend = wp.SdlBackend(only_vid=None)
    devices = backend.open()
    assert [d["vid"] for d in devices] == [0x0EB7, 0x046D]
    assert devices[0]["fanatec"] is True
    assert wp.os.environ["SDL_DIRECTINPUT_ENABLED"] == "0"
    assert backend.info["directinput_driver"] == "0"

    fake.guids.append(FANATEC_GUID)
    fake.queue.append(type("e", (), {"type": fake.JOYDEVICEADDED, "device_index": 2})())
    events = []
    rows = backend.poll(1.0, events)
    assert [e["event"] for e in events] == ["added"]
    assert len(rows) == 3

    fake.queue.append(type("e", (), {"type": fake.JOYDEVICEREMOVED, "instance_id": 101})())
    events = []
    rows = backend.poll(2.0, events)
    assert [e["event"] for e in events] == ["removed"] and len(rows) == 2


def test_sdl_only_vid_filters_late_devices(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygame", FakePygame([FANATEC_GUID, OTHER_GUID]))
    monkeypatch.setattr(wp.time, "sleep", lambda s: None)
    assert [d["vid"] for d in wp.SdlBackend(only_vid=0x046D).open()] == [0x046D]
