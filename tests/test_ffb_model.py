"""The vJoy PID -> DirectInput passthrough model, fed with packet fixtures.

Each fixture in tests/fixtures/vjoy_ffb_packets/ replays through the model
and must reproduce its .expected.txt call log line for line. Fixtures say
in their header whether they were recorded on the simrig or built from the
vJoy SDK layout.
"""

from __future__ import annotations

import struct
from dataclasses import replace

import pytest

from simsteer.core import ffb_model as fm
from simsteer.runtime.ffb.vjoy_ffb import crosscheck
import ffb_fixtures as fx

NAMES = fx.fixture_names()


def decoded(name: str) -> list[tuple[dict, fm.Record]]:
    return [(p, fm.decode_packet(p["cmd"], bytes.fromhex(p["data"]))) for p in fx.load(name)["packets"]]


def feed(model: fm.FfbModel, t: float, pkt: tuple[int, bytes]) -> list[fm.DiCall]:
    return model.apply(fm.decode_packet(*pkt), t)


def texts(calls: list[fm.DiCall]) -> list[str]:
    return [c.text() for c in calls]


# --------------------------------------------------------------------------
# fixtures


def test_fixtures_exist():
    assert "synthetic_all_types" in NAMES and "synthetic_directions" in NAMES


@pytest.mark.parametrize("name", NAMES)
def test_fixture_replays_to_expected_log(name):
    _, lines = fm.run_packets(fx.load(name)["packets"])
    expected = fx.expected_path(name).read_text(encoding="utf-8").splitlines()
    assert lines == expected


@pytest.mark.parametrize("name", NAMES)
def test_fixture_names_its_source(name):
    source = fx.load(name)["source"]
    assert source["kind"] in ("recorded", "synthetic")
    assert name.startswith(source["kind"])
    if source["kind"] == "synthetic":
        assert fx.SYNTHETIC_NOTE in source["note"]


@pytest.mark.parametrize("name", NAMES)
def test_recorded_packets_decode_like_the_dll(name):
    """Recorded fixtures carry the DLL's own Ffb_h_* reading of each packet."""
    checked = 0
    for p, rec in decoded(name):
        if p.get("dll") is not None:
            assert crosscheck(rec, p["dll"]) == [], p["data"]
            checked += 1
    if fx.load(name)["source"]["kind"] == "recorded":
        assert checked == len(fx.load(name)["packets"])


ROUND_TRIP = [n for n in NAMES if (fx.FIXTURES / f"{n}.sender.txt").exists()]
IGNORED = {"axes", "direction", "trigger_button", "iterations", "duration", "gain"}


@pytest.mark.parametrize("name", ROUND_TRIP)
def test_recorded_round_trip_matches_the_sender(name):
    """Each effect the sender started on vJoy starts on the wheel with the
    same parameters, apart from the safety window and gain rounding."""
    model = fm.FfbModel()
    bridged = []
    for p in fx.load(name)["packets"]:
        bridged += texts(model.apply(fm.decode_packet(p["cmd"], bytes.fromhex(p["data"])), p["t"]))
    sent = fx.starts(fx.sender_calls(fx.FIXTURES / f"{name}.sender.txt"))
    got = fx.starts(bridged)
    assert [i for i, _ in got] == [i for i, _ in sent]
    for (_, want), (_, have) in zip(sent, got):
        endless = want.get("duration") == "INFINITE"
        assert have.get("duration") == ("200000" if endless else want.get("duration"))
        assert abs(int(have.get("gain", 10000)) - int(want.get("gain", 10000))) <= 20
        if want["iterations"] != "INFINITE":
            assert have["iterations"] == want["iterations"]
        if endless and "envelope.fade_time" in want:
            want = dict(want, **{"envelope.fade_time": "0"})
        assert {k: v for k, v in have.items() if k not in IGNORED} == \
            {k: v for k, v in want.items() if k not in IGNORED}


def test_fixtures_cover_every_packet_and_effect_type():
    records = [rec for name in NAMES for _, rec in decoded(name)]
    kinds = {type(r) for r in records}
    for cls in (fm.CreateNewEffect, fm.EffectReport, fm.Envelope, fm.Condition, fm.Periodic,
                fm.ConstantForce, fm.RampForce, fm.EffectOperation, fm.BlockFree, fm.DeviceControl,
                fm.DeviceGain, fm.HostQuery, fm.Unknown):
        assert cls in kinds, cls.__name__
    assert {r.effect_type for r in records if isinstance(r, fm.CreateNewEffect)} >= set(fm.EFFECT_KINDS)
    assert {r.control for r in records if isinstance(r, fm.DeviceControl)} == set(fm.CTRL_TO_DISFFC)
    assert {r.op for r in records if isinstance(r, fm.EffectOperation)} == {fm.OP_START, fm.OP_SOLO, fm.OP_STOP}


def test_synthetic_counters():
    model, _ = fm.run_packets(fx.load("synthetic_all_types")["packets"])
    c = model.counters
    assert (c.unknown, c.orphans, c.adapted) == (7, 1, 1)
    assert [u.reason for u in model.unknown_log] == [
        "unsupported packet type CustomForceData", "unsupported packet type DownloadSample",
        "unsupported packet type SetCustomForce", "Constant shorter than 4 bytes", "device control 9",
        "effect type 12", "effect operation 7",
    ]
    assert model.status() == "FFB pass 7 unknown 1 adapted 1 orphan"


# --------------------------------------------------------------------------
# decoding


def test_decode_reads_ffb_h_offsets():
    _, data = fx.effect_report(3, 8, duration=1234, gain=200, axes=fm.AXIS_X | fm.AXIS_Y, d1=100, d2=200,
                               trigger=2, repeat=5, sample=6, delay=7)
    assert fm.decode_packet(fm.IOCTL_HID_WRITE_REPORT, data) == fm.EffectReport(
        ebi=3, effect_type=8, duration_ms=1234, trigger_repeat_ms=5, sample_period_ms=6, start_delay_ms=7,
        gain=200, trigger_button=2, axes=3, direction=(100, 200))
    assert fm.packet_type(fm.IOCTL_HID_WRITE_REPORT, data) == fm.PT_EFFREP
    assert fm.device_id(data) == 1


def test_decode_sign_extends_16_bit_fields():
    assert fm.decode_packet(*fx.constant(1, -3000)) == fm.ConstantForce(1, -3000)
    assert fm.decode_packet(*fx.ramp(1, -10000, 10000)) == fm.RampForce(1, -10000, 10000)
    assert fm.decode_packet(*fx.periodic(1, 10000, -5000, 35999, 70000)) == fm.Periodic(1, 10000, -5000, 35999, 70000)
    assert fm.decode_packet(*fx.condition(1, 0, -100, -200, 300, 400, 500, 600)) == \
        fm.Condition(1, 0, -100, -200, 300, 400, 500, 600)
    # what Ffb_h_Eff_Constant would report for the same bytes
    raw = struct.unpack_from("<H", fx.constant(1, -3000)[1], 2)[0]
    assert raw == 62536


def test_create_new_effect_reads_driver_block_index():
    assert fm.decode_packet(*fx.new_effect(4, 9)) == fm.CreateNewEffect(effect_type=4, ebi=9)


def test_feature_reports_are_host_queries():
    for rid in (2, 3, 4):
        rec = fm.decode_packet(*fx.feature(rid))
        assert rec == fm.HostQuery(0x10 + rid)
        assert fm.FfbModel().apply(rec, 0.0) == []


@pytest.mark.parametrize("pkt,reason", [
    (fx.raw(fm.IOCTL_HID_WRITE_REPORT, [0x17, 1, 0, 0]), "unsupported packet type CustomForceData"),
    (fx.raw(fm.IOCTL_HID_WRITE_REPORT, [0x15, 1]), "Constant shorter than 4 bytes"),
    (fx.raw(fm.IOCTL_HID_WRITE_REPORT, [0x15]), "packet shorter than 2 bytes"),
    (fx.control(0), "device control 0"),
    (fx.new_effect(12, 3), "effect type 12"),
    (fx.effect_report(1, 0), "effect type 0"),
    (fx.effect_op(1, 4), "effect operation 4"),
])
def test_unknown_packets_are_counted_and_never_forwarded(pkt, reason):
    model = fm.FfbModel()
    feed(model, 0.0, fx.new_effect(1, 1))
    assert feed(model, 0.1, pkt) == []
    assert model.counters.unknown == 1
    assert model.unknown_log[-1].reason == reason
    assert model.status() == "FFB pass 1 unknown"


# --------------------------------------------------------------------------
# unit conversions


def test_gain_conversion():
    assert fm.gain_to_di(0) == 0
    assert fm.gain_to_di(255) == 10000
    assert fm.gain_to_di(128) == 5020
    values = [fm.gain_to_di(g) for g in range(256)]
    assert values == sorted(values)
    assert all(abs(v - g * 10000 / 255) <= 0.5 for g, v in enumerate(values))


def test_time_conversion():
    assert fm.hid_duration_to_us(0) == 0
    assert fm.hid_duration_to_us(32767) == 32_767_000
    assert fm.hid_duration_to_us(0x8000) == fm.INFINITE
    assert fm.hid_duration_to_us(0xFFFF) == fm.INFINITE
    assert fm.ms_to_us(40) == 40_000


def test_direction_conversion():
    assert fm.direction_to_centidegrees(0) == 0
    assert fm.direction_to_centidegrees(0x7FFF) == 36000
    assert fm.direction_to_centidegrees(round(0x7FFF / 4)) == 9000


@pytest.mark.parametrize("axes,d1,factor,adapted", [
    (fm.AXIS_X, 0, 1.0, False),
    (fm.AXIS_X, round(0x7FFF / 2), -1.0, True),
    (fm.AXIS_X | fm.AXIS_Y, 0, 1.0, False),
    (fm.AXIS_Y, 0, 0.0, True),
    (fm.DIRECTION_ENABLE, round(0x7FFF / 4), 1.0, False),
    (fm.DIRECTION_ENABLE, round(0x7FFF * 3 / 4), -1.0, True),
    (fm.DIRECTION_ENABLE, 0, 0.0, True),
])
def test_x_factor(axes, d1, factor, adapted):
    assert fm.x_factor(axes, d1, 0) == (factor, adapted)


def test_negative_direction_folds_into_the_force():
    model = fm.FfbModel()
    feed(model, 0.0, fx.new_effect(4, 1))
    feed(model, 0.0, fx.periodic(1, 2000, 300, 9000, 50))
    calls = feed(model, 0.0, fx.effect_report(1, 4, duration=100, axes=fm.AXIS_X, d1=round(0x7FFF / 2)))
    assert calls[0].params.type_specific == fm.DiPeriodic(2000, -300, 27000, 50_000)


# --------------------------------------------------------------------------
# the game's lifecycle


def start_constant(model, t=0.0, duration=fx.INFINITE_MS, ebi=1, magnitude=3000, **kw):
    out = []
    out += feed(model, t, fx.new_effect(1, ebi))
    out += feed(model, t, fx.constant(ebi, magnitude))
    out += feed(model, t, fx.effect_report(ebi, 1, duration=duration, **kw))
    out += feed(model, t, fx.effect_op(ebi, fm.OP_START))
    return out


def test_create_parameters_start_stop_free_in_order():
    model = fm.FfbModel()
    calls = start_constant(model, duration=500)
    calls += feed(model, 0.2, fx.constant(1, -1000))
    calls += feed(model, 0.3, fx.effect_op(1, fm.OP_STOP))
    calls += feed(model, 0.4, fx.block_free(1))
    assert [c.op for c in calls] == ["CreateEffect", "SetParameters", "SetParameters", "Start",
                                     "SetParameters", "Stop", "Release"]
    first, complete = calls[1], calls[2]
    assert first.flags & fm.DIEP_NODOWNLOAD and first.flags & fm.DIEP_AXES
    assert not complete.flags & fm.DIEP_NODOWNLOAD
    assert complete.params == fm.DiParams(duration=500_000, gain=10000, type_specific=fm.DiConstant(3000))
    assert calls[4].text() == "SetParameters ebi=1 flags=TYPESPECIFICPARAMS constant(magnitude=-1000)"
    assert model.slots == {}


def test_unchanged_effect_report_sends_only_what_changed():
    model = fm.FfbModel()
    start_constant(model, duration=500)
    assert feed(model, 0.1, fx.effect_report(1, 1, duration=500)) == []
    calls = feed(model, 0.2, fx.effect_report(1, 1, duration=500, gain=128))
    assert texts(calls) == ["SetParameters ebi=1 flags=GAIN gain=5020"]


def test_solo_and_loop_count():
    model = fm.FfbModel()
    start_constant(model, duration=100)
    calls = feed(model, 0.0, fx.effect_op(1, fm.OP_SOLO, 3))
    assert texts(calls) == ["Start ebi=1 iterations=3 flags=SOLO"]


def test_device_gain_and_controls():
    model = fm.FfbModel()
    assert texts(feed(model, 0.0, fx.gain(64))) == ["SetGain 2510"]
    for code, name in ((1, "SETACTUATORSON"), (2, "SETACTUATORSOFF"), (3, "STOPALL"),
                       (5, "PAUSE"), (6, "CONTINUE"), (4, "RESET")):
        assert texts(feed(model, 0.0, fx.control(code))) == [f"Command {name}"]


def test_reset_releases_every_effect():
    model = fm.FfbModel()
    start_constant(model, ebi=1)
    start_constant(model, ebi=2)
    calls = feed(model, 0.1, fx.control(fm.CTRL_DEVRST))
    assert texts(calls) == ["Command RESET", "Release ebi=1", "Release ebi=2"]
    assert model.slots == {} and model.next_rearm() is None


def test_packets_for_unseen_effects_are_orphans():
    model = fm.FfbModel()
    assert feed(model, 0.0, fx.constant(5, 1000)) == []
    assert (model.counters.orphans, model.counters.unknown) == (1, 0)


# --------------------------------------------------------------------------
# safety window


def test_infinite_effect_plays_in_a_200_ms_window():
    model = fm.FfbModel()
    calls = start_constant(model)
    assert calls[2].params.duration == 200_000
    assert model.next_rearm() == pytest.approx(0.1)
    assert model.rearm(0.05) == []
    assert texts(model.rearm(0.1)) == ["SetParameters ebi=1 flags=DURATION|START duration=200000"]
    assert model.next_rearm() == pytest.approx(0.2)
    feed(model, 0.15, fx.effect_op(1, fm.OP_STOP))
    assert model.next_rearm() is None and model.rearm(1.0) == []


def test_force_drops_within_250_ms_when_renewals_stop():
    """The longest the wheel can keep a force after the bridge's last call."""
    model = fm.FfbModel()
    start_constant(model)
    t, last = 0.0, []
    armed_until = []
    while t < 2.0:
        t = round(t + 0.01, 3)
        for c in model.rearm(t):
            armed_until.append(t + c.params.duration / 1e6)
            last.append(t)
    assert all(b - a <= 0.2 + 1e-9 for a, b in zip(last, armed_until))
    gaps = [b - a for a, b in zip(last, last[1:])]
    assert max(gaps) <= 0.1 + 1e-9  # renewed long before the window closes


def test_rearm_continues_the_wave():
    model = fm.FfbModel()
    feed(model, 0.0, fx.new_effect(4, 1))
    feed(model, 0.0, fx.periodic(1, 2000, 0, 0, 40))
    feed(model, 0.0, fx.effect_report(1, 4))
    feed(model, 0.0, fx.effect_op(1, fm.OP_START))
    (call,) = model.rearm(0.1)
    assert call.flags == fm.DIEP_DURATION | fm.DIEP_START | fm.DIEP_TYPESPECIFICPARAMS
    assert call.params.type_specific.phase == 18000  # 100 ms of a 40 ms period is 2.5 turns


def test_rearm_continues_the_attack_and_never_fades():
    model = fm.FfbModel()
    feed(model, 0.0, fx.new_effect(1, 1))
    feed(model, 0.0, fx.envelope(1, 0, 1000, 300, 100))
    feed(model, 0.0, fx.constant(1, 3000))
    calls = feed(model, 0.0, fx.effect_report(1, 1))
    assert calls[0].params.envelope == fm.DiEnvelope(0, 300_000, 1000, 0)
    feed(model, 0.0, fx.effect_op(1, fm.OP_START))
    (call,) = model.rearm(0.1)
    assert call.params.envelope == fm.DiEnvelope(1000, 200_000, 1000, 0)
    model.rearm(0.2)
    model.rearm(0.3)
    (call,) = model.rearm(0.4)
    assert call.params.envelope == fm.DiEnvelope(3000, 0, 1000, 0)


def test_rearm_counts_down_the_start_delay():
    model = fm.FfbModel()
    start_constant(model, delay=150)
    (call,) = model.rearm(0.1)
    assert call.flags & fm.DIEP_STARTDELAY and call.params.start_delay == 50_000
    (call,) = model.rearm(0.2)
    assert call.params.start_delay == 0


def test_endless_loop_of_a_finite_effect_renews_on_a_loop_boundary():
    model = fm.FfbModel()
    feed(model, 0.0, fx.new_effect(4, 1))
    feed(model, 0.0, fx.periodic(1, 2000, 0, 0, 30))
    feed(model, 0.0, fx.effect_report(1, 4, duration=30))
    calls = feed(model, 0.0, fx.effect_op(1, fm.OP_START, fm.LOOP_INFINITE))
    assert texts(calls) == ["Start ebi=1 iterations=7 flags=0"]  # 7 x 30 ms covers 200 ms
    assert model.next_rearm() == pytest.approx(0.12)
    assert model.rearm(0.11) == []
    assert texts(model.rearm(0.12)) == ["Start ebi=1 iterations=7 flags=0"]


def test_finite_effects_are_not_renewed():
    model = fm.FfbModel()
    start_constant(model, duration=500)
    assert model.next_rearm() is None
    assert model.rearm(1.0) == []


def test_pause_holds_renewals_and_continue_shifts_them():
    model = fm.FfbModel()
    start_constant(model)
    feed(model, 0.05, fx.control(fm.CTRL_DEVPAUSE))
    assert model.rearm(0.5) == [] and model.next_rearm() is None
    feed(model, 1.0, fx.control(fm.CTRL_DEVCONT))
    assert model.next_rearm() == pytest.approx(1.05)


def test_stop_all_ends_renewals():
    model = fm.FfbModel()
    start_constant(model)
    feed(model, 0.05, fx.control(fm.CTRL_STOPALL))
    assert model.next_rearm() is None


# --------------------------------------------------------------------------
# replug


def test_replay_rebuilds_the_wheel():
    model = fm.FfbModel()
    feed(model, 0.0, fx.gain(128))
    start_constant(model, ebi=1)
    start_constant(model, ebi=2, duration=100)
    calls = model.replay(5.0)
    assert [c.op for c in calls] == ["SetGain", "Command", "CreateEffect", "SetParameters", "Start",
                                     "CreateEffect", "SetParameters"]
    assert calls[0].value == 5020 and calls[1].text() == "Command SETACTUATORSON"
    assert calls[3].params == fm.DiParams(duration=200_000, gain=10000, type_specific=fm.DiConstant(3000))
    assert not calls[3].flags & fm.DIEP_NODOWNLOAD
    assert model.next_rearm() == pytest.approx(5.1)


def test_replay_keeps_actuators_off_and_pause():
    model = fm.FfbModel()
    feed(model, 0.0, fx.control(fm.CTRL_DISACT))
    feed(model, 0.0, fx.control(fm.CTRL_DEVPAUSE))
    assert texts(model.replay(1.0)) == ["SetGain 10000", "Command SETACTUATORSOFF", "Command PAUSE"]


# --------------------------------------------------------------------------
# call text


def test_call_text_names_only_flagged_fields():
    params = fm.DiParams(duration=fm.INFINITE, gain=5000, type_specific=fm.DiRamp(-1, 1),
                         envelope=fm.DiEnvelope(1, 2, 3, 4))
    call = fm.DiCall("SetParameters", 3, "RampForce", fm.DIEP_DURATION | fm.DIEP_ENVELOPE, params)
    assert call.text() == ("SetParameters ebi=3 flags=DURATION|ENVELOPE duration=INFINITE "
                           "envelope(attack_level=1 attack_time=2 fade_level=3 fade_time=4)")
    assert fm.DiCall("SetParameters", 1, flags=fm.DIEP_ENVELOPE, params=replace(params, envelope=None)).text() == \
        "SetParameters ebi=1 flags=ENVELOPE envelope=none"
    assert fm.DiCall("Start", 1, iterations=fm.INFINITE).text() == "Start ebi=1 iterations=INFINITE flags=0"
