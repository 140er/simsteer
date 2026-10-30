"""vJoy FFB packet fixtures: encoders for the SDK byte layout and the
synthetic scenarios.

Synthetic fixtures are built from the vJoy 2.2.2.0 SDK: the report IDs in
public.h, the Ffb_h_* byte offsets in vJoyInterface.cpp and the field
sizes in the PID descriptor (apps/vJoyConf/hidReportDescFfb.h). They are
labelled "synthetic from SDK" and are to be replaced by recorded packets
where a recording covers the same case. Recorded fixtures come from
`python -m simsteer.runtime.ffb.vjoy_ffb --record` while
`tools/ffb_sender.py --device vjoy` plays its script.

    python tests/ffb_fixtures.py            # write the synthetic fixtures
    python tests/ffb_fixtures.py --update   # also rewrite every .expected.txt
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from simsteer.core import ffb_model as fm  # noqa: E402

FIXTURES = ROOT / "fixtures" / "vjoy_ffb_packets"
SYNTHETIC_NOTE = "synthetic from SDK, to be replaced by recorded packets when Jeffrey allows"
DEV = 1 << 4
W, F = fm.IOCTL_HID_WRITE_REPORT, fm.IOCTL_HID_SET_FEATURE
NONE_BUTTON = 0xFF
INFINITE_MS = 0xFFFF


def effect_report(ebi, etype, duration=INFINITE_MS, gain=255, axes=fm.AXIS_X, d1=0, d2=0,
                  trigger=NONE_BUTTON, repeat=0, sample=0, delay=0):
    return W, struct.pack("<BBBHHHHBBBHH", DEV | fm.PT_EFFREP, ebi, etype, duration, repeat, sample, delay,
                          gain, trigger, axes, d1, d2)


def envelope(ebi, attack_level, fade_level, attack_ms, fade_ms):
    return W, struct.pack("<BBHHII", DEV | fm.PT_ENVREP, ebi, attack_level, fade_level, attack_ms, fade_ms)


def condition(ebi, block, offset, pos, neg, psat, nsat, dead):
    return W, struct.pack("<BBBhhhHHH", DEV | fm.PT_CONDREP, ebi, block, offset, pos, neg, psat, nsat, dead)


def periodic(ebi, magnitude, offset, phase, period_ms):
    return W, struct.pack("<BBHhHI", DEV | fm.PT_PRIDREP, ebi, magnitude, offset, phase, period_ms)


def constant(ebi, magnitude):
    return W, struct.pack("<BBh", DEV | fm.PT_CONSTREP, ebi, magnitude)


def ramp(ebi, start, end):
    return W, struct.pack("<BBhh", DEV | fm.PT_RAMPREP, ebi, start, end)


def effect_op(ebi, op, loops=1):
    return W, bytes([DEV | fm.PT_EFOPREP, ebi, op, loops])


def block_free(ebi):
    return W, bytes([DEV | fm.PT_BLKFRREP, ebi])


def control(code):
    return W, bytes([DEV | fm.PT_CTRLREP, code])


def gain(value):
    return W, bytes([DEV | fm.PT_GAINREP, value])


def new_effect(etype, ebi):
    # Host sends type and a 10-bit byte count; the vJoy driver writes the
    # allocated block index over byte 2 (driver/sys/hid.c).
    return F, bytes([DEV | 0x01, etype, ebi, 0x00])


def feature(report_id, payload=b"\x00"):
    return F, bytes([DEV | report_id]) + payload


def raw(cmd, data):
    return cmd, bytes(data)


def _doc(note: str, steps: list[tuple[float, tuple[int, bytes]]]) -> dict:
    return {
        "source": {"kind": "synthetic", "note": f"{SYNTHETIC_NOTE}. {note}",
                   "sdk": "BrunnerInnovation vJoy v2.2.2.0 SDK.zip, vjoyinterface.h, public.h"},
        "packets": [{"t": round(t, 6), "cmd": cmd, "data": data.hex(), "dll": None} for t, (cmd, data) in steps],
    }


def all_types() -> dict:
    """Every packet type the model handles, plus the ones it must drop."""
    s: list[tuple[float, tuple[int, bytes]]] = []

    def at(t, pkt):
        s.append((t, pkt))

    at(0.000, control(fm.CTRL_DEVRST))
    at(0.001, control(fm.CTRL_ENACT))
    at(0.002, gain(255))
    # constant without end, live update, stop
    at(0.010, new_effect(1, 1))
    at(0.011, constant(1, -3000))
    at(0.012, effect_report(1, 1))
    at(0.013, effect_op(1, fm.OP_START))
    at(0.100, constant(1, 2500))
    at(0.350, effect_op(1, fm.OP_STOP))
    # sine with envelope, finite, solo, twice
    at(0.360, new_effect(4, 2))
    at(0.361, envelope(2, 1000, 0, 100, 50))
    at(0.362, periodic(2, 2000, -500, 9000, 40))
    at(0.363, effect_report(2, 4, duration=300, gain=191))
    at(0.364, effect_op(2, fm.OP_SOLO, 2))
    # spring on X and Y without end, started with loop count 255
    at(0.400, new_effect(8, 3))
    at(0.401, condition(3, 0, 0, 3000, 3000, 5000, 5000, 100))
    at(0.402, condition(3, 1, 0, 3000, 3000, 5000, 5000, 100))
    at(0.403, effect_report(3, 8, axes=fm.AXIS_X | fm.AXIS_Y))
    at(0.404, effect_op(3, fm.OP_START, 255))
    at(0.500, control(fm.CTRL_DEVPAUSE))
    at(0.600, control(fm.CTRL_DEVCONT))
    at(0.700, gain(128))
    # ramp with a start delay
    at(0.800, new_effect(2, 4))
    at(0.801, ramp(4, -2000, 2000))
    at(0.802, effect_report(4, 2, duration=200, delay=50))
    at(0.803, effect_op(4, fm.OP_START))
    # the other waves and conditions
    t = 0.850
    for ebi, etype in ((5, 3), (6, 5), (7, 6), (8, 7)):
        at(t, new_effect(etype, ebi))
        at(t + 0.001, periodic(ebi, 1500, 0, 0, 30))
        at(t + 0.002, effect_report(ebi, etype, duration=100))
        at(t + 0.003, effect_op(ebi, fm.OP_START))
        t += 0.01
    for ebi, etype in ((9, 9), (10, 10), (11, 11)):
        at(t, new_effect(etype, ebi))
        at(t + 0.001, condition(ebi, 0, 0, 2000, 2000, 10000, 10000, 0))
        at(t + 0.002, effect_report(ebi, etype, duration=100))
        at(t + 0.003, effect_op(ebi, fm.OP_START))
        t += 0.01
    at(1.000, control(fm.CTRL_STOPALL))
    for i, ebi in enumerate(range(1, 12)):
        at(1.010 + i * 0.001, block_free(ebi))
    # dropped: custom force packets, host queries, malformed packets
    at(1.100, raw(W, [DEV | fm.PT_CSTMREP, 1, 0, 0]))
    at(1.101, raw(W, [DEV | fm.PT_SMPLREP, 0, 0]))
    at(1.102, raw(W, [DEV | fm.PT_SETCREP, 1, 0, 0, 0, 0]))
    at(1.103, feature(0x02, b"\x00\x00\x00"))
    at(1.104, feature(0x03, b"\x00\x00\x00"))
    at(1.105, feature(0x04, b"\x00"))
    at(1.106, raw(W, [DEV | fm.PT_CONSTREP, 1]))
    at(1.107, control(9))
    at(1.108, new_effect(12, 12))
    at(1.109, effect_op(12, 7))
    at(1.110, constant(42, 1000))
    at(1.200, control(fm.CTRL_DISACT))
    at(1.201, control(fm.CTRL_ENACT))
    at(1.202, new_effect(1, 1))
    at(1.203, constant(1, 1000))
    at(1.204, effect_report(1, 1))
    at(1.205, effect_op(1, fm.OP_START))
    at(1.300, control(fm.CTRL_DEVRST))
    return _doc("Covers every PID packet type and the drops.", s)


def directions() -> dict:
    """Effects aimed somewhere other than plain X, folded onto the wheel."""
    s: list[tuple[float, tuple[int, bytes]]] = []

    def at(t, pkt):
        s.append((t, pkt))

    deg = lambda d: round(d * 0x7FFF / 360)  # noqa: E731
    cases = [
        (1, 1, fm.DIRECTION_ENABLE, deg(90)),    # polar east: +X
        (2, 1, fm.DIRECTION_ENABLE, deg(270)),   # polar west: -X
        (3, 1, fm.DIRECTION_ENABLE, deg(45)),    # polar north-east: 0.707 X
        (4, 1, fm.AXIS_X, deg(180)),             # X only, turned round: -X
        (5, 1, fm.AXIS_Y, 0),                    # Y only: nothing on X
        (6, 4, fm.AXIS_X | fm.AXIS_Y, deg(180)), # sine, spherical 180: -X
        (7, 6, fm.AXIS_X | fm.AXIS_Y, deg(180)), # sawtooth up, spherical 180: -X
    ]
    t = 0.0
    for ebi, etype, axes, d1 in cases:
        at(t, new_effect(etype, ebi))
        if etype == 1:
            at(t + 0.001, constant(ebi, 4000))
        else:
            at(t + 0.001, periodic(ebi, 2000, 300, 0, 50))
        at(t + 0.002, effect_report(ebi, etype, duration=100, axes=axes, d1=d1))
        at(t + 0.003, effect_op(ebi, fm.OP_START))
        t += 0.02
    # a direction change while playing
    at(t, effect_report(1, 1, duration=100, axes=fm.DIRECTION_ENABLE, d1=deg(270)))
    return _doc("Polar, spherical, Y-only and turned-round directions.", s)


SYNTHETIC = {"synthetic_all_types": all_types, "synthetic_directions": directions}


# --------------------------------------------------------------------------
# Round trip: what the sender asked of vJoy against what the bridge asks of
# the wheel, compared effect by effect at every start.

_TOKEN = re.compile(r"(\w+)\(([^)]*)\)|(\w+)=(\S+)")


def parse_call(text: str) -> tuple[str, int, dict[str, str]]:
    op, _, rest = text.partition(" ")
    fields: dict[str, str] = {}
    for block, inner, key, value in _TOKEN.findall(rest):
        if block:
            for k, v in re.findall(r"(\w+)=(\S+)", inner):
                fields[f"{block}.{k}"] = v
            fields["kind"] = block
        else:
            fields[key] = value
    return op, int(fields.pop("ebi", 0)), fields


def starts(calls: list[str]) -> list[tuple[int, dict[str, str]]]:
    """(creation index, accumulated parameters) at every Start, in order."""
    state: dict[int, dict[str, str]] = {}
    order: dict[int, int] = {}
    out = []
    for text in calls:
        op, ebi, fields = parse_call(text)
        if op == "CreateEffect":
            order[ebi] = len(order)
            state[ebi] = {"type": fields["type"]}
        elif op == "SetParameters" and ebi in state:
            fields.pop("flags", None)
            if "envelope" in fields:  # envelope=none
                state[ebi] = {k: v for k, v in state[ebi].items() if not k.startswith("envelope.")}
            state[ebi].update(fields)
        elif op == "Start" and ebi in state:
            out.append((order[ebi], dict(state[ebi], iterations=fields["iterations"])))
    return out


def sender_calls(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return [line.split("\t")[1] for line in lines if line.count("\t") >= 3 and not line.split("\t")[2].startswith("DIERR")]


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


def expected_path(name: str) -> Path:
    return FIXTURES / f"{name}.expected.txt"


def main(argv: list[str]) -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, build in SYNTHETIC.items():
        (FIXTURES / f"{name}.json").write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    if "--update" in argv:
        for name in fixture_names():
            _, lines = fm.run_packets(load(name)["packets"])
            expected_path(name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
