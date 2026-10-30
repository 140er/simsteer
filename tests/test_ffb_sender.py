"""tools/ffb_sender.py stops as soon as the device refuses a gain write, and
`--probe` plays one capped constant force and nothing else."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from simsteer.core import ffb_model as fm
from simsteer.runtime.ffb import dinput_out as di

ROOT = Path(__file__).resolve().parent.parent
UNSUPPORTED = 0x80004001  # DIERR_UNSUPPORTED


@pytest.fixture
def sender():
    spec = importlib.util.spec_from_file_location("ffb_sender", ROOT / "tools" / "ffb_sender.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RefusingSink(di.LogSink):
    def __init__(self, refused_value: int) -> None:
        super().__init__()
        self.refused_value = refused_value

    def execute(self, call):
        if call.op == "SetGain" and call.value == self.refused_value:
            self.calls.append(call)
            return UNSUPPORTED
        return super().execute(call)


def test_refused_first_gain_plays_nothing(sender, capsys):
    sink = RefusingSink(500)
    rc = sender.play(sink, sender.default_script(), 0, 5, None)
    assert sink.lines() == ["SetGain 500", "Command STOPALL"]
    assert rc == sender.EXIT_GAIN_REFUSED
    assert "refused device gain 500 (DIERR_UNSUPPORTED)" in capsys.readouterr().err


def test_refused_gain_mid_script_stops_the_script_and_its_repeats(sender, capsys):
    steps = [
        (0.0, fm.DiCall("SetGain", value=10000)),
        (0.0, fm.DiCall("CreateEffect", 1, "ConstantForce")),
        (0.0, fm.DiCall("Start", 1, iterations=1)),
        (0.0, fm.DiCall("SetGain", value=8000)),
        (0.0, fm.DiCall("Start", 1, iterations=1)),
    ]
    sink = RefusingSink(400)
    rc = sender.play(sink, steps, 0.5, 5, None)
    assert sink.lines() == ["SetGain 500", "CreateEffect ebi=1 type=ConstantForce", "Start ebi=1 iterations=1 flags=0",
                            "SetGain 400", "Command STOPALL"]
    assert rc == sender.EXIT_GAIN_REFUSED
    assert "refused device gain 400" in capsys.readouterr().err


class ProbeSink(di.LogSink):
    def __init__(self, results=None, refused_value=None) -> None:
        super().__init__(results=results or {})
        self.axis_id = 0
        self.last_error = None
        self.refused_value = refused_value

    def execute(self, call):
        if call.op == "SetGain" and call.value == self.refused_value:
            self.calls.append(call)
            return UNSUPPORTED
        return super().execute(call)


class Console:
    def __init__(self, interactive: bool) -> None:
        self.interactive = interactive

    def isatty(self) -> bool:
        return self.interactive


@pytest.fixture
def windows(sender, monkeypatch):
    """main() on a pretend Windows console; every sink it opens is recorded."""
    world = {"sinks": [], "prompts": [], "slept": [], "sink": ProbeSink()}

    def make_sink(*_a, **_k):
        world["sinks"].append(world["sink"])
        return world["sink"]

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdin", Console(True))
    monkeypatch.setattr("builtins.input", lambda prompt="": world["prompts"].append(prompt))
    monkeypatch.setattr(di, "DirectInputSink", make_sink)
    monkeypatch.setattr(sender.time, "sleep", world["slept"].append)
    return world


PROBE = ["--device", "fanatec", "--probe", "--ffb-test-cap", "5"]
PROBE_CALLS = [
    "SetGain 500",
    "CreateEffect ebi=1 type=ConstantForce",
    "SetParameters ebi=1 flags=DURATION|SAMPLEPERIOD|GAIN|TRIGGERBUTTON|TRIGGERREPEATINTERVAL|AXES|DIRECTION"
    "|TYPESPECIFICPARAMS|STARTDELAY duration=1000000 sample_period=0 gain=10000 trigger_button=NONE"
    " trigger_repeat=0 axes=1 direction=CARTESIAN{0} constant(magnitude=2000) start_delay=0",
    "Start ebi=1 iterations=1 flags=0",
    "Stop ebi=1",
    "Release ebi=1",
    "Command STOPALL",
]


def test_probe_sets_the_capped_gain_then_plays_one_constant_force_for_a_second(sender, windows, capsys):
    rc = sender.main(PROBE)
    out = capsys.readouterr().out
    sink = windows["sink"]
    assert rc == 0
    assert sink.lines() == PROBE_CALLS
    assert [c.op for c in sink.calls].count("CreateEffect") == 1
    assert [c.op for c in sink.calls].count("Start") == 1
    assert windows["slept"] == [1.0]
    assert len(windows["prompts"]) == 1
    assert ("magnitude 2000 of 10000, cap 5 percent, expected peak 1 percent of the wheel's peak torque"
            in out.splitlines()[0])
    assert "probe: all 6 calls returned OK" in out


def test_probe_magnitude_sets_the_force_and_the_expected_peak(sender, windows, capsys):
    rc = sender.main([*PROBE, "--probe-magnitude", "3000"])
    assert rc == 0
    assert "constant(magnitude=3000)" in windows["sink"].lines()[2]
    assert "expected peak 1.5 percent" in capsys.readouterr().out


def test_probe_stops_and_unloads_when_interrupted_while_playing(sender, windows, monkeypatch):
    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(sender.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        sender.main(PROBE)
    assert windows["sink"].lines()[-3:] == ["Stop ebi=1", "Release ebi=1", "Command STOPALL"]


def test_probe_reports_a_failed_call_and_still_stops(sender, windows, capsys):
    windows["sink"] = ProbeSink(results={"Start": UNSUPPORTED})
    rc = sender.main(PROBE)
    assert rc == 1
    assert windows["slept"] == []
    assert windows["sink"].lines()[-3:] == ["Stop ebi=1", "Release ebi=1", "Command STOPALL"]
    assert "probe: 1 of 6 calls failed: Start DIERR_UNSUPPORTED" in capsys.readouterr().out


def test_probe_refused_gain_sends_no_effect_call(sender, windows, capsys):
    windows["sink"] = ProbeSink(refused_value=500)
    rc = sender.main(PROBE)
    assert rc == sender.EXIT_GAIN_REFUSED
    assert windows["sink"].lines() == ["SetGain 500"]
    assert "refused device gain 500" in capsys.readouterr().err


@pytest.mark.parametrize("magnitude", ["0", "-1", "3001", "10000", "1.5", "much"])
def test_probe_magnitude_out_of_range_is_refused_before_open(sender, windows, magnitude):
    with pytest.raises(SystemExit) as e:
        sender.main([*PROBE, "--probe-magnitude", magnitude])
    assert e.value.code == 2
    assert windows["sinks"] == [] and windows["prompts"] == []


def test_probe_without_a_console_plays_nothing(sender, windows, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", Console(False))
    rc = sender.main(PROBE)
    assert rc == 2
    assert windows["sinks"] == [] and windows["prompts"] == []
    assert "needs someone at the console" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [["--print-script"], ["--list"], ["--seconds", "5"], ["--log", "x.txt"]])
def test_probe_and_the_script_are_mutually_exclusive(sender, windows, extra):
    with pytest.raises(SystemExit) as e:
        sender.main([*PROBE, *extra])
    assert e.value.code == 2
    assert windows["sinks"] == []


def test_probe_magnitude_needs_probe(sender, windows):
    with pytest.raises(SystemExit):
        sender.main(["--device", "fanatec", "--ffb-test-cap", "5", "--probe-magnitude", "1000"])
    assert windows["sinks"] == []
