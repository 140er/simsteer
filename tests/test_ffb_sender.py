"""tools/ffb_sender.py stops as soon as the device refuses a gain write."""

from __future__ import annotations

import importlib.util
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
