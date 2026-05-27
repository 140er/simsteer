"""Manual driver inputs (throttle, brake, override steering) controlled
via the tuner GUI.

Deliberately NOT persisted — every run starts at zero / off so we don't
restart the loop with yesterday's 70% throttle (or hard-left manual
steering override) still active.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ManualInputs:
    throttle: float = 0.0          # 0..1, sent to the right trigger
    brake: float = 0.0             # 0..1, sent to the left trigger
    steer: float = 0.0             # -1..+1, axis value when override is on
    steer_override: bool = False   # when True, `steer` replaces the model's
                                   # output. Useful for AC where the game can
                                   # only be bound to one input source at a
                                   # time, so manual driving without
                                   # disengaging the pad needs this slider.
    long_override: bool = False    # when True, `throttle`/`brake` replace the
                                   # longitudinal controller's output. Default
                                   # off so AI throttle/brake fires whenever
                                   # the pad is engaged.
