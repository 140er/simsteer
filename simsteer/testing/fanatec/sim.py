"""`FanatecSim`: a simulated Fanatec base behind SimSteer's real DirectInput
and vJoy code."""

from __future__ import annotations

import threading
import time
from typing import Callable

from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.ffb import vjoy_ffb as vj
from simsteer.testing.fanatec.device import SimDevice
from simsteer.testing.fanatec.dinput import DirectInputBackend
from simsteer.testing.fanatec.ffb import FfbLog
from simsteer.testing.fanatec.profiles import PROFILES, VJOY
from simsteer.testing.fanatec.vjoy import SimGame, SimVJoyDll


class FanatecSim:
    """Simulated Fanatec base(s), the vJoy device and the game behind it.

        with FanatecSim(profile="dd2", live_instance=1) as sim:
            sim.wheel.play(Track().ramp(steer_deg=15, over=0.1))
            ...                         # run SimSteer code
            sim.ffb.assert_force_never_above(0.15)
            sim.assert_no_leaks()

    profile: "dd2" (two DirectInput instances, as a Podium DD2 shows up),
    "single" (one instance), "two_bases", "none" (only vJoy).
    live_instance: which DD2 instance carries the input: 0, 1 or "both".
    pedals_inverted: pedals read high at rest and low when pressed.
    vjoy: simulate vJoyInterface.dll too (False keeps the installed one).
    vjoy_listed: list the vJoy device in DirectInput, as Windows does.
    clock: seconds, `time.monotonic` by default; everything (tracks,
    effect durations, the event log) uses it.
    events: a callable or a JSONL path that receives every FFB event.
    """

    def __init__(self, profile: str = "dd2", *, live_instance=0, rim_range_deg: float = 900,
                 pedals_inverted: bool = False, reset_restores_gain: bool = True, vjoy: bool = True,
                 vjoy_listed: bool = True, vjoy_options: dict | None = None,
                 clock: Callable[[], float] | None = None, events=None) -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; one of {', '.join(PROFILES)}")
        self.lock = threading.RLock()
        self.clock = clock or time.monotonic
        self._writer = None
        if isinstance(events, str) or hasattr(events, "__fspath__"):
            self._writer = FfbLog.writer(events)
            events = self._writer
        self._events = events
        self.bases = [SimDevice(self, instances, live_instance=live_instance, rim_range_deg=rim_range_deg,
                                pedals_inverted=pedals_inverted, reset_restores_gain=reset_restores_gain)
                      for instances in PROFILES[profile]]
        self.devices = list(self.bases)
        if vjoy_listed:
            self.devices.append(SimDevice(self, (VJOY,), vjoy=True))
        self.vjoy = SimVJoyDll(self, **(vjoy_options or {})) if vjoy else None
        self.game = SimGame(self.vjoy)
        self.di = DirectInputBackend(self)
        self.installed = False

    @property
    def wheel(self) -> SimDevice:
        """The (first) Fanatec base."""
        if not self.bases:
            raise RuntimeError("this profile has no Fanatec base")
        return self.bases[0]

    @property
    def ffb(self) -> FfbLog:
        return self.wheel.ffb

    @property
    def calls(self):
        return self.di.calls

    @property
    def DirectInput(self):  # noqa: N802 - stands in for dinput_out.DirectInput
        """The `dinput_out.DirectInput` stand-in, for code that takes a factory."""
        return self.di.DirectInput

    def _forward_event(self, event: dict) -> None:
        if self._events is not None:
            self._events(event)

    def vjoy_values(self) -> dict[int, int]:
        if self.vjoy is None:
            return {}
        return dict(self.vjoy.devices[min(self.vjoy.devices)].values)

    def vjoy_buttons(self) -> set[int]:
        if self.vjoy is None:
            return set()
        return {b - 1 for b in self.vjoy.devices[min(self.vjoy.devices)].buttons}

    # ---- install

    def install(self) -> FanatecSim:
        if self.installed:
            raise RuntimeError("already installed")
        di.install_backend(self.di)
        if self.vjoy is not None:
            try:
                vj.install_backend(self.vjoy)
            except Exception:
                di.install_backend(None)
                raise
        self.installed = True
        return self

    def uninstall(self) -> None:
        if self.installed:
            di.install_backend(None)
            if self.vjoy is not None:
                vj.install_backend(None)
            self.installed = False
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> FanatecSim:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.uninstall()

    # ---- whole-session checks

    def assert_no_leaks(self) -> None:
        """Every DirectInput, device and effect object was released."""
        live = self.di.live_handles()
        assert not live, "not released: " + ", ".join(live)
        windows = self.di.windows
        assert not windows, f"window(s) not destroyed: {sorted(windows)}"

    def assert_single_thread(self) -> None:
        """Every DirectInput object was used only on the thread that created it."""
        bad = self.di.thread_violations
        assert not bad, f"{len(bad)} call(s) off the owner thread: {bad[:5]}"

    def assert_no_force_writes(self) -> None:
        for base in self.bases:
            base.ffb.assert_no_force_writes()
