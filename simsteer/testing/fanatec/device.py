"""One simulated physical device: input state, faults and the force engine.

A Fanatec base owns its DirectInput instances (two for a DD2), the input
the rim and pedals report, the motor (device gain, actuators, pause,
safety switch) and every effect created on any of its instances. The COM
layer in `dinput.py` only marshals calls to the methods here.
"""

from __future__ import annotations

from simsteer.core import ffb_model as fm
from simsteer.runtime.ffb import dinput_out as di
from simsteer.testing.fanatec.ffb import NOMINAL, FfbLog, level_of
from simsteer.testing.fanatec.profiles import InstanceProfile, Obj
from simsteer.testing.fanatec.track import PEDALS, Compiled, Track

PEDAL_NAMES = {"Accelerator": "throttle", "Brake": "brake", "Clutch": "clutch"}
REST = {"steer_deg": 0.0, "throttle": 0.0, "brake": 0.0, "clutch": 0.0, "pov": -1}
VJOY_USAGES = (0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37)  # X Y Z Rx Ry Rz Slider Dial
VJOY_MIN, VJOY_MAX = 0x0001, 0x8000

DI_OK = di.DI_OK
DI_DOWNLOADSKIPPED = 0x00000003
DI_NOEFFECT = di.S_FALSE
DIERR_ACQUIRED = 0x800700AA
DIERR_UNSUPPORTED = 0x80004001
DIERR_NOTINITIALIZED = 0x80070015
# DirectInput leaves starting or changing an effect on a paused device
# undefined; the simulator refuses it with this code so tests notice.
DIERR_PAUSED = di.DIERR_INVALIDPARAM

DIGFFS_EMPTY = 0x00000001
DIGFFS_STOPPED = 0x00000002
DIGFFS_PAUSED = 0x00000004
DIGFFS_ACTUATORSON = 0x00000010
DIGFFS_ACTUATORSOFF = 0x00000020
DIGFFS_POWERON = 0x00000040
DIGFFS_SAFETYSWITCHON = 0x00000100
DIGFFS_SAFETYSWITCHOFF = 0x00000200
DIGFFS_DEVICELOST = 0x80000000


class SimEffect:
    """One IDirectInputEffect on a device."""

    def __init__(self, device: SimDevice, handle: int, owner, kind: str) -> None:
        self.device, self.handle, self.owner, self.kind = device, handle, owner, kind
        self.params = fm.DiParams()
        self.has_axes = self.has_ts = False
        self.downloaded = self.playing = self.released = False
        self.started_at = 0.0
        self.iterations = 1

    @property
    def complete(self) -> bool:
        return self.has_axes and (self.has_ts or self.kind == "CustomForce")

    def _until(self) -> float | None:
        d = self.params.duration
        if d == fm.INFINITE or self.iterations == fm.INFINITE:
            return None
        return self.started_at + (self.params.start_delay + d * self.iterations) / fm.DI_SECONDS

    def _emit_play(self, t: float) -> None:
        begin = max(t, self.started_at + self.params.start_delay / fm.DI_SECONDS)
        self.device.ffb.add({"t": t, "ev": "play", "effect": self.handle, "kind": self.kind,
                             "level": level_of(self.kind, self.params), "start": begin, "until": self._until()})

    def is_playing(self, t: float) -> bool:
        if not self.playing:
            return False
        until = self._until()
        return until is None or t < until

    def update(self, t: float, params: fm.DiParams, flags: int) -> int:
        if self.device.paused and not flags & fm.DIEP_NODOWNLOAD:
            return DIERR_PAUSED  # a paused device's effects cannot be modified
        self.params = params
        if flags & fm.DIEP_AXES:
            self.has_axes = True
        if flags & fm.DIEP_TYPESPECIFICPARAMS:
            self.has_ts = True
        if flags & fm.DIEP_NODOWNLOAD:
            return DI_OK
        exclusive = self.owner.exclusive_acquired
        if not exclusive or not self.complete:
            if flags & fm.DIEP_START:
                return di.DIERR_NOTEXCLUSIVEACQUIRED if not exclusive else di.DIERR_INCOMPLETEEFFECT
            return DI_DOWNLOADSKIPPED
        self.downloaded = True
        if flags & fm.DIEP_START:
            return self.start(t, 1, 0)
        if self.is_playing(t):
            self._emit_play(t)
        return DI_OK

    def start(self, t: float, iterations: int, flags: int) -> int:
        if not self.owner.exclusive_acquired:
            return di.DIERR_NOTEXCLUSIVEACQUIRED
        if self.device.paused:
            return DIERR_PAUSED  # no new effects start while paused
        if not self.complete:
            return di.DIERR_INCOMPLETEEFFECT
        if flags & fm.DIES_SOLO:
            for other in self.device.effects:
                if other is not self:
                    other.stop(t)
        self.downloaded, self.playing = True, True
        self.started_at, self.iterations = t, iterations
        self._emit_play(t)
        return DI_OK

    def stop(self, t: float) -> int:
        if self.playing:
            self.playing = False
            self.device.ffb.add({"t": t, "ev": "end", "effect": self.handle})
        return DI_OK

    def unload(self, t: float) -> int:
        self.stop(t)
        self.downloaded = False
        return DI_OK

    def status(self, t: float) -> int:
        return 1 if self.is_playing(t) else 0  # DIEGES_PLAYING


class SimDevice:
    """A physical device behind one or more DirectInput instances."""

    def __init__(self, sim, instances: tuple[InstanceProfile, ...], *, live_instance=0, rim_range_deg=900.0,
                 pedals_inverted=False, reset_restores_gain=True, reset_disables_actuators=True,
                 vjoy=False) -> None:
        self.sim = sim
        self.instances = tuple(instances)
        self.objects = [p.objects() for p in self.instances]
        self.live_instance = live_instance
        self.rim_range_deg = float(rim_range_deg)
        self.pedals_inverted = pedals_inverted
        self.reset_restores_gain = reset_restores_gain
        self.reset_disables_actuators = reset_disables_actuators
        self.is_vjoy = vjoy
        self.ffb = FfbLog(on_event=sim._forward_event)
        self.effects: list[SimEffect] = []
        self.plugged = self.pedals_plugged = True
        self.generation = 0
        self.exclusive = None  # a device handle, "game" or None
        self.gain, self.autocenter = NOMINAL, 1
        self.actuators, self.paused, self.safety = True, False, True
        self._gate = True
        self._static = dict(REST)
        self._buttons: dict[int, float | None] = {}  # button -> release time
        self._track: Compiled | None = None
        self._track_start = self._track_seen = 0.0
        self._faults: list[list] = []
        self.refuse_gain = None  # callable(value) -> True to refuse a device gain

    @property
    def name(self) -> str:
        return self.instances[0].name

    # ---- input (thread safe through the simulator's lock)

    def set(self, **values) -> None:
        """Jump to these values now; ends any track that is playing."""
        with self.sim.lock:
            now = self.sim.clock()
            self._advance(now)
            if self._track is not None:
                self._static.update({k: v for k, v in self._track_values(now).items() if k in REST})
                self._track = None
            for name, v in values.items():
                if name not in REST:
                    raise ValueError(f"unknown input {name!r}")
                self._static[name] = v

    def play(self, track: Track) -> Compiled:
        """Start `track` now, from the current state; returns the compiled track."""
        with self.sim.lock:
            now = self.sim.clock()
            start = self.state(now)
            self.set()  # fold any running track first
            compiled = track.compile({k: start[k] for k in ("steer_deg", *PEDALS)})
            # -1: a plug event at the track's first instant is still due.
            self._track, self._track_start, self._track_seen = compiled, now, -1.0
            return compiled

    @property
    def track_done(self) -> bool:
        with self.sim.lock:
            return self._track is None or self.sim.clock() - self._track_start >= self._track.duration

    def press(self, button: int, hold_s: float | None = None) -> None:
        with self.sim.lock:
            self._buttons[button] = None if hold_s is None else self.sim.clock() + hold_s

    def release(self, button: int) -> None:
        with self.sim.lock:
            self._buttons.pop(button, None)

    def _track_values(self, now: float) -> dict:
        return self._track.values(now - self._track_start) if self._track is not None else {}

    def _advance(self, now: float) -> None:
        """Apply track plug events and button releases that are due."""
        if self._track is not None:
            rel = now - self._track_start
            for ch, setter in (("plugged", self._set_plugged), ("pedals_plugged", self._set_pedals_plugged)):
                for t, v in self._track.transitions(ch, self._track_seen, rel):
                    setter(bool(v), self._track_start + t)
            self._track_seen = max(self._track_seen, rel)
        for b, until in list(self._buttons.items()):
            if until is not None and now >= until:
                del self._buttons[b]

    def state(self, now: float | None = None) -> dict:
        """What the rim and pedals do now: steer_deg, throttle, brake, clutch,
        pov, buttons."""
        with self.sim.lock:
            now = self.sim.clock() if now is None else now
            self._advance(now)
            s = dict(self._static)
            buttons = set(self._buttons)
            for ch, v in self._track_values(now).items():
                if ch.startswith("button:"):
                    (buttons.add if v else buttons.discard)(int(ch.split(":", 1)[1]))
                elif ch in s:
                    s[ch] = v
            s["buttons"] = buttons
            return s

    def is_live(self, index: int) -> bool:
        return self.is_vjoy or self.live_instance in ("both", index)

    def axis_value(self, index: int, obj: Obj, s: dict) -> float:
        """Normalised 0..1 position of an axis object."""
        if self.is_vjoy:
            values = self.sim.vjoy_values()
            usage = VJOY_USAGES[obj.number] if obj.number < len(VJOY_USAGES) else None
            raw = values.get(usage, 0x4000)
            return (raw - VJOY_MIN) / (VJOY_MAX - VJOY_MIN)
        if not self.is_live(index):
            s = REST
        if obj.name == "Wheel Axis":
            return min(1.0, max(0.0, 0.5 + s["steer_deg"] / self.rim_range_deg))
        pedal = PEDAL_NAMES.get(obj.name)
        if pedal is not None or obj.name == "Combined Pedals":
            p = {k: (min(1.0, max(0.0, s[k])) if self.pedals_plugged else 0.0) for k in PEDALS}
            if obj.name == "Combined Pedals":
                return 0.5 + (p["brake"] - p["throttle"]) / 2
            return 1.0 - p[pedal] if self.pedals_inverted else p[pedal]
        return 0.5

    def button_down(self, index: int, number: int, s: dict) -> bool:
        if self.is_vjoy:
            return number in self.sim.vjoy_buttons()
        return self.is_live(index) and number in s["buttons"]

    def pov(self, index: int, s: dict) -> int:
        if self.is_vjoy or not self.is_live(index) or s["pov"] is None or s["pov"] < 0:
            return 0xFFFFFFFF
        return int(s["pov"]) % 36000

    # ---- faults

    def unplug(self) -> None:
        with self.sim.lock:
            self._set_plugged(False, self.sim.clock())

    def replug(self) -> None:
        with self.sim.lock:
            self._set_plugged(True, self.sim.clock())

    def unplug_pedals(self) -> None:
        with self.sim.lock:
            self._set_pedals_plugged(False, self.sim.clock())

    def replug_pedals(self) -> None:
        with self.sim.lock:
            self._set_pedals_plugged(True, self.sim.clock())

    def _set_pedals_plugged(self, on: bool, t: float) -> None:
        self.pedals_plugged = on

    def _set_plugged(self, on: bool, t: float) -> None:
        if on == self.plugged:
            return
        self.plugged = on
        if on:
            # A replugged base is a new USB device: new device objects, its
            # own default gain and autocentre, nobody holding it.
            self.generation += 1
            self.exclusive = None
            self.gain, self.autocenter = NOMINAL, 1
            self.actuators, self.paused = True, False
            self.ffb.add({"t": t, "ev": "gain", "value": self.gain})
        else:
            for e in self.effects:
                e.unload(t)
            self.effects = []
        self._update_gate(t, "replugged" if on else "unplugged")

    def hold_exclusive(self, on: bool = True) -> bool:
        """The game takes (or drops) the base exclusively. False if SimSteer
        holds it already."""
        with self.sim.lock:
            if not on:
                if self.exclusive == "game":
                    self.exclusive = None
                return True
            if self.exclusive not in (None, "game"):
                return False
            self.exclusive = "game"
            return True

    def safety_switch(self, on: bool) -> None:
        """The base's own safety switch: off cuts the motor."""
        with self.sim.lock:
            self.safety = on
            self._update_gate(self.sim.clock(), "safety switch")

    def emergency_stop(self) -> None:
        self.safety_switch(False)

    def fail(self, method: str, hr: int, times: int = 1) -> None:
        """Make the next `times` calls of `method` on this device return `hr`."""
        with self.sim.lock:
            self._faults.append([method, hr & 0xFFFFFFFF, times])

    def take_fault(self, method: str) -> int | None:
        for f in self._faults:
            if f[0] == method and f[2] > 0:
                f[2] -= 1
                return f[1]
        return None

    # ---- motor

    def _update_gate(self, t: float, why: str) -> None:
        on = self.plugged and self.actuators and not self.paused and self.safety
        if on != self._gate:
            self._gate = on
            self.ffb.add({"t": t, "ev": "gate", "on": on, "why": why})

    def set_gain(self, t: float, value: int) -> int:
        if value > NOMINAL:
            return di.DIERR_INVALIDPARAM
        if self.refuse_gain is not None and self.refuse_gain(value):
            return DIERR_UNSUPPORTED
        self.gain = value
        self.ffb.add({"t": t, "ev": "gain", "value": value})
        return DI_OK

    def command(self, t: float, flag: int) -> int:
        if flag == fm.DISFFC_STOPALL:
            for e in self.effects:
                e.stop(t)
        elif flag == fm.DISFFC_RESET:
            # DirectInput: effects are removed and the actuators disabled.
            for e in self.effects:
                e.unload(t)
            self.paused, self.actuators = False, not self.reset_disables_actuators
            if self.reset_restores_gain and self.gain != NOMINAL:
                self.gain = NOMINAL
                self.ffb.add({"t": t, "ev": "gain", "value": NOMINAL})
        elif flag == fm.DISFFC_PAUSE:
            self.paused = True
        elif flag == fm.DISFFC_CONTINUE:
            if not self.paused:
                return di.DIERR_INVALIDPARAM  # CONTINUE is only valid while paused
            self.paused = False
        elif flag == fm.DISFFC_SETACTUATORSOFF:
            self.actuators = False
        elif flag == fm.DISFFC_SETACTUATORSON:
            self.actuators = True
        else:
            return di.DIERR_INVALIDPARAM
        self._update_gate(t, fm.DISFFC_NAMES.get(flag, hex(flag)))
        return DI_OK

    def ff_state(self, t: float) -> int:
        playing = any(e.is_playing(t) for e in self.effects)
        state = DIGFFS_POWERON
        state |= 0 if self.effects else DIGFFS_EMPTY
        state |= 0 if playing else DIGFFS_STOPPED
        state |= DIGFFS_PAUSED if self.paused else 0
        state |= DIGFFS_ACTUATORSON if self.actuators else DIGFFS_ACTUATORSOFF
        state |= DIGFFS_SAFETYSWITCHON if self.safety else DIGFFS_SAFETYSWITCHOFF
        return state | (0 if self.plugged else DIGFFS_DEVICELOST)

    def unacquired(self, t: float, handle) -> None:
        """An exclusive Unacquire or Release stops and unloads that handle's effects."""
        for e in self.effects:
            if e.owner is handle:
                e.unload(t)
        if self.exclusive is handle:
            self.exclusive = None

    def playing(self, t: float | None = None) -> list[SimEffect]:
        t = self.sim.clock() if t is None else t
        return [e for e in self.effects if e.is_playing(t)]

    def force(self, t: float | None = None) -> float:
        """The force fraction the motor gives now (from the event log)."""
        return self.ffb.force_at(self.sim.clock() if t is None else t)
