"""Force feedback passthrough model: vJoy PID packets in, DirectInput calls out.

Pure: no ctypes, no Windows. The vJoy side (`runtime/ffb/vjoy_ffb.py`)
hands raw packets in, the DirectInput side (`runtime/ffb/dinput_out.py`)
executes the calls that come out, and the bridge thread joins the two.

Packet layout follows the vJoy 2.2.2.0 SDK (BrunnerInnovation fork):
`SDK/inc/vjoyinterface.h` (FFBPType, FFBEType, FFBOP, FFB_CTRL and the
FFB_EFF_* structs), `SDK/inc/public.h` (HID_ID_* report IDs) and the
byte offsets the `Ffb_h_*` helpers read in
`apps/common/vJoyInterface/vJoyInterface.cpp`. Field ranges and units
come from the PID descriptor vJoyConf installs,
`apps/vJoyConf/hidReportDescFfb.h`. The helpers do not sign-extend the
16-bit signed fields (magnitude, offsets, coefficients), so this decoder
reads the bytes itself. The bridge cross-checks it against the DLL.

Every packet maps to the DirectInput call a game would have made on the
wheel itself, with the same effect type, parameters, envelope, gain and
start or stop order. Exceptions are listed where they happen: a device
reset also releases every effect, and parameters that only make sense on
vJoy's second FFB axis are adapted to the wheel's single axis and counted.

Unit conversions, vJoy PID -> DirectInput:

    duration, trigger repeat,      u16 ms, 0..32767       x 1000 -> us
      sample period, start delay   duration > 32767 (HID null, 0xFFFF) -> INFINITE
    effect gain, device gain       u8 0..255              round(g * 10000 / 255)
    constant magnitude, ramp       s16 -10000..10000      identity
      start and end, periodic
      offset, condition offset
      and coefficients
    periodic magnitude, envelope   u16 0..10000           identity
      levels, saturations,
      dead band
    periodic phase                 u16 0..35999 (1/100 deg)  identity
    periodic period                u32 ms                 x 1000 -> us
    envelope attack, fade time     u32 ms                 x 1000 -> us
    direction                      u16 0..32767 = 0..360 deg  log * 36000 / 32767,
                                                          folded into the sign and
                                                          size of the X component
    loop count                     u8                     iterations; 255 -> INFINITE

Safety: an effect that would play forever (INFINITE duration or loop
count 255) plays on the wheel as a 200 ms effect that `rearm()` renews
every 100 ms. If SimSteer dies, the wheel's force stops within 200 ms.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from typing import Union

# --------------------------------------------------------------------------
# vJoy PID packet types (vjoyinterface.h FFBPType, public.h HID_ID_*)

IOCTL_HID_WRITE_REPORT = 0x000B000F
IOCTL_HID_SET_FEATURE = 0x000B0191

PT_EFFREP = 0x01
PT_ENVREP = 0x02
PT_CONDREP = 0x03
PT_PRIDREP = 0x04
PT_CONSTREP = 0x05
PT_RAMPREP = 0x06
PT_CSTMREP = 0x07
PT_SMPLREP = 0x08
PT_EFOPREP = 0x0A
PT_BLKFRREP = 0x0B
PT_CTRLREP = 0x0C
PT_GAINREP = 0x0D
PT_SETCREP = 0x0E
PT_NEWEFREP = 0x11
PT_BLKLDREP = 0x12
PT_POOLREP = 0x13
PT_STATEREP = 0x14

PACKET_NAMES = {
    PT_EFFREP: "EffectReport", PT_ENVREP: "Envelope", PT_CONDREP: "Condition",
    PT_PRIDREP: "Periodic", PT_CONSTREP: "Constant", PT_RAMPREP: "Ramp",
    PT_CSTMREP: "CustomForceData", PT_SMPLREP: "DownloadSample",
    PT_EFOPREP: "EffectOperation", PT_BLKFRREP: "BlockFree",
    PT_CTRLREP: "DeviceControl", PT_GAINREP: "DeviceGain",
    PT_SETCREP: "SetCustomForce", PT_NEWEFREP: "CreateNewEffect",
    PT_BLKLDREP: "BlockLoad", PT_POOLREP: "PoolReport", PT_STATEREP: "StateReport",
}

# FFBEType -> DirectInput effect GUID name (dinput.h GUID_*).
EFFECT_KINDS = {
    1: "ConstantForce", 2: "RampForce", 3: "Square", 4: "Sine", 5: "Triangle",
    6: "SawtoothUp", 7: "SawtoothDown", 8: "Spring", 9: "Damper",
    10: "Inertia", 11: "Friction",
}
PERIODIC_KINDS = frozenset({"Square", "Sine", "Triangle", "SawtoothUp", "SawtoothDown"})
CONDITION_KINDS = frozenset({"Spring", "Damper", "Inertia", "Friction"})

OP_START, OP_SOLO, OP_STOP = 1, 2, 3

CTRL_ENACT, CTRL_DISACT, CTRL_STOPALL, CTRL_DEVRST, CTRL_DEVPAUSE, CTRL_DEVCONT = 1, 2, 3, 4, 5, 6

# Effect report byte 13: Axes Enable X (bit 0), Y (bit 1), Direction Enable (bit 2).
AXIS_X, AXIS_Y, DIRECTION_ENABLE = 0x01, 0x02, 0x04

HID_TIME_MAX = 0x7FFF
HID_DIRECTION_MAX = 0x7FFF
LOOP_INFINITE = 0xFF

# --------------------------------------------------------------------------
# DirectInput constants (dinput.h)

INFINITE = 0xFFFFFFFF
DI_FFNOMINALMAX = 10000
DI_SECONDS = 1_000_000
DI_DEGREES = 100
DIEB_NOTRIGGER = 0xFFFFFFFF

DIEFF_OBJECTIDS = 0x00000001
DIEFF_OBJECTOFFSETS = 0x00000002
DIEFF_CARTESIAN = 0x00000010
DIEFF_POLAR = 0x00000020
DIEFF_SPHERICAL = 0x00000040

DIEP_DURATION = 0x00000001
DIEP_SAMPLEPERIOD = 0x00000002
DIEP_GAIN = 0x00000004
DIEP_TRIGGERBUTTON = 0x00000008
DIEP_TRIGGERREPEATINTERVAL = 0x00000010
DIEP_AXES = 0x00000020
DIEP_DIRECTION = 0x00000040
DIEP_ENVELOPE = 0x00000080
DIEP_TYPESPECIFICPARAMS = 0x00000100
DIEP_STARTDELAY = 0x00000200
DIEP_START = 0x20000000
DIEP_NORESTART = 0x40000000
DIEP_NODOWNLOAD = 0x80000000

DIEP_NAMES = (
    (DIEP_DURATION, "DURATION"), (DIEP_SAMPLEPERIOD, "SAMPLEPERIOD"), (DIEP_GAIN, "GAIN"),
    (DIEP_TRIGGERBUTTON, "TRIGGERBUTTON"), (DIEP_TRIGGERREPEATINTERVAL, "TRIGGERREPEATINTERVAL"),
    (DIEP_AXES, "AXES"), (DIEP_DIRECTION, "DIRECTION"), (DIEP_ENVELOPE, "ENVELOPE"),
    (DIEP_TYPESPECIFICPARAMS, "TYPESPECIFICPARAMS"), (DIEP_STARTDELAY, "STARTDELAY"),
    (DIEP_START, "START"), (DIEP_NORESTART, "NORESTART"), (DIEP_NODOWNLOAD, "NODOWNLOAD"),
)

DIES_SOLO = 0x00000001

DISFFC_RESET = 0x00000001
DISFFC_STOPALL = 0x00000002
DISFFC_PAUSE = 0x00000004
DISFFC_CONTINUE = 0x00000008
DISFFC_SETACTUATORSON = 0x00000010
DISFFC_SETACTUATORSOFF = 0x00000020

DISFFC_NAMES = {
    DISFFC_RESET: "RESET", DISFFC_STOPALL: "STOPALL", DISFFC_PAUSE: "PAUSE",
    DISFFC_CONTINUE: "CONTINUE", DISFFC_SETACTUATORSON: "SETACTUATORSON",
    DISFFC_SETACTUATORSOFF: "SETACTUATORSOFF",
}

CTRL_TO_DISFFC = {
    CTRL_ENACT: DISFFC_SETACTUATORSON, CTRL_DISACT: DISFFC_SETACTUATORSOFF,
    CTRL_STOPALL: DISFFC_STOPALL, CTRL_DEVRST: DISFFC_RESET,
    CTRL_DEVPAUSE: DISFFC_PAUSE, CTRL_DEVCONT: DISFFC_CONTINUE,
}

# The safety window for effects that would otherwise never end.
ARMED_US = 200_000
REARM_AFTER_S = 0.100
UNKNOWN_LOG_MAX = 100
TIME_EPS = 1e-6

# --------------------------------------------------------------------------
# Unit conversions


def ms_to_us(ms: int) -> int:
    return ms * 1000


def hid_duration_to_us(ms: int) -> int:
    return INFINITE if ms > HID_TIME_MAX else ms * 1000


def gain_to_di(gain: int) -> int:
    """u8 0..255 to DirectInput 0..10000, rounded half up."""
    return (gain * DI_FFNOMINALMAX * 2 + 255) // (255 * 2)


def direction_to_centidegrees(logical: int) -> int:
    """Direction instance, logical 0..32767 = physical 0..36000."""
    return (logical * 36000 * 2 + HID_DIRECTION_MAX) // (HID_DIRECTION_MAX * 2)


def x_factor(axes: int, d1: int, d2: int) -> tuple[float, bool]:
    """How much of the effect lands on the wheel's one axis (X).

    Returns (factor in [-1, 1], adapted). DirectInput gives a single-axis
    effect no direction (rglDirection {0}); its sign is in the magnitude.
    A game that names only X with a zero direction is passed through
    exactly. Anything else is projected onto X with DirectInput's
    conventions: polar angles start at north (-Y) and turn clockwise, so X
    is sin(angle); spherical angles start at +X, so X is cos(angle).
    """
    if axes & DIRECTION_ENABLE:
        f = _sin_centidegrees(direction_to_centidegrees(d1))
    elif axes & AXIS_X and not axes & AXIS_Y:
        if d1 == 0:
            return 1.0, False
        f = _sin_centidegrees(direction_to_centidegrees(d1) + 9000)
    elif axes & AXIS_Y and not axes & AXIS_X:
        return 0.0, True
    elif axes & AXIS_X:
        f = _sin_centidegrees(direction_to_centidegrees(d1) + 9000)
    else:
        return 1.0, True
    return f, f != 1.0


def _sin_centidegrees(angle: int) -> float:
    """sin of an angle in 1/100 degree. Within one HID step (1.1) of a
    quarter turn the value is taken as exact, so a game's east or west is
    passed through as exactly +1 or -1."""
    quarter, rest = divmod(angle, 9000)
    if rest <= 1:
        return (0.0, 1.0, 0.0, -1.0)[quarter % 4]
    if rest >= 8999:
        return (1.0, 0.0, -1.0, 0.0)[quarter % 4]
    return math.sin(math.radians(angle / DI_DEGREES))


# --------------------------------------------------------------------------
# Typed records, one per PID packet


@dataclass(frozen=True)
class CreateNewEffect:
    effect_type: int
    ebi: int


@dataclass(frozen=True)
class EffectReport:
    ebi: int
    effect_type: int
    duration_ms: int
    trigger_repeat_ms: int
    sample_period_ms: int
    start_delay_ms: int
    gain: int
    trigger_button: int
    axes: int
    direction: tuple[int, int]


@dataclass(frozen=True)
class Envelope:
    ebi: int
    attack_level: int
    fade_level: int
    attack_time_ms: int
    fade_time_ms: int


@dataclass(frozen=True)
class Condition:
    ebi: int
    block: int
    center_offset: int
    positive_coefficient: int
    negative_coefficient: int
    positive_saturation: int
    negative_saturation: int
    dead_band: int


@dataclass(frozen=True)
class Periodic:
    ebi: int
    magnitude: int
    offset: int
    phase: int
    period_ms: int


@dataclass(frozen=True)
class ConstantForce:
    ebi: int
    magnitude: int


@dataclass(frozen=True)
class RampForce:
    ebi: int
    start: int
    end: int


@dataclass(frozen=True)
class EffectOperation:
    ebi: int
    op: int
    loop_count: int


@dataclass(frozen=True)
class BlockFree:
    ebi: int


@dataclass(frozen=True)
class DeviceControl:
    control: int


@dataclass(frozen=True)
class DeviceGain:
    gain: int


@dataclass(frozen=True)
class HostQuery:
    """Block Load, Pool and State feature reports. The vJoy driver answers
    them itself; nothing goes to the wheel."""

    packet_type: int


@dataclass(frozen=True)
class Unknown:
    packet_type: int
    cmd: int
    data: bytes
    reason: str


Record = Union[
    CreateNewEffect, EffectReport, Envelope, Condition, Periodic, ConstantForce,
    RampForce, EffectOperation, BlockFree, DeviceControl, DeviceGain, HostQuery, Unknown,
]


def packet_type(cmd: int, data: bytes) -> int:
    """Ffb_h_Type: low nibble of the report ID, +0x10 for features."""
    return (data[0] & 0x0F) + (0x10 if cmd == IOCTL_HID_SET_FEATURE else 0)


def device_id(data: bytes) -> int:
    """Ffb_h_DeviceID: high nibble of the report ID."""
    return (data[0] & 0xF0) >> 4


# Minimum payload length per type: the Ffb_h_* size checks minus the
# 8-byte FFB_DATA header.
_MIN_LEN = {
    PT_EFFREP: 16, PT_ENVREP: 14, PT_CONDREP: 15, PT_PRIDREP: 12, PT_CONSTREP: 4,
    PT_RAMPREP: 6, PT_EFOPREP: 4, PT_BLKFRREP: 2, PT_CTRLREP: 2, PT_GAINREP: 2,
    PT_NEWEFREP: 4, PT_BLKLDREP: 2, PT_POOLREP: 2, PT_STATEREP: 2,
}


def decode_packet(cmd: int, data: bytes) -> Record:
    """One vJoy FFB packet (FFB_DATA.data, without the 8-byte size and cmd
    header) to its typed record. Offsets match the Ffb_h_* helpers."""
    data = bytes(data)
    if len(data) < 2:
        return Unknown(-1, cmd, data, "packet shorter than 2 bytes")
    tp = packet_type(cmd, data)
    need = _MIN_LEN.get(tp)
    if need is None:
        return Unknown(tp, cmd, data, f"unsupported packet type {PACKET_NAMES.get(tp, hex(tp))}")
    if len(data) < need:
        return Unknown(tp, cmd, data, f"{PACKET_NAMES[tp]} shorter than {need} bytes")

    def u16(i: int) -> int:
        return data[i] | data[i + 1] << 8

    def s16(i: int) -> int:
        return struct.unpack_from("<h", data, i)[0]

    def u32(i: int) -> int:
        return struct.unpack_from("<I", data, i)[0]

    ebi = data[1]
    if tp == PT_EFFREP:
        if data[2] not in EFFECT_KINDS:
            return Unknown(tp, cmd, data, f"effect type {data[2]}")
        axes = data[13]
        d2 = u16(16) if len(data) >= 18 else 0
        return EffectReport(
            ebi=ebi, effect_type=data[2], duration_ms=u16(3), trigger_repeat_ms=u16(5),
            sample_period_ms=u16(7), start_delay_ms=u16(9), gain=data[11],
            trigger_button=data[12], axes=axes, direction=(u16(14), d2),
        )
    if tp == PT_ENVREP:
        return Envelope(ebi, u16(2), u16(4), u32(6), u32(10))
    if tp == PT_CONDREP:
        return Condition(ebi, data[2], s16(3), s16(5), s16(7), u16(9), u16(11), u16(13))
    if tp == PT_PRIDREP:
        return Periodic(ebi, u16(2), s16(4), u16(6), u32(8))
    if tp == PT_CONSTREP:
        return ConstantForce(ebi, s16(2))
    if tp == PT_RAMPREP:
        return RampForce(ebi, s16(2), s16(4))
    if tp == PT_EFOPREP:
        if data[2] not in (OP_START, OP_SOLO, OP_STOP):
            return Unknown(tp, cmd, data, f"effect operation {data[2]}")
        return EffectOperation(ebi, data[2], data[3])
    if tp == PT_BLKFRREP:
        return BlockFree(ebi)
    if tp == PT_CTRLREP:
        if data[1] not in CTRL_TO_DISFFC:
            return Unknown(tp, cmd, data, f"device control {data[1]}")
        return DeviceControl(data[1])
    if tp == PT_GAINREP:
        return DeviceGain(data[1])
    if tp == PT_NEWEFREP:
        if data[1] not in EFFECT_KINDS:
            return Unknown(tp, cmd, data, f"effect type {data[1]}")
        # The vJoy driver writes the allocated block index into byte 2
        # (driver/sys/hid.c, Ffb_ProcessPacket); 0 means the pool is full.
        return CreateNewEffect(effect_type=data[1], ebi=data[2])
    return HostQuery(tp)


# --------------------------------------------------------------------------
# DirectInput side


@dataclass(frozen=True)
class DiEnvelope:
    attack_level: int
    attack_time: int
    fade_level: int
    fade_time: int


@dataclass(frozen=True)
class DiConstant:
    magnitude: int


@dataclass(frozen=True)
class DiRamp:
    start: int
    end: int


@dataclass(frozen=True)
class DiPeriodic:
    magnitude: int
    offset: int
    phase: int
    period: int


@dataclass(frozen=True)
class DiCondition:
    offset: int
    positive_coefficient: int
    negative_coefficient: int
    positive_saturation: int
    negative_saturation: int
    dead_band: int


TypeSpecific = Union[DiConstant, DiRamp, DiPeriodic, DiCondition, None]


@dataclass(frozen=True)
class DiParams:
    """A DIEFFECT for the wheel's single force feedback axis. The output
    names the axis by object ID (DIEFF_OBJECTIDS), because the DD2 rejects
    DIEFF_OBJECTOFFSETS, and gives it no direction (DIEFF_CARTESIAN {0}),
    as DirectInput documents for one-axis effects."""

    duration: int = 0
    sample_period: int = 0
    gain: int = DI_FFNOMINALMAX
    trigger_button: int = DIEB_NOTRIGGER
    trigger_repeat: int = 0
    start_delay: int = 0
    envelope: DiEnvelope | None = None
    type_specific: TypeSpecific = None


def _flag_names(flags: int) -> str:
    names = [name for bit, name in DIEP_NAMES if flags & bit]
    return "|".join(names) or "0"


def _ts_text(ts: TypeSpecific) -> str:
    if isinstance(ts, DiConstant):
        return f"constant(magnitude={ts.magnitude})"
    if isinstance(ts, DiRamp):
        return f"ramp(start={ts.start} end={ts.end})"
    if isinstance(ts, DiPeriodic):
        return f"periodic(magnitude={ts.magnitude} offset={ts.offset} phase={ts.phase} period={ts.period})"
    if isinstance(ts, DiCondition):
        return (f"condition(offset={ts.offset} positive_coefficient={ts.positive_coefficient} "
                f"negative_coefficient={ts.negative_coefficient} positive_saturation={ts.positive_saturation} "
                f"negative_saturation={ts.negative_saturation} dead_band={ts.dead_band})")
    return "none"


def _time_text(us: int) -> str:
    return "INFINITE" if us == INFINITE else str(us)


@dataclass(frozen=True)
class DiCall:
    """One DirectInput call. `text()` is the canonical log line that tests
    and lanes compare byte for byte; it names only the fields `flags` sends."""

    op: str
    ebi: int = 0
    kind: str = ""
    flags: int = 0
    params: DiParams | None = None
    iterations: int = 1
    value: int = 0

    def text(self) -> str:
        if self.op == "CreateEffect":
            return f"CreateEffect ebi={self.ebi} type={self.kind}"
        if self.op == "SetParameters":
            p = self.params or DiParams()
            parts = [f"SetParameters ebi={self.ebi} flags={_flag_names(self.flags)}"]
            if self.flags & DIEP_DURATION:
                parts.append(f"duration={_time_text(p.duration)}")
            if self.flags & DIEP_SAMPLEPERIOD:
                parts.append(f"sample_period={p.sample_period}")
            if self.flags & DIEP_GAIN:
                parts.append(f"gain={p.gain}")
            if self.flags & DIEP_TRIGGERBUTTON:
                tb = "NONE" if p.trigger_button == DIEB_NOTRIGGER else str(p.trigger_button)
                parts.append(f"trigger_button={tb}")
            if self.flags & DIEP_TRIGGERREPEATINTERVAL:
                parts.append(f"trigger_repeat={p.trigger_repeat}")
            if self.flags & DIEP_AXES:
                parts.append("axes=1")
            if self.flags & DIEP_DIRECTION:
                parts.append("direction=CARTESIAN{0}")
            if self.flags & DIEP_ENVELOPE:
                e = p.envelope
                parts.append("envelope=none" if e is None else
                             f"envelope(attack_level={e.attack_level} attack_time={e.attack_time} "
                             f"fade_level={e.fade_level} fade_time={e.fade_time})")
            if self.flags & DIEP_TYPESPECIFICPARAMS:
                parts.append(_ts_text(p.type_specific))
            if self.flags & DIEP_STARTDELAY:
                parts.append(f"start_delay={p.start_delay}")
            return " ".join(parts)
        if self.op == "Start":
            flags = "SOLO" if self.flags & DIES_SOLO else "0"
            return f"Start ebi={self.ebi} iterations={_time_text(self.iterations)} flags={flags}"
        if self.op in ("Stop", "Release"):
            return f"{self.op} ebi={self.ebi}"
        if self.op == "Command":
            return f"Command {DISFFC_NAMES.get(self.flags, hex(self.flags))}"
        if self.op == "SetGain":
            return f"SetGain {self.value}"
        return f"{self.op} ebi={self.ebi}"


# --------------------------------------------------------------------------
# The game's effect table


@dataclass
class Slot:
    """One effect the game created on vJoy, in DirectInput units, before the
    safety window is applied."""

    ebi: int
    kind: str
    report: EffectReport | None = None
    duration: int = 0
    sample_period: int = 0
    gain: int = DI_FFNOMINALMAX
    trigger_repeat: int = 0
    start_delay: int = 0
    factor: float = 1.0
    envelope: DiEnvelope | None = None
    type_specific: TypeSpecific = None
    has_x_condition: bool = False
    params_sent: bool = False
    downloaded: bool = False
    playing: bool = False
    iterations: int = 1
    solo: bool = False
    armed_at: float = 0.0
    active_since: float = 0.0
    ends_at: float | None = None
    armed_iterations: int = 1

    @property
    def complete(self) -> bool:
        if self.report is None:
            return False
        if self.kind in CONDITION_KINDS:
            return self.has_x_condition
        return self.type_specific is not None

    @property
    def unbounded(self) -> bool:
        return self.duration == INFINITE or self.iterations == INFINITE

    def active(self, now: float) -> bool:
        return self.playing and (self.ends_at is None or now < self.ends_at)

    def set_end(self) -> None:
        if self.unbounded:
            self.ends_at = None
        else:
            self.ends_at = max(self.armed_at, self.active_since) + self.duration * self.iterations / DI_SECONDS


@dataclass
class Counters:
    packets: int = 0
    forwarded: int = 0
    unknown: int = 0
    orphans: int = 0
    adapted: int = 0
    calls: int = 0
    rearms: int = 0


def _sustain(ts: TypeSpecific) -> int:
    if isinstance(ts, DiConstant):
        return abs(ts.magnitude)
    if isinstance(ts, DiRamp):
        return max(abs(ts.start), abs(ts.end))
    if isinstance(ts, DiPeriodic):
        return ts.magnitude
    return 0


def _fold(ts: TypeSpecific, kind: str, factor: float) -> TypeSpecific:
    """Apply the X factor to a force. Exact for factor 1 and -1 except a
    negated sawtooth, which DirectInput can only express as the other
    sawtooth type; there it shifts by half a period."""
    if factor == 1.0 or ts is None or isinstance(ts, DiCondition):
        return ts
    if isinstance(ts, DiConstant):
        return DiConstant(round(ts.magnitude * factor))
    if isinstance(ts, DiRamp):
        return DiRamp(round(ts.start * factor), round(ts.end * factor))
    if isinstance(ts, DiPeriodic):
        phase = (ts.phase + 18000) % 36000 if factor < 0 else ts.phase
        return DiPeriodic(round(ts.magnitude * abs(factor)), round(ts.offset * factor), phase, ts.period)
    return ts


class FfbModel:
    """The game's effect table and the packet -> DirectInput mapping.

    `apply(record, now)` returns the calls for one packet. `rearm(now)`
    returns the calls that renew effects playing in the safety window.
    `replay(now)` rebuilds the wheel after it was unplugged. `now` is a
    monotonic time in seconds supplied by the caller.
    """

    def __init__(self, armed_us: int = ARMED_US, rearm_after_s: float = REARM_AFTER_S) -> None:
        self.armed_us = armed_us
        self.rearm_after_s = rearm_after_s
        self.slots: dict[int, Slot] = {}
        self.device_gain = DI_FFNOMINALMAX
        self.paused_at: float | None = None
        self.actuators_on = True
        self.counters = Counters()
        self.unknown_log: list[Unknown] = []

    def _log_unknown(self, rec: Unknown) -> None:
        if len(self.unknown_log) < UNKNOWN_LOG_MAX:
            self.unknown_log.append(rec)

    # ---- device parameters for a slot

    def _device_params(self, s: Slot, now: float) -> DiParams:
        envelope = s.envelope
        ts = _fold(s.type_specific, s.kind, s.factor)
        start_delay = s.start_delay
        duration = s.duration
        if s.duration == INFINITE:
            duration = self.armed_us
            if envelope is not None:
                # A game effect without an end never fades.
                envelope = replace(envelope, fade_time=0)
            if s.playing:
                start_delay = max(0, round((s.active_since - s.armed_at) * DI_SECONDS))
                active = max(0.0, s.armed_at - s.active_since)
                if envelope is not None:
                    envelope = self._attack_from(envelope, ts, active)
                if isinstance(ts, DiPeriodic) and ts.period:
                    turns = round(active * DI_SECONDS * 36000 / ts.period)
                    ts = replace(ts, phase=(ts.phase + turns) % 36000)
        return DiParams(
            duration=duration, sample_period=s.sample_period, gain=s.gain,
            trigger_button=DIEB_NOTRIGGER, trigger_repeat=s.trigger_repeat,
            start_delay=start_delay, envelope=envelope, type_specific=ts,
        )

    @staticmethod
    def _attack_from(env: DiEnvelope, ts: TypeSpecific, active_s: float) -> DiEnvelope:
        """The rest of a linear attack that started `active_s` ago."""
        elapsed = round(active_s * DI_SECONDS)
        if env.attack_time <= elapsed:
            return replace(env, attack_time=0, attack_level=_sustain(ts))
        sustain = _sustain(ts)
        level = env.attack_level + (sustain - env.attack_level) * elapsed / env.attack_time
        return replace(env, attack_level=round(level), attack_time=env.attack_time - elapsed)

    def _set_params(self, s: Slot, flags: int, now: float) -> list[DiCall]:
        if not s.params_sent:
            flags |= DIEP_AXES | DIEP_DIRECTION
            s.params_sent = True
        if not s.complete:
            flags |= DIEP_NODOWNLOAD
        else:
            s.downloaded = True
        return [DiCall("SetParameters", s.ebi, s.kind, flags, self._device_params(s, now))]

    # ---- packets

    def apply(self, rec: Record, now: float) -> list[DiCall]:
        c = self.counters
        c.packets += 1
        if isinstance(rec, Unknown):
            c.unknown += 1
            self._log_unknown(rec)
            return []
        calls = self._apply(rec, now)
        if calls is None:
            # A packet for an effect created before the bridge started, or
            # an operation or control code outside the PID ranges.
            c.orphans += 1
            return []
        c.forwarded += 1
        c.calls += len(calls)
        return calls

    def _apply(self, rec: Record, now: float) -> list[DiCall] | None:
        if isinstance(rec, HostQuery):
            return []
        if isinstance(rec, DeviceGain):
            self.device_gain = gain_to_di(rec.gain)
            return [DiCall("SetGain", value=self.device_gain)]
        if isinstance(rec, DeviceControl):
            return self._control(rec.control, now)
        if isinstance(rec, CreateNewEffect):
            kind = EFFECT_KINDS[rec.effect_type]
            if rec.ebi == 0:
                # vJoy's pool was full; the game got an error, nothing to create.
                return []
            calls = []
            if rec.ebi in self.slots:
                calls.append(DiCall("Release", rec.ebi))
            self.slots[rec.ebi] = Slot(rec.ebi, kind)
            calls.append(DiCall("CreateEffect", rec.ebi, kind))
            return calls
        s = self.slots.get(getattr(rec, "ebi", 0))
        if s is None:
            return None
        if isinstance(rec, EffectReport):
            return self._effect_report(s, rec, now)
        if isinstance(rec, Envelope):
            s.envelope = DiEnvelope(rec.attack_level, ms_to_us(rec.attack_time_ms),
                                    rec.fade_level, ms_to_us(rec.fade_time_ms))
            return self._set_params(s, DIEP_ENVELOPE, now)
        if isinstance(rec, ConstantForce):
            s.type_specific = DiConstant(rec.magnitude)
            return self._set_params(s, DIEP_TYPESPECIFICPARAMS, now)
        if isinstance(rec, RampForce):
            s.type_specific = DiRamp(rec.start, rec.end)
            return self._set_params(s, DIEP_TYPESPECIFICPARAMS, now)
        if isinstance(rec, Periodic):
            s.type_specific = DiPeriodic(rec.magnitude, rec.offset, rec.phase, ms_to_us(rec.period_ms))
            return self._set_params(s, DIEP_TYPESPECIFICPARAMS, now)
        if isinstance(rec, Condition):
            if rec.block & 0x0F:
                # The block for vJoy's Y axis; the wheel has none.
                self.counters.adapted += 1
                return []
            s.type_specific = DiCondition(rec.center_offset, rec.positive_coefficient, rec.negative_coefficient,
                                          rec.positive_saturation, rec.negative_saturation, rec.dead_band)
            s.has_x_condition = True
            return self._set_params(s, DIEP_TYPESPECIFICPARAMS, now)
        if isinstance(rec, EffectOperation):
            return self._operation(s, rec, now)
        if isinstance(rec, BlockFree):
            del self.slots[s.ebi]
            return [DiCall("Release", s.ebi)]
        return None

    def _effect_report(self, s: Slot, rec: EffectReport, now: float) -> list[DiCall]:
        old = s.report
        factor, adapted = x_factor(rec.axes, *rec.direction)
        trigger = 1 <= rec.trigger_button <= 8
        if adapted or trigger:
            self.counters.adapted += 1
        flags = 0
        if old is None or old.duration_ms != rec.duration_ms:
            flags |= DIEP_DURATION
        if old is None or old.sample_period_ms != rec.sample_period_ms:
            flags |= DIEP_SAMPLEPERIOD
        if old is None or old.gain != rec.gain:
            flags |= DIEP_GAIN
        if old is None or old.trigger_button != rec.trigger_button:
            flags |= DIEP_TRIGGERBUTTON
        if old is None or old.trigger_repeat_ms != rec.trigger_repeat_ms:
            flags |= DIEP_TRIGGERREPEATINTERVAL
        if old is None or old.start_delay_ms != rec.start_delay_ms:
            flags |= DIEP_STARTDELAY
        if factor != s.factor and s.type_specific is not None:
            # The direction now lives in the folded force.
            flags |= DIEP_TYPESPECIFICPARAMS
        s.report = rec
        s.duration = hid_duration_to_us(rec.duration_ms)
        s.sample_period = ms_to_us(rec.sample_period_ms)
        s.gain = gain_to_di(rec.gain)
        s.trigger_repeat = ms_to_us(rec.trigger_repeat_ms)
        s.start_delay = ms_to_us(rec.start_delay_ms)
        s.factor = factor
        if old is not None and (old.duration_ms > HID_TIME_MAX) != (rec.duration_ms > HID_TIME_MAX) and s.envelope:
            flags |= DIEP_ENVELOPE
        if s.active(now) and flags & DIEP_DURATION:
            s.set_end()
        if not flags and s.params_sent:
            return []
        return self._set_params(s, flags, now)

    def _operation(self, s: Slot, rec: EffectOperation, now: float) -> list[DiCall]:
        if rec.op == OP_STOP:
            s.playing = False
            return [DiCall("Stop", s.ebi)]
        if rec.op not in (OP_START, OP_SOLO):
            return None
        solo = rec.op == OP_SOLO
        if solo:
            for other in self.slots.values():
                other.playing = False
        s.iterations = INFINITE if rec.loop_count == LOOP_INFINITE else rec.loop_count
        s.solo = solo
        s.playing = True
        s.armed_at = now
        s.active_since = now + s.start_delay / DI_SECONDS
        flags = DIES_SOLO if solo else 0
        if s.duration != INFINITE and s.iterations == INFINITE:
            s.armed_iterations = self._looped_iterations(s)
        else:
            s.armed_iterations = 1 if s.duration == INFINITE else s.iterations
        s.set_end()
        return [DiCall("Start", s.ebi, s.kind, flags, iterations=s.armed_iterations)]

    def _looped_iterations(self, s: Slot) -> int:
        """Enough loops of a finite effect to cover the safety window."""
        if s.duration == 0:
            return 1
        return max(1, math.ceil(self.armed_us / s.duration))

    def _control(self, control: int, now: float) -> list[DiCall] | None:
        cmd = CTRL_TO_DISFFC.get(control)
        if cmd is None:
            return None
        calls = [DiCall("Command", flags=cmd)]
        if control == CTRL_DEVRST:
            # vJoy freed every block (Ffb_ResetPIDData) and DirectInput
            # invalidates every effect on reset, so release ours too.
            calls += [DiCall("Release", ebi) for ebi in sorted(self.slots)]
            self.slots.clear()
            self.paused_at = None
            self.actuators_on = True
        elif control == CTRL_STOPALL:
            for s in self.slots.values():
                s.playing = False
        elif control == CTRL_DEVPAUSE:
            if self.paused_at is None:
                self.paused_at = now
        elif control == CTRL_DEVCONT:
            if self.paused_at is not None:
                shift = now - self.paused_at
                for s in self.slots.values():
                    if s.playing:
                        s.armed_at += shift
                        s.active_since += shift
                        if s.ends_at is not None:
                            s.ends_at += shift
                self.paused_at = None
        elif control == CTRL_ENACT:
            self.actuators_on = True
        elif control == CTRL_DISACT:
            self.actuators_on = False
        return calls

    # ---- safety window

    def next_rearm(self) -> float | None:
        """When `rearm` next has work, or None."""
        if self.paused_at is not None:
            return None
        times = [self._due(s) for s in self.slots.values() if s.playing and s.unbounded and s.downloaded]
        return min(times) if times else None

    def _due(self, s: Slot) -> float:
        if s.duration != INFINITE and s.duration:
            # Renew a looped effect on a loop boundary so the wave stays whole.
            period = s.duration / DI_SECONDS
            return s.armed_at + math.ceil(self.rearm_after_s / period - 1e-9) * period
        return s.armed_at + self.rearm_after_s

    def rearm(self, now: float) -> list[DiCall]:
        if self.paused_at is not None:
            return []
        calls: list[DiCall] = []
        for ebi in sorted(self.slots):
            s = self.slots[ebi]
            if not (s.active(now) and s.unbounded and s.downloaded) or now + TIME_EPS < self._due(s):
                continue
            if s.duration != INFINITE:
                s.armed_at = self._due(s)
                calls.append(DiCall("Start", s.ebi, s.kind, 0, iterations=s.armed_iterations))
            else:
                s.armed_at = now
                flags = DIEP_DURATION | DIEP_START
                if s.start_delay:
                    flags |= DIEP_STARTDELAY
                if s.envelope is not None:
                    flags |= DIEP_ENVELOPE
                if isinstance(s.type_specific, DiPeriodic):
                    flags |= DIEP_TYPESPECIFICPARAMS
                calls.append(DiCall("SetParameters", s.ebi, s.kind, flags, self._device_params(s, now)))
        self.counters.rearms += len(calls)
        self.counters.calls += len(calls)
        return calls

    # ---- rebuild after the wheel came back

    def replay(self, now: float) -> list[DiCall]:
        """Calls that put the current effect table on a freshly opened wheel:
        device gain, every effect with its parameters, and the effects that
        play without end restarted where they were. Finite effects that were
        playing are not restarted; they would replay from their beginning."""
        calls = [DiCall("SetGain", value=self.device_gain)]
        if not self.actuators_on:
            calls.append(DiCall("Command", flags=DISFFC_SETACTUATORSOFF))
        for ebi in sorted(self.slots):
            s = self.slots[ebi]
            calls.append(DiCall("CreateEffect", ebi, s.kind))
            s.params_sent = False
            s.downloaded = False
            resume = s.active(now) and s.unbounded and s.complete
            if not resume:
                s.playing = False
            else:
                s.armed_at = now
            if s.report is None and s.type_specific is None and s.envelope is None:
                continue
            flags = DIEP_DURATION | DIEP_SAMPLEPERIOD | DIEP_GAIN | DIEP_TRIGGERBUTTON | \
                DIEP_TRIGGERREPEATINTERVAL | DIEP_STARTDELAY
            if s.type_specific is not None:
                flags |= DIEP_TYPESPECIFICPARAMS
            if s.envelope is not None:
                flags |= DIEP_ENVELOPE
            calls += self._set_params(s, flags, now)
            if resume:
                calls.append(DiCall("Start", ebi, s.kind, 0, iterations=s.armed_iterations))
        if self.paused_at is not None:
            calls.append(DiCall("Command", flags=DISFFC_PAUSE))
        self.counters.calls += len(calls)
        return calls

    def status(self) -> str:
        """HUD text, e.g. `FFB pass 0 unknown`."""
        c = self.counters
        text = f"FFB pass {c.unknown} unknown"
        if c.adapted:
            text += f" {c.adapted} adapted"
        if c.orphans:
            text += f" {c.orphans} orphan"
        return text


def run_packets(packets, tick_s: float = 0.01, tail_s: float = 0.5) -> tuple[FfbModel, list[str]]:
    """Feed recorded packets through a fresh model as the bridge would,
    renewing the safety window every `tick_s`. `packets` are dicts with
    `t` (seconds), `cmd` and `data` (hex), the fixture format. Returns the
    model and the call log, one `<seconds> <call>` line per call."""
    model = FfbModel()
    lines: list[str] = []
    tick = 0.0

    def emit(t: float, calls: list[DiCall]) -> None:
        lines.extend(f"{t:.3f} {c.text()}" for c in calls)

    for p in packets:
        t = float(p["t"])
        while tick + tick_s <= t:
            tick += tick_s
            emit(tick, model.rearm(tick))
        emit(t, model.apply(decode_packet(int(p["cmd"]), bytes.fromhex(p["data"])), t))
    end = tick + tail_s
    while tick + tick_s <= end:
        tick += tick_s
        emit(tick, model.rearm(tick))
    return model, lines

