"""What each simulated device looks like to DirectInput.

The Podium DD2 (USB 0EB7:0007) enumerates as two DirectInput instances
with the same product name. Both are force feedback devices with one
actuator on "Wheel Axis" (object ID 0x01000002, 10 N, 256 steps) and the
same effect list; they differ in axis names, axis count (8 and 12) and
button count (108 and 63). The object lists below are copied from a real
enumeration (a game's DirectInput log and `list_ff_devices` on a DD2).
Which instance carries live input, and the pedals' rest direction, are
not known yet, so both are options of `FanatecSim`.

Instance GUIDs are synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from simsteer.runtime.ffb import dinput_out as di

GUID_X, GUID_Y, GUID_Z, GUID_RX, GUID_RY, GUID_RZ, GUID_SLIDER, _ = di._AXIS_GUIDS
GUID_BUTTON = "A36D02F0-C9F3-11CF-BFC7-444553540000"
GUID_POV = "A36D02F2-C9F3-11CF-BFC7-444553540000"
GUID_CUSTOM_FORCE = "13541C2B-8E33-11D0-9AD0-00A0C9A06E35"
AXIS_TYPES = {"X": GUID_X, "Y": GUID_Y, "Z": GUID_Z, "Rx": GUID_RX, "Ry": GUID_RY, "Rz": GUID_RZ,
              "Slider": GUID_SLIDER}

DIDFT_ABSAXIS = 0x02
DIDFT_PSHBUTTON = 0x04
DIDFT_POV = 0x10
DIDOI_FFACTUATOR = 0x01
DIDOI_ASPECTPOSITION = 0x100

DI8DEVTYPE_DRIVING_HID = 0x00010216  # driving, dual pedals, HID

# The effect names a DD2 reports, in EnumEffects order, by GUID.
DD2_EFFECTS = (
    ("ConstantForce", "Constant"), ("RampForce", "Ramp Force"), ("Square", "Square Wave"),
    ("Sine", "Sine Wave"), ("Triangle", "Triangle Wave"), ("SawtoothUp", "Sawtooth Up Wave"),
    ("SawtoothDown", "Sawtooth Down Wave"), ("Spring", "Spring"), ("Damper", "Damper"),
    ("Inertia", "Inertia"), ("Friction", "Friction"), ("CustomForce", "CustomForce"),
)
VJOY_EFFECTS = tuple((kind, f"GUID_{kind}") for kind, _ in DD2_EFFECTS[:-1])


def effect_guid(kind: str) -> str:
    return GUID_CUSTOM_FORCE if kind == "CustomForce" else di.EFFECT_GUIDS[kind]


@dataclass(frozen=True)
class Obj:
    """One DirectInput object (axis, button or POV) of an instance."""

    kind: str  # "axis" | "button" | "pov"
    guid: str
    name: str
    number: int  # instance number within its kind
    actuator: bool = False

    @property
    def dw_type(self) -> int:
        base = {"axis": DIDFT_ABSAXIS, "button": DIDFT_PSHBUTTON, "pov": DIDFT_POV}[self.kind]
        return base | self.number << 8 | (di.DIDFT_FFACTUATOR if self.actuator else 0)

    @property
    def flags(self) -> int:
        if self.kind != "axis":
            return 0
        return DIDOI_ASPECTPOSITION | (DIDOI_FFACTUATOR if self.actuator else 0)


@dataclass(frozen=True)
class InstanceProfile:
    """One DirectInput instance of a device."""

    instance: str
    name: str
    vid: int
    pid: int
    axes: tuple[tuple[str, str], ...]  # (axis type, object name) in object order
    buttons: int
    povs: int = 0
    effects: tuple[tuple[str, str], ...] = ()
    ff_driver_version: int = 0
    ff_period_us: int = 1000
    actuators: tuple[str, ...] = ("Wheel Axis",)
    max_force: int = 10

    @property
    def ff(self) -> bool:
        return bool(self.effects)

    def objects(self) -> list[Obj]:
        out = [Obj("axis", AXIS_TYPES[t], name, i, actuator=self.ff and name in self.actuators)
               for i, (t, name) in enumerate(self.axes)]
        out += [Obj("pov", GUID_POV, "Hat Switch", i) for i in range(self.povs)]
        out += [Obj("button", GUID_BUTTON, f"Button {i}", i) for i in range(self.buttons)]
        return out


DD2_NAME = "FANATEC Podium Wheel Base DD2"
DD2_VID, DD2_PID = di.FANATEC_VID, 0x0007

DD2_A = InstanceProfile(
    instance="5E1A0001-0000-11F1-8000-53494D534D30",
    name=DD2_NAME, vid=DD2_VID, pid=DD2_PID,
    axes=(("X", "Wheel Axis"), ("Y", "Accelerator"), ("Rx", "X Rotation"), ("Ry", "Y Rotation"),
          ("Rz", "Brake"), ("Slider", "Combined Pedals"), ("Slider", "Clutch"), ("Slider", "Dial")),
    buttons=108, povs=1, effects=DD2_EFFECTS, ff_driver_version=33363,
)
DD2_B = InstanceProfile(
    instance="5E1A0002-0000-11F1-8000-53494D534D30",
    name=DD2_NAME, vid=DD2_VID, pid=DD2_PID,
    axes=(("X", "Wheel Axis"), ("Y", "Z Axis"), ("Rx", "X Rotation"), ("Ry", "Y Rotation"),
          ("Rz", "Brake"), ("Slider", "Accelerator"), ("Slider", "Clutch"), ("Slider", "Combined Pedals"),
          ("Slider", "Dial"), ("Slider", "Dial"), ("Slider", "Dial"), ("Slider", "Dial")),
    buttons=63, povs=1, effects=DD2_EFFECTS, ff_driver_version=0,
)
VJOY = InstanceProfile(
    instance="5E1A00FF-0000-11F1-8000-53494D534D30",
    name="vJoy Device", vid=di.VJOY_VID, pid=di.VJOY_PID,
    axes=(("X", "X Axis"), ("Y", "Y Axis"), ("Z", "Z Axis"), ("Rx", "X Rotation"), ("Ry", "Y Rotation"),
          ("Rz", "Z Rotation"), ("Slider", "Slider"), ("Slider", "Dial")),
    buttons=32, effects=VJOY_EFFECTS, ff_driver_version=1, ff_period_us=1_000_000,
    actuators=("X Axis", "Y Axis"), max_force=0,
)


def with_instance(profile: InstanceProfile, n: int) -> InstanceProfile:
    """The same device with another instance GUID (a second base)."""
    from dataclasses import replace

    head, rest = profile.instance.split("-", 1)
    return replace(profile, instance=f"{int(head, 16) + 0x100 * n:08X}-{rest}")


# Profile name -> one tuple of instances per simulated base.
PROFILES: dict[str, tuple[tuple[InstanceProfile, ...], ...]] = {
    "dd2": ((DD2_A, DD2_B),),
    "single": ((DD2_A,),),
    "two_bases": ((DD2_A,), (with_instance(DD2_A, 1),)),
    "none": (),
}
