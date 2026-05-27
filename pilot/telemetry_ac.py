"""Assetto Corsa shared-memory reader.

AC publishes three pages of shared memory while the car is in a session:

    Local\\acpmf_physics    SPageFilePhysics  (~712 bytes)
    Local\\acpmf_graphics   SPageFileGraphic  (~1576 bytes)
    Local\\acpmf_static     SPageFileStatic   (~782 bytes)

We only need the physics page. Field offsets below are computed from the
struct layout in the AC SDK header (`SharedMemory.h`), checked against
the AC ShareMemory wiki + every public Python AC telemetry library.

Same interface as `pilot.telemetry.Telemetry` so `main.py` can swap one
for the other based on which game is running. Methods mostly mirror
ETS2's, with one quirk: AC doesn't expose road-wheel angle directly, so
`wheel_angle_rad` derives it from yaw rate / speed via the bicycle
model. The wheelbase value cancels out in the LiveParams + controller
inversion as long as you pass the same one in both places (we read it
from `ControllerConfig.wheelbase_m`).
"""

from __future__ import annotations

import math
import mmap
import struct

# Physics page
SHARED_MEM_NAME = "Local\\acpmf_physics"
SHARED_MEM_SIZE = 4 * 1024   # struct is ~712 bytes; map a generous slab

# Field offsets in SPageFilePhysics. All floats unless noted.
OFF_PACKET_ID = 0           # int — non-zero when AC is producing data
OFF_GAS = 4
OFF_BRAKE = 8
OFF_FUEL = 12
OFF_GEAR = 16               # int
OFF_RPMS = 20               # int
OFF_STEER_ANGLE = 24        # float -1..+1 — normalized input
OFF_SPEED_KMH = 28
# velocity[3] @ 32, accG[3] @ 44, wheelSlip[4] @ 56, wheelLoad[4] @ 72,
# wheelsPressure[4] @ 88, wheelAngularSpeed[4] @ 104, tyreWear[4] @ 120,
# tyreDirtyLevel[4] @ 136, tyreCoreTemperature[4] @ 152, camberRAD[4] @ 168,
# suspensionTravel[4] @ 184, drs @ 200, tc @ 204, heading @ 208,
# pitch @ 212, roll @ 216, cgHeight @ 220, carDamage[5] @ 224,
# numberOfTyresOut @ 244, pitLimiterOn @ 248, abs @ 252, kersCharge @ 256,
# kersInput @ 260, autoShifterOn @ 264, rideHeight[2] @ 268,
# turboBoost @ 276, ballast @ 280, airDensity @ 284, airTemp @ 288,
# roadTemp @ 292, localAngularVel[3] @ 296, finalFF @ 308.
OFF_LOCAL_ANG_VEL = 296     # float[3] rad/s — [pitch, yaw, roll] in vehicle frame


class ACTelemetry:
    """Optional live source of v_ego, yaw rate, and steering for AC."""

    def __init__(self) -> None:
        self._mm: mmap.mmap | None = None
        try:
            self._mm = mmap.mmap(-1, SHARED_MEM_SIZE, SHARED_MEM_NAME,
                                 access=mmap.ACCESS_READ)
        except OSError:
            self._mm = None
            return
        if not self._is_active():
            self._mm.close()
            self._mm = None

    @property
    def available(self) -> bool:
        return self._mm is not None

    def _is_active(self) -> bool:
        if self._mm is None:
            return False
        try:
            packet_id = struct.unpack_from("<i", self._mm, OFF_PACKET_ID)[0]
        except struct.error:
            return False
        return packet_id > 0

    def _read_float(self, offset: int) -> float | None:
        if self._mm is None:
            return None
        try:
            return float(struct.unpack_from("<f", self._mm, offset)[0])
        except struct.error:
            return None

    def speed_mps(self) -> float | None:
        kmh = self._read_float(OFF_SPEED_KMH)
        if kmh is None:
            return None
        return kmh / 3.6

    def yaw_rate_rad_s(self) -> float | None:
        # localAngularVel = [pitch_rate, yaw_rate, roll_rate] in vehicle frame
        # (Y axis is up in AC, so [1] = around Y = yaw).
        if self._mm is None:
            return None
        try:
            arr = struct.unpack_from("<3f", self._mm, OFF_LOCAL_ANG_VEL)
            return float(arr[1])
        except struct.error:
            return None

    def game_steer(self) -> float | None:
        """Normalized steering input in [-1, +1]. Same source as user_steer
        in AC — there's no separate 'after game-side filter' value."""
        return self._read_float(OFF_STEER_ANGLE)

    def user_steer(self) -> float | None:
        return self._read_float(OFF_STEER_ANGLE)

    def wheel_angle_rad(self, wheelbase_m: float = 1.0) -> float | None:
        """Synthesized from yaw rate via the bicycle model:
            wheel_angle = atan(yaw_rate * wheelbase / v)

        AC doesn't publish a road-wheel angle, so we infer it from the
        observed yaw response. Pass the same `wheelbase_m` as the
        controller uses; the value cancels out in the LiveParams fit +
        controller inversion, so any consistent number works (in
        principle), but matching the controller's value keeps the
        learned `scale` close to physical units.
        """
        v = self.speed_mps()
        yaw = self.yaw_rate_rad_s()
        if v is None or yaw is None:
            return None
        if abs(v) < 1.0:
            # Below 1 m/s the bicycle-model inversion blows up. Return
            # 0 rather than None so LiveParams' MIN_AXIS gate handles
            # the rest.
            return 0.0
        return math.atan(yaw * wheelbase_m / v)

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
