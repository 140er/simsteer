"""SCS shared-memory telemetry, for Euro Truck Simulator 2 and American
Truck Simulator.

RenCloud's scs-telemetry plugin publishes a `scsTelemetryMap_t` struct at
`Local\\SCSTelemetry` when the game loads it. Both SCS games use the same
plugin and layout.

The telemetry field offsets are ported verbatim from v1's proven
reader (PLUGIN_REVID 12 layout); do not "tidy" them — they are
computed from the C++ struct and a one-byte drift silently corrupts
speed/steering reads.

References:
  https://github.com/RenCloud/scs-sdk-plugin
"""

from __future__ import annotations

import math
import struct
import time
from collections.abc import Callable

from simsteer.runtime.telemetry._shm import Buffer, SharedPage, open_windows_page


SHARED_MEM_NAME = "Local\\SCSTelemetry"
SHARED_MEM_SIZE = 32 * 1024

# Field offsets in scsTelemetryMap_t (PLUGIN_REVID 12). See header file
# tracked at .scs-sdk-plugin/scs-telemetry/inc/scs-telemetry-common.hpp.
OFF_SDK_ACTIVE = 0          # bool  (1 byte, then 3 bytes pad)
OFF_PAUSED = 4              # bool  (1 byte, then 3 bytes pad)
OFF_TIME = 8                # uint64 — non-zero when plugin is alive
OFF_PLUGIN_REV = 40         # uint32 telemetry_plugin_revision
OFF_GAME = 52               # uint32 (1=ETS2, 2=ATS, 0=unknown)
# 4th zone (floats) starts at 700. truck_f starts at 948 after common_f.scale
# (4 bytes) and config_f (244 bytes). truck_f layout:
#   speed, engineRpm, userSteer, userThrottle, userBrake, userClutch,
#   gameSteer, gameThrottle, gameBrake, gameClutch, cruiseControlSpeed, ...
OFF_TRUCK_SPEED = 948             # float m/s (negative when reversing)
OFF_TRUCK_ENGINE_RPM = 952        # float
OFF_TRUCK_USER_STEER = 956        # float -1..+1 — physical input from wheel/pad
OFF_TRUCK_GAME_STEER = 972        # float -1..+1 — final game-applied steering
# truck_wheelSteering[16] starts after the per-wheel suspension and velocity
# arrays. truck_f starts at 948; the 30 individual truck_f scalars take
# 120 bytes (948..1067), then 4 wheel arrays of 16 floats precede steering:
#   wheelSuspDeflection 1072, wheelVelocity 1136, wheelSteering 1200.
OFF_TRUCK_WHEEL_STEERING = 1200   # float[16] radians, per wheel
# 6th zone (vectors) starts at 1640. config_fv (228 bytes) -> truck_fv at 1868.
# truck_fv layout: lv_acc(12) av_acc(12) acc(12) aa_acc(12) cabinAV(12) cabinAA(12)
OFF_CABIN_AV_Y = 1920             # float rad/s, yaw rate of the cabin (z-axis)


class ScsTelemetry:
    def __init__(self, *, opener: Callable[[str, int], Buffer | None] = open_windows_page,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._page = SharedPage(SHARED_MEM_NAME, SHARED_MEM_SIZE, _is_active,
                                opener=opener, clock=clock)

    @property
    def _mm(self) -> Buffer | None:
        return self._page.get()

    @property
    def available(self) -> bool:
        return self._mm is not None

    def _read_float(self, offset: int) -> float | None:
        mm = self._mm
        if mm is None:
            return None
        try:
            return float(struct.unpack_from("<f", mm, offset)[0])
        except struct.error:
            return None

    def speed_mps(self) -> float | None:
        """Forward speed in m/s. Negative when reversing. None if no plugin."""
        v = self._read_float(OFF_TRUCK_SPEED)
        return v

    def game_steer(self) -> float | None:
        """Current game-applied steering in [-1, +1] (after sensitivity etc).
        Useful for closed-loop angle control; not used yet."""
        return self._read_float(OFF_TRUCK_GAME_STEER)

    def user_steer(self) -> float | None:
        """Raw input steering in [-1, +1] before in-game smoothing/sensitivity."""
        return self._read_float(OFF_TRUCK_USER_STEER)

    def wheel_angle_rad(self, wheelbase_m: float = 1.0) -> float | None:
        """Mean steering angle of the front (steering) wheels, in radians.

        SCS publishes `truck.wheel.steering` in *rotations* (one full turn
        = 1.0; the SDK header documents a typical range of <-0.25, 0.25>).
        We multiply by 2π to convert to radians, which is what the
        controller and LiveParams assume their inputs are in. Without
        this, LiveParams fits axis-vs-rotations and the controller
        inverts that fit using a target in radians — units cancel
        wrong and the commanded axis comes out ~2π× too small, so the
        truck barely steers.

        Reads truck_wheelSteering[0..3] and returns the mean (in rad) of
        non-zero entries. Most trucks steer on wheels 0 and 1 (front
        axle); some 8x4 trucks steer the second axle too.

        `wheelbase_m` is accepted for interface compatibility with
        `ACTelemetry.wheel_angle_rad` (which derives the angle from
        yaw rate); ETS2 reports the real value so the arg is unused.
        """
        del wheelbase_m
        mm = self._mm
        if mm is None:
            return None
        try:
            arr = struct.unpack_from("<4f", mm, OFF_TRUCK_WHEEL_STEERING)
            nz = [a for a in arr if abs(a) > 1e-6]
            mean_rot = float(sum(nz) / len(nz)) if nz else 0.0
            return mean_rot * 2.0 * math.pi
        except struct.error:
            return None

    def yaw_rate_rad_s(self) -> float | None:
        """Cabin angular velocity around the vertical axis (rad/s)."""
        return self._read_float(OFF_CABIN_AV_Y)

    def close(self) -> None:
        self._page.close()


def _is_active(mm: Buffer) -> bool:
    # The page can outlive the plugin with stale data; the plugin clears
    # sdkActive on shutdown.
    sdk_active = mm[OFF_SDK_ACTIVE] != 0
    time_val = struct.unpack_from("<Q", mm, OFF_TIME)[0]
    return sdk_active and time_val > 0

