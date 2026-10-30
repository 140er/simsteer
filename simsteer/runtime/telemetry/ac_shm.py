"""Assetto Corsa shared-memory telemetry, also read by Assetto Corsa
Competizione.

While a session runs, AC publishes `SPageFilePhysics` at
`Local\\acpmf_physics`. Only the physics page is read. Offsets come from
Kunos' shared memory reference for AC (`SPageFilePhysics` in its
`SharedMemory.h`). Kunos' ACC shared memory documentation keeps the same
layout for every field read here.

AC has no road-wheel angle, so `wheel_angle_rad` derives one from yaw rate
with the bicycle model. The wheelbase cancels between that and the
controller's inversion as long as both use the same value, which the loop
ensures by passing `ControllerConfig.wheelbase_m`.

Reference:
  https://www.assettocorsa.net/forum/index.php?threads/shared-memory-reference.3352/
"""

from __future__ import annotations

import math
import struct
import time
from collections.abc import Callable
from typing import NamedTuple

from simsteer.runtime.telemetry._shm import Buffer, SharedPage, open_windows_page

SHARED_MEM_NAME = "Local\\acpmf_physics"
# The struct is about 712 bytes in AC and 800 in ACC; map a generous slab.
SHARED_MEM_SIZE = 4 * 1024

OFF_PACKET_ID = 0           # int32, counts physics steps, 0 before a session
OFF_STEER_ANGLE = 24        # float, normalized steering input -1..+1
OFF_SPEED_KMH = 28          # float
OFF_LOCAL_ANG_VEL = 296     # float[3] rad/s in the car frame, Y is up
PHYSICS_LEN = OFF_LOCAL_ANG_VEL + 12

# packetId stops counting when the session is paused or the game has quit
# while SimSteer still holds the page. Physics runs at hundreds of Hz.
STALE_AFTER_S = 1.0
MIN_SYNTH_SPEED_MPS = 1.0

_HEAD = struct.Struct("<i")
_STEER_SPEED = struct.Struct("<2f")
_YAW = struct.Struct("<f")


class AcPhysics(NamedTuple):
    packet_id: int
    steer: float
    speed_mps: float
    yaw_rate_rad_s: float


def parse_ac_physics(buf: Buffer) -> AcPhysics:
    """The fields SimSteer reads from an `SPageFilePhysics` buffer."""
    if len(buf) < PHYSICS_LEN:
        raise ValueError(f"physics page is {len(buf)} bytes, need {PHYSICS_LEN}")
    (packet_id,) = _HEAD.unpack_from(buf, OFF_PACKET_ID)
    steer, kmh = _STEER_SPEED.unpack_from(buf, OFF_STEER_ANGLE)
    (yaw,) = _YAW.unpack_from(buf, OFF_LOCAL_ANG_VEL + 4)
    return AcPhysics(packet_id, steer, kmh / 3.6, yaw)


def synth_wheel_angle(yaw_rate: float, speed_mps: float, wheelbase_m: float) -> float:
    """Bicycle model: atan(yaw_rate * wheelbase / v). 0 below 1 m/s, where
    the inversion blows up; LiveParams' MIN_AXIS gate handles the rest."""
    if abs(speed_mps) < MIN_SYNTH_SPEED_MPS:
        return 0.0
    return math.atan(yaw_rate * wheelbase_m / speed_mps)


class AcTelemetry:
    def __init__(self, *, opener: Callable[[str, int], Buffer | None] = open_windows_page,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_id: int | None = None
        self._last_change = 0.0
        self._page = SharedPage(SHARED_MEM_NAME, SHARED_MEM_SIZE, _has_session,
                                opener=opener, clock=clock)

    def _read(self) -> AcPhysics | None:
        buf = self._page.get()
        if buf is None:
            self._last_id = None
            return None
        p = parse_ac_physics(buf)
        now = self._clock()
        if p.packet_id != self._last_id:
            self._last_id = p.packet_id
            self._last_change = now
        elif now - self._last_change > STALE_AFTER_S:
            return None
        return p

    @property
    def available(self) -> bool:
        return self._read() is not None

    def speed_mps(self) -> float | None:
        p = self._read()
        return None if p is None else p.speed_mps

    def yaw_rate_rad_s(self) -> float | None:
        p = self._read()
        return None if p is None else p.yaw_rate_rad_s

    def game_steer(self) -> float | None:
        """AC has no separate post-filter steering, so this is the input."""
        p = self._read()
        return None if p is None else p.steer

    def user_steer(self) -> float | None:
        return self.game_steer()

    def wheel_angle_rad(self, wheelbase_m: float = 1.0) -> float | None:
        p = self._read()
        if p is None:
            return None
        return synth_wheel_angle(p.yaw_rate_rad_s, p.speed_mps, wheelbase_m)

    def close(self) -> None:
        self._page.close()


def _has_session(buf: Buffer) -> bool:
    return len(buf) >= PHYSICS_LEN and _HEAD.unpack_from(buf, OFF_PACKET_ID)[0] > 0
