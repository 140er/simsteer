"""Forza Data Out telemetry, for Forza Horizon 4 and 5.

With Data Out on, Forza sends one UDP packet per physics frame to the
address and port set in its HUD and Gameplay settings. The layout is
Turn 10's Forza Motorsport 7 "Dash" format, a 232-byte Sled block plus
79 bytes of dash fields (311 bytes). Forza Horizon 4 and 5 insert 12
bytes between the two blocks and send 324 bytes, so every dash field
sits 12 bytes later. The variant is told apart by length. A Sled-only
packet (232 bytes) still gives speed, as the length of the velocity
vector, and yaw rate, but no steering.

The adapter binds one socket for its whole life and reads it on a
background thread, so the loop never blocks on the network. A packet
older than `STALE_AFTER_S`, or one with IsRaceOn = 0 (menus, pause,
loading), counts as not available.

Layout sources: Turn 10's FM7 Data Out specification and the original
`pilot/telemetry_forza.py` reader, including its Horizon offset adjustment.
https://forums.forza.net/t/forza-motorsport-7-data-out-feature-details/74013
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import NamedTuple

from simsteer.runtime.telemetry.ac_shm import synth_wheel_angle

DEFAULT_HOST = ""
DEFAULT_PORT = 7777

OFF_IS_RACE_ON = 0          # int32, 0 in menus, pause and loading
OFF_VELOCITY = 32           # float[3] m/s, car frame
OFF_ANG_VEL_Y = 48          # float rad/s, yaw rate (Y is up), left positive
OFF_SPEED = 244             # float m/s, FM7 layout
OFF_STEER = 308             # int8 -127..127, FM7 layout
SLED_LEN = 232
FM7_DASH_LEN = 311
HORIZON_DASH_LEN = 324
HORIZON_SHIFT = 12

STALE_AFTER_S = 0.5
BIND_RETRY_S = 5.0

_RACE_ON = struct.Struct("<i")
_VEL_YAW = struct.Struct("<3f4xf")    # velocity x, y, z, skip ang_vel x, ang_vel y
_SPEED = struct.Struct("<f")
_STEER = struct.Struct("<b")


class ForzaDash(NamedTuple):
    race_on: bool
    speed_mps: float
    yaw_rate_rad_s: float
    steer: float | None


def parse_forza_dash(packet: bytes) -> ForzaDash | None:
    """The fields SimSteer reads from one Data Out packet, or None when it
    is shorter than the Sled block."""
    n = len(packet)
    if n < SLED_LEN:
        return None
    (race_on,) = _RACE_ON.unpack_from(packet, OFF_IS_RACE_ON)
    vx, vy, vz, yaw = _VEL_YAW.unpack_from(packet, OFF_VELOCITY)
    if not math.isfinite(yaw):
        return None
    if n >= HORIZON_DASH_LEN:
        shift = HORIZON_SHIFT
    elif n >= FM7_DASH_LEN:
        shift = 0
    else:
        speed = math.hypot(vx, vy, vz)
        return ForzaDash(race_on != 0, speed, yaw, None) if math.isfinite(speed) else None
    (speed,) = _SPEED.unpack_from(packet, OFF_SPEED + shift)
    if not math.isfinite(speed):
        return None
    (steer,) = _STEER.unpack_from(packet, OFF_STEER + shift)
    return ForzaDash(race_on != 0, speed, yaw, max(-1.0, steer / 127.0))


class ForzaTelemetry:
    def __init__(self, port: int = DEFAULT_PORT, host: str = DEFAULT_HOST, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.host = host
        self.port = port
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: tuple[float, ForzaDash] | None = None
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._next_bind = 0.0
        self._bind_error: str | None = None
        self._bind()

    def _bind(self) -> None:
        now = self._clock()
        if self._sock is not None or self._stop.is_set() or now < self._next_bind:
            return
        self._next_bind = now + BIND_RETRY_S
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind((self.host, self.port))
        except OSError as e:
            s.close()
            if self._bind_error is None:
                print(f"forza: cannot listen on UDP port {self.port} ({e}); is another "
                      f"telemetry app using it? Retrying every {BIND_RETRY_S:.0f} s.")
            self._bind_error = str(e)
            return
        s.settimeout(0.25)
        self._sock = s
        self._bind_error = None
        self.port = s.getsockname()[1]
        print(f"forza: listening on UDP port {self.port}")
        self._thread = threading.Thread(target=self._recv_loop, args=(s,),
                                        name="forza-telemetry", daemon=True)
        self._thread.start()

    def _recv_loop(self, s: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                data = s.recv(2048)
            except TimeoutError:
                continue
            except OSError:
                return
            dash = parse_forza_dash(data)
            if dash is not None:
                with self._lock:
                    self._latest = (self._clock(), dash)

    def _read(self) -> ForzaDash | None:
        if self._sock is None:
            self._bind()
            return None
        with self._lock:
            latest = self._latest
        if latest is None or self._clock() - latest[0] > STALE_AFTER_S:
            return None
        return latest[1] if latest[1].race_on else None

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
        """Steering in [-1, +1], right positive. Forza has no separate
        pre-filter input. LiveParams' fitted scale absorbs the sign."""
        p = self._read()
        return None if p is None else p.steer

    def user_steer(self) -> float | None:
        return None

    def wheel_angle_rad(self, wheelbase_m: float = 1.0) -> float | None:
        p = self._read()
        if p is None:
            return None
        return synth_wheel_angle(p.yaw_rate_rad_s, p.speed_mps, wheelbase_m)

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
