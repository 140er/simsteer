"""Forza (Horizon 5 / Motorsport 7+) telemetry reader.

Forza's "Data Out" feature sends a binary UDP packet at ~60 Hz to a
host:port the user configures in-game. Two packet layouts exist:

  - FM7  Dash: 311 bytes. Sled (232) + 79 bytes of Dash fields.
  - FH4 / FH5 Dash: 324 bytes. Sled (232) + 12-byte HorizonPlaceholder
    + 79 bytes of Dash fields.

Forza Horizon 4 and 5 insert a 12-byte HorizonPlaceholder block between
the Sled and the Dash payload, which shifts Speed / Steer / pedals /
position by exactly 12 bytes from their FM7 offsets. We detect the
packet variant by total length and add the correct offset adjustment
on the fly — no version flag needed.

Same duck-typed interface as `Telemetry` (ETS2) and `ACTelemetry`:
    .available
    .speed_mps()
    .yaw_rate_rad_s()
    .game_steer()
    .user_steer()
    .wheel_angle_rad(wheelbase_m)
    .close()

Reads run on a background daemon thread so the inference loop never
blocks on socket I/O. The thread always keeps the most recent packet;
property accessors snapshot it under a lock.

To enable in Forza:
    Settings → HUD and Gameplay → Data Out → ON
        Data Out IP Address  = 127.0.0.1
        Data Out IP Port     = 7777   (or whatever you pass --forza-port)
        Data Out Packet Format = Dash (NOT Sled — we need both halves)
"""

from __future__ import annotations

import math
import socket
import struct
import threading

# Sled-section offsets (bytes from packet start). These are identical
# across FM7 / FH4 / FH5 — the Sled is the first 232 bytes in all of
# them.
OFF_IS_RACE_ON = 0          # int — 0 in menus / paused, 1 when racing
OFF_VEL_X = 32              # 3 floats, m/s, vehicle-local frame
OFF_ANG_VEL_X = 44          # 3 floats, rad/s
OFF_ANG_VEL_Y = 48          # ← yaw rate (Y axis is up in Forza)
OFF_YAW = 56                # rad — vehicle heading vs world

# Dash-section offsets — given in their FM7 layout. For FH4/FH5,
# `_dash_offset` adds the 12-byte HorizonPlaceholder shift.
OFF_SPEED_FM7 = 244         # m/s — scalar speed
OFF_STEER_FM7 = 308         # signed byte, -127..+127
OFF_ACCEL_FM7 = 303         # byte 0..255 — throttle input
OFF_BRAKE_FM7 = 304         # byte 0..255

# Packet length thresholds.
MIN_SLED_LEN = 232          # Sled-only — we accept but lose Speed/Steer
MIN_FM7_DASH_LEN = 311      # FM7 Dash (Sled + 79)
MIN_HORIZON_DASH_LEN = 324  # FH4/FH5 Dash (Sled + 12 placeholder + 80)
HORIZON_SHIFT = 12          # bytes inserted between Sled and Dash in FH

DEFAULT_HOST = "0.0.0.0"  # listen on all interfaces — Forza usually sends to 127.0.0.1
DEFAULT_PORT = 7777


class ForzaTelemetry:
    """UDP receiver for Forza's Data Out stream.

    The constructor binds a non-blocking socket and starts a receiver
    thread. Access methods are safe to call from any thread; they snapshot
    the latest received packet under a small lock.
    """

    def __init__(self, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT) -> None:
        self._sock: socket.socket | None = None
        self._last_packet: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.settimeout(0.5)
            self._sock = s
        except OSError as e:
            print(f"forza: UDP bind failed on {host}:{port} ({e}); "
                  f"is another telemetry app already using this port?")
            self._sock = None
            return
        self._thread = threading.Thread(
            target=self._recv_loop, name="forza-telem", daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            # Accept Sled-only too — we'll just lose Speed + Steer if
            # the user picked the wrong packet format. Better than
            # silently dropping the stream.
            if len(data) >= MIN_SLED_LEN:
                with self._lock:
                    self._last_packet = data

    # ----- public interface -----

    @property
    def available(self) -> bool:
        """True if we've received a packet AND Forza says it's currently
        producing real data (IsRaceOn flag). Goes False during menus,
        loading screens, pauses."""
        p = self._snapshot()
        if p is None:
            return False
        try:
            return struct.unpack_from("<i", p, OFF_IS_RACE_ON)[0] != 0
        except struct.error:
            return False

    def speed_mps(self) -> float | None:
        """Vehicle speed in m/s. Prefers the scalar Speed field from
        the Dash packet; falls back to the Sled velocity magnitude if
        only the Sled half is available. Handles FM7 vs FH4/FH5
        offset differences automatically."""
        p = self._snapshot()
        if p is None:
            return None
        speed_off = self._dash_offset(p, OFF_SPEED_FM7)
        if speed_off is not None and len(p) >= speed_off + 4:
            try:
                return float(struct.unpack_from("<f", p, speed_off)[0])
            except struct.error:
                pass
        # Sled fallback — magnitude of (vx, vy, vz). Less clean since
        # it includes lateral / vertical components, but matches
        # forward speed within a fraction of a percent for normal
        # driving.
        try:
            vx, vy, vz = struct.unpack_from("<3f", p, OFF_VEL_X)
            return float(math.sqrt(vx * vx + vy * vy + vz * vz))
        except struct.error:
            return None

    def yaw_rate_rad_s(self) -> float | None:
        """Yaw rate in rad/s (rotation about world up axis).

        Forza convention: positive AngularVelocityY = counter-clockwise
        when viewed from above = LEFT turn — matches openpilot's
        left-positive convention. No sign flip needed.
        """
        p = self._snapshot()
        if p is None:
            return None
        try:
            return float(struct.unpack_from("<f", p, OFF_ANG_VEL_Y)[0])
        except struct.error:
            return None

    def game_steer(self) -> float | None:
        """Steering input in [-1, +1]. Forza encodes this as a signed
        byte (-127..+127); we normalize to a float. Positive = right
        (Xbox convention — opposite of openpilot's left-positive
        wheel angle; LiveParams' learned scale absorbs the sign).

        FH4/FH5: lives at offset 320 (= FM7's 308 + 12 placeholder).
        FM7: lives at offset 308. Auto-detected by packet length."""
        p = self._snapshot()
        if p is None:
            return None
        steer_off = self._dash_offset(p, OFF_STEER_FM7)
        if steer_off is None or len(p) < steer_off + 1:
            return None
        try:
            raw = struct.unpack_from("<b", p, steer_off)[0]
            return float(raw) / 127.0
        except struct.error:
            return None

    def user_steer(self) -> float | None:
        """Forza doesn't expose pre-filter input separately from the
        game-applied steer, so this returns the same value as
        game_steer(). LiveParams fits against gameSteer for ETS2; for
        Forza it's the same source either way."""
        return self.game_steer()

    def wheel_angle_rad(self, wheelbase_m: float = 1.0) -> float | None:
        """Synthesized from yaw rate via the bicycle model — Forza
        doesn't publish a road-wheel angle. Same approach as AC:

            wheel_angle = atan(yaw_rate * wheelbase / v_ego)

        Pass the controller's `wheelbase_m` so the value used to
        synthesize the wheel angle matches the value used to invert
        it. Below 1 m/s the bicycle inversion blows up, so we return
        0; LiveParams' MIN_AXIS gate handles the rest of the warmup.
        """
        v = self.speed_mps()
        yaw = self.yaw_rate_rad_s()
        if v is None or yaw is None:
            return None
        if abs(v) < 1.0:
            return 0.0
        return math.atan(yaw * wheelbase_m / v)

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # ----- internals -----

    def _snapshot(self) -> bytes | None:
        with self._lock:
            return self._last_packet

    @staticmethod
    def _dash_offset(packet: bytes, fm7_offset: int) -> int | None:
        """Return the offset of a Dash-section field whose FM7 layout
        position is `fm7_offset`. For FH4/FH5 (packet >= 324 bytes),
        the field has shifted by HORIZON_SHIFT (12) bytes due to the
        HorizonPlaceholder block inserted between Sled and Dash.
        Returns None if the packet is too short to contain a Dash
        payload at all (Sled-only)."""
        n = len(packet)
        if n >= MIN_HORIZON_DASH_LEN:
            return fm7_offset + HORIZON_SHIFT
        if n >= MIN_FM7_DASH_LEN:
            return fm7_offset
        return None
