"""The telemetry contract every game profile's reader satisfies.

A `Telemetry` is the loop's live window into the game world: forward
speed, the cabin yaw rate, the front-wheel angle, and the steering
input the game actually applied. The loop reads these every frame to
drive longitudinal control, feed LiveCalib, and fit LiveParams.

This is a `Protocol`, not a base class: each game's reader (ETS2's
shared-memory map, AC's UDP packet, Forza's Data Out) implements the
methods structurally and does **not** import or subclass anything here.
That keeps the games package free of a runtime dependency on the
loop — no circular import between the registry and the frame loop.

The units are fixed across games so the control math stays game-
agnostic: speed in m/s, angles in radians, yaw rate in rad/s, steering
in [-1, +1]. A reader that natively reports other units (ETS2 publishes
wheel steering in rotations; AC has no wheel-angle field and derives one
from yaw rate) converts at this boundary.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Telemetry(Protocol):
    """Live per-frame signals from the running game.

    Every reader returns `None` from the value methods when the datum
    is momentarily unreadable (plugin handshake not complete, stale
    map), so the loop can fall back to a default rather than crash.
    """

    @property
    def available(self) -> bool:
        """True once the game's telemetry source is live and handshook."""
        ...

    def speed_mps(self) -> float | None:
        """Forward speed, m/s. Negative when reversing."""
        ...

    def game_steer(self) -> float | None:
        """Steering the game actually applied, in [-1, +1] (post input
        filter / sensitivity). This is the LiveParams regression target."""
        ...

    def user_steer(self) -> float | None:
        """Raw human steering input, in [-1, +1], before in-game
        smoothing/sensitivity."""
        ...

    def wheel_angle_rad(self, wheelbase_m: float = 1.0) -> float | None:
        """Mean front-wheel steering angle, radians. `wheelbase_m` is
        used only by readers that synthesize the angle from yaw rate
        (AC); readers that report it directly (ETS2) ignore the arg."""
        ...

    def yaw_rate_rad_s(self) -> float | None:
        """Cabin/chassis yaw rate about the vertical axis, rad/s."""
        ...

    def close(self) -> None:
        """Release the underlying handle (shared memory, socket)."""
        ...
