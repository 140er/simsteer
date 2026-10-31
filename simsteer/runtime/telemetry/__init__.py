"""Telemetry: the reader contract and the adapter kinds a profile can name.

A `Telemetry` is the loop's live window into the game world: forward
speed, the cabin yaw rate, the front-wheel angle, and the steering
input the game actually applied. The loop reads these every frame to
drive longitudinal control, feed LiveCalib, and fit LiveParams.

This is a `Protocol`, not a base class: each adapter (the SCS shared
memory map, later AC's and Forza's) implements the methods structurally.
A profile's `telemetry.kind` picks the adapter from `KINDS`, and the other
keys of its `[telemetry]` table become the adapter's keyword arguments.
Adapters are imported only when opened, so loading profiles stays cheap.

The units are fixed across games so the control math stays game-
agnostic: speed in m/s, angles in radians, yaw rate in rad/s, steering
in [-1, +1]. A reader that natively reports other units (ETS2 publishes
wheel steering in rotations; AC has no wheel-angle field and derives one
from yaw rate) converts at this boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    from simsteer.games.profile import TelemetrySpec


@runtime_checkable
class Telemetry(Protocol):
    """Live per-frame signals from the running game.

    Every reader returns `None` from the value methods when the datum
    is momentarily unreadable (plugin handshake not complete, stale
    map), so the loop can fall back to a default rather than crash.

    Readers may also expose `vehicle_id() -> str | None` for binding
    saved calibration to a vehicle. Without it, a saved Check requires
    confirmation at the next launch.
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


class NoTelemetry:
    """For games without a telemetry source."""

    available = False

    def speed_mps(self) -> float | None:
        return None

    def game_steer(self) -> float | None:
        return None

    def user_steer(self) -> float | None:
        return None

    def wheel_angle_rad(self, wheelbase_m: float = 1.0) -> float | None:
        return None

    def yaw_rate_rad_s(self) -> float | None:
        return None

    def close(self) -> None:
        pass


def _scs_shm() -> Telemetry:
    from simsteer.runtime.telemetry.scs_shm import ScsTelemetry
    return ScsTelemetry()


class Kind(NamedTuple):
    # Keys a profile may set in `[telemetry]` next to `kind`, with types.
    params: Mapping[str, type]
    open: Callable[..., Telemetry]


KINDS: dict[str, Kind] = {
    "scs_shm": Kind({}, _scs_shm),
    "none": Kind({}, NoTelemetry),
}


def open_telemetry(spec: TelemetrySpec) -> Telemetry:
    return KINDS[spec.kind].open(**spec.params)
