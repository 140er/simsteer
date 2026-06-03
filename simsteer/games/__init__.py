"""Game profiles — one self-contained module per game.

Each module defines a `GameProfile` subclass decorated with `@register`;
the registry auto-discovers them. Import the registry helpers and base
types from this package.
"""

from simsteer.games.base import (
    GameDefaults,
    GameProfile,
    SetupStep,
    StepResult,
    StepStatus,
    all_profiles,
    discover,
    get_profile,
    register,
)

__all__ = [
    "GameDefaults",
    "GameProfile",
    "SetupStep",
    "StepResult",
    "StepStatus",
    "all_profiles",
    "discover",
    "get_profile",
    "register",
]
