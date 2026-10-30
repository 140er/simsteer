"""Games are TOML profiles; see `docs/PROFILES.md`."""

from simsteer.games.base import (
    Registry,
    all_profiles,
    get_profile,
    load_registry,
    registry,
    reload,
    user_dir,
)
from simsteer.games.profile import GameProfile, ProfileError, parse_profile

__all__ = [
    "GameProfile",
    "ProfileError",
    "Registry",
    "all_profiles",
    "get_profile",
    "load_registry",
    "parse_profile",
    "registry",
    "reload",
    "user_dir",
]
