"""The game registry: every profile SimSteer knows, built-in and user.

Built-in profiles ship in `simsteer/games/profiles/*.toml`. User profiles
in `data_dir()/games/*.toml` are added after them, and one whose `id`
matches a built-in replaces it. A file that fails to parse is skipped and
its `ProfileError` kept in `Registry.errors`, so one bad file never hides
the others.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from simsteer.games.profile import GameProfile, ProfileError, load_profile
from simsteer.paths import data_dir

BUILTIN_DIR = Path(__file__).with_name("profiles")


def user_dir() -> Path:
    return data_dir() / "games"


@dataclass(frozen=True)
class Registry:
    profiles: tuple[GameProfile, ...]
    errors: tuple[ProfileError, ...] = ()

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.profiles]

    def get(self, game_id: str | None) -> GameProfile | None:
        return next((p for p in self.profiles if p.id == game_id), None)

    def detect(self, running: Iterable[str]) -> GameProfile | None:
        """The first profile with one of its processes in `running`."""
        names = {n.lower() for n in running}
        return next((p for p in self.profiles
                     if any(n in names for n in p.process_names)), None)

    def fallback(self, last: str | None = None) -> GameProfile | None:
        """What `auto` drives when no known game is running: the game
        driven last, else ETS2, else the first tested profile."""
        return self.get(last) or self.get("ets2") or next(
            (p for p in self.profiles if p.tested),
            self.profiles[0] if self.profiles else None)


def _scan(folder: Path, builtin: bool) -> tuple[list[GameProfile], list[ProfileError]]:
    profiles: list[GameProfile] = []
    errors: list[ProfileError] = []
    if not folder.is_dir():
        return profiles, errors
    seen: dict[str, Path] = {}
    for path in sorted(folder.glob("*.toml")):
        try:
            p = load_profile(path, builtin=builtin)
        except ProfileError as e:
            errors.append(e)
            continue
        if p.id in seen:
            errors.append(ProfileError(path, "id", f"{p.id!r} is already defined "
                                                   f"in {seen[p.id].name}"))
            continue
        seen[p.id] = path
        profiles.append(p)
    return profiles, errors


def load_registry(builtin: Path = BUILTIN_DIR, user: Path | None = None) -> Registry:
    profiles, errors = _scan(builtin, builtin=True)
    users, user_errors = _scan(user_dir() if user is None else user, builtin=False)
    by_id = {p.id: i for i, p in enumerate(profiles)}
    for p in users:
        if p.id in by_id:
            profiles[by_id[p.id]] = p
        else:
            profiles.append(p)
    return Registry(tuple(profiles), tuple(errors + user_errors))


_cache: dict[Path, Registry] = {}


def registry() -> Registry:
    """The registry for the current data dir, loaded once per dir."""
    key = user_dir()
    if key not in _cache:
        _cache[key] = load_registry(user=key)
    return _cache[key]


def reload() -> Registry:
    _cache.clear()
    return registry()


def all_profiles() -> list[GameProfile]:
    return list(registry().profiles)


def get_profile(game_id: str) -> GameProfile | None:
    return registry().get(game_id)
