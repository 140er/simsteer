"""The game-profile system: one self-contained class per supported game.

A `GameProfile` collapses everything game-specific into a single place —
the process names used to detect it, how to read its telemetry, where its
true camera FOV lives, the declarative auto-fix setup steps, and its
physical/control defaults. A registry auto-discovers every profile module
in this package, so **adding a game is adding one file here**; nothing
else in the codebase needs editing.

This replaces v1's scattered per-game logic: the duck-typed telemetry
trio, the `open_telemetry` if/elif dispatch, the argparse `--game`
choices, and the per-game preflight branches.
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class GameDefaults:
    """Per-game physical and control seeds.

    Every field here is a STATIC knob (see the static-knob discipline in
    `simsteer.core.learners`): set once, never closed-loop tuned at
    runtime. Only the steering-rack fit (LiveParams) and camera pose
    (LiveCalib) learn online, because they observe signals independent of
    what they tune. `fov_h_deg` is a seed/fallback only — the real value
    is read from the game's own config or measured once by the
    FovResolver (Phase 3), then frozen.
    """

    # Physical wheelbase (front-to-rear axle), metres. Sets the
    # curvature -> steer-angle bicycle relation.
    wheelbase_m: float
    # Scales the model's commanded curvature into device authority.
    steer_authority: float = 1.0
    # Seconds of pure-pursuit lookahead.
    lookahead_s: float = 0.15
    # Seconds the command leads the plan, to mask actuation lag.
    curvature_anticipation_s: float = 0.0
    # Horizontal FOV seed/fallback, degrees (real value resolved later).
    fov_h_deg: float = 75.0
    # Short note shown in onboarding about the expected camera view.
    camera_hint: str = ""


class StepStatus(Enum):
    """Outcome of a setup step's read-only `check()`."""

    OK = "ok"                  # already satisfied; nothing to do
    NEEDS_FIX = "needs_fix"    # not satisfied, but apply() can fix it
    CANT_AUTOFIX = "cant_fix"  # not satisfied; the user must act
    UNKNOWN = "unknown"        # couldn't determine (e.g. game not found)


@dataclass
class StepResult:
    status: StepStatus
    detail: str = ""


class SetupStep(ABC):
    """One declarative auto-fix action in a game's first-run setup.

    `check()` is read-only and reports status; `apply()` performs the fix
    when one is possible. The onboarding UI (Phase 4) walks a profile's
    steps, surfaces their status, and offers one-click fixes for the ones
    that report NEEDS_FIX.
    """

    id: str
    title: str

    @abstractmethod
    def check(self) -> StepResult:
        """Read-only: report whether this step is satisfied."""
        ...

    def apply(self) -> tuple[bool, str]:
        """Attempt the fix. Returns (succeeded, message). Default: no
        automatic fix — the user must perform this step manually."""
        return False, "Manual step — no automatic fix available."


class GameProfile(ABC):
    """Everything the runtime needs to know about one game.

    Subclasses declare the class attributes below and implement
    `make_telemetry()`. `resolve_fov()` and `setup_steps()` are optional
    (sensible no-op defaults). Decorate the subclass with `@register` so
    the registry discovers it.
    """

    # Stable identifier used in settings and as the registry key.
    id: str
    # Human-readable name for the UI.
    display_name: str
    # Process image names that indicate this game is running.
    process_names: tuple[str, ...]
    # Physical + control seeds.
    defaults: GameDefaults

    @abstractmethod
    def make_telemetry(self):
        """Construct this game's telemetry reader (ported in Phase 1)."""
        ...

    def resolve_fov(self) -> float | None:
        """Read the game's true horizontal FOV (degrees) from its own
        config, if the game exposes it. Return None when the FOV must be
        measured one-shot or set manually instead (Phase 3)."""
        return None

    def setup_steps(self) -> list[SetupStep]:
        """Declarative auto-fix steps for first-run setup (Phase 4)."""
        return []


# --- Registry: auto-discovers every profile module in this package -------

_REGISTRY: dict[str, type[GameProfile]] = {}
_discovered = False


def register(cls: type[GameProfile]) -> type[GameProfile]:
    """Class decorator that adds a `GameProfile` subclass to the registry."""
    if not getattr(cls, "id", ""):
        raise ValueError(f"{cls.__name__} must define a non-empty `id`")
    _REGISTRY[cls.id] = cls
    return cls


def discover() -> None:
    """Import every profile module in this package so their `@register`
    decorators run. Idempotent; safe to call repeatedly."""
    global _discovered
    if _discovered:
        return
    import simsteer.games as pkg

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name == "base" or info.name.startswith("_"):
            continue
        importlib.import_module(f"{pkg.__name__}.{info.name}")
    _discovered = True


def all_profiles() -> list[GameProfile]:
    """Instantiate every registered profile."""
    discover()
    return [cls() for cls in _REGISTRY.values()]


def get_profile(game_id: str) -> GameProfile | None:
    """Instantiate the profile with this id, or None if unknown."""
    discover()
    cls = _REGISTRY.get(game_id)
    return cls() if cls is not None else None
