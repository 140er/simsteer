"""Central registry for every user-facing keybinding.

The handlers themselves live in main.py / debug/overlay.py and consume
key codes from `cv2.waitKey` or `pilot.global_keys`. This module owns
the metadata (label, description, group) and the `is_enabled` predicate
that gates each handler against `settings.disabled_hotkeys`.

Adding a new hotkey:
  1. Pick a stable string id (snake_case).
  2. Add a `Hotkey(...)` entry to `HOTKEYS` below.
  3. In the handler, wrap the action with `if hk_enabled("your_id", settings): ...`.

The id is the persistent contract — display text can change without
breaking saved `disabled_hotkeys` lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pilot.settings import Settings


Group = str


@dataclass(frozen=True)
class Hotkey:
    id: str               # stable persistence key
    key: str              # display name ("INSERT", "NumPad 4", "R", "1")
    action: str           # short human label ("Engage / Disengage")
    description: str      # one-line explainer for the tuner
    group: Group          # tab grouping: Driving / View / Calibration / Binding
    scope: str            # "global" (fires while game focused) or "overlay"
    danger: bool = False  # red-tagged in the UI (test fire, reset, etc.)


# ----- registry -----
#
# Order matters — the tab renders rows in this order, grouped by `group`
# in first-appearance order. Keep "Driving" first since that's what new
# users want to see.

HOTKEYS: list[Hotkey] = [
    # Driving
    Hotkey("engage", "INSERT", "Engage / Disengage",
           "Toggle AI control. Refuses to engage while the camera is "
           "still calibrating; press R to start over.",
           "Driving", "global"),
    Hotkey("lane_change_left", "NumPad 4", "Lane change LEFT",
           "Sustained desire pulse to the model. Hold-time tunable on "
           "the Lane Change tab.",
           "Driving", "global"),
    Hotkey("lane_change_right", "NumPad 6", "Lane change RIGHT",
           "Sustained desire pulse to the model.",
           "Driving", "global"),
    Hotkey("nav_queue_left", "Page Up", "NAV: queue LEFT maneuver",
           "Adds a left maneuver to the NAV queue; fires at the trigger "
           "distance configured in the Lane Change tab.",
           "Driving", "global"),
    Hotkey("nav_queue_right", "Page Down", "NAV: queue RIGHT maneuver",
           "Adds a right maneuver to the NAV queue.",
           "Driving", "global"),
    Hotkey("nav_clear", "End", "NAV: clear queue",
           "Drops any pending NAV maneuvers.",
           "Driving", "global"),

    # View
    Hotkey("quit", "Q", "Quit",
           "Closes the overlay window and exits the loop.",
           "View", "overlay"),
    Hotkey("view_toggle", "V", "Toggle model-view / capture-view",
           "Switches the overlay between the model's warped input and "
           "the raw capture.",
           "View", "overlay"),
    Hotkey("input_inset", "I", "Toggle model-input thumbnail",
           "Shows/hides the small preview of what the model is "
           "actually seeing.",
           "View", "overlay"),
    Hotkey("hud_mode", "H", "Toggle HUD: user / dev",
           "User mode is the clean panel layout; dev mode is the "
           "stack-of-debug-text dump.",
           "View", "overlay"),

    # Calibration
    Hotkey("mirror_sign", "M", "Mirror lateral sign",
           "Flip the steering / overlay convention if everything is "
           "left-right inverted on a fresh install.",
           "Calibration", "overlay"),
    Hotkey("save_calib", "C", "Save calibration.json",
           "Persists the current camera pose + lateral_sign immediately.",
           "Calibration", "overlay"),
    Hotkey("reset_calib_defaults", "0", "Reset calibration to defaults",
           "Restores pitch / yaw / FOV / height to dataclass defaults.",
           "Calibration", "overlay", danger=True),
    Hotkey("recalibrate", "R", "Recalibrate",
           "Wipe LiveCalib + LiveParams + first-drive flag and restart "
           "the wizard. Force-disengages first.",
           "Calibration", "overlay", danger=True),
    Hotkey("fov_minus", "[", "FOV −1°",
           "Narrow capture FOV by 1°. FOV is static — set it to match "
           "your in-game horizontal FOV.",
           "Calibration", "overlay"),
    Hotkey("fov_plus", "]", "FOV +1°",
           "Widen capture FOV.",
           "Calibration", "overlay"),
    Hotkey("pitch_minus", ",", "Pitch −0.5°",
           "Tilt camera UP. Positive pitch = looking down.",
           "Calibration", "overlay"),
    Hotkey("pitch_plus", ".", "Pitch +0.5°",
           "Tilt camera DOWN.",
           "Calibration", "overlay"),
    Hotkey("height_minus", ";", "Camera height −5 cm",
           "Lower mount height.",
           "Calibration", "overlay"),
    Hotkey("height_plus", "'", "Camera height +5 cm",
           "Raise mount height.",
           "Calibration", "overlay"),

    # Binding (while disengaged — for ETS2's controller wizard)
    Hotkey("wiggle_lx", "1", "Wiggle Left Stick X",
           "Drives the input at full deflection so ETS2's binding "
           "wizard catches it. Disengaged only.",
           "Binding (disengaged)", "overlay"),
    Hotkey("wiggle_rx", "2", "Wiggle Right Stick X",
           "", "Binding (disengaged)", "overlay"),
    Hotkey("wiggle_lt", "3", "Wiggle Left Trigger (brake)",
           "", "Binding (disengaged)", "overlay"),
    Hotkey("wiggle_rt", "4", "Wiggle Right Trigger (throttle)",
           "", "Binding (disengaged)", "overlay"),
    Hotkey("wiggle_a", "5", "Wiggle A",
           "", "Binding (disengaged)", "overlay"),
    Hotkey("wiggle_b", "6", "Wiggle B",
           "", "Binding (disengaged)", "overlay"),
    Hotkey("wiggle_x", "7", "Wiggle X",
           "", "Binding (disengaged)", "overlay"),
    Hotkey("wiggle_y", "8", "Wiggle Y",
           "", "Binding (disengaged)", "overlay"),
    Hotkey("wiggle_lb", "9", "Wiggle LB",
           "", "Binding (disengaged)", "overlay"),
    Hotkey("wiggle_rb", "0", "Wiggle RB",
           "", "Binding (disengaged)", "overlay"),

    # Test
    Hotkey("test_fire", "T", "Test fire steering ±0.5",
           "Sends a quick left-right axis sweep to confirm the virtual "
           "pad is reaching the game. Force-engages briefly.",
           "Diagnostics", "overlay", danger=True),
]


# Map for O(1) lookup by id, built once at import.
_BY_ID: dict[str, Hotkey] = {h.id: h for h in HOTKEYS}

# Map from key-display-name to id for some convenience reverse-lookups.
KEY_TO_ID: dict[str, str] = {h.key: h.id for h in HOTKEYS}


def by_id(hotkey_id: str) -> Hotkey | None:
    return _BY_ID.get(hotkey_id)


def hk_enabled(hotkey_id: str, settings: "Settings | None") -> bool:
    """True when the hotkey should fire. Default-true if settings is
    None (called before Settings is wired) or the id is unknown."""
    if settings is None:
        return True
    return hotkey_id not in (settings.disabled_hotkeys or [])


def groups_in_order() -> list[Group]:
    """Return groups in their first-appearance order from HOTKEYS."""
    seen: list[Group] = []
    for h in HOTKEYS:
        if h.group not in seen:
            seen.append(h.group)
    return seen


def hotkeys_in_group(group: Group) -> list[Hotkey]:
    return [h for h in HOTKEYS if h.group == group]
