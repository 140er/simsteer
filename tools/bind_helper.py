"""Interactive ETS2 binding helper.

ETS2's "press the input you want to bind" prompt only catches actual
gamepad activity. Our virtual pad sits centered while disengaged, so
nothing fires. This tool lets you wiggle one input on demand:

    1. python -m tools.bind_helper          (or double-click bind_helper.bat)
    2. Pick the axis/button to wiggle.
    3. You get a 4-second countdown — alt-tab to ETS2, open the bind
       prompt, then come back focus is no longer needed.
    4. The tool wiggles that input for ~3 s. ETS2 detects it. Done.
    5. Pick another, or `q` to quit.

Triggers wiggle 0 ↔ full. Sticks wiggle ±1 along the chosen axis.
Buttons rapidly press/release.
"""

from __future__ import annotations

import sys
import time

# NOTE: vgamepad is NOT imported at module top on purpose. `import
# vgamepad` connects to the ViGEm bus as a side effect and raises
# VIGEM_ERROR_BUS_NOT_FOUND if the ViGEm driver isn't installed. Since
# pilot.main imports this module eagerly at startup, a top-level import
# here would hard-crash the whole app before the preflight check can
# report the missing driver gracefully. We import vgamepad lazily inside
# the functions that actually need a pad, and build the button table on
# first use via `_buttons()`. (`from __future__ import annotations` keeps
# the `vg.VX360Gamepad` type hints as strings, so they don't need vg
# imported either.)

INPUTS: list[tuple[str, str]] = [
    ("left_stick_x",   "Left stick — X (steering on a typical setup)"),
    ("left_stick_y",   "Left stick — Y"),
    ("right_stick_x",  "Right stick — X"),
    ("right_stick_y",  "Right stick — Y"),
    ("left_trigger",   "Left trigger (LT)"),
    ("right_trigger",  "Right trigger (RT)"),
    ("button_a",       "Button A"),
    ("button_b",       "Button B"),
    ("button_x",       "Button X"),
    ("button_y",       "Button Y"),
    ("button_lb",      "LB shoulder"),
    ("button_rb",      "RB shoulder"),
    ("button_back",    "Back / View"),
    ("button_start",   "Start / Menu"),
    ("dpad_up",        "D-pad up"),
    ("dpad_down",      "D-pad down"),
    ("dpad_left",      "D-pad left"),
    ("dpad_right",     "D-pad right"),
]

_BUTTONS_CACHE: dict | None = None


def _buttons() -> dict:
    """name -> XUSB_BUTTON map, built lazily on first call. Importing
    vgamepad here (not at module top) keeps this module side-effect-free
    to import — see the NOTE above."""
    global _BUTTONS_CACHE
    if _BUTTONS_CACHE is None:
        import vgamepad as vg
        x = vg.XUSB_BUTTON
        _BUTTONS_CACHE = {
            "button_a":     x.XUSB_GAMEPAD_A,
            "button_b":     x.XUSB_GAMEPAD_B,
            "button_x":     x.XUSB_GAMEPAD_X,
            "button_y":     x.XUSB_GAMEPAD_Y,
            "button_lb":    x.XUSB_GAMEPAD_LEFT_SHOULDER,
            "button_rb":    x.XUSB_GAMEPAD_RIGHT_SHOULDER,
            "button_back":  x.XUSB_GAMEPAD_BACK,
            "button_start": x.XUSB_GAMEPAD_START,
            "dpad_up":      x.XUSB_GAMEPAD_DPAD_UP,
            "dpad_down":    x.XUSB_GAMEPAD_DPAD_DOWN,
            "dpad_left":    x.XUSB_GAMEPAD_DPAD_LEFT,
            "dpad_right":   x.XUSB_GAMEPAD_DPAD_RIGHT,
        }
    return _BUTTONS_CACHE


def center(pad: vg.VX360Gamepad) -> None:
    pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    pad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
    pad.left_trigger_float(value_float=0.0)
    pad.right_trigger_float(value_float=0.0)
    for b in _buttons().values():
        pad.release_button(button=b)
    pad.update()


def _set(pad: vg.VX360Gamepad, kind: str, value: float) -> None:
    """Drive one input to `value` (range -1..+1 for sticks, 0..1 for triggers,
    0/1 for buttons)."""
    if kind == "left_stick_x":
        pad.left_joystick_float(x_value_float=value, y_value_float=0.0)
    elif kind == "left_stick_y":
        pad.left_joystick_float(x_value_float=0.0, y_value_float=value)
    elif kind == "right_stick_x":
        pad.right_joystick_float(x_value_float=value, y_value_float=0.0)
    elif kind == "right_stick_y":
        pad.right_joystick_float(x_value_float=0.0, y_value_float=value)
    elif kind == "left_trigger":
        pad.left_trigger_float(value_float=max(0.0, value))
    elif kind == "right_trigger":
        pad.right_trigger_float(value_float=max(0.0, value))
    else:
        buttons = _buttons()
        if kind in buttons:
            btn = buttons[kind]
            if value > 0.5:
                pad.press_button(button=btn)
            else:
                pad.release_button(button=btn)


def hold(pad: vg.VX360Gamepad, kind: str, value: float, seconds: float) -> None:
    """Hold `kind` at `value` for `seconds`, refreshing the pad every 50 ms
    so ETS2 sees a steady input across multiple polling frames."""
    _set(pad, kind, value)
    pad.update()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(0.05)
        pad.update()  # repeated updates keep the value visible across game polls


def wiggle(pad: vg.VX360Gamepad, kind: str) -> None:
    """One slow wiggle: full positive (1 s), back to center (0.3 s), full
    negative (1 s), back to center. ETS2's binding wizard catches the first
    sustained half; the second half tells it the input is bipolar (an axis,
    not a button) so it binds the right way around.
    """
    if kind in _buttons():
        # Buttons: just press for a second.
        hold(pad, kind, 1.0, 1.0)
        _set(pad, kind, 0.0)
        pad.update()
        return

    # Triggers: 0 -> 1 -> 0
    if "trigger" in kind:
        hold(pad, kind, 1.0, 1.2)
        _set(pad, kind, 0.0)
        pad.update()
        return

    # Sticks: positive then negative.
    hold(pad, kind, +1.0, 1.0)
    _set(pad, kind, 0.0)
    pad.update()
    time.sleep(0.3)
    hold(pad, kind, -1.0, 1.0)
    _set(pad, kind, 0.0)
    pad.update()


def menu() -> str | None:
    print("\nWhich input should I wiggle?")
    for i, (_, desc) in enumerate(INPUTS, 1):
        print(f"  {i:2}. {desc}")
    print("   q. quit")
    choice = input("> ").strip().lower()
    if choice in ("q", "quit", "exit", ""):
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(INPUTS):
        return INPUTS[int(choice) - 1][0]
    print("not a valid choice")
    return ""


def main() -> int:
    try:
        import vgamepad as vg
        pad = vg.VX360Gamepad()
    except Exception as e:
        print(f"could not create virtual pad — is ViGEm Bus installed? {e}")
        return 1

    # Warmup: ViGEm sometimes ignores the first update after creation.
    center(pad)
    time.sleep(0.2)
    center(pad)

    print("\nETS2 binding helper — virtual Xbox 360 pad ready.")
    print("Important: close pilot.main first if it's running — only one")
    print("virtual pad should be active at a time, otherwise ETS2 may")
    print("listen to the wrong one.\n")
    print("Workflow:")
    print("  1. In ETS2: Options > Controls. Make sure the virtual Xbox")
    print("     controller is selected as a 'Selected control device'.")
    print("  2. Click the binding field you want to set (it'll show 'Press...').")
    print("  3. Alt-tab back here and pick the input below.")
    print("  4. You get a 5 s countdown — alt-tab to ETS2 during it.\n")

    while True:
        kind = menu()
        if kind is None:
            break
        if not kind:
            continue
        for n in (5, 4, 3, 2, 1):
            print(f"  wiggling in {n}…", end="\r", flush=True)
            time.sleep(1.0)
        print(f"  wiggling {kind}…             ")
        wiggle(pad, kind)
        print("  done. Click another field or pick another input.\n")

    center(pad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
