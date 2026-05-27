"""Diagnostic: prints every nav/control keypress GlobalKeys detects.

Run this BEFORE you launch pilot.main if the nav keys (PageUp/PageDown/End)
aren't doing anything. It tells you whether:

  - the keypress is reaching us at all (you'll see a line per press), or
  - it's not being detected (no lines = your keyboard/OS isn't sending
    the VK codes we watch for, probably because a keyboard mode / Fn
    layer is remapping them, or another app is suppressing them).

While running, press: PageUp, PageDown, End, Insert, ← / →. Each one
should produce a printed line on press AND release. Run for ~10s then
Ctrl-C to quit.

If a key prints nothing on your machine, switching its binding in
pilot.main is the fix — pick something that DOES print here.
"""
from __future__ import annotations

import sys
import time

from pilot.global_keys import (
    KEY_END, KEY_INSERT, KEY_LEFT, KEY_PAGEDOWN, KEY_PAGEUP, KEY_RIGHT,
    KEY_F1, KEY_F2, KEY_F3, KEY_F4,
    KEY_DELETE, KEY_HOME,
    GlobalKeys,
)

# Watch every potentially-useful nav key. The label is what we'll
# print on press/release so you know which one fired.
WATCH = {
    KEY_INSERT:   "Insert",
    KEY_DELETE:   "Delete",
    KEY_HOME:     "Home",
    KEY_END:      "End",
    KEY_PAGEUP:   "PageUp",
    KEY_PAGEDOWN: "PageDown",
    KEY_LEFT:     "Left arrow",
    KEY_RIGHT:    "Right arrow",
    KEY_F1:       "F1",
    KEY_F2:       "F2",
    KEY_F3:       "F3",
    KEY_F4:       "F4",
    ord("q"):     "Q",
}


def main() -> int:
    print("Global-key diagnostic. Press any of these keys; each press will")
    print("print a line. Ctrl-C to quit.")
    print()
    print("Watching:")
    for ch, label in sorted(WATCH.items(), key=lambda kv: kv[1]):
        vk = ch & 0xFF if ch & 0x100 else (ord(chr(ch).upper()) if chr(ch).isalpha() else ch)
        print(f"  {label:12s}  id=0x{ch:X}  -> VK 0x{vk:02X}")
    print()
    g = GlobalKeys(set(WATCH.keys()))
    # Track release edges too, by polling the down state directly.
    prev_down = {ch: False for ch in WATCH}
    try:
        while True:
            pressed = g.poll()
            # Check release edges separately (poll() only reports rising edges)
            for ch in WATCH:
                vk = g._vk[ch]
                down = bool(g._user32.GetAsyncKeyState(vk) & 0x8000)
                if not down and prev_down[ch]:
                    print(f"  release: {WATCH[ch]}")
                prev_down[ch] = down
            for ch in pressed:
                print(f"  PRESS:   {WATCH[ch]}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
