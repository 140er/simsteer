"""Smoke test: LiveParams skips updates when the user's wheel/keyboard
diverges from what the AI commanded.

Scenario:
- AI sends commanded_axis = 0.05 (gentle steering) every frame
- gameSteer = 0.05 + user_input
  - 200 frames: no user input → gameSteer matches, fit updates
  - 200 frames: user input = +0.5 → gameSteer = 0.55, gate rejects
  - 200 frames: no user input again → fit resumes
"""
from __future__ import annotations

import numpy as np

from pilot.liveparams import LiveParams


def main() -> None:
    lp = LiveParams()
    lp.reset()

    true_scale = 0.10
    commanded = 0.05  # what the AI sends

    def step(commanded_axis: float, user_input: float, label: str = "") -> None:
        game_steer = commanded_axis + user_input
        # Synthesize wheel angle from total steering (lag-correlated
        # for cleanliness — we just pass current to simulate steady).
        wheel = true_scale * game_steer
        lp.update(game_steer, 20.0, wheel,
                  commanded_axis=commanded_axis)
        if label:
            print(f"  {label}: samples={lp.session_samples} "
                  f"rej_a={lp.rej_assist} scale={lp.scale:+.4f}")

    print("=== phase 1: no user input — fit should accumulate samples ===")
    for i in range(200):
        step(commanded, 0.0)
    print(f"  end: samples={lp.session_samples} scale={lp.scale:+.4f} "
          f"rej_a={lp.rej_assist}  (expect samples~200, scale~0.10)")

    print()
    print("=== phase 2: user adds +0.5 input — all should be rejected ===")
    pre = lp.session_samples
    pre_rej = lp.rej_assist
    for i in range(200):
        step(commanded, 0.5)
    added = lp.session_samples - pre
    rejected = lp.rej_assist - pre_rej
    print(f"  end: samples added={added} rej_a added={rejected} "
          f"scale={lp.scale:+.4f}  (expect added=0, rej_a=200)")

    print()
    print("=== phase 3: user backs off — fit resumes ===")
    pre = lp.session_samples
    for i in range(200):
        step(commanded, 0.0)
    added = lp.session_samples - pre
    print(f"  end: samples added={added} scale={lp.scale:+.4f} "
          f"(expect added~200, scale still ~0.10)")

    print()
    print("=== phase 4: disengaged (commanded_axis=None) — fit accepts ===")
    # When commanded_axis is None, the assist gate is bypassed so the
    # human's pure-input samples can train the fit when AI is off.
    pre = lp.session_samples
    pre_rej = lp.rej_assist
    for i in range(200):
        game_steer = 0.3  # human is driving
        wheel = true_scale * 0.3
        lp.update(game_steer, 20.0, wheel, commanded_axis=None)
    added = lp.session_samples - pre
    rejected = lp.rej_assist - pre_rej
    print(f"  end: samples added={added} rej_a added={rejected}  "
          f"(expect added~200, rej_a=0)")


if __name__ == "__main__":
    main()
