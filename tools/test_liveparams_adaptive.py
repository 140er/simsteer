"""Smoke test for the new LiveParams adaptive behaviors:

1. Adaptive forgetting: when innovations are large, FORGET drops so
   the fit re-converges faster than the legacy 50 s window.
2. Watchdog: persistent disagreement triggers a P reset.
3. Trouble detector: a sudden speed collapse causes updates to be
   skipped (crash / off-road).
"""
from __future__ import annotations

import numpy as np

from pilot.liveparams import (
    BAD_FIT_PATIENCE, FORGET, FORGET_FLOOR, LiveParams,
    TROUBLE_SPEED_DROP_FRAC,
)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 2000

    # Smooth low-freq steering signal, like real driving.
    raw = rng.standard_normal(n + 30)
    steer = np.convolve(raw, np.ones(20) / 20.0, mode="same") * 1.5

    print("=== test 1: adaptive forgetting accelerates re-convergence ===")
    # Phase 1: 500 samples at true_scale=0.10, fit converges.
    # Phase 2: true_scale ABRUPTLY changes to 0.30 (different car).
    # Compare convergence speed with adaptive vs hypothetical fixed FORGET.
    lp = LiveParams()
    lp.reset()
    true_scale = 0.10
    for i in range(500):
        wheel = true_scale * steer[i - 5] if i >= 5 else 0.0
        lp.update(float(steer[i]), 20.0, float(wheel))
    print(f"  after 500 samples @ scale=0.10: fit scale={lp.scale:+.4f}")

    # Now change the dynamics.
    print(f"  -- dynamics CHANGE: true_scale 0.10 -> 0.30 --")
    true_scale = 0.30
    snapshots = {}
    for i in range(500, 2000):
        wheel = true_scale * steer[i - 5] if i >= 5 else 0.0
        lp.update(float(steer[i]), 20.0, float(wheel))
        if (i - 500) in (50, 100, 200, 500, 1000, 1499):
            snapshots[i - 500] = (lp.scale, lp.adaptive_forget(),
                                   lp.watchdog_resets,
                                   lp._consecutive_bad_fit)
    for samples_since_change, (s, fr, wd, cb) in snapshots.items():
        print(f"  +{samples_since_change:4d} samples after change: "
              f"scale={s:+.4f}  forget={fr:.4f}  wd_resets={wd}  cb={cb}")
    print(f"  (target was +0.30)")

    print()
    print("=== test 2: trouble detector skips during sudden speed loss ===")
    lp = LiveParams()
    lp.reset()
    # Warm up at 25 m/s.
    for i in range(100):
        lp.update(float(steer[i % n]), 25.0, 0.10 * float(steer[(i - 5) % n]))
    print(f"  warmed up @ 25 m/s:  samples={lp.session_samples}  rej_t={lp.rej_trouble}")

    # Simulate a crash: speed collapses to 5 m/s.
    print(f"  simulating crash: speed drops 25 -> 5 m/s")
    pre_trouble_samples = lp.session_samples
    for i in range(100, 200):
        lp.update(float(steer[i % n]), 5.0, 0.10 * float(steer[(i - 5) % n]))
    crash_samples_added = lp.session_samples - pre_trouble_samples
    print(f"  after 100 crash samples:  samples added={crash_samples_added}  "
          f"rej_t={lp.rej_trouble}  (expect samples_added=0, rej_t>0)")

    # Speed recovers — fit should resume.
    print(f"  recovery: speed back to 25 m/s for 100 samples")
    pre_recovery_samples = lp.session_samples
    for i in range(200, 300):
        lp.update(float(steer[i % n]), 25.0, 0.10 * float(steer[(i - 5) % n]))
    recovery_samples_added = lp.session_samples - pre_recovery_samples
    print(f"  after 100 recovery samples:  samples added={recovery_samples_added}  "
          f"(expect samples_added > 0, fit resumed)")

    print()
    print("=== test 3: watchdog reset on persistent bad fit ===")
    lp = LiveParams()
    lp.reset()
    # Warm to convergence at scale=0.10.
    for i in range(500):
        wheel = 0.10 * steer[i - 5] if i >= 5 else 0.0
        lp.update(float(steer[i]), 20.0, float(wheel))
    pre_resets = lp.watchdog_resets
    # Now feed contradictory data: wheel says scale=-0.10 (sign flipped).
    print(f"  pre-contradiction: scale={lp.scale:+.4f}  wd={pre_resets}")
    for i in range(500, 1000):
        wheel = -0.10 * steer[i - 5]
        lp.update(float(steer[i]), 20.0, float(wheel))
    print(f"  after 500 contradictory samples: scale={lp.scale:+.4f}  "
          f"wd_resets={lp.watchdog_resets - pre_resets}  "
          f"(expect >=1 reset, scale moves toward -0.10)")


if __name__ == "__main__":
    main()
