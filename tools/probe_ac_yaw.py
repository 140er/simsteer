"""Probe AC's shared memory while you turn the steering wheel.

The driving model + LiveParams need real yaw rate. AC's
`localAngularVel[3]` is in the vehicle frame, but the per-axis order
isn't clearly documented across versions — could be [pitch, yaw, roll]
or [pitch, roll, yaw] or other.

This script polls all three components plus steer + speed for a few
seconds. Drive in a CIRCLE (or just turn the wheel hard while moving)
during the polling. Whichever component shows the strongest linear
correlation with `steerAngle` is the yaw axis. Output tells you which
index to set in pilot/telemetry_ac.py.

    python -m tools.probe_ac_yaw           # 10 second poll
"""

from __future__ import annotations

import struct
import sys
import time

import numpy as np

from pilot.telemetry_ac import (
    OFF_LOCAL_ANG_VEL, OFF_PACKET_ID, OFF_SPEED_KMH, OFF_STEER_ANGLE,
    SHARED_MEM_NAME, SHARED_MEM_SIZE,
)


def main(seconds: float = 10.0) -> int:
    import mmap
    try:
        mm = mmap.mmap(-1, SHARED_MEM_SIZE, SHARED_MEM_NAME,
                       access=mmap.ACCESS_READ)
    except OSError as e:
        print(f"can't open {SHARED_MEM_NAME}: {e}")
        return 1

    samples: list[tuple[float, float, float, float, float]] = []
    print(f"polling for {seconds:.0f}s — turn the wheel left + right while "
          f"the car is rolling")
    t_end = time.perf_counter() + seconds
    last_pkt = -1
    while time.perf_counter() < t_end:
        pkt = struct.unpack_from("<i", mm, OFF_PACKET_ID)[0]
        if pkt == last_pkt:
            time.sleep(0.005)
            continue
        last_pkt = pkt
        steer = struct.unpack_from("<f", mm, OFF_STEER_ANGLE)[0]
        speed = struct.unpack_from("<f", mm, OFF_SPEED_KMH)[0]
        avx, avy, avz = struct.unpack_from("<3f", mm, OFF_LOCAL_ANG_VEL)
        samples.append((steer, speed, avx, avy, avz))
        time.sleep(0.01)

    mm.close()
    if len(samples) < 50:
        print(f"only {len(samples)} samples — is AC running and rolling?")
        return 1

    arr = np.array(samples, dtype=np.float64)
    steer = arr[:, 0]
    speed = arr[:, 1]
    avx, avy, avz = arr[:, 2], arr[:, 3], arr[:, 4]

    print(f"\ngot {len(samples)} samples, "
          f"speed range {speed.min():.1f}-{speed.max():.1f} km/h, "
          f"steer range {steer.min():+.2f} to {steer.max():+.2f}")

    if abs(steer.std()) < 0.02:
        print("steer barely moved — turn the wheel more next time.")
        return 1

    # Pearson correlation of each angular velocity axis with steer.
    for label, av in [("avx", avx), ("avy", avy), ("avz", avz)]:
        if av.std() < 1e-5:
            print(f"  {label}: std=0  (axis didn't move)")
            continue
        r = float(np.corrcoef(steer, av)[0, 1])
        print(f"  {label}: std={av.std():.4f}  correlation with steer = {r:+.3f}")

    # Yaw rate has the highest |correlation| with steer (positive or
    # negative depending on AC's sign convention).
    cors = {
        "avx (index 0)": float(np.corrcoef(steer, avx)[0, 1]) if avx.std() > 1e-5 else 0.0,
        "avy (index 1)": float(np.corrcoef(steer, avy)[0, 1]) if avy.std() > 1e-5 else 0.0,
        "avz (index 2)": float(np.corrcoef(steer, avz)[0, 1]) if avz.std() > 1e-5 else 0.0,
    }
    best_label, best_r = max(cors.items(), key=lambda x: abs(x[1]))
    print(f"\nstrongest correlation: {best_label}  (r = {best_r:+.3f})")
    print(f"=> yaw rate is in localAngularVel{best_label.split()[1]}")
    print(f"   sign is {'normal' if best_r > 0 else 'INVERTED'} relative to steer "
          f"(LiveParams handles either way).")
    return 0


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    raise SystemExit(main(secs))
