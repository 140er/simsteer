"""One-shot, open-loop FOV solver for ETS2 (the safe auto-FOV).

The model is fed a warp that crops the captured frame to the model's
expected HFOV using `calib.fov_h_deg`. If that FOV is wrong, the crop is
the wrong width, the world appears zoomed in/out, and the model's
perceived forward speed `pose[0]` diverges from real speed by exactly the
zoom factor:

    vx_model / v_ego = tan(FOV_set / 2) / tan(FOV_true / 2)

So we can solve the true FOV from a single straight-line measurement:

    FOV_true = 2 * atan( tan(FOV_set / 2) / ratio )

This is NOT the removed LiveFov (a continuous closed loop that fed itself
because the model's vx is a function of the very FOV it was tuning). We
measure the ratio once over a straight stretch, solve, write it, and
stop. A one-shot open-loop solve cannot run away.

    python tools/measure_fov.py            # measure + save
    python tools/measure_fov.py --dry-run  # measure only, don't write

Drive STRAIGHT on a highway at > ~30 km/h while it collects.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from simsteer.core.calibration import Calibration
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue
from simsteer.games.ets2 import ETS2Telemetry
from simsteer.runtime.capture import Capture, CaptureConfig

GAME = "ets2"
NEED_SAMPLES = 60       # straight+fast samples before we trust the solve
MIN_SPEED_MPS = 8.0     # ~29 km/h — below this the pose vx is noisy
MAX_YAW_RAD_S = 0.06    # ~3.4 deg/s — only measure on near-straight road
WARMUP_FRAMES = 25      # let the recurrent model's context buffer fill
TIMEOUT_S = 150.0


def main() -> int:
    dry = "--dry-run" in sys.argv

    calib = Calibration.load(GAME)
    print(f"current FOV: {calib.fov_h_deg:.1f}deg  "
          f"(image {calib.image_w}x{calib.image_h})")

    tel = ETS2Telemetry()
    if not tel.available:
        print("waiting for ETS2 telemetry (load into an active drive)...")
        for _ in range(40):           # ~20 s; handshake is one-shot per reader
            tel.close()
            time.sleep(0.5)
            tel = ETS2Telemetry()
            if tel.available:
                break
        if not tel.available:
            print("ETS2 telemetry NOT available — start ETS2, load into a "
                  "drive (not a menu), then rerun.")
            return 1

    fq = FrameQueue()
    model = DrivingModel()
    print(f"vision: {model.active_provider}   policy: {model.policy_provider}")
    print("Drive STRAIGHT on a highway at > ~30 km/h. Collecting samples...")

    ratios: list[float] = []
    t0 = time.time()
    i = 0
    last_status = 0.0
    with Capture(CaptureConfig(target_fps=20)) as cap:
        while time.time() - t0 < TIMEOUT_S and len(ratios) < NEED_SAMPLES:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.02)
                continue
            i += 1
            n, w = fq.push(frame, calib)
            vo, po = model.step(n, w)
            d = decode(vo, po)
            if i <= WARMUP_FRAMES:
                continue
            v = tel.speed_mps()
            yaw = tel.yaw_rate_rad_s()
            if v is None or yaw is None:
                continue
            now = time.time()
            if v < MIN_SPEED_MPS or abs(yaw) > MAX_YAW_RAD_S:
                if now - last_status > 0.5:
                    print(f"  waiting for straight+fast: v={v * 3.6:5.1f} km/h "
                          f"yaw={yaw:+.3f} rad/s  (have {len(ratios)} samples)")
                    last_status = now
                continue
            vx = float(d.pose[0])
            if vx <= 0.1:
                continue
            ratios.append(vx / v)
            if now - last_status > 0.5:
                print(f"  collecting: {len(ratios)}/{NEED_SAMPLES}  "
                      f"ratio~{np.median(ratios):.3f}  "
                      f"(vx={vx:4.1f} v={v:4.1f} m/s)")
                last_status = now

    tel.close()

    if len(ratios) < 10:
        print(f"\nnot enough straight+fast samples ({len(ratios)}). "
              f"Get on a highway, hold it straight at speed, and rerun.")
        return 1

    ratio = float(np.median(ratios))
    fov_old = calib.fov_h_deg
    fov_new = math.degrees(2.0 * math.atan(
        math.tan(math.radians(fov_old) / 2.0) / ratio))
    clamped = max(40.0, min(150.0, fov_new))

    print(f"\nmeasured vx_model/v_ego = {ratio:.3f}  (n={len(ratios)} samples)")
    print(f"  ratio>1 = FOV set too wide; ratio<1 = too narrow")
    print(f"  solved FOV: {fov_old:.1f}deg -> {fov_new:.1f}deg"
          + (f" (clamped to {clamped:.1f})" if clamped != fov_new else ""))

    if dry:
        print("  --dry-run: not writing.")
        return 0

    calib.fov_h_deg = clamped
    calib.save(game=GAME)
    print(f"SAVED calibration_{GAME}.json  fov_h_deg={clamped:.1f}  "
          f"(VFOV={calib.fov_v_deg:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
