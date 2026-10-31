"""Generate neutral upgrade fixtures with V0.12's actual save methods.

Run with a checkout of V0.12 at V012_COMMIT as the only argument.
The checked-in JSON is consumed by tests; regeneration is explicit.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

V012_COMMIT = "8f9ae83e6c11968d1e4aa0293d39db48a77359c2"
MODEL_IDENTITY = {"model": "upgrade-fixture", "sha256": "0" * 64}


def main():
    source = Path(sys.argv[1]).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != V012_COMMIT:
        raise SystemExit(f"Expected V0.12 commit {V012_COMMIT}")
    sys.path[:0] = [str(source), str(source / "tests")]
    output = Path(__file__).resolve().parent
    os.environ["SIMSTEER_DATA_DIR"] = str(output / "camera")

    import math
    import numpy as np

    from simsteer.app.settings import Settings
    from simsteer.calib.flow import Obs, Step
    from simsteer.calib.score import evaluate
    from simsteer.core.calibration import Calibration
    from simsteer.core.learners.livecalib import CalStatus, LiveCalib
    from simsteer.core.learners.liveparams import LiveParams
    from test_calib_flow import Driver, new_flow
    from test_calib_score import _drive
    from test_steering_direction import _probe_output

    for step in ("camera", "check", "done"):
        directory = output / step
        directory.mkdir(exist_ok=True)
        for path in directory.glob("*.json"):
            path.unlink()

    cal = Calibration(image_w=1920, image_h=1080, fov_h_deg=75.)
    lc = LiveCalib(game="ets2", clock=lambda: 0.)
    lp = LiveParams(game="ets2", device_kind="gamepad", lag_frames=0, clock=lambda: 0.)
    flow = new_flow(lc, lp, cal, model_identity=MODEL_IDENTITY)
    flow._finish_fov("manual")

    def camera_sample():
        lc.update(cal, pose=np.array([10., .1, -.1]),
                  road_transform=np.array([0., 0., 1.2]), actual_yaw_rate=0.,
                  v_ego=10., pose_std=np.array([.01, .02, .01]),
                  road_transform_std=np.full(3, .01), game_steer=0.)

    for _ in range(30):
        camera_sample()
    flow.tick(Obs(now=.05, dt=.05, v=10.))
    flow.attempts = 3
    flow._camera_no_progress_s = 2.5
    flow.save()
    cal.save(game="ets2")
    lc.save_state("ets2")
    lp.save()
    Settings(game="ets2", max_width=1280, disabled_hotkeys=["hud"]).save()
    assert flow.step is Step.CAMERA

    # Learn real Camera and Steering state from synthetic observations.
    os.environ["SIMSTEER_DATA_DIR"] = str(output / "done")
    for _ in range(280):
        camera_sample()
    assert lc.cal_status is CalStatus.CALIBRATED
    flow.tick(Obs(now=.1, dt=.05, v=10.))
    for speed in (12., 24.):
        for i in range(400):
            wheel = .02 * math.sin(i / 30)
            lp.update(10 * wheel, speed, wheel)
    assert lp.trusted()
    driver = Driver(flow)
    for i in range(80):
        wheel = .01 if i % 40 < 20 else -.01
        driver.frames(1, wheel=wheel, observed_axis=4 * wheel,
                      model_curvature=wheel / 2.5, engaged=False)
    assert flow.direction.ready
    _probe_output(flow, -1)
    driver.frames(1)
    assert flow.step is Step.CHECK
    flow._finish_check(evaluate(_drive()))
    assert flow.step is Step.DONE

    flow.save()
    cal.save(game="ets2")
    lc.save_state("ets2")
    lp.save()
    Settings(game="ets2", max_width=1280, disabled_hotkeys=["hud"]).save()

    os.environ["SIMSTEER_DATA_DIR"] = str(output / "check")
    for path in (output / "done").glob("*.json"):
        if not path.name.startswith("calibflow"):
            shutil.copyfile(path, output / "check" / path.name)
    flow._enter(Step.CHECK)
    flow.save()


if __name__ == "__main__":
    main()
