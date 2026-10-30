"""Replay a clip through the loop and report how calibration went.

Runs `loop.run()` in process with the replay source, telemetry and clock,
like `python -m simsteer --source`, plus what calibration lanes need:
a state dir that outlives the run, scripted keys, a starting FOV or
steering fit, and the step times. It works on revisions with the old
first-drive wizard too, so parent and head can be compared:

    python tools/calib_replay.py --clip ~/.cache/simsteer/clips/synthetic-calib \\
        --passive --log out.jsonl
    python tools/calib_replay.py --tree ../parent --fake-pad --passive ...

The last line printed is a JSON summary: first clip time of each step
(head) or of camera calibrated and wizard DONE (parent), the check
verdict, and the final FOV, camera pose and steering fit.

Keys are `WHEN:KEY` pairs, comma separated. WHEN is a frame number or
`STEP+N`, N frames after the flow first enters STEP (fov, camera,
steering, check). KEY is r, insert, backspace, q or a single character.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = {"r": ord("r"), "q": ord("q"), "backspace": 8,
        "insert": 0x100 | 0x2D}


class ScriptedKeys:
    """Keys by processed frame number; the loop polls once per frame.
    With `accept_fov`, INSERT is pressed whenever a measured FOV waits."""

    def __init__(self, script: dict[int, list[int]], holder: dict,
                 accept_fov: bool = False, accept_after: int = 0,
                 settings=None, passive_from_steering: bool = False) -> None:
        self.script = script
        self.holder = holder
        self.accept_fov = accept_fov
        self.accept_after = accept_after
        self.settings = settings
        self.passive_from_steering = passive_from_steering
        self.frame = 0
        self.entered: dict[str, int] = {}
        self._proposed_at: int | None = None

    def poll(self) -> list[int]:
        self.frame += 1
        keys = list(self.script.get(self.frame, []))
        flow = self.holder.get("flow")
        if flow is None:
            return keys
        self.entered.setdefault(flow.step.value, self.frame)
        for when, codes in self.script.items():
            if isinstance(when, str):
                step, n = when.split("+")
                if self.entered.get(step) is not None and \
                        self.frame == self.entered[step] + int(n):
                    keys += codes
        if self.passive_from_steering:
            # Like ETS2 with a gamepad: the fit only learns once SimSteer
            # would drive, which a replay stands in for with the recording.
            set_passive(self.settings, flow.step.value in ("steering", "check"))
        if self.accept_fov and getattr(flow, "fov_proposal", None) is not None:
            if self._proposed_at is None:
                self._proposed_at = self.frame
            if self.frame - self._proposed_at >= self.accept_after:
                keys.append(KEYS["insert"])
                self._proposed_at = None
        return keys


class FakePad:
    """A connected, disengaged pad, so a wizard-era loop fits passively."""

    kind = "gamepad"
    last_error = None
    engaged = False

    def __init__(self, *a, **k) -> None:
        pass

    def engage(self) -> None:
        self.engaged = True

    def disengage(self) -> None:
        self.engaged = False

    def center(self) -> None:
        pass

    def set_steering(self, *a, **k) -> None:
        pass

    def set_throttle_brake(self, *a, **k) -> None:
        pass


def set_passive(settings, on: bool) -> None:
    """`passive_fit` since game profiles, `passive_fit_ets2` before."""
    name = "passive_fit" if hasattr(settings, "passive_fit") else "passive_fit_ets2"
    setattr(settings, name, on)


def parse_keys(text: str | None) -> dict[int | str, list[int]]:
    out: dict[int | str, list[int]] = {}
    for part in filter(None, (text or "").split(",")):
        when, key = part.split(":")
        code = KEYS.get(key.lower(), ord(key[0]) if len(key) == 1 else None)
        if code is None:
            raise SystemExit(f"unknown key {key!r}")
        out.setdefault(int(when) if when.isdigit() else when, []).append(code)
    return out


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--tree", type=Path, default=ROOT,
                    help="checkout to run (default: this one)")
    ap.add_argument("--state", type=Path, default=None,
                    help="data dir kept across runs (default: fresh, deleted)")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--keys", default=None)
    ap.add_argument("--accept-fov", action="store_true",
                    help="press INSERT when a measured FOV is proposed")
    ap.add_argument("--accept-after", type=int, default=0, metavar="FRAMES",
                    help="wait this long before accepting the FOV")
    ap.add_argument("--passive-from-steering", action="store_true",
                    help="fit steering from the recording only from the "
                         "Steering step on")
    ap.add_argument("--fov", type=float, default=None,
                    help="start from this FOV")
    ap.add_argument("--seed", default=None, metavar="A,B,C,SAMPLES",
                    help="start from this steering fit")
    ap.add_argument("--device", default="gamepad")
    ap.add_argument("--passive", action="store_true",
                    help="fit steering from the recorded drive")
    ap.add_argument("--fake-pad", action="store_true",
                    help="give the loop a disengaged pad")
    ap.add_argument("--generic", action="store_true",
                    help="run the game as a profile without a steering fit")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--snap-dir", type=Path, default=None,
                    help="open the window and save the HUD at each step")
    args = ap.parse_args(argv)

    clip = args.clip.expanduser().resolve()
    tree = args.tree.expanduser().resolve()
    state = args.state.expanduser().resolve() if args.state else None
    tmp = None
    if state is None:
        tmp = state = Path(tempfile.mkdtemp(prefix="simsteer-calib-"))
    state.mkdir(parents=True, exist_ok=True)
    os.environ["SIMSTEER_DATA_DIR"] = str(state)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(tree))
    os.chdir(tree)

    meta = read_json(clip / "clip.json") or {}
    game = meta.get("game", "ets2")
    if args.fov is not None:
        p = state / f"calibration_{game}.json"
        cal = read_json(p) or {}
        cal["fov_h_deg"] = args.fov
        p.write_text(json.dumps(cal, indent=2))
    if args.seed:
        a, b, c, n = (float(x) for x in args.seed.split(","))
        (state / f"liveparams_{game}_{args.device}.json").write_text(json.dumps(
            {"a": a, "b": b, "c": c, "samples": int(n)}, indent=2))

    import simsteer.runtime.loop as loop
    from simsteer.app.settings import Settings
    from simsteer.runtime.replay import ReplayClock, ReplaySource, ReplayTelemetry

    has_flow = hasattr(loop, "CalibFlow")
    clock = ReplayClock()
    marks: dict[str, float] = {}
    holder: dict = {}

    if has_flow:
        base = loop.CalibFlow

        class Flow(base):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                holder["flow"] = self
        loop.CalibFlow = Flow
    else:
        base_wiz = loop.Wizard

        class Wiz(base_wiz):
            def tick(self):
                from simsteer.core.learners.livecalib import CalStatus
                if (self._lc.cal_status == CalStatus.CALIBRATED
                        and "camera_done" not in marks):
                    marks["camera_done"] = clock.now()
                if self.phase == 0 and "wizard_done" not in marks:
                    marks["wizard_done"] = clock.now()
                super().tick()
        loop.Wizard = Wiz

    target = game
    if args.generic:
        import dataclasses

        from simsteer.games import get_profile
        from simsteer.games.profile import TelemetrySpec
        target = dataclasses.replace(get_profile(game), telemetry=TelemetrySpec("none"))
    if args.fake_pad:
        loop.DeviceManager = FakePad

    snaps: list[str] = []
    if args.snap_dir is not None:
        args.snap_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        shown = loop.show_scaled
        seen: set[str] = set()

        def show(img, max_w):
            flow = holder.get("flow")
            if flow is not None:
                key = flow.step.value
                if getattr(flow, "fov_proposal", None) is not None:
                    key += "-proposal"
                elif (key == "steering" and flow.direction.ready
                      and not flow.output_direction.ready):
                    key += "-output-direction"
                elif flow.calibrating and flow.progress >= 0.5:
                    key += "-mid"
                if key not in seen:
                    seen.add(key)
                    out = args.snap_dir / f"{len(seen):02d}-{key}.png"
                    cv2.imwrite(str(out), img)
                    snaps.append(str(out))
            shown(img, max_w)
            # Scripted keys replace WindowKeys, whose waitKey repaints.
            cv2.waitKey(1)
        loop.show_scaled = show

    settings = Settings(no_tuner=True, no_gamepad=not args.fake_pad,
                        device=args.device)
    set_passive(settings, args.passive)
    tel = ReplayTelemetry(clip, clock)
    log = args.log.expanduser().resolve() if args.log else (state / "calib-replay.jsonl")
    keys = ScriptedKeys(parse_keys(args.keys), holder, args.accept_fov,
                        args.accept_after, settings, args.passive_from_steering)
    rc = loop.run(settings, target, source=ReplaySource(clip, clock),
                  make_telemetry=lambda: tel, clock=clock, keys=keys,
                  headless=args.snap_dir is None, max_frames=args.frames,
                  log_path=str(log))

    steps: dict[str, float] = {}
    last = None
    with log.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            last = row
            step = row.get("calib")
            if step is not None and step not in steps:
                steps[step] = row["t"]
    if has_flow:
        steps.pop("fov", None)
        marks.update({f"{k}_start": v for k, v in steps.items()})
    summary = {
        "rev": "flow" if has_flow else "wizard",
        "rc": rc,
        "frames": last["frame"] if last else 0,
        "clip_s": last["t"] if last else 0.0,
        "marks": marks,
        "final_step": last.get("calib") if last else None,
        "final_pct": last.get("calib_pct") if last else None,
    }
    cal = read_json(state / f"calibration_{game}.json") or {}
    summary["camera"] = {k: cal.get(k) for k in ("fov_h_deg", "pitch_deg",
                                                  "yaw_deg", "height_m")}
    summary["steering"] = read_json(state / f"liveparams_{game}_{args.device}.json")
    flow_state = read_json(state / f"calibflow_{game}_{args.device}.json")
    if flow_state:
        summary["result"] = flow_state.get("result")
        summary["fov"] = {k: v for k, v in flow_state.get("fov", {}).items()
                          if k != "solver"}
        solver = flow_state.get("fov", {}).get("solver", {})
        summary["fov"]["estimate"] = solver.get("estimate")
        summary["fov"]["candidates"] = solver.get("candidates")
        summary["fov"]["result"] = solver.get("result")
    if snaps:
        summary["snaps"] = snaps
    print(json.dumps(summary))
    if tmp is not None:
        with contextlib.suppress(OSError):
            shutil.rmtree(tmp)
    return rc


if __name__ == "__main__":
    sys.exit(main())
