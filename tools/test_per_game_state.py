"""Smoke test for per-game state files (liveparams/controller/calibration)."""
from __future__ import annotations

from pathlib import Path

from pilot.calibration import Calibration, resolve_state_path
from pilot.controller import ControllerConfig
from pilot.liveparams import LiveParams


def main() -> None:
    root = Path(".").resolve()

    print("=== resolve_state_path ===")
    print(f"liveparams default: {resolve_state_path('liveparams').name}")
    print(f"liveparams ets2:    {resolve_state_path('liveparams', 'ets2').name}")
    print(f"liveparams forza:   {resolve_state_path('liveparams', 'forza').name}")
    print(f"controller forza:   {resolve_state_path('controller', 'forza').name}")
    print(f"calibration ac:     {resolve_state_path('calibration', 'ac').name}")

    print("\n=== per-game ControllerConfig save/load ===")
    ets2_cfg = ControllerConfig.load(game="ets2")
    ets2_cfg.steer_max = 0.8
    ets2_cfg.save(game="ets2")

    forza_cfg = ControllerConfig.load(game="forza")
    forza_cfg.steer_max = 0.5
    forza_cfg.save(game="forza")

    ets2_p = resolve_state_path("controller", "ets2")
    forza_p = resolve_state_path("controller", "forza")
    print(f"ets2  file:  exists={ets2_p.exists()}  name={ets2_p.name}")
    print(f"forza file:  exists={forza_p.exists()}  name={forza_p.name}")

    ets2_back = ControllerConfig.load(game="ets2")
    forza_back = ControllerConfig.load(game="forza")
    print(f"ets2  reloaded steer_max: {ets2_back.steer_max}   (expect 0.8)")
    print(f"forza reloaded steer_max: {forza_back.steer_max}   (expect 0.5)")

    print("\n=== per-game LiveParams save/load ===")
    ets2_lp = LiveParams(game="ets2")
    ets2_lp.x[0] = 0.123
    ets2_lp.save()

    forza_lp = LiveParams(game="forza")
    forza_lp.x[0] = -0.456
    forza_lp.save()

    ets2_back = LiveParams(game="ets2")
    forza_back = LiveParams(game="forza")
    print(f"ets2  reloaded scale: {ets2_back.scale:+.3f}  (expect +0.123)")
    print(f"forza reloaded scale: {forza_back.scale:+.3f}  (expect -0.456)")

    print("\n=== legacy fallback for fresh game ===")
    ac_cfg = ControllerConfig.load(game="ac")
    print(
        f"ac steer_max (no ac file yet): {ac_cfg.steer_max}  "
        f"(falls back to legacy controller.json)")

    for p in ("controller_ets2.json", "controller_forza.json",
              "liveparams_ets2.json", "liveparams_forza.json"):
        fp = root / p
        if fp.exists():
            fp.unlink()
            print(f"cleaned {p}")


if __name__ == "__main__":
    main()
