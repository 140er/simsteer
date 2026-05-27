"""Verify the SCS telemetry plugin is working.

Run this with ETS2 loaded into a save (truck in world). It opens the
shared memory and prints speed, RPM, and steering input every 200 ms
until you Ctrl+C. If you see speed change as you drive, telemetry is
wired up correctly.

    python -m tools.test_telemetry
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pilot.telemetry import (
    OFF_GAME, OFF_PLUGIN_REV, OFF_TIME, OFF_TRUCK_ENGINE_RPM,
    Telemetry,
)


def main() -> int:
    tel = Telemetry()
    if not tel.available:
        print("Telemetry NOT available. Check that:")
        print("  1. ETS2 is running and you're loaded into a game (truck in world)")
        print("  2. scs-telemetry.dll is in <ETS2>/bin/win_x64/plugins/")
        print("  3. game.log.txt mentions scs-telemetry loading")
        return 1

    # Header info — only need to read once.
    import struct
    mm = tel._mm
    plugin_rev = struct.unpack_from("<I", mm, OFF_PLUGIN_REV)[0]
    game = struct.unpack_from("<I", mm, OFF_GAME)[0]
    game_name = {0: "unknown", 1: "ETS2", 2: "ATS"}.get(game, f"game={game}")
    print(f"plugin revision: {plugin_rev}    game: {game_name}")
    print()
    print(f"{'time':>10} {'speed m/s':>10} {'km/h':>7} {'rpm':>7} "
          f"{'gameSteer':>11} {'userSteer':>11}")

    try:
        while True:
            t = struct.unpack_from("<Q", mm, OFF_TIME)[0]
            speed = tel.speed_mps()
            rpm = struct.unpack_from("<f", mm, OFF_TRUCK_ENGINE_RPM)[0]
            gs = tel.game_steer()
            us = tel.user_steer()
            kph = (speed or 0.0) * 3.6
            print(f"{t:>10} {speed:>10.2f} {kph:>7.1f} {rpm:>7.0f} "
                  f"{(gs or 0):>+11.3f} {(us or 0):>+11.3f}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        tel.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
