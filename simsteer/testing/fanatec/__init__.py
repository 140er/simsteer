"""A simulated Fanatec wheel base for tests without hardware.

SimSteer meets the wheel only through DirectInput (`dinput_out`) and the
vJoy DLL (`vjoy_ffb`). `FanatecSim` installs stand-ins for both, so the
real code runs unchanged: device detection by VID/PID and name, the DD2's
two same-name DirectInput instances, data formats, acquire and loss,
the force feedback sink and bridge. Tests drive the rim, pedals and
buttons with `Track`s and check what the base received with `FfbLog`.

Only tests and `python -m simsteer.testing.fanatec` install it; the app
never does.
"""

from simsteer.testing.fanatec.device import SimDevice, SimEffect
from simsteer.testing.fanatec.ffb import FfbLog, level_of
from simsteer.testing.fanatec.profiles import DD2_A, DD2_B, PROFILES, VJOY
from simsteer.testing.fanatec.sim import FanatecSim
from simsteer.testing.fanatec.track import Track
from simsteer.testing.fanatec.vjoy import SimGame, SimVJoyDll, load_packets

__all__ = ["DD2_A", "DD2_B", "FanatecSim", "FfbLog", "PROFILES", "SimDevice", "SimEffect", "SimGame",
           "SimVJoyDll", "Track", "VJOY", "level_of", "load_packets"]
