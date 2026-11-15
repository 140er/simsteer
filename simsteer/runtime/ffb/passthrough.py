"""Force feedback pass-through inside SimSteer, for wheel output.

SimSteer owns the vJoy device the game steers with, so a separate bridge
process cannot receive the game's force feedback. This runs the same
`FfbBridge` in-process: the game's effects reach the real wheel only
while SimSteer is engaged and fault-free, with the device gain capped
(`--ffb-cap`, 1 to 15 percent). Any disarm (disengage, fault, exit)
closes the wheel on the bridge thread, which stops and unloads every
effect within one bridge step.

A failure to start never stops driving: it is reported once and SimSteer
continues without force feedback.
"""

from __future__ import annotations

import sys
from typing import Callable

from simsteer.runtime.ffb.bridge import FfbBridge, validate_cap


class FfbPassthrough:
    def __init__(self, cap_percent: float, vjoy_device: int = 1,
                 say: Callable[[str], None] = print,
                 make_bridge: Callable[[], tuple[FfbBridge, object]] | None = None) -> None:
        self.cap = validate_cap(cap_percent)
        self.vjoy_device = vjoy_device
        self.say = say
        self._make_bridge = make_bridge or self._windows_bridge
        self.bridge: FfbBridge | None = None
        self._receiver = None
        self._last_status = ""

    def _windows_bridge(self) -> tuple[FfbBridge, object]:
        if sys.platform != "win32":
            raise OSError("force feedback pass-through needs Windows")
        from simsteer.runtime.ffb import dinput_out as di
        from simsteer.runtime.ffb import vjoy_ffb

        rx = vjoy_ffb.VJoyFfbReceiver(self.vjoy_device)
        check = rx.open()
        if not check.ok:
            rx.close()
            raise OSError(f"vJoy FFB: {check.row}" + (f". Fix: {check.fix}" if check.fix else ""))
        try:
            bridge = FfbBridge(rx, di.DirectInputSink(), self.cap, attended=True, armed=False,
                               list_devices=di.attached_ff_devices, say=self.say)
        except Exception:
            rx.close()
            raise
        return bridge, rx

    def start(self) -> bool:
        try:
            self.bridge, self._receiver = self._make_bridge()
            self.bridge.start()
        except Exception as e:  # never stop driving over force feedback
            self.say(f"FFB disabled: {e}")
            self.bridge = None
            self._close_receiver()
            return False
        if not self.bridge.running:
            self.say(f"FFB disabled: {self.bridge.status}")
            self.close()
            return False
        self.say(f"FFB pass-through ready, cap {self.cap:g}% of the wheel's peak torque, "
                 "active only while engaged")
        return True

    def update(self, active: bool) -> None:
        """Call every loop tick and from every stop path."""
        if self.bridge is None:
            return
        self.bridge.arm(active and self.bridge.running)
        status = self.bridge.status
        if status != self._last_status:
            self._last_status = status
            self.say(status)

    @property
    def status(self) -> str:
        return "" if self.bridge is None else self.bridge.status

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.arm(False)
            self.bridge.stop()
            if self.bridge.running:
                # Its thread still uses the receiver; the wheel is already
                # disarmed and the process exit ends both.
                self.say("FFB bridge did not stop in time; vJoy receiver left open")
                self.bridge = None
                self._receiver = None
                return
            self.bridge = None
        self._close_receiver()

    def _close_receiver(self) -> None:
        if self._receiver is not None:
            try:
                self._receiver.close()
            finally:
                self._receiver = None
