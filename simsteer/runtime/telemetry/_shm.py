"""A named shared-memory page a game publishes, opened when the game is up.

The adapter keeps one `SharedPage` for the session. Until the game has
published live data, `get()` retries the open at most every `retry_s`,
so the loop can poll `available` every frame without recreating readers.
"""

from __future__ import annotations

import mmap
import struct
import sys
import time
from collections.abc import Callable

Buffer = mmap.mmap | bytearray | memoryview
RETRY_S = 1.0


def open_windows_page(name: str, size: int) -> Buffer | None:
    if sys.platform != "win32":  # named mappings are a Windows API
        return None
    try:
        return mmap.mmap(-1, size, name, access=mmap.ACCESS_READ)
    except OSError:
        return None


class SharedPage:
    def __init__(self, name: str, size: int, is_live: Callable[[Buffer], bool], *,
                 opener: Callable[[str, int], Buffer | None] = open_windows_page,
                 retry_s: float = RETRY_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.name = name
        self.size = size
        self._is_live = is_live
        self._opener = opener
        self._retry_s = retry_s
        self._clock = clock
        self._buf: Buffer | None = None
        self._next_try = 0.0
        self.get()

    def get(self) -> Buffer | None:
        """The page while the game reports live data, else None."""
        if self._buf is None:
            now = self._clock()
            if now < self._next_try:
                return None
            self._next_try = now + self._retry_s
            self._buf = self._opener(self.name, self.size)
            if self._buf is None:
                return None
        try:
            live = self._is_live(self._buf)
        except (ValueError, IndexError, struct.error):
            live = False
        if not live:
            # Opening a name nobody publishes creates an empty page; never
            # hold one, so the game creates the page itself on launch.
            self.close()
        return self._buf

    def close(self) -> None:
        if self._buf is not None and hasattr(self._buf, "close"):
            self._buf.close()
        self._buf = None
