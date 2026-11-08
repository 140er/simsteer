"""Manual lane-change lifetime and re-arming for modeld desire pulses."""

from __future__ import annotations

from collections.abc import Sequence


class ManualLaneChange:
    def __init__(self) -> None:
        self.index: int | None = None
        self._until = 0.0
        self._committed = False
        self._last_desire: int | None = None
        self._release_pending = False

    @property
    def active(self) -> bool:
        return self.index is not None

    def request(self, index: int, now: float, hold_s: float) -> None:
        if index not in (3, 4):
            raise ValueError("manual lane changes require desire 3 or 4")
        self.index = index
        self._until = now + hold_s
        self._committed = False
        self._release_pending = index == self._last_desire

    def reset(self) -> None:
        self.index = None
        self._until = 0.0
        self._committed = False
        self._release_pending = False
        # Keep the last model input: cancellation alone is not a zero step.

    def expire(self, now: float) -> None:
        if self.active and now >= self._until:
            self.reset()

    def model_desire(self, nav_index: int | None) -> int | None:
        """Select once per model step, including any neutral re-arm step."""
        index = self.index if self.active else nav_index
        if self._release_pending:
            index = None
            self._release_pending = False
        self._last_desire = index
        return index

    def observe(self, desire_state: Sequence[float], threshold: float) -> bool:
        """Cut a committed manual request when the model finishes it."""
        if not self.active or self._last_desire != self.index:
            return False
        probability = float(desire_state[3]) + float(desire_state[4])
        if probability > threshold:
            self._committed = True
        elif self._committed and probability < 0.1:
            self.reset()
            return True
        return False
