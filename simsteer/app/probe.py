"""Active system-identification for the steering rack.

Problem: passive LiveParams fitting only learns from whatever steering
happens naturally. On a straight highway with the AI just keeping lane,
the commanded axis is near zero, `rej_small_axis` fires constantly, and
the fit barely moves — minutes of driving with no progress.

Fix: during the wizard's steering-fit phase (Phase B), superimpose a
small sinusoidal perturbation on top of the AI's command. This excites
the rack across its operating range so every frame contributes a useful
(axis, wheel_angle) pair. Once `LiveParams.trusted()` flips True, the
probe stops automatically.

Trade-off: while probing, the truck weaves by a small (~1 cm at 20 m/s
peak) lateral amount. The perturbation is symmetric (sine) so the net
drift over a cycle is ~0. The user sees "the truck is wiggling slightly
while it's testing my steering" — communicated via the wizard banner.

Safety gates:
- Disabled when not engaged, when LiveParams is already trusted, during
  lane changes, at low/high speed, or when the AI is already commanding
  a non-trivial steer. These prevent the probe from compounding with
  legitimate steering input.
- Amplitude scaled down at higher speeds where small wheel changes
  translate to large lateral acceleration.
"""

from __future__ import annotations

import math


class SteeringProbe:
    """Sinusoidal axis perturbation. Stateful: holds the phase, last
    output, and reason the last call decided to gate (for HUD)."""

    # Peak perturbation in axis units (±). 0.03 is a small enough signal
    # that the truck doesn't visibly swerve, but large enough to break
    # past the `min_axis` rejection gate in LiveParams.
    BASE_AMPLITUDE_AXIS = 0.03
    PERIOD_S = 3.0                  # full sine cycle
    # Speed window — outside this, probing isn't useful or isn't safe.
    MIN_SPEED_MPS = 6.0             # below: probes feel like jerky inputs
    MAX_SPEED_MPS = 30.0            # above: small wheel = large lat accel
    SPEED_SCALE_REF_MPS = 12.0      # amplitude scales as ref/v above this
    # If AI is already commanding a non-trivial steer, don't add noise.
    MAX_AI_AXIS_FOR_PROBE = 0.10

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._t = 0.0
        self.last_offset = 0.0
        self.last_gate = "init"
        self.cycles_done = 0
        # Total accepted "active" frames — for the HUD ETA / trust check.
        self.active_frames = 0

    def tick(self, dt: float,
             ai_axis: float,
             v_ego: float | None,
             in_lane_change: bool,
             engaged: bool,
             lp_trusted: bool,
             wizard_steering_phase: bool) -> float:
        """Compute this frame's probe offset (axis units). Returns 0
        when any safety gate or relevance gate is active.

        Caller adds the offset to `ai_steer` to form the final commanded
        axis, and passes that summed value to LiveParams as
        `commanded_axis` so the intervention detector sees no mismatch.
        """
        if not self.enabled:
            return self._gate("disabled")
        if not engaged:
            return self._gate("disengaged")
        if lp_trusted:
            return self._gate("lp_trusted")
        if not wizard_steering_phase:
            # Outside of the wizard's calibration window, leave the
            # truck alone. (Users can extend probing with a future
            # always-probe flag if they want.)
            return self._gate("not_wizard_phase_b")
        if in_lane_change:
            return self._gate("lane_change")
        if v_ego is None:
            return self._gate("no_speed")
        if v_ego < self.MIN_SPEED_MPS:
            return self._gate(f"slow v={v_ego:.1f}")
        if v_ego > self.MAX_SPEED_MPS:
            return self._gate(f"fast v={v_ego:.1f}")
        if abs(ai_axis) > self.MAX_AI_AXIS_FOR_PROBE:
            return self._gate(f"ai_busy ax={ai_axis:+.2f}")

        # All gates passed — advance phase and emit the sine value.
        self._t += dt
        # Track full cycles for the HUD.
        if self._t >= self.PERIOD_S:
            self.cycles_done += int(self._t // self.PERIOD_S)
            self._t = self._t % self.PERIOD_S
        amplitude = self._amplitude_for_speed(v_ego)
        self.last_offset = amplitude * math.sin(
            2.0 * math.pi * self._t / self.PERIOD_S)
        self.last_gate = "active"
        self.active_frames += 1
        return self.last_offset

    def reset(self) -> None:
        """Called by the R hotkey alongside the calibrator resets."""
        self._t = 0.0
        self.last_offset = 0.0
        self.last_gate = "reset"
        self.cycles_done = 0
        self.active_frames = 0

    # ----- helpers -----

    def _gate(self, reason: str) -> float:
        self.last_offset = 0.0
        self.last_gate = reason
        return 0.0

    def _amplitude_for_speed(self, v_ego: float) -> float:
        """Scale amplitude down above SPEED_SCALE_REF_MPS so lateral
        acceleration stays bounded across the speed window."""
        if v_ego <= self.SPEED_SCALE_REF_MPS:
            return self.BASE_AMPLITUDE_AXIS
        return self.BASE_AMPLITUDE_AXIS * (self.SPEED_SCALE_REF_MPS / v_ego)
