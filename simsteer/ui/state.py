"""Shared UI state and the calibration boundary, independent of Tk/OpenCV."""

from __future__ import annotations

import time
from dataclasses import dataclass


def developer_mode(settings) -> bool:
    return settings.hud_mode == "dev"


def set_developer_mode(settings, enabled: bool) -> None:
    settings.hud_mode = "dev" if enabled else "user"
    settings.save()


def toggle_developer_mode(settings) -> None:
    """The H handler and the tuner switch share the persisted flag."""
    set_developer_mode(settings, not developer_mode(settings))


def restart_required(startup_game: str, settings) -> bool:
    return settings.game != startup_game


def status_label(state) -> str:
    """The toolbar and HUD name the same state, including the failing check."""
    if state.engaged:
        return "Engaged"
    if state.engaged_label.upper() == "NO GAMEPAD":
        return "No gamepad"
    if state.telemetry_status == "Unavailable":
        return "No telemetry"
    if any("frame rate too low" in warning for warning in state.warnings):
        return "Low fps"
    if state.calibration:
        return (
            "Calibrating"
            if state.calibration.tone == "calibrating"
            else state.calibration.step_name
        )
    return state.engaged_label.capitalize()


@dataclass(frozen=True)
class CalibrationView:
    """One presentation contract for both calibration implementations.

    Progress is within the named step (0..1). An unknown ETA stays None;
    the UI never invents a countdown. Score is a developer readout.
    """

    step_name: str
    instruction: str
    progress: float = 0.0
    time_left_s: float | None = None
    key_prompt: str = ""
    score: float | None = None
    steps: tuple[str, ...] = ("FOV", "Camera", "Steering", "Check")
    step_index: int = 0
    tone: str = "calibrating"

    def __post_init__(self):
        object.__setattr__(self, "progress", max(0.0, min(1.0, self.progress)))
        if self.time_left_s is not None:
            object.__setattr__(self, "time_left_s", max(0.0, self.time_left_s))


def calibration_view(
    *, wizard=None, routine=None, flow=None, v_ego=None, engaged=False
) -> CalibrationView | None:
    """Adapt sources here; renderers do not import or interrogate learners.

    ``flow`` is the public surface of the guided FOV/Camera/Steering/Check
    flow. Keeping it optional lets that flow replace the V0.12 observers
    with one call-site change, without coupling the layout to its types.
    """
    if flow is not None:
        panel = flow.panel(v_ego, engaged)
        if panel is None:
            return None
        steps = tuple(s.value.title() for s in flow.steps)
        steps = tuple("FOV" if s == "Fov" else s for s in steps)
        step = flow.step.value.title()
        step = "FOV" if step == "Fov" else step
        proposal = flow.fov_proposal if step == "FOV" else None
        prompt = ""
        instruction = flow.hint(v_ego, engaged)
        if proposal is not None:
            instruction = f"FOV {proposal:.0f} measured, set {flow.calib.fov_h_deg:.0f}"
            prompt = f"Insert  Use {proposal:.0f}    Backspace  Keep {flow.calib.fov_h_deg:.0f}"
        result = flow.result or {}
        if panel.tone == "passed":
            step, instruction, prompt = "Calibrated", "", ""
        return CalibrationView(
            step,
            instruction if panel.tone == "passed" else instruction or panel.title,
            panel.progress,
            flow.eta_s(),
            prompt,
            result.get("score"),
            steps,
            steps.index(step) if step in steps else len(steps) - 1,
            panel.tone,
        )

    if routine is not None and routine.active:
        from simsteer.app.calibration_routine import _STEPS

        index = routine.current_step
        if 0 <= index < len(_STEPS):
            step = _STEPS[index]
            elapsed = max(0.0, time.monotonic() - routine.step_started_ts)
            return CalibrationView(
                step.name.title(),
                step.prompt,
                elapsed / step.max_duration_s,
                step.max_duration_s - elapsed,
                steps=tuple(s.name.title() for s in _STEPS),
                step_index=index,
            )

    if wizard is not None and wizard.active:
        from simsteer.app.wizard import WizardPhase

        phase = wizard.phase
        if phase == WizardPhase.DONE and time.time() < wizard._ready_banner_until:
            return CalibrationView(
                "Calibrated", "", 1.0, steps=("Camera", "Steering"), step_index=1, tone="passed"
            )
        if phase != WizardPhase.DONE:
            camera = phase == WizardPhase.CAMERA
            progress = wizard._calib_frac(wizard._lc) if camera else wizard._lp_frac(wizard._lp)
            return CalibrationView(
                "Camera" if camera else "Steering",
                wizard.hint(v_ego),
                progress,
                wizard.eta_seconds(),
                "" if camera or engaged else "Insert  Engage",
                steps=("Camera", "Steering"),
                step_index=0 if camera else 1,
            )
    return None
