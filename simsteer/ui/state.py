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


def restart_required(startup_game: str, selected_game: str) -> bool:
    return selected_game != startup_game


def status_label(state) -> str:
    """The toolbar and HUD name the same state, including the failing check."""
    if state.engaged:
        return state.driver or "Engaged"
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
    return state.driver or state.engaged_label.capitalize()


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
    steps: tuple[str, ...] = ("FOV", "Camera", "Steering", "Direction", "Check", "Done")
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

    ``flow`` owns calibration and output authority. The direction check is
    a substate of Steering in that engine and a separate visible step here.
    Legacy observers remain supported when their source is available.
    """
    if flow is not None:
        panel = flow.panel(v_ego, engaged)
        if panel is None:
            return None
        steps = ["FOV" if s.value == "fov" else s.value.title() for s in flow.steps]
        if "Steering" in steps:
            steps.insert(steps.index("Steering") + 1, "Direction")
        steps.append("Done")
        step = "FOV" if flow.step.value == "fov" else flow.step.value.title()
        proposal = flow.fov_proposal if step == "FOV" else None
        prompt = ""
        instruction = flow.hint(v_ego, engaged)
        progress, eta, tone = panel.progress, flow.eta_s(), panel.tone
        if proposal is not None:
            instruction = f"FOV {proposal:.0f} measured, set {flow.calib.fov_h_deg:.0f}"
            prompt = f"Insert  Use {proposal:.0f}    Backspace  Keep {flow.calib.fov_h_deg:.0f}"
        if (step == "Steering" and flow.direction.ready
                and not flow.output_direction.ready):
            step = "Direction"
            output = flow.output_direction
            progress, eta = output.progress, None
            if output.active:
                eta = max(0.0, 4 * output.PULSE_S - output.elapsed)
                prompt = "Insert  Cancel"
            elif v_ego is not None and abs(v_ego) < output.STOP_SPEED_MPS:
                prompt = "Insert  Retry" if output.reason else "Insert  Start"
                if instruction.startswith("Press INSERT"):
                    instruction = "Check steering direction"
            if output.reason.startswith("No steering response"):
                tone = "failed"
        elif step in ("Steering", "Check") and not engaged:
            if instruction == "Press INSERT and let SimSteer drive":
                instruction, prompt = "Let SimSteer drive", "Insert  Engage"
            elif instruction == "Press INSERT for the check drive":
                instruction, prompt = "Check drive", "Insert  Engage"
        result = flow.result or {}
        index = steps.index(step) if step in steps else steps.index("Check")
        if tone == "passed":
            step, instruction, prompt, index = "Calibrated", "", "", steps.index("Done")
        elif flow.step.value == "failed":
            step, prompt = "Not calibrated", "R  Recalibrate"
            instruction = result.get("reason", instruction).removesuffix(". R restarts")
        return CalibrationView(
            step,
            instruction,
            progress,
            eta,
            prompt,
            result.get("score"),
            tuple(steps),
            index,
            tone,
        )

    if routine is not None and routine.active:
        steps = getattr(routine, "steps", None)
        if steps is None:
            from simsteer.app.calibration_routine import _STEPS

            steps = _STEPS

        index = routine.current_step
        if 0 <= index < len(steps):
            step = steps[index]
            elapsed = max(0.0, time.monotonic() - routine.step_started_ts)
            return CalibrationView(
                step.name.title(),
                step.prompt,
                elapsed / step.max_duration_s,
                step.max_duration_s - elapsed,
                steps=tuple(s.name.title() for s in steps),
                step_index=index,
            )

    if wizard is not None and wizard.active:
        phase = wizard.phase.name
        if phase == "DONE" and time.time() < wizard._ready_banner_until:
            return CalibrationView(
                "Calibrated", "", 1.0, steps=("Camera", "Steering"), step_index=1, tone="passed"
            )
        if phase != "DONE":
            camera = phase == "CAMERA"
            progress = wizard._calib_frac(wizard._lc) if camera else wizard._lp_frac(wizard._lp)
            return CalibrationView(
                "Camera" if camera else "Steering",
                wizard.hint(v_ego).partition(" — ")[0],
                progress,
                wizard.eta_seconds(),
                "" if camera or engaged else "Insert  Engage",
                steps=("Camera", "Steering"),
                step_index=0 if camera else 1,
            )
    return None
