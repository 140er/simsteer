# SimSteer 1.0 MVP plan

SimSteer 1.0 targets Windows: download the zip, choose a game, complete
setup and a guided calibration, then drive with a gamepad or a Fanatec
wheel and pedals. Setup, calibration and the physical wheel path are
release requirements.

The approved scope is:

- Add games through TOML profiles, with telemetry for Assetto Corsa,
  Forza and Le Mans Ultimate, and a generic steer-only fallback for games
  without telemetry. Generic mode leaves throttle and brake to the player.
- Use one setup flow for game selection, device binding, models, drivers
  and game checks, with a clear fix for each failed check.
- Resume calibration per game and device, automatically correct steering
  inversion, and finish with a scored check drive.
- Read physical Fanatec inputs, pass them through vJoy, and hand control
  back safely. Preserve the game's force feedback while disengaged;
  make the rim follow the AI within torque caps while engaged, with optional
  force mixing.
- Remove the legacy `pilot/` path after parity and replacement flows are
  verified. Release 1.0 only after real game and hardware validation.

## Current evidence

Merged code and completed product validation are separate milestones.
Pending device or driving checks remain required even when code is merged.

| Area | Evidence and remaining gate |
|---|---|
| Local checks and replay | [#4](https://github.com/140er/simsteer/pull/4) and [#6](https://github.com/140er/simsteer/pull/6) merged; check scripts, deterministic replay and pinned model fetching are available. |
| Profiles | [#7](https://github.com/140er/simsteer/pull/7) merged into V0.12; adding a profile is documented in [PROFILES.md](../PROFILES.md). |
| Force feedback | Bridge code in [#5](https://github.com/140er/simsteer/pull/5) and the single-effect probe in [#8](https://github.com/140er/simsteer/pull/8) are merged; physical passthrough, feel and stop-path validation remain required. |
| Physical wheel probe | [#3](https://github.com/140er/simsteer/pull/3) is open; live wheel and pedal evidence is still needed before selecting the input backend. |
| Source launch and signing | [#9](https://github.com/140er/simsteer/pull/9) is merged into V0.12. |
| AC and Forza telemetry | [#10](https://github.com/140er/simsteer/pull/10) is open. Windows timing passes; real drives and Forza menu validation where the game is available, otherwise recorded as unavailable, independent review and final-head checks remain required. |
| Guided calibration | [#11](https://github.com/140er/simsteer/pull/11) is open and blocked from merge until calibration detects and corrects inverted steering, persists the correction per game/device, tests inverted input and supplies fresh review evidence. |

## Phases and dependencies

These are work units, not claims of completion. The evidence above records
what has landed; each remaining unit must satisfy the merge gates below.

| Phase | Depends on | Completion evidence |
|---|---|---|
| 1A: Local checks | None | Lint, regression tests and CLI smoke pass; tests keep user state outside the repository. |
| 1B: Deterministic replay | 1A | Recorded and synthetic clips replay headlessly, repeatable outputs match, and model downloads verify pinned hashes. |
| 1C: Physical wheel probe | None | Wheel and pedal readings work while a game has focus; supervised force and interruption tests establish a usable backend. |
| 2A: Data profiles | 1B | Built-in and user TOML profiles load, invalid profiles identify their errors, and existing ETS2 behavior is preserved. |
| 2B: AC and Forza telemetry | 2A | Parser tests, telemetry loss/recovery and Windows replay pass. Real drives and Forza menu checks are required where the game is available, otherwise recorded as unavailable. |
| 2C: Generic mode | 2B | A game without telemetry can engage for steering only, with model speed, conservative steering limits and no pedal output. |
| 2D: Further telemetry | 2C and confirmed games available for validation | Le Mans Ultimate and the other supported games have documented telemetry checks and real-drive evidence; unvalidated profiles stay marked untested. |
| 3A: Actionable checks | 2D | Model, driver, rendering and game checks identify fixes; repeated fixes leave the same valid state. |
| 3B: Unified setup | 3A and 3C | Game, Device, Checks, FOV and Bind steps resume after interruption and expose calibration status and recalibration. |
| 3C: Guided calibration | 1B; integrate profiles from 2A | FOV, Camera, Steering and Check stages resume per game/device; inversion is corrected and saved; a two-minute scored drive passes or identifies what to redo. |
| 3D: Legacy removal | 3B | Golden-output parity passes; `pilot/`, obsolete debug tools and launch paths are removed; source and release docs use the replacement flow. |
| 4A: Physical input | 1A and 1C hardware evidence | Wheel, pedals and engage buttons work in the background, with range, inversion, deadband and disconnect tests. |
| 4B: vJoy handover | 2C and 4A | Manual input passes through; brake and steering overrides disengage safely; throttle override and the handover ramp work in real driving. |
| 4C: Unchanged game FFB | 1A | Effect traces and supervised comparisons match native feedback while disengaged; unknown packet counts stay zero and force stops after a crash. |
| 4D: Engaged rim following | 4B, 4C and 3C | Torque-capped rim following, optional force mixing, transitions, physical override and stall shutdown pass supervised tests. |
| 5A: Release 1.0 | 3D and 4D | A fresh Windows zip completes setup, calibration and real drives with both device paths; generic mode and the published tested-game list are verified. |

Calibration can progress alongside telemetry work. The setup window
integrates its results once the checks and calibration flows are ready.

## Merge and release gates

- Require passing repository checks, independent review and applicable
  replay, live-game, device and safety evidence at the exact proposed head.
  A pending hardware check is pending evidence, never a passing result.
- Changes to setup, calibration, generic mode, handover or force feedback
  need maintainer review of screenshots and video as applicable. Fixes
  need fresh evidence for the behavior they change.
- Keep the HUD and tuner overhaul behind a separate selection gate. A
  maintainer must choose a concrete proposal before an implementation or
  build PR begins.
- Target contribution PRs at V0.12. Update pushed branches by merging
  their base; do not force-push or rebase published work.
- Release only when the packaged application passes the real-game and
  hardware checks, documentation links work, and each game is accurately
  marked tested or untested. Unit tests alone do not establish readiness.

### Performance acceptance

On Windows, **head replay non-model p95 must be <=50 ms AND model p95
must be <=50 ms**, independently, for every tested profile. Required
metrics must be present, valid and finite; model medians cannot replace
model p95.

Linux replay and model timings must still be measured and reported, but
never block timing acceptance. Parent/head and cross-profile replay changes,
parser timings, replay calibration duration, build size and live FPS are
informational. The first live calibration still targets under 10 minutes;
functional calibration lane predicates remain required. Existing startup,
calibration-quality, device, hardware, safety and force-feedback acceptance
checks also remain required, including physical versus gamepad loop cost,
bridged versus native feedback lag and force-feedback loop cost comparisons.

### Hardware safety

Any test that can move the wheel requires a person present and explicit
approval to start. Begin at 5% force, increase by at most 2.5 percentage
points with approval at each step, and never exceed 15%. Use a torque-based
rim-follow cap appropriate to the base, enforce the 15% ceiling and verify
that interruption or a stalled process stops force. New driver installation,
removal or configuration changes require explicit approval.

## Contributor entry points

Run [scripts/check.sh](../../scripts/check.sh) with
`bash scripts/check.sh`, or [scripts/check.ps1](../../scripts/check.ps1)
on Windows. These cover lint, tests and CLI smoke. Verification runs
locally rather than through GitHub Actions.

Use `python -m simsteer --list-games` to inspect profiles. Replay support
lives in [simsteer/runtime/replay.py](../../simsteer/runtime/replay.py),
with measurements in [tools/replay_bench.py](../../tools/replay_bench.py).
Evaluate both required p95 metrics against the Windows policy above.

See [PROFILES.md](../PROFILES.md),
[simsteer/runtime/telemetry/](../../simsteer/runtime/telemetry/),
[simsteer/runtime/ffb/](../../simsteer/runtime/ffb/),
[SETUP.md](../SETUP.md) and [RELEASE.md](../../RELEASE.md) for existing
contributor references. Updating setup and release instructions is part of
phases 3D and 5A.
