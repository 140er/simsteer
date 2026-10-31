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
| Guided calibration | [#11](https://github.com/140er/simsteer/pull/11) is merged into V0.12: calibration detects and corrects inverted steering and saves the correction per game and device. Real-game calibration drives remain pending, and the Check stage needs the certification work in [hardening](#v012-hardening) item FE-08. |
| Status bar, camera mod, live-run fixes, settings, model contract | [#13](https://github.com/140er/simsteer/pull/13), [#14](https://github.com/140er/simsteer/pull/14), [#15](https://github.com/140er/simsteer/pull/15), [#16](https://github.com/140er/simsteer/pull/16), [#17](https://github.com/140er/simsteer/pull/17) and [#19](https://github.com/140er/simsteer/pull/19) are merged into V0.12. |
| Hardening | Two independent reviews of V0.12 (31 Oct) found output-safety, model-contract and certification gaps. [V0.12 hardening](#v012-hardening) lists every item; no engaged session runs before its P0 items merge. |

## Phases and dependencies

These are work units, not claims of completion. The evidence above records
what has landed; each remaining unit must satisfy the merge gates below.
The hardening steps 0A–0M in [V0.12 hardening](#v012-hardening) come first;
where they change a phase's completion evidence, the table says so.

| Phase | Depends on | Completion evidence |
|---|---|---|
| 1A: Local checks | None | Lint, regression tests and CLI smoke pass; tests keep user state outside the repository. |
| 1B: Deterministic replay | 1A | Recorded and synthetic clips replay headlessly, repeatable outputs match, and model downloads verify pinned hashes. |
| 1C: Physical wheel probe | None | Wheel and pedal readings work while a game has focus; supervised force and interruption tests establish a usable backend. |
| 2A: Data profiles | 1B | Built-in and user TOML profiles load, invalid profiles identify their errors, and existing ETS2 behavior is preserved. |
| 2B: AC and Forza telemetry | 2A | Parser tests, telemetry loss/recovery and Windows replay pass. Real drives and Forza menu checks are required where the game is available, otherwise recorded as unavailable. |
| 2C: Generic mode | 2B | A game without telemetry can engage for steering only, with model speed, conservative steering limits and no pedal output. |
| 2D: Further telemetry | 2C and confirmed games available for validation | Le Mans Ultimate and the other supported games have documented telemetry checks and real-drive evidence; unvalidated profiles stay marked untested. |
| 3A: Actionable checks | 2D and 0G | Model, driver, rendering and game checks identify fixes; repeated fixes leave the same valid state. The driver installer is pinned and hash-checked, installs only the selected backend and fails loudly; binding checks read neutral back from the device; checks pass from both shipped entry points (FE-09). |
| 3B: Unified setup | 3A and 3C | Game, Device, Checks, FOV and Bind steps resume after interruption and expose calibration status and recalibration. |
| 3C: Guided calibration | 1B; integrate profiles from 2A; 0D, 0E and 0F | FOV, Camera, Steering and Check stages resume per game/device; inversion is corrected and saved; a two-minute scored drive passes or identifies what to redo. Check returns PASS, FAIL or INSUFFICIENT_EVIDENCE, scores bias and spread, judges a frozen fit, and FAILED removes engagement (FE-08). The FOV step holds its accuracy below 20 Hz (FE-06); Camera reports why it is blocked and ends instead of waiting forever (FE-07). Real-game lanes run in the supervised session (FE-15). |
| 3D: Legacy removal | 3B | Golden-output parity passes; `pilot/`, obsolete debug tools and launch paths are removed; source and release docs use the replacement flow. Legacy launchers go earlier (FE-22) and tracked calibration files earlier still (FE-10). |
| 4A: Physical input | 1A and 1C hardware evidence | Wheel, pedals and engage buttons work in the background, with range, inversion, deadband and disconnect tests. |
| 4B: vJoy handover | 2C and 4A; 0I and 0J | Manual input passes through; brake and steering overrides disengage safely; throttle override and the handover ramp work in real driving. One output owner writes every device and expires stale commands (FE-12); a device swap keeps the old device on failure (FE-11); device-output bench lanes cover takeovers, loss and stalls (FE-13). |
| 4C: Unchanged game FFB | 1A | Effect traces and supervised comparisons match native feedback while disengaged; unknown packet counts stay zero and force stops after a crash. Before integration: a queue staleness policy, a `stop()` that completes, and binding to the device identity (FE-26). |
| 4D: Engaged rim following | 4B, 4C and 3C | Torque-capped rim following, optional force mixing, transitions, physical override and stall shutdown pass supervised tests. |
| 5A: Release 1.0 | 3D and 4D | A fresh Windows zip completes setup, calibration and real drives with both device paths; generic mode and the published tested-game list are verified. Every hardening item is merged or listed as a known limit, and the supervised session (FE-15) has passed on the release output code. |

Calibration can progress alongside telemetry work. The setup window
integrates its results once the checks and calibration flows are ready.

## V0.12 hardening

Two independent reviews of V0.12 (31 October 2026) agree on the core
problems: output authority that is not verified, a model adapter that
matches the pinned openpilot release only in shape, calibration that can
certify without evidence, and two parallel implementations. Every P0 and P1
finding and the process advice is an item below, with its acceptance tests.

Rules:

- Each item lands as a PR into V0.12, never `main`. Its gate is an
  independent review that passes on the current head plus passing
  repository checks at that head.
- P0 items (FE-01 to FE-05) merge before any engaged session, gamepad or
  wheel, supervised or not.
- An item is done when its acceptance tests exist at a merged V0.12
  revision.
- "Device-output level" means the test checks what reaches the device: the
  fake gamepad report, the simulated vJoy axes or the pedal values. An
  intermediate controller value is not enough.
- New steps are named 0A–0M. They harden the V0.12 base and come before
  further engaged work. Phases 1A–5A stay as they are.

Order:

1. Now: 0B ([#19](https://github.com/140er/simsteer/pull/19)) has merged;
   0A output safety and 0C simple warp are in progress; 0F
   ([#20](https://github.com/140er/simsteer/pull/20)) is in review;
   [#18](https://github.com/140er/simsteer/pull/18) is in a fix round. #18 and 0A share the loop and output layer, so whichever merges
   second takes V0.12 in and keeps both test sets passing.
2. After #18 and 0A: 0J (single output owner), then 0I (session ownership).
3. After 0B and 0C: 0D (time base), then 0E (Camera). 0G
   (setup contract) and 0H (state) are independent.
4. 0K bench lanes grow with each output item. 0L, the maintainer's
   disengaged clip corpus, can happen any time. 0M, the supervised engaged
   session, comes last.

### Items

| ID | Item | Priority | Found by | Step / PR | Depends on | Status (31 Oct) |
|---|---|---|---|---|---|---|
| FE-01 | Finite guards from decode to device | P0 | both reviews | 0A | none | in progress |
| FE-02 | Stop path that needs no successful frame, neutral on exit, no `os._exit` | P0 | both | 0A | none | in progress |
| FE-03 | `T` test key off in player mode; bind wiggle through the loop's output path | P0 | both | 0A | none | in progress |
| FE-04 | Model contract at the pinned release | P0 | both | 0B, [#19](https://github.com/140er/simsteer/pull/19) | none | merged; recorded-frames report waits on FE-14 |
| FE-05 | Simple-warp pitch geometry | P0 | one | 0C | none | in progress |
| FE-06 | 20 Hz assumption and the FOV solver | P1 | both | 0D | FE-04, FE-05 | not started |
| FE-07 | Camera bootstrap, blocked reason, no-progress outcome | P1 | both | 0E | FE-04, FE-05, FE-06 | not started |
| FE-08 | Certification without evidence; FAILED policy | P1 | both | 0F, [#20](https://github.com/140er/simsteer/pull/20) | none | in review |
| FE-09 | Setup as an executable contract | P1 | mainly one | 0G, feeds 3A | none | not started |
| FE-10 | State: context binding, atomic saves, per-user data, untracked calibration files | P1 | both | 0H | FE-01 | not started |
| FE-11 | Session ownership and device-swap transaction | P1 | both | 0I | #18, FE-12 | not started |
| FE-12 | Single output owner with command expiry for every device | P1 | both | 0J | #18, FE-01–03 | not started |
| FE-13 | Device-output bench lanes | process | one | 0K | [#21](https://github.com/140er/simsteer/pull/21) (simulated base), tests from FE-01–03, FE-11, FE-12 | not started; #21 in review |
| FE-14 | Disengaged clip corpus recorded by the maintainer | process | both | 0L, maintainer-scheduled | a merged recording build | waiting on the maintainer |
| FE-15 | One supervised engaged session with a checklist | process | one | 0M, maintainer-scheduled | FE-01–05, FE-08, FE-13, FE-14 | waiting on dependencies |
| FE-16 | Unattended GUI automation is not engaged evidence | process | one | rule; scopes [#22](https://github.com/140er/simsteer/pull/22) | none | in effect |
| FE-17 | One merged revision per live run | process | one | rule | none | in effect |
| FE-18 | Backlog for findings outside a PR's diff | process | one | rule | none | not started |
| FE-19 | Safety checklist for output PRs | process | one | rule | none | in effect for output PRs |
| FE-20 | One-time audit of the inherited core against upstream | process | one | review task | FE-04 | not started |
| FE-21–27 | P2 list below | P2 | see list | see list | see list | not started |

### P0: before any engaged session

**FE-01. Finite guards (0A).**

- A NaN or infinity injected at each stage (decoded model output, each
  telemetry field, controller output, a learner sample, a device setter)
  gives, on that tick or the next, a neutral gamepad report or centred vJoy
  steering with pedals released, disengages, and shows one banner naming
  the source. One device-output test per stage, for both device paths.
- One NaN frame followed by 100 clean frames gives the clean-frame steering
  again; the rate limiter no longer keeps full lock.
- A NaN wheel sample leaves the steering fit and its covariance unchanged
  and finite.
- Non-finite state is never saved, and a saved file holding NaN or Infinity
  is rejected on load without a crash.
- No device wrapper turns NaN into full output by clamping.

**FE-02. Minimum stop path (0A).**

- A capture call that never returns still lets the loop handle the engage
  key and disengage within 200 ms, with neutral output within 200 ms.
- Frames that do not change for 250 ms (video capture modes repeat frames)
  count as stale: disengage and neutral.
- Telemetry whose timestamp stops for 250 ms, or that reports pause, gives
  disengage and neutral. The SCS reader checks the plugin revision at start.
- The engage key works when no frame arrived.
- Normal exit, Ctrl+C, console close, logoff and shutdown write neutral
  before the process ends.
- Save & Restart shuts down in order and neutralises first; no `os._exit`
  remains in `simsteer/`.
- A hard kill runs no handler, so vJoy keeps its last value. This stays a
  recorded risk, checked at standstill in FE-15, and it blocks a release.
  It clears only with a mitigation proven by a measured kill test on real
  hardware, or with an explicit maintainer decision to ship it as a known
  limit.

**FE-03. Test-fire key (0A).**

- With developer mode off, `T` writes nothing to any device.
- With developer mode on, it acts only while disengaged and stopped, never
  blocks the loop, and writes through the loop's existing output writer,
  never from the UI thread.
- Bind-wiggle helpers write through that same writer, stop when engagement
  starts, and periodic centring no longer cuts them off. The single output
  owner arrives in 0J (FE-12), which checks these paths again.

**FE-04. Model contract (0B, [#19](https://github.com/140er/simsteer/pull/19), merged).**

- Traffic convention one-hot at `int(is_rhd)`, with the default meaning
  right-hand traffic.
- Desire is a rising-edge pulse; a history test shows one pulse per
  lane-change intent.
- Lane-line probabilities, lead layout and exponentiated standard
  deviations match upstream.
- Golden tests decode a fixture with distinct values in every tensor region
  and compare against a reference parser transcribed from upstream at the
  pinned revision, with the source cited.
- An offline real-model before/after report on recorded frames. Still
  open: #19's report used a synthetic replay, so this waits on FE-14's
  clips.

**FE-05. Simple-warp geometry (0C).**

- Focal length from the original capture width, and the principal point
  shifted for asymmetric crops.
- At 2560 px, 120° and 2° pitch, the horizon in both the narrow and wide
  views lands within 0.5 model px of the perspective warp (today 26.6 and
  10.7 px off). Parametrised over pitch, FOV and aspect ratio.
- Saved simple-warp pitch is invalidated by a schema change, so Camera
  measures again.
- Offline real-model before/after report.

### P1

**FE-06. Time base and FOV solver (0D).**

- Feature and desire queues use frame timestamps, or the loop paces at
  20 Hz and blocks engagement while the rate over 2 s stays below 18 Hz,
  with the reason shown.
- Replaying one clip at 10, 15, 18 and 20 Hz keeps model speed within ±5%
  of the 20 Hz run, or blocks engagement with the rate reason.
- The FOV solver normalises by the measured time span: a true 90° solves
  within ±1.5° at 15–20 Hz.
- Buffers reset after frame gaps, pauses and geometry or FOV changes.
- Live runs report the frame period at native resolution with the HUD
  rendering (informational).

**FE-07. Camera (0E).**

- Bootstrap matches upstream at the pin: certainty gates are skipped until
  upstream's block count, with the source cited.
- A structured blocked reason shows in the status bar and tuner, and
  counts per reason are saved.
- Three minutes of eligible driving without progress ends in NO_PROGRESS
  with the top reasons and what to change.

**FE-08. Certification (0F, [#20](https://github.com/140er/simsteer/pull/20)).**

- Each capability declares required measurements; a missing one gives
  INSUFFICIENT_EVIDENCE, never PASS.
- Mean lateral bias and spread are both scored; a constant 1 m offset
  fails. Out-of-lane samples count as excursions instead of being dropped.
- The Check judges a frozen fit (or a snapshot), not one that keeps
  adapting during the Check.
- FAILED and every invalidation remove engagement; a device-output test
  shows zero AI output in FAILED. The maintainer confirms this policy
  before merge.
- A saved PASS is invalidated when FOV, camera pose, device or binding,
  control settings or the schema change.
- A trusted fit needs left/right and speed coverage, not only a sample
  count.

**FE-09. Setup contract (0G, feeds 3A).**

- Checks detect the game's relative or absolute steering setting and its
  deadzone, and name the fix.
- Binding checks read neutral back from the device after centring.
- The checks pass from `python -m simsteer --check` and from the
  packaged executable.
- The driver installer pins versions and hashes, installs only the selected
  backend, and fails loudly. It pins one vJoy build by version and hash,
  and that build passes the vJoy force-feedback lanes on current Windows 11.
  README and SETUP point to the same build.

**FE-10. State (0H).**

- All state writes are atomic; a kill during a write leaves the previous
  file loadable.
- A truncated or corrupt controller or calibration file starts with
  defaults and a banner instead of a crash.
- Calibration carries a context fingerprint (game, device and binding,
  vehicle where telemetry gives it, model hash, capture size, FOV, warp
  mode, schema). A mismatch blocks engagement until recalibration.
- Source runs keep state in the per-user data directory; a run from a fresh
  clone leaves `git status` clean.
- `calibration_*.json` and `controller_*.json` leave the repository;
  examples ship as `*.example.json` if needed.

**FE-11. Session ownership (0I).**

- The UI sends commands through a queue that the loop applies between
  ticks; no UI-thread write reaches the controller, learners or devices.
- A device swap disengages, goes neutral, opens the new device and only
  then closes the old one; on failure the old device stays.
- A swap never re-engages, and no frame during a swap emits non-neutral
  output.

**FE-12. Single output owner (0J, after #18).**

- One output owner writes every device kind, generalising #18's relay.
- AI commands carry a timestamp; one older than 150 ms gives neutral
  output on every device kind. A device-output test stalls the loop for
  300 ms.
- Exit and crash paths request neutral through the owner.

### Process

- **FE-13. Bench lanes (0K).** A separate open step: #21 provides the
  simulated Fanatec base, and the lanes come with the tests of FE-01–03,
  FE-11 and FE-12. One test marker inside the repository check
  runs the device-output scenarios on the fake gamepad and the simulated
  Fanatec base: NaN per stage, capture stall, stale frames, stale or paused
  telemetry, the engage key without frames, exit events, the test key,
  a failed device swap, stall expiry, brake and rim takeovers, and a pedal
  map that does not match the device. It runs on Linux and Windows and
  passes at the revision the supervised session uses.
- **FE-14. Clip corpus (0L, maintainer-scheduled).** The maintainer drives
  by hand, disengaged, with recording on: at least 15 minutes of ETS2
  highway and 10 minutes of Assetto Corsa with curves, lane changes and
  traffic, plus telemetry. The clips stay private and back offline
  real-model regression for FE-04 to FE-08. Claims about real frames cite
  these clips, not synthetic ones.
- **FE-15. Supervised engaged session (0M, maintainer-scheduled).** It
  starts after FE-01–05 and FE-08 have merged, FE-13 passes and FE-14 is
  recorded; for the wheel path also after #18. One merged revision for the
  whole session. A written checklist covers preflight, standstill stop
  checks (engage key, brake, test key, hard kill with pedals at rest),
  guided calibration to Check, engaged highway driving with takeovers, and
  every exit path. Force feedback stays off unless a separately approved
  lane runs under the hardware safety rules. Each line ends PASS, FAIL or
  NOT RUN with a log or video; any FAIL ends engaged driving for that
  session.
- **FE-16. No unattended engaged evidence.** Unattended scripted runs of
  commercial games are diagnostics, setup and restore tools only. They never
  check an engaged-driving lane. [#22](https://github.com/140er/simsteer/pull/22)
  is reviewed on that basis and claims no engaged evidence.
- **FE-17. One merged revision per live run.** Live runs use one merged
  V0.12 revision or one PR head on V0.12, never an ad-hoc combination of
  open PRs, and name it in the log.
- **FE-18. Pre-existing findings.** A review finding outside the PR's diff
  becomes a backlog entry with severity and location. Open P0 entries block
  FE-15.
- **FE-19. Output PR checklist.** PRs that touch output, input, the loop,
  force feedback or the controller answer: finite inputs, a stop path
  without frames, a single writer, command expiry, neutral on every exit,
  nothing blocking the loop, and no test path that writes around the owner.
- **FE-20. Core audit.** After FE-04, one audit of the inherited core
  (decode, curvature law, calibration, warp, learners) against the pinned
  upstream revision. Findings go to FE-18.

### P2

- **FE-21.** Reproducible environment: a lock file, declared optional
  dependencies, a pinned SCS plugin recipe and DirectML session options.
- **FE-22.** Legacy and docs: remove the `pilot` launchers and doc
  references now, delete the tree in 3D, bring README, SETUP, RELEASE,
  ARCHITECTURE and this plan's status up to V0.12, and add a short
  contributor contract with the FE-19 checklist.
- **FE-23.** Steering-fit identifiability: wheel and wheel·v² are collinear
  at constant speed; decide whether intervention learning is a takeover.
- **FE-24.** Curvature law: adopt the pinned upstream law or drop the
  "direct port" claim and document the delay settings.
- **FE-25.** Camera resolution guidance: HFOV ≤ 2·atan(W/1820), and pad
  narrow sources instead of stretching them.
- **FE-26.** Force feedback before integration: queue staleness, a `stop()`
  that completes, and binding to the device identity.
- **FE-27.** Small items: `lookahead_s` defaults that disagree, global keys
  reading only the high bit, and AEB without hysteresis.

### PR gates

| PR | Items | Before merge into V0.12 |
|---|---|---|
| [#18](https://github.com/140er/simsteer/pull/18) | then FE-12, FE-11 | Its review's two high findings, the drift-absorption finding and the reset-to-midpoint finding fixed; passing review and checks at the head; FE-19 checklist. Hardware lanes stay supervised. |
| [#19](https://github.com/140er/simsteer/pull/19) | FE-04 | Merged into V0.12. The recorded-frames report stays open under FE-04 until FE-14. |
| [#20](https://github.com/140er/simsteer/pull/20) | FE-08 | FE-08 acceptance, with the frozen fit or a named follow-up; maintainer confirms the FAILED policy; passing review and checks. |
| [#21](https://github.com/140er/simsteer/pull/21) | FE-13 | Passing review and checks. |
| [#22](https://github.com/140er/simsteer/pull/22) | FE-16 | Passing review and checks, plus one live Windows run of start, backup, restore and virtual display. No engaged-evidence claims. |
| [#10](https://github.com/140er/simsteer/pull/10) | P0 base | Existing gates, plus signed AC speed, documented yaw sign and the `forza_port` test conflict fixed; merges after the P0 items. |
| [#3](https://github.com/140er/simsteer/pull/3) | none | Unchanged; kept as a tool. |

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
- Hardening work targets V0.12; nothing merges into `main` before the
  release. Each hardening PR needs an independent review that passes on its
  current head and passing repository checks at that head.
- No engaged session runs before the P0 hardening items have merged.
  Unattended scripted game runs are diagnostics only and never count as
  engaged-driving evidence. Every live run uses one merged revision and
  names it.

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
Live runs also report the frame period at native capture resolution with
the HUD rendering (FE-06); it is informational and does not change these
gates.

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
