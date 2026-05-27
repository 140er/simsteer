# SimSteer

Drive Euro Truck Simulator 2, Forza Horizon 5, and Assetto Corsa with
comma.ai's openpilot vision + planning model. Captures the game window,
runs the model, and steers via a virtual gamepad (ViGEm) or virtual
wheel (vJoy).

## Documentation

| Doc | Read it for |
|---|---|
| **[docs/SETUP.md](docs/SETUP.md)** | **Start here.** Full from-scratch setup: Python, deps, model fetch, drivers, per-game config, camera FOV, first run. |
| [docs/TUNING.md](docs/TUNING.md) | Every tuner knob, the calibration workflow, and a symptom→fix cheat sheet. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the capture→model→controller→output pipeline works and what's a comma port vs ours. |
| [RELEASE.md](RELEASE.md) | Maintainer release / build / zip checklist. |

There are two ways to run SimSteer:
- **From source** (you have this folder + Python 3.11) → follow
  [docs/SETUP.md](docs/SETUP.md).
- **From a pre-built `SimSteer.exe`** (someone handed you the
  `dist\SimSteer\` zip) → the *End-user install* section below.

## End-user install

1. **Download** the latest `SimSteer-<version>.zip` from
   [Releases](https://github.com/140er/simsteer/releases) and
   extract it anywhere (e.g. `C:\SimSteer\`).

2. **Unblock the .exe.** Windows quarantines unsigned binaries downloaded
   from the internet:
   - Right-click `SimSteer.exe` → Properties → tick **Unblock** → OK
   - Or, on first launch, click *More info → Run anyway* on the
     SmartScreen popup.

3. **Install the output driver** for your chosen device, then **reboot**:
   - **Gamepad (default)** → ViGEm Bus Driver (emulates an Xbox 360 pad):
     https://github.com/nefarius/ViGEmBus/releases — run the `.msi`.
   - **Wheel** (`--device wheel`) → vJoy (emulates a DirectInput wheel;
     ETS2 skips its gamepad rack assist for a linear response):
     https://github.com/njz3/vJoy/releases — run the installer.

   You only need the one matching how you'll steer. Gamepad is the
   default.

4. **Game-specific setup** (only do the ones you'll play):

   <details>
   <summary><b>ETS2</b></summary>

   - **SCS Telemetry plugin**: SimSteer bundles it. On first launch
     a dialog will offer to install it into your ETS2 plugins folder.
     If that fails (UAC denied), the plugin is at `prereqs\scs-telemetry.dll`
     inside the install folder — copy it manually into
     `<ETS2>\bin\win_x64\plugins\`.
   - **Steering deadzone must be 0.** ETS2 defaults to 16%, which silences
     the AI's small steering inputs. Fix in-game:
     Options → Controls → find the Steering deadzone slider → set to 0%.
     SimSteer will warn you on startup if it detects a non-zero deadzone.
   - **Camera**: F1 (interior cab).
   - **Window mode**: windowed or borderless (not exclusive fullscreen).
   - **Controller binding**: with SimSteer running, ETS2 will see the
     virtual Xbox 360 pad as a new device. Bind steering axis → Left Stick X.
     Use the wiggle keys (1-0) on the overlay window to identify each input.
   </details>

   <details>
   <summary><b>Forza Horizon 5</b></summary>

   - **Enable Data Out** in-game:
     - Settings → HUD and Gameplay → Data Out → **ON**
     - Data Out IP Address: `127.0.0.1`
     - Data Out IP Port: `7777`
     - Data Out Packet Format: **Dash**
   - **Camera**: bumper or cockpit (dashcam-like).
   - **Window mode**: windowed or borderless.
   </details>

   <details>
   <summary><b>Assetto Corsa</b></summary>

   - Launch AC through **Content Manager** (https://acstuff.ru/app/) —
     stock launcher works but Content Manager is more reliable.
   - **Camera**: cockpit / interior.
   - **Window mode**: windowed or borderless.
   </details>

5. **Run** `SimSteer.exe`. The launcher tells you what's missing.

6. **Set the camera FOV** (once, by hand — there is no auto-FOV). In the
   tuner's **Camera & Calibration** section, set **Capture VFOV** to match
   your in-game field of view (ETS2: Options → Gameplay → Camera; AC:
   Options → Video → Camera FOV; Forza: Settings → Difficulty → Camera
   FOV). Then drive a straight road at speed and check the HUD **FOV**
   line: `ratio vx_model/v_ego` should sit near `1.00`. `>1.05` = FOV too
   high (narrow it); `<0.95` = too low (widen it). **Wrong FOV is the #1
   cause of the plan veering off the road** — pitch/yaw/height
   auto-calibrate, FOV does not. See [docs/TUNING.md](docs/TUNING.md).

7. **Calibrate** (first launch per game):
   - Drive normally on highway for ~10-15 minutes.
   - A progress bar at the top of the overlay tracks calibration.
   - Don't yank the wheel — the calibration auto-pauses when it detects
     human steering input.
   - When you hear the READY chime (if you've dropped audio files into
     `assets/` — see [Audio](#audio)) and the banner turns green, press
     **INSERT** to engage.

8. **Drive.** Press **INSERT** to engage/disengage at any time.

### Configuration

SimSteer opens a small tuner window alongside the overlay. The first
tab — **Setup** — has everything a new user needs to pick:

- **Game**: Auto-detect / ETS2 / AC / Forza. *Game change requires
  restart* (click **Save & Restart**; SimSteer re-launches itself).
- **Output device**: Gamepad (ViGEm) or Wheel (vJoy). Applies on next
  launch; live device-swap is on the Manual tab.
- **Active steering probing** (default on): the small ±0.03 axis wiggle
  the wizard uses during Phase B.
- **Passive LiveParams fit on ETS2 while disengaged** (default off):
  risky — pollutes the gamepad fit if you drive with a wheel.
- **Force-engage** (dev only): bypass the calibration / FPS /
  telemetry gate.

Settings persist to `%LOCALAPPDATA%\SimSteer\settings.json`. CLI flags
still override settings for the current run, so a one-off
`SimSteer.exe --game forza` doesn't overwrite your saved default.

To skip the tuner window entirely, pass `--no-tuner` or set
`no_tuner: true` in `settings.json` (you'll lose the Setup tab too,
so leave this off unless you're scripting).

### Hotkeys

| Key | Action |
|---|---|
| `INSERT` | Engage / disengage (global — works while the game is focused) |
| `NumPad 4 / 6` | Lane change LEFT / RIGHT (global) |
| `PgUp / PgDn` | NAV: queue LEFT / RIGHT maneuver (global) |
| `End` | NAV: clear queue (global) |
| `Q` | Quit (overlay window) |
| `R` | Reset calibration — wipes LiveCalib + LiveParams + wizard flag, restarts the first-drive wizard. Force-disengages first. (overlay) |
| `V` | Toggle model-view / capture-view (overlay) |
| `M` | Mirror lateral sign (overlay) |
| `[` `]` `,` `.` `;` `'` | Manual calibration tweaks (overlay) |
| `1`-`0` | Wiggle controller inputs for ETS2 binding (overlay, while disengaged) |

### Active steering probing

During the wizard's Phase B (camera done, steering not yet trusted), SimSteer
superimposes a small sinusoidal perturbation (±0.03 axis at ≤12 m/s, scaled
down at higher speeds) on top of the AI's steering command. This excites the
rack across its operating range so `LiveParams` gets useful `(axis, wheel)`
pairs every frame even on a perfectly straight highway — where natural
driving would produce no signal at all (`rej_small_axis` rejecting every
sample). Empirically this converts straight-line driving from 0 useful
samples to ~80% accepted, cutting steering-fit convergence from "many minutes
of cornering" to "~15 seconds on a freeway."

Safety: the probe is gated on engaged + wizard Phase B + speed in
`[6, 30] m/s` + no lane change + the AI not already commanding a large
steer. Once `LiveParams.trusted()` flips True, the probe stops on its own.

The HUD's `PROBE` line shows the current probe offset, the gate reason
when inactive, and total active frames. Disable entirely with `--no-probe`.

### Calibration recovery

If calibration goes bad — symptoms include the truck wandering, lanes
drawn wrong, or persistent INVALID state in the LIVECALIB HUD line —
you have three escape hatches:

1. **Press `R`** in the overlay window. Wipes both calibrators and the
   wizard flag, restarts the first-drive wizard from zero.
2. **Quit and relaunch with `--reset-calib`**: equivalent to `R` but
   applied at startup before any state loads.
3. **Manual**: delete `%LOCALAPPDATA%\SimSteer\livecalib_state_<game>.json`,
   `liveparams_<game>_<device>.json`, and `first_drive_done_<game>.flag`.

LiveCalib also auto-recovers if its smoothed estimate goes INVALID for
3+ consecutive blocks — it resets itself and the wizard hint surfaces
the reason so you know the bar jumping back to 0% wasn't a glitch.

## Troubleshooting

- **Truck doesn't steer (ETS2)**: deadzone isn't 0. SimSteer warns on
  startup if it can detect this.
- **"Cannot engage — frame rate too low"**: DirectML failed to load and
  vision is on CPU. Install onnxruntime-directml:
  `pip install onnxruntime-directml` (dev install) or reinstall the
  bundle (release install).
- **"Game telemetry not detected"**: SCS plugin missing (ETS2), Data Out
  off (Forza), or AC not in a session.
- **AC / ETS2 launched as admin**: launch SimSteer as admin too —
  shared-memory mappings live in per-session namespaces.
- **AV quarantines the .exe mid-extraction**: unsigned PyInstaller
  bundles are a common false-positive. Whitelist `SimSteer\` and
  re-extract.

## Audio

SimSteer plays optional chimes on engage/disengage/ready. Files live
under `assets/` next to the .exe:

- `engage.wav`
- `disengage.wav`
- `denied.wav`
- `ready.wav`

Drop your own `.wav` files in; SimSteer gracefully no-ops if a file is
missing. Keep them short (under ~1 s) and quiet — loud chimes while
driving are dangerous.

## Where data lives

| | Read-only | Read/write |
|---|---|---|
| Models, bundled DLLs | `SimSteer\models\`, `SimSteer\prereqs\` | — |
| Calibration / steering fits / wizard flags | — | `%LOCALAPPDATA%\SimSteer\` |
| Audio | `SimSteer\assets\*.wav` | — |

To start calibration over for a game, delete the matching files in
`%LOCALAPPDATA%\SimSteer\` (e.g. `livecalib_state_ets2.json`,
`liveparams_ets2_gamepad.json`, `first_drive_done_ets2.flag`).

## Dev install

For working on SimSteer itself (not just running it). This is the quick
version — [docs/SETUP.md](docs/SETUP.md) has the full walkthrough
including drivers, per-game config, and the manual FOV step.

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
pip install pyinstaller       # only needed to build .exe
pip install pyvjoy            # only needed for --device wheel
python tools\fetch_model.py
```

Run directly:

```powershell
python -m pilot.main
python -m pilot.main --no-gamepad     # dry run, no virtual pad output
python -m pilot.main --device wheel   # vJoy wheel instead of ViGEm pad
python -m pilot.main --force-engage   # bypass calibration gate (DEV ONLY)
```

Build the .exe:

```powershell
.\build.bat
```

Output lands in `dist\SimSteer\`. See [RELEASE.md](RELEASE.md) for the
full release checklist.

## What it does (architecture)

- ~20 Hz screen capture (DXGI desktop duplication via dxcam)
- Frame warp onto openpilot's fixed virtual cameras (`warp.py`)
- Two-stage ONNX inference: `driving_vision.onnx` (DirectML) → `driving_policy.onnx` (CPU)
- Plan + lane decoding (openpilot's postprocess math)
- Feed-forward lateral controller with online steering-rack fitting
  (`LiveParams`, ports openpilot's `paramsd`) + a slow closed-loop
  wheel-angle trim
- Online camera-pose calibration (`LiveCalib`, ports openpilot's
  `calibrationd`). Camera **FOV is static / manual** — set once to match
  the game.
- Virtual output: ViGEm Xbox 360 pad or vJoy wheel

Steering timing (`lookahead_s` = actuator delay, `curvature_anticipation_s`
= lead buffer) and `steer_authority` are **static knobs**, matching how
openpilot ships per-car constants rather than tuning them online. Four
earlier online auto-tuners were removed for running away — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#removed-auto-tuners-and-why).

The model is comma.ai's; the bridge is ours.

## License

This project's code is released under the [MIT License](LICENSE).

It is an **unofficial** hobby project and is **not affiliated with,
endorsed by, or supported by comma.ai**. The driving model
(`driving_vision.onnx` / `driving_policy.onnx`) is fetched at setup time
from comma.ai's [openpilot](https://github.com/commaai/openpilot) repo
and remains subject to openpilot's own license — it is not redistributed
here. Third-party drivers (ViGEm, vJoy) and their tools are the property
of their respective authors and are installed separately, not bundled in
this repository.

Driving a vehicle — virtual or real — with software carries risk. Use in
games only, at your own risk; see the warranty disclaimer in the LICENSE.
