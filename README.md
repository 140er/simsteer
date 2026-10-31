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
| [docs/PROFILES.md](docs/PROFILES.md) | Adding a game: the TOML profile file and every key in it. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the capture→model→controller→output pipeline works and what's a comma port vs ours. |
| [RELEASE.md](RELEASE.md) | Maintainer release / build / zip checklist. |

There are two ways to run SimSteer:
- **From source** (you have this folder + Python 3.11) → follow
  [docs/SETUP.md](docs/SETUP.md).
- **From a pre-built `SimSteer.exe`** (someone handed you the
  `dist\SimSteer\` zip) → the *End-user install* section below.

If Smart App Control blocks `SimSteer.exe`, [run from source](#run-from-source).

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
   - **Use Wheel (absolute) steering and set steering deadzone to 0%.**
     ETS2 defaults to 16%, which silences
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

6. **Calibrate** (first drive per game and device). SimSteer walks you through four steps at the top of the overlay, as
   in `CALIBRATE 2/4 Camera 63% ~3 min`. The target is under 10 minutes of
   highway driving.
   1. **FOV.** Drive straight above 30 km/h. SimSteer measures your FOV
      once and shows it. **INSERT** accepts it, **BACKSPACE** keeps
      yours. Setting FOV by hand during this step (tuner, `[` `]`) uses
      that value.
   2. **Camera.** Drive manually on a highway with small steering
      inputs; big inputs pause it.
   3. **Steering.** First drive gentle left and right bends manually.
      SimSteer compares the model, input axis and measured wheel angle
      to detect inverted input. Then stop on level ground with the parking
      brake on, release steering and press **INSERT**: four small symmetric
      steering pulses check device-to-game direction while SimSteer stays
      disengaged. Movement or player steering
      aborts the check. After verification, press **INSERT** again to drive.
      Both mappings save for this game and device. It asks for
      a straight stretch, gentle curves, two lane changes and a slow
      turn while it fits the steering.
   4. **Check.** Two minutes of engaged highway driving. It ends with
      `CALIBRATED score 0.86`, or names the step to redo and why and
      goes back to it.

   Progress saves as you go, so quitting resumes where you stopped.
   **R** starts over at FOV.

7. **Drive.** Press **INSERT** to engage/disengage at any time.

### Configuration

SimSteer opens a small tuner window alongside the overlay. The first
tab — **Setup** — has everything a new user needs to pick:

- **Game**: Auto, or one of the game profiles (`--list-games` prints
  them). Auto drives the game that is running at launch. *Game change
  requires restart* (click **Save & Restart**; SimSteer re-launches
  itself). Adding a game is one TOML file, see
  [docs/PROFILES.md](docs/PROFILES.md).
- **Output device**: Gamepad (ViGEm) or Wheel (vJoy). Applies on next
  launch; live device-swap is on the Manual tab.
- **Active steering probing** (default on): the small ±0.03 axis wiggle
  used during the calibration's Steering step.
- **Passive LiveParams fit while disengaged** (default off): for games
  whose profile does not allow it, such as ETS2. Risky there — pollutes
  the gamepad fit if you drive with a wheel.
- **Force-engage** (dev only): bypass the calibration / FPS /
  telemetry gate.

Settings persist to `%LOCALAPPDATA%\SimSteer\settings.json`. CLI flags
still override settings for the current run, so a one-off
`SimSteer.exe --game ats` doesn't overwrite your saved default.

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
| `R` | Recalibrate: wipes the camera pose and steering fit and restarts calibration at FOV. Force-disengages first. (overlay) |
| `BACKSPACE` | Keep your FOV instead of the measured one (overlay, during calibration) |
| `V` | Toggle model-view / capture-view (overlay) |
| `M` | Mirror road overlay (also steering without wheel telemetry) |
| `[` `]` `,` `.` `;` `'` | Manual calibration tweaks (overlay) |
| `1`-`0` | Wiggle controller inputs for ETS2 binding (overlay, while disengaged) |

### Active steering probing

Stop on level ground with the parking brake on. Output direction is checked
below 0.5 m/s with four alternating ±0.10 steering holds of three seconds
each. Only the final second of each hold counts, allowing game input lag
and smoothing. The check ends within 14
seconds or shows a one-line reason. It sends steering only, with AI
disengaged, and remains available with `--no-probe`.

After verification, during the calibration's Steering step SimSteer adds
a small sinusoidal perturbation (±0.03 axis at ≤12 m/s, scaled down at higher
speeds) to AI steering until the rack fit is trusted. This excites the
rack across its operating range so `LiveParams` gets useful `(axis, wheel)`
pairs every frame even on a perfectly straight highway — where natural
driving would produce no signal at all (`rej_small_axis` rejecting every
sample). Empirically this converts straight-line driving from 0 useful
samples to ~80% accepted, cutting steering-fit convergence from "many minutes
of cornering" to "~15 seconds on a freeway."

Safety: the probe is gated on engaged + the Steering step + speed in
`[6, 30] m/s` + no lane change + the AI not already commanding a large
steer. Once `LiveParams.trusted()` flips True, the probe stops on its own.

The HUD's `PROBE` line shows the current probe offset, the gate reason
when inactive, and total active frames. Disable this rack probe with `--no-probe`.

### Calibration recovery

If calibration goes bad — symptoms include the truck wandering, lanes
drawn wrong, or persistent INVALID state in the LIVECALIB HUD line —
you have three escape hatches:

1. **Press `R`** in the overlay window, or **Recalibrate** in the
   tuner's Setup section. Both restart calibration at FOV.
2. **Manual**: delete `%LOCALAPPDATA%\SimSteer\calibflow_<game>_<device>.json`,
   `livecalib_state_<game>.json` and `liveparams_<game>_<device>.json`.

LiveCalib also auto-recovers if its smoothed estimate goes INVALID for
3+ consecutive blocks. It resets itself, calibration goes back to the
Camera step, and the hint says why the bar jumped back to 0%.

## Run from source

Use this when Windows says Smart App Control blocked `SimSteer.exe`.
Smart App Control blocks unsigned apps it doesn't know, and SimSteer
isn't code-signed yet. Keep Smart App Control on: older Windows
configurations require a reset to turn it back on after disabling it.
[Recent Windows updates allow re-enabling it without a reset](https://support.microsoft.com/en-us/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions).

1. Download **Source code (zip)** from the latest
   [release](https://github.com/140er/simsteer/releases) and extract it.
2. Double-click `run-from-source.bat`.

The first run sets up a private Python 3.11 in `.venv`, installs
SimSteer's packages and fetches the models, which takes a few minutes.
Later runs start straight away. It needs [uv](https://docs.astral.sh/uv/)
and offers to install it if it's missing. Arguments pass through, for
example `run-from-source.bat --game ets2`.

Follow the driver, game, camera and calibration steps above, launching
with `run-from-source.bat` in step 5. Details are in
[docs/SETUP.md](docs/SETUP.md#run-from-source-smart-app-control).

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
| Calibration / steering fits | — | `%LOCALAPPDATA%\SimSteer\` |
| Audio | `SimSteer\assets\*.wav` | — |

To start calibration over for a game, delete the matching files in
`%LOCALAPPDATA%\SimSteer\` (e.g. `livecalib_state_ets2.json`,
`liveparams_ets2_gamepad.json`, `calibflow_ets2_gamepad.json`).

## Dev install

For working on SimSteer itself (not just running it). This is the quick
version — [docs/SETUP.md](docs/SETUP.md) has the full walkthrough
including drivers, per-game config, and guided calibration.

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
- Guided camera-pose calibration (`LiveCalib`, based on openpilot's
  `calibrationd`). FOV comes from the game's config, a measurement the
  player accepts, or manual entry.
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
