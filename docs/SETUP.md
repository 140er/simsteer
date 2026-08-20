# SimSteer — Setup from source

This is the complete walkthrough for getting SimSteer running on a fresh
Windows machine, starting from the zipped source folder. If you were
handed a pre-built `SimSteer.exe` instead, skip to
[Drivers](#3-install-the-output-driver) and
[Per-game setup](#5-per-game-setup) — the rest is for building/running
from source.

> **What this is**: a bridge that screen-captures a driving game, runs
> comma.ai's openpilot vision+planning model on the frame, and steers
> the car through a virtual gamepad (ViGEm) or virtual wheel (vJoy).
> Supports Euro Truck Simulator 2, Forza Horizon 5, and Assetto Corsa.

---

## 0. What you need

| Requirement | Notes |
|---|---|
| **Windows 10/11** | The whole input + capture stack is Windows-only (ViGEm, vJoy, DXGI capture). |
| **Python 3.11** | Hard requirement — wheels for `onnxruntime-directml`, `vgamepad` etc. target 3.11. `py -3.11 --version` to check. |
| **A GPU that supports DirectML** | AMD / NVIDIA / Intel all work via DirectML. Without it vision falls back to CPU at ~4–8 FPS (too slow to engage). The original rig is an AMD 7900 XTX. |
| **~2 GB free disk** | venv + ONNX models + build output. |
| **One of the supported games** | ETS2, Forza Horizon 5, or Assetto Corsa. |

---

## 1. Unpack + create the virtual environment

```powershell
# from wherever you extracted the zip, e.g. C:\SimSteer\
cd "C:\SimSteer"

py -3.11 -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pulls:

| Package | Why |
|---|---|
| `numpy>=1.26,<2.0` | tensor math / decoding |
| `opencv-python>=4.9` | image warp, overlay drawing |
| `onnxruntime-directml>=1.17` | GPU inference (vision model) |
| `dxcam>=0.0.5` | fast DXGI screen capture |
| `mss>=9.0` | fallback screen capture |
| `vgamepad>=0.1.0` | ViGEm virtual Xbox 360 pad (gamepad output) |
| `customtkinter>=5.2,<6.0` | the tuner window |

**Wheel-mode only:** if you intend to use the vJoy wheel device
(`--device wheel`) instead of the gamepad, also:

```powershell
pip install pyvjoy
```

> **Path-with-spaces note**: this project is often developed in a path with a space, e.g.
> `C:\Sim Tools\SimSteer\`. Always quote the venv activate
> path, or activate once and run commands from the activated shell.

---

## 2. Fetch the model

The ONNX models aren't committed (they're ~59 MB and come from comma.ai's
repo). Fetch them once:

```powershell
python tools\fetch_model.py
```

This sparse-clones openpilot, LFS-pulls `driving_vision.onnx` (~45 MB)
and `driving_policy.onnx` (~14 MB) into `models\`, then cleans up the
temp clone. Verify:

```powershell
Get-ChildItem models\*.onnx
```

You should see both files. The app does a fatal preflight check if
they're missing or truncated.

---

## 3. Install the output driver

SimSteer drives the game through a **virtual** controller. Pick based on
how you want to steer:

### Gamepad (default — `--device gamepad`)
**ViGEm Bus Driver** — emulates an Xbox 360 pad.
- Download: https://github.com/nefarius/ViGEmBus/releases
- Run the `.msi`, **reboot**.

### Wheel (`--device wheel`)
**vJoy** — emulates a DirectInput wheel. ETS2 treats it as a real wheel
and skips its speed-sensitive gamepad rack assist, giving a linear
axis→wheel response (often nicer to tune).
- Download: https://github.com/njz3/vJoy/releases
- Run the installer, **reboot**.
- Also `pip install pyvjoy` (see step 1).

### Fanatec (`--device fanatec`)
**For Fanatec wheel owners** — **DirectInput Force Feedback motor control**

SimSteer uses **DirectInput Force Feedback** to send constant force effects to
your Fanatec wheel motor, attempting to physically turn the rim when engaged.

- **Requirements**:
  - **Fanatec driver**: https://fanatec.com/en-us/technology/firmware-update
  - Wheel in **PC mode** (check Fanatec Control Panel)
  - **vJoy driver** (fallback if FFB unavailable - see Wheel section above)
  - `pip install pygame` (for wheel detection)
  - `pip install pyvjoy` (for vJoy fallback)

- **How it works**:
  - When **engaged**: SimSteer acquires DirectInput in BACKGROUND|EXCLUSIVE mode
    and sends constant force effects to physically turn the wheel rim toward
    the AI's target steering angle
  - When **disengaged**: Motor releases (all effects stopped) so you can take
    manual control
  - Game continues to read your wheel position, so ETS2/AC/Forza follow the
    physical rim whether AI or human is steering
  
- **FFB Mode vs vJoy Fallback**:
  - **FFB motor mode** (preferred): Physical wheel actually moves
  - **vJoy fallback**: If FFB acquisition fails (no wheel detected, game has
    exclusive foreground lock, or FFB initialization error), SimSteer
    automatically falls back to vJoy virtual output (coexistence mode)
  - Check console output or tuner status to see which mode is active

- **Implementation Status**:
  - DirectInput FFB motor control is implemented via DirectInput8 COM interfaces
    using Python ctypes
  - Actual hardware validation with Fanatec wheels is pending
  - If you have a Fanatec wheel, please test and report results
  - Code includes proper HRESULT checking, device enumeration, acquisition,
    effect creation, and force updates
  - Fallback to vJoy is automatic and seamless if FFB cannot be acquired

- **Limitations**:
  - DirectInput FFB requires BACKGROUND|EXCLUSIVE access. Most modern games
    read wheel position via shared DirectInput and use FFB separately, so
    this usually works. If game holds FOREGROUND|EXCLUSIVE FFB, SimSteer
    can't acquire and falls back to vJoy.
  - DIERR_NOTACQUIRED / DIERR_INPUTLOST errors trigger automatic re-acquisition
    attempts. Persistent failures fall back to vJoy.

- **Future: Fanatec SDK**:
  - Fanatec provides proprietary FullForce SDK (`EndorFanatecSdk64.dll`) that
    may allow simultaneous game + SimSteer control without DirectInput conflicts
  - Requires game developer SDK access (gamedev@fanatec.com)
  - Current implementation uses standard DirectInput as portable solution

> You only need the one matching your `--device`. Gamepad is the default
> and the simplest to get going.

### Shortcut: the driver scripts

Instead of installing by hand, run the helper (installs **both** ViGEm
and vJoy + `pyvjoy` into the venv; self-elevates, one UAC prompt):

```powershell
.\tools\install_drivers.ps1
```

To remove them again — useful when a real wheel (e.g. a Fanatec) fights
the always-present virtual devices:

```powershell
.\tools\uninstall_drivers.ps1
```

Both want a reboot afterward so the kernel drivers load / unload cleanly.

---

## 4. (ETS2 only) Install the SCS telemetry plugin

SimSteer reads speed + wheel angle from ETS2 via an SCS telemetry plugin.

- If a built `prereqs\scs-telemetry.dll` is present, the app offers a
  one-click install on first launch (it copies the DLL into
  `<ETS2>\bin\win_x64\plugins\`). Accept the UAC prompt.
- If that DLL isn't in your copy, grab a release from
  https://github.com/RenCloud/scs-sdk-plugin/releases and drop the
  64-bit `scs-telemetry.dll` into `<ETS2>\bin\win_x64\plugins\` yourself.

Forza and AC need no plugin (see per-game setup below).

---

## 5. Per-game setup

Only do the game(s) you'll actually drive.

### Euro Truck Simulator 2
- **Steering deadzone MUST be 0.** ETS2 defaults to ~16%, which eats the
  AI's small steering inputs and makes driving look broken.
  *Options → Controls → Steering deadzone → 0%.* The app warns on startup
  if it detects a non-zero value.
- **Camera**: interior cab (F1).
- **Window mode**: windowed or borderless — **not** exclusive fullscreen
  (screen capture needs a composited window).
- **Controller binding**: with SimSteer running, ETS2 sees the virtual
  pad/wheel as a new device. Bind the steering axis to it. Use the
  wiggle keys (`1`–`0`) on the overlay window while disengaged to make
  one input move so ETS2's binding wizard can detect it.

### Forza Horizon 5
- *Settings → HUD and Gameplay → Data Out → **ON***
- Data Out IP Address: `127.0.0.1`
- Data Out IP Port: `7777` (must match `--forza-port`)
- Data Out Packet Format: **Dash**
- Camera: bumper or cockpit. Window: windowed/borderless.

### Assetto Corsa
- Launch through **Content Manager** (https://acstuff.ru/app/) — more
  reliable shared-memory telemetry than the stock launcher.
- Camera: cockpit/interior. Window: windowed/borderless.

---

## 6. Camera FOV — automatic detection + manual verification

> **CRITICAL**: Wrong FOV used to be the #1 cause of the plan veering off
> the road. SimSteer now includes **AUTOMATIC FOV detection** to fix this!

### Automatic FOV Detection (NEW!)

SimSteer will automatically measure and correct your FOV during the first
drive:

1. **Just drive normally on the highway** for 30-60 seconds at 15+ mph
2. Watch the HUD **"auto-FOV"** line for progress (60 samples needed)
3. When complete, SimSteer applies the correction automatically
4. You'll see a banner: "auto-FOV: XX → YY deg (vx/v_ego=N.NN)"

The auto-FOV system:
- Measures `vx_model / v_ego` ratio on straight driving
- Solves the true FOV once and freezes (cannot run away)
- Saves the result automatically

### Manual FOV Setup (Optional)

If you prefer manual control or want to verify auto-FOV worked correctly:

1. Find your in-game FOV setting:
   - **ETS2**: Options → Gameplay → Camera → *Field of view*.
   - **AC**: Options → Video → *Camera FOV*.
   - **Forza**: Settings → Difficulty → *Camera FOV*.
2. In the SimSteer tuner, open **Camera & Calibration** and set
   **Capture VFOV** so it matches. (The tuner takes a *vertical* FOV and
   derives horizontal from your capture aspect ratio.)
3. Verify while driving: the HUD's **FOV** line shows
   `ratio vx_model/v_ego`. Drive a straight road at highway speed:
   - `~1.00` (±5%) → FOV is right.
   - `> 1.05` → FOV too high, narrow it.
   - `< 0.95` → FOV too low, widen it.

**Note**: Pitch, yaw, and camera height *do* auto-calibrate (LiveCalib).
Only FOV needs manual setting if you disable auto-FOV.

---

## 7. Run it

From the activated venv:

```powershell
python -m pilot.main                 # default: gamepad, auto-detect game
python -m pilot.main --device wheel  # vJoy wheel output
python -m pilot.main --game ets2     # force a specific game
python -m pilot.main --no-gamepad    # overlay only, no virtual device (dry run)
```

See [all CLI flags](#cli-flags) below.

On launch you get an **overlay window** (the camera view with the model's
plan drawn on it) and a **tuner window** (collapsible sections of live
knobs). The console / launcher tells you what's missing if a preflight
check fails.

### First drive (calibration)
1. Drive **manually** on a highway for ~10–15 minutes
2. **Setup wizard provides clear guidance**:
   - Banner shows progress percentage and requirements
   - Speed: must maintain 15+ mph for calibration
   - Steering: drive straight with gentle inputs
   - Warning indicators (⚠️) show if you're too slow or steering too much
   - Success indicators (✓) confirm good calibration progress
3. Calibration auto-pauses when it detects hard steering
4. Watch for two milestones:
   - Camera calibration (pitch/yaw/height)
   - Steering calibration (rack fit via LiveParams)
5. When the banner shows "✓ READY — Press INSERT to engage autonomous
   driving", press **INSERT** to engage
6. **INSERT** toggles engage/disengage any time

See [docs/TUNING.md](TUNING.md) for what every knob does and how to fix
bad driving, and [docs/ARCHITECTURE.md](ARCHITECTURE.md) for how the
pipeline works.

---

## 8. (Optional) Build a standalone .exe

To produce a no-Python-required bundle (for sharing with non-developers):

```powershell
& ".\.venv\Scripts\Activate.ps1"
pip install pyinstaller
python tools\fetch_model.py     # models must be present to bundle them
.\build.bat
```

Output: `dist\SimSteer\` containing `SimSteer.exe` (windowed) and
`SimSteer-debug.exe` (console). See [RELEASE.md](../RELEASE.md) for the
full release/zip checklist.

> **Close any running `SimSteer.exe` before rebuilding** — PyInstaller
> can't overwrite a locked exe and the build will fail on "Access is
> denied."

---

## CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--game {auto,ets2,ac,forza}` | auto | Which game's telemetry to read. `auto` tries ETS2 → AC → Forza. |
| `--device {gamepad,wheel,fanatec}` | gamepad | Output device. gamepad = ViGEm Xbox 360; wheel = vJoy; fanatec = Fanatec + vJoy coexistence. |
| `--vjoy-device N` | 1 | vJoy device index when `--device wheel` or `--device fanatec`. |
| `--forza-port N` | 7777 | UDP port Forza Data Out targets. Must match in-game. |
| `--max-width N` | 1600 | Cap overlay window width (px). |
| `--no-gamepad` | off | Run overlay + model but open no virtual device. Skips ViGEm/vJoy preflight. Dry-run. |
| `--no-tuner` | off | Don't open the tuner window. (You lose the Setup tab too.) |
| `--no-probe` | off | Disable the active-steering probe used during the calibration wizard. |
| `--no-auto-fov` | off | Disable automatic FOV detection (forces manual FOV setup). |
| `--passive-fit-ets2` | off | Allow LiveParams to learn while disengaged on ETS2 (risky — see TUNING.md). |
| `--force-engage` | off | **Dev only.** Bypass the calibration / FPS / telemetry engage gate. |
| `--reset-calib` | — | Wipe LiveCalib + LiveParams + wizard flag at startup and recalibrate from zero. |

CLI flags override saved settings for that run only;
`%LOCALAPPDATA%\SimSteer\settings.json` holds the persistent defaults.

---

## Where data lives

| | Location |
|---|---|
| Models, bundled DLLs | `models\`, `prereqs\` (read-only) |
| Per-user calibration / steering fits / wizard flags | `%LOCALAPPDATA%\SimSteer\` (read/write) |
| Audio cues | `assets\*.wav` (optional, read-only) |

To restart calibration for a game, delete its files in
`%LOCALAPPDATA%\SimSteer\` (`livecalib_state_<game>.json`,
`liveparams_<game>_<device>.json`, `first_drive_done_<game>.flag`) — or
just press `R` in the overlay.

## Troubleshooting

- **Truck doesn't steer (ETS2)** → deadzone isn't 0. The preflight dialog
  shows a detailed warning with step-by-step fix instructions.
- **"Cannot engage — frame rate too low"** → DirectML didn't load, vision
  is on CPU. Preflight shows a prominent warning. Reinstall
  `onnxruntime-directml`.
- **Plan veers way off the road / truck won't hold a lane** → Auto-FOV
  should fix this automatically. If it persists, check the HUD FOV ratio
  manually (see [step 6](#6-camera-fov--automatic-detection--manual-verification)).
- **"Game telemetry not detected"** → SCS plugin missing (ETS2), Data Out
  off (Forza), or AC not in a session. Preflight provides specific
  instructions for each case.
- **Game launched as admin** → launch SimSteer as admin too (shared
  memory lives in per-session namespaces). Preflight warns about this for AC.
- **Build fails "Access is denied" on SimSteer.exe** → a copy is still
  running; close it first.
- **Fanatec wheel not detected** → Check preflight messages for detailed
  troubleshooting. Ensure driver installed, wheel in PC mode, powered on.
  SimSteer will automatically fall back to vJoy mode if FFB control fails.
- **Fanatec wheel detected but doesn't move physically** → FFB motor control
  failed, running in vJoy fallback mode. Check console for "FFB motor control"
  vs "vJoy fallback" status. Common causes: game holding exclusive foreground
  FFB, DirectInput ctypes bindings incomplete (work in progress).
- **Fanatec FFB too weak / too strong** → Adjust Fanatec tuning menu FOR
  (Force) setting. SimSteer sends proportional constant force based on angle
  error. Your base's FOR/FFB settings scale the output torque.
