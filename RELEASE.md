# Release checklist

For maintainers cutting a new SimSteer release.

## 1. Pre-release

- [ ] Bump `pilot/version.py:__version__`.
- [ ] If shipping the SCS plugin DLL: build the latest from
      `.scs-sdk-plugin/` and drop the 64-bit `scs-telemetry.dll` at
      `prereqs/scs-telemetry.dll`. Confirm `.scs-sdk-plugin/LICENSE`
      still permits redistribution.
- [ ] If adding/refreshing audio cues: drop `.wav` files into `assets/`.
- [ ] If adding/refreshing screenshots: drop them into `docs/` with the
      names referenced by `pilot/preflight.py`
      (`forza-data-out.png`, `ets2-deadzone.png`, `ac-content-manager.png`).

## 2. Build

```powershell
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
pip install pyinstaller
python tools\fetch_model.py            # ensure models/ has both .onnx files
.\build.bat
```

Output: `dist\SimSteer\` (~280 MB unpacked).

## 3. Smoke test on a clean Windows 11 VM (no Python installed)

Critical — DirectML / ViGEm / mmap behave differently on machines that
aren't the dev's.

1. Copy `dist\SimSteer\` to the VM.
2. Run `SimSteer-debug.exe` (console version) and verify:
   - Fatal preflight: ViGEm not installed → Tk modal, then exit 2.
3. Install ViGEm in the VM, reboot, run `SimSteer.exe` (release).
4. With no game running: HUD shows 3 yellow warnings, INSERT blocks
   with "game telemetry not detected".
5. Launch ETS2 (or your game of choice). Verify:
   - Telemetry connects (HUD shows the game name in title bar).
   - For ETS2: if SCS plugin missing, install dialog appears with
     [Install] button. Click Install → DLL copies to plugins folder.
   - For ETS2: if deadzone non-zero, warning appears with menu path.
6. Wizard banner appears: `FIRST DRIVE — 0% — drive manually on highway`.
7. Drive ~12 min. Verify:
   - Progress climbs.
   - When you manually yank the wheel, calibration pauses
     (rej_human increments, smoothed pitch/yaw stay stable).
   - At ~5 min: phase B transition — banner switches to
     `Camera ready — press INSERT and drive gently`.
   - INSERT engages with yellow `ENGAGED — steering fit warming` banner.
   - At ~10-12 min: READY banner + chime (if `assets/ready.wav` is present).
8. Quit. Relaunch. Verify wizard does NOT reappear (flag file is honored).
9. Quit mid-wizard at ~50% in a fresh game profile. Relaunch.
   Progress resumes within ~1% of where it stopped (LiveCalib persistence).

## 4. Path-with-space regression

Build from a path containing a space (e.g. `C:\Sim Tools\SimSteer\`) and install
the output into `C:\Program Files\SimSteer\`. Verify writes land in
`%LOCALAPPDATA%\SimSteer\` (per `pilot.paths.data_dir()`).

## 5. Zip + upload

```powershell
$ver = (Get-Content pilot\version.py | Select-String '__version__\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
Compress-Archive -Path dist\SimSteer -DestinationPath "SimSteer-$ver.zip"
```

Upload to GitHub Releases with release notes. README's *End-user install*
section is the user-facing docs; reference it.

## Sharing source (not the built exe)

To hand the **source** to another developer (so they build/run it
themselves), use the helper — it `git archive`s HEAD, so only tracked
files go in (no `.venv`, `dist\`, `build\`, `models\*.onnx`, or per-user
state):

```powershell
.\tools\make_source_zip.ps1                # source only (small)
.\tools\make_source_zip.ps1 -IncludeModels # also bundle models\ (~59 MB)
```

Commit first — the archive reflects HEAD. The recipient extracts and
follows [docs/SETUP.md](docs/SETUP.md). Without `-IncludeModels` they run
`python tools\fetch_model.py` to pull the ONNX models.

## Known quirks to call out in release notes

- Unsigned binary — SmartScreen and many AV products flag on first run.
  Document the unblock flow in the release description.
- No auto-update — users need to download new releases manually.
- Calibration is per-user, per-game — first-run wizard always shows on
  a fresh install.

## Future (post-MVP)

- EV code signing certificate (~$300/yr) to remove SmartScreen friction.
- Auto-update channel (Tauri, electron-updater, or simple Github API
  polling).
- Bundle ViGEm/vJoy MSIs (currently links only).
