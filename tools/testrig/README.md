# Windows game test sessions

Run from a source checkout with Python 3.11 or later, in the logged-in Windows
console desktop. The tools use the standard library and Windows PowerShell;
the optional black overlay also uses Python's Tk support. Steam must already be
running and signed in, and the selected game must already be installed and owned.
The tools do not install drivers, change accounts, accept agreements, buy games,
change update settings, reboot, or turn monitors off.

```powershell
python -m tools.testrig guard
python -m tools.testrig ac prepare --session practice-01 --track magione --car lotus_elise_sc
python -m tools.testrig ac start --session practice-01
python -m tools.testrig ac ready --session practice-01 --timeout 60
python -m tools.testrig ac status --session practice-01
python -m tools.testrig ac stop --session practice-01
python -m tools.testrig ac restore --session practice-01
```

`start` includes preparation when no state exists, then waits for readiness.
`prepare` is useful for reviewing the configuration before launch. Repeat either
with the same options to resume/check the existing session. `ready` requires a
visible owned game window and multiple changing shared-memory packets; an open
launcher, paused simulation, or a stale mapping is insufficient. Its default
deadline is 180 seconds, with a maximum of 600 seconds and no automatic retries.

Both `stop` and `restore` close the session's processes, clean up its display,
and restore its settings. Repeat them after interruption to resume cleanup.
A completed session never rolls back later user edits. Use a new `--session`
name for the next test.

Exit code 0 means the requested operation completed. Code 2 means refusal or
failure; stderr explains the blocked step. JSON on stdout includes the phase and
state file location. Keep state, backups, logs and reports private.

## Guard and ownership

The read-only guard refuses unrelated Steam games, known racing games, VR,
recording/streaming software and OBS, or user activity within the last 120 seconds
(`--idle-seconds` can select 60–3600). It checks the current interactive console
session and refuses service or remote desktops. Keep remote automation input-free
while the owner is away. The guard runs before backups, before edits, and again
immediately before Steam launch. Readiness continues checking unrelated software
and pending restarts, while allowing interaction with the owned test game.

Restart checks include Windows Update `RebootRequired`, component servicing
`RebootPending`, nonempty `PendingFileRenameOperations`, update reboot indicators
and deadlines, and scheduled/running UpdateOrchestrator reboot tasks. Missing,
stale, malformed or access-denied inventory refuses startup. Nothing clears these
indicators automatically.

Launch uses `steam://rungameid/<appid>`. Ownership is recorded before checking
readiness: only newly created expected executables, with matching paths, creation
times, a recorded launch window and Steam/owned-launcher ancestry are adopted.
AC's directly started practice simulation is recorded against its launching
process as well. Cleanup verifies creation time and path through the same process
handle used to close/terminate it. It never kills by image name or closes Steam.
An unowned game process blocks restoration.

Standard named sign-in, ownership, EULA and firewall dialogs produce distinct
errors. Custom-rendered or localized game dialogs may instead produce the bounded
readiness timeout; the tool does not guess at buttons or enter credentials.

If a crash leaves Steam launch intent but no process whose ownership can be
established, restoration refuses. Cancel that pending launch in Steam, confirm
that it cannot subsequently start, then use:

```powershell
python -m tools.testrig ac restore --session practice-01 --launch-cancelled
```

This confirmation clears only the unresolved launch intent. It grants no
permission to kill an unowned process.

## Assetto Corsa

The adapter discovers Steam libraries from the registry and `libraryfolders.vdf`,
checks the app manifest, and resolves Documents through the Windows known-folder
API, including redirected folders. It backs up the game's Documents tree,
installation `cfg`, and active Steam user's app-specific userdata before editing.

The temporary configuration selects X360 input, zero steering deadzone/filter,
gamma 1, zero speed sensitivity, no wheel force feedback or pad rumble, automatic
clutch/shifting, and a single offline practice session. The default car/track are
Lotus Elise SC and Magione; `--layout` selects an installed track layout. The
camera is cockpit, with `--hfov` converted to vertical FOV for `--width/--height`.
Both Documents and installation defaults are covered by the backup.

Steam opens the AC launcher; once its owned window is available the adapter starts
`acs.exe` against the prepared `cfg/race.ini`. No mouse-coordinate launcher flow
is needed. Readiness opens the existing physics and graphics mappings without
creating them, requires live practice status, and observes advancing packet IDs.

The tools configure the virtual pad but do not create or drive one. The test
runner owns its input device and must identify its XInput slot separately.

## Euro Truck Simulator 2

Supply a disposable **local, offline test profile with a playable stock-game
save**. `profile.sii` must be plaintext `SiiNunit`, declare `is_local: true`, and
contain no account/online credentials or enabled cloud flag. Encrypted `ScsC`
metadata must be exported/decrypted before use; this tool neither downloads nor
runs a decoder. A profile created through the game's first-run wizard without a
playable save is insufficient. Cloud profile folders are refused.

```powershell
python -m tools.testrig ets2 prepare --session truck-01 --profile-source .\test-data\offline-profile --xinput-slot 0
python -m tools.testrig ets2 start --session truck-01
python -m tools.testrig ets2 stop --session truck-01
```

Preparation creates a new local `SimSteer Test` profile in a private session home,
copying only the seed's saves and game control settings. The seed is never edited.
Only the new profile gets Wheel/absolute steering (`c_relatsteer=0`), 0% steering
deadzone, the chosen XInput binding, disabled additional joystick devices, no force
feedback, and the repository's `comma_wide_cam.scs` enabled as its sole mod. The
game starts windowed, with start-in-truck enabled. The camera mod changes the
bumper view; the runner selects bumper camera (key 6) after readiness if needed.

An existing compatible SCS telemetry plugin is required. Alternatively supply a
trusted `--telemetry-plugin .\test-data\scs-telemetry.dll`; the destination DLL or
previously absent plugin directory is backed up and restored. No plugin is fetched
from the network. Readiness requires changing live ETS2 SDK time, the test profile
selection, and the repository camera mount in the new home directory's game log.

Steam must forward the isolated `-homedir` and `-profile` launch parameters. The
adapter verifies the actual process arguments and stops with a specific error if
`-homedir` is ignored. There is no direct-executable fallback. The normal Documents
and app-specific Steam userdata trees are also backed up, covering an erroneous
Steam launch before that verification. Steam launch-option/sign-in dialogs require
manual resolution outside the session. No cloud or World of Trucks sign-in is
performed, and no existing cloud profile is copied.

## Virtual display and physical cover

The VDD must already be installed and disabled. Specify its exact virtual display
adapter instance ID; the tool rejects physical adapters and already enabled
devices. The caller needs permission to enable/disable that device.

```powershell
python -m tools.testrig ac prepare --session display-01 --display vdd --vdd-device "<virtual adapter instance ID>" --width 1920 --height 1080
python -m tools.testrig ac start --session display-01
python -m tools.testrig ac stop --session display-01
```

The display worker performs this sequence:

1. Persist the original active `QueryDisplayConfig` paths/modes, physical primary,
   and display rectangles before changing any device.
2. Enable only the selected VDD with `Enable-PnpDevice` and verify its state.
3. Extend the desktop, retaining every physical display at its existing resolution
   and relative position, and use `ChangeDisplaySettingsEx(CDS_SET_PRIMARY)` to put
   the VDD at the origin as primary. Verify physical displays remain secondary.
4. Cover only the physical displays with opaque, nonactivating black windows using
   `WDA_EXCLUDEFROMCAPTURE`. The physical monitors remain powered on and awake.
5. Start the temporary windowed configuration at the requested VDD resolution,
   then place the owned simulation window borderless on that primary display.
6. On stop, failure or expiry, restore the saved topology and physical primary with
   `SetDisplayConfig`, disable the VDD, close the covers, and restore game settings.
   A failed exact restore also attempts physical-primary recovery, and cleanup
   attempts every independent step. Failed steps remain explicit in the receipt.

`--display vdd` includes the physical overlay. `--overlay` alone covers the
existing physical displays without changing topology. Overlay exclusion requires
Windows 10 build 19041 or later. The worker verifies affinity, but the live runner
must still verify capture shows the actual game on the intended output. Do not
assume an existing capture selection follows a new primary display automatically.

The independent worker holds its startup parent's process handle. A startup crash
triggers display cleanup. After readiness it stays alive until stop, game exit or
its bounded `--max-seconds` lifetime (default/max 1800). On an unattended failure
it attempts game/config recovery too. A killed worker or OS failure can prevent
cleanup code from running; `restore` uses the saved receipt to recover. Do not
delete its state until topology, VDD disable and file restoration all verify.

## Recovery and verification

State defaults to `SimSteer/testrig` under LocalAppData; `--state-dir` selects a
different private root. A cross-process lock serializes commands. Atomic JSON
records launch intent, process identities, every backup's absence/type/hash, and
restoration progress. Backups are verified before any edit and again before
restoration. Symlinks, junctions and other reparse points are refused.

Restoration preserves absent paths, empty directories and file bytes, checks for
remaining game writers and changing Steam files, verifies SHA-256 hashes, and
checks them again after a settling interval. If another writer is active, a backup
is corrupt, or a display operation cannot be verified, retain the journal and
resolve the stated failure before retrying `restore`. It never overwrites the
original baseline with a later test configuration.

Linux tests exercise discovery, journals, guards, launch ownership, readiness,
profile isolation and display ordering/failure recovery using fakes. They do not
establish that a particular Windows driver, Steam client, game build or capture
backend works. `scripts/check_windows_imports.py` also imports all test-rig modules
on Linux without invoking a Windows API.

New games should add a `GameSpec` and an adapter implementing validation, backup
targets, preparation, launch arguments, process validation and scenario readiness.
Forza Horizon 5 and Le Mans Ultimate are guard entries, not launchable adapters yet.

API reference: [SetWindowDisplayAffinity](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowdisplayaffinity),
[ChangeDisplaySettingsEx](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-changedisplaysettingsexw),
[SetDisplayConfig](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setdisplayconfig).
