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
`prepare` is useful for reviewing the configuration before launch. Repeat it
with the same options to check an active preparation. A finished session refuses
preparation; use a new `--session`. `ready` requires a
visible owned game window and multiple changing shared-memory packets; an open
launcher, paused simulation, or a stale mapping is insufficient. Its default
deadline is 180 seconds, with a maximum of 600 seconds and no automatic retries.

Both `stop` and `restore` close the session's processes, clean up its display,
and restore its settings. Repeat them after interruption to resume cleanup.
A completed session never rolls back later user edits. Use a new `--session`
name for the next test.

Exit code 0 means the requested operation completed. Code 2 means refusal or
failure; stderr explains the blocked step. JSON on stdout includes the phase and
state file location. An otherwise completed restore with owner conflicts also
returns 2, with phase `restored-with-conflicts` and the conflicting paths on
stderr. Retrying preserves that result; inspect the retained originals rather
than assuming that all settings were restored. Keep state, backups, logs and
reports private.

## Guard and ownership

Both `python -m tools.testrig guard` and `<game> guard` are accepted. The guard
refuses known games/VR apps, an unowned borderless fullscreen foreground window,
active capture signals, or user activity within the last 120 seconds
(`--idle-seconds` can select 60–3600). It checks the current interactive console
session and refuses service or remote desktops. Keep remote automation input-free
while the owner is away. The guard runs before backups, before edits, and again
immediately before Steam launch. Readiness continues checking unrelated software
and pending restarts, while allowing interaction with the owned test game.

Restart checks include Windows Update `RebootRequired`, component servicing
`RebootPending`, update reboot indicators/deadlines, and running or imminent
UpdateOrchestrator reboot tasks (within 15 minutes). `PendingFileRenameOperations`
alone and more distant scheduled tasks produce warnings. Missing,
stale, malformed or access-denied inventory refuses startup. Nothing clears these
indicators automatically.

Idle NVIDIA overlays, GameBar, the Oculus service, desktop streaming services and
Steam-installed tools such as Wallpaper Engine do not block merely by being
resident. Owned game helper ancestry is recognized too. Active-output checks use
visible OBS output controls/current OBS logs, Steam streaming logs/helpers,
Sunshine client connections, compatible Parsec logs, and visible sharing controls
where available. Idle OBS is allowed.

Steam Remote Play log events must belong to the running Steam process. Older
events are ignored; the latest `Client version` marker provides the boundary if
process birth is unavailable. Explicit stream starts activate the log signal;
a later session end/stop clears it. Audio diagnostics and disabled game-recording
messages do not start a stream. Unparseable timestamps need a current streaming
process or visible output control to corroborate them. Streaming helpers and other
positive capture signals still refuse independently.

Capture coverage is:

| Tool | Signals read |
| --- | --- |
| OBS Studio | Visible output controls; current standard or portable log for recording, streaming, replay buffer and virtual camera; media-file growth in configured profile output folders |
| Streamlabs | Visible output controls; current `logs` or `node-obs/logs` output log; configured output folders from compatible profile INI or `settings.json` output-path keys |
| NVIDIA Share/Overlay and capture helper | Visible recording/streaming indicators; process encoder and CPU counters |
| Game Bar and `bcastdvr` | Visible recording indicators; explicit record command flags; process encoder and CPU counters, including the running capture service's PID (service presence alone is allowed) |
| Discord/Teams | Visible Go Live/sharing/stop-sharing controls; process encoder and CPU counters |
| Any other process using a hardware encoder | VideoEncode GPU activity, including encoder work attributed to a container/helper process |

For those capture processes, positive output controls/log events, growing media
files or VideoEncode GPU activity refuse startup. When output state is unreadable,
explicit recording command flags or CPU usage of at least 25% of one core also
refuse. Unreadable output state without active signs produces a warning, continues,
and records the warning in session status. Hardware-encoder activity is collected
for all processes, including unknown recorder names. Unowned encoder activity with
unreadable output state refuses with the process name, including background replay
capture. Verified session-owned processes are exempt. An open app alone never
proves capture.
CPU/GPU counters, custom paths, localized or custom-rendered controls, and hidden
sharing UI may be unavailable. These are supported signals, not a guarantee that
every recording mode is observable; verify them before live use.

A maximized browser/editor with a caption is allowed by the unknown-game
heuristic, including on a monitor without a taskbar. Known game names still block;
the shell's exclusive Direct3D fullscreen indication also blocks. Unknown borderless
fullscreen apps remain conservative refusals. `GetLastInputInfo` does not cover
wheel or pad input.

Launch uses `steam://rungameid/<appid>`. Ownership is recorded before checking
readiness: only newly created expected executables, with matching paths, creation
times, a recorded launch window and Steam/owned-launcher ancestry are adopted.
AC's directly started practice simulation is recorded against its launching
process as well. Cleanup verifies creation time and path through the same process
handle used to close/terminate it. It never kills by image name or closes Steam.
An unowned game process blocks restoration.

Standard named sign-in, ownership, EULA, firewall and Steam launch-confirmation dialogs produce distinct
errors. Custom-rendered or localized game dialogs may instead produce the bounded
readiness timeout; the tool does not guess at buttons or enter credentials.
Steam launch confirmation requires explicit confirmation text or a known dialog
class/title combination. Normal owned `Launching...`, `Preparing to launch` and
Steam notification windows are allowed; an owner HWND alone is not a prompt.

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
API, including redirected folders. It backs up only the six `.ini` files it edits
in each of Documents `cfg` and installation `cfg`. Replays, setups, other files and
Steam userdata are never backed up, deleted or rolled back.

The temporary configuration selects X360 input, zero steering deadzone/filter,
gamma 1, zero speed sensitivity, no wheel force feedback or pad rumble, automatic
clutch/shifting, and a single offline practice session. The default car/track are
Lotus Elise SC and Magione; `--layout` selects an installed track layout. The
camera is cockpit, with `--hfov` converted to vertical FOV for `--width/--height`.
Both sets of edited files are covered by the per-file manifest. Missing config
files are seeded individually; the rest of a configuration directory is not copied.

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
trusted `--telemetry-plugin .\test-data\scs-telemetry.dll`; only the destination DLL
is backed up and restored. Its parent directory is never removed. No plugin is fetched
from the network. Readiness requires changing live ETS2 SDK time, the test profile
selection, and the repository camera mount in the new home directory's game log.

Steam must forward the isolated `-homedir` and `-profile` launch parameters. The
adapter verifies the actual process arguments and stops with a specific error if
`-homedir` is ignored. There is no direct-executable fallback. The normal Documents
and Steam userdata trees are never restored: any unexpected changes made there by
Steam or the game remain visible for investigation. Validate argument forwarding
before relying on isolation. Steam launch-option/sign-in dialogs require
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
2. Enable only the selected VDD with `Enable-PnpDevice` and verify its state. Match
   its PnP instance against the CCD adapter interface path and GDI source name,
   rather than the hardware ID returned by `EnumDisplayDevices`. Multiple matching
   outputs are refused. Wait up to 10 seconds for the adapter's asynchronous CCD
   source to appear; an unavailable source fails with a bounded error.
3. Extend the desktop, retaining every physical display at its existing resolution
   and relative position, and use `ChangeDisplaySettingsEx(CDS_SET_PRIMARY)` to put
   the VDD at the origin as primary. Verify physical displays remain secondary.
4. Cover only the physical displays with opaque, nonactivating black windows using
   `WDA_EXCLUDEFROMCAPTURE`. The physical monitors remain powered on and awake.
5. Start the temporary windowed configuration at the requested VDD resolution,
   then place the owned simulation window borderless on that primary display.
6. On stop, failure or expiry, restore the saved topology and physical primary with
   `SetDisplayConfig`, disable the VDD, close the covers, and restore game settings.
   A failed exact restore attempts physical-primary recovery. After VDD disable,
   stable monitor/adapter paths remap changed display identifiers and reapply the
   original physical rectangles, modes and primary. A verified fallback is accepted
   and reported. Cleanup attempts every independent step; unresolved failures stay
   explicit in the receipt.

`--display vdd` includes the physical overlay. `--overlay` alone covers the
existing physical displays without changing topology. Overlay exclusion requires
Windows 10 build 19041 or later. The worker verifies affinity, but the live runner
must still verify capture shows the actual game on the intended output. Do not
assume an existing capture selection follows a new primary display automatically.

An independent session worker also runs when no display option is selected. It
holds its startup parent's process handle and watches the simulation using process
handles, keeping PowerShell inventory off the overlay's message pump. A dedicated
keyboard/mouse hook thread decides owner activity and filters injected automation.
`GetLastInputInfo` provides an availability and heartbeat cross-check: a changed
input tick not seen by either hook within one second, or a stalled hook pump,
aborts as owner presence. Injected hook events participate in the heartbeat but
do not count as owner input. Real owner input closes the cover and restores the
display, then stops the owned game and recovers configuration. This does not capture
key values or pointer positions. Creating `display-stop` in the session directory
also requests complete cleanup. After readiness the worker stays alive until stop,
owner input, game exit or its bounded `--max-seconds` lifetime (default/max 1800).
A killed worker or OS failure can prevent
cleanup code from running; `restore` uses the saved receipt to recover. Do not
delete its state until topology, VDD disable and file restoration all verify.

Before `status`, `guard`, `prepare` or `start`, the CLI scans the selected state
root for incomplete VDD receipts from an earlier Windows boot. It restores those
display layouts first, disables the recorded VDD, and reports the recovery. A
failure refuses the command. A session with a live recorded worker or owned process
is always left alone, comparing PID, creation time and path, including a second
check under the command lock before recovery. Boot identity uses the boot counter
when available. The fallback requires a boot-time change exceeding five minutes
and an uptime discontinuity; clock drift alone is insufficient. Without enough
evidence, use explicit `restore` for an orphaned session. There
is no automatic startup task: after an unexpected reboot, run one of these
commands with the original state root before using the rig. This recovery restores
displays only; use the session's `restore` command for game settings.

## Recovery and verification

State defaults to `SimSteer/testrig` under LocalAppData; `--state-dir` selects a
different private root. A cross-process lock serializes commands. Atomic JSON
records launch intent, process identities, each edited file's original mode/hash,
the exact bytes intended for each write, and restoration progress. A separately
published `manifest.json` carries the same recovery information. Revision numbers
select the newest valid copy if publication was interrupted. Sharing-violation
reads/replacements retry for about two seconds. Backups are verified before edits
and restoration. Symlinks, junctions and other reparse points are refused.

Restoration operates on individual files and never deletes a game/owner folder.
After the owned writers exit, the journal seals each edited file's last session
hash. Game rewrites belong to the session only when the receipt proves monitor
coverage through writer shutdown: a recorded monitor start, clean stop reason,
coverage timestamp and no monitor errors. A live monitor can prove coverage during
autonomous shutdown. Only writes after both the tool's write and monitor start
qualify. The tool also checks every edited file when monitoring begins; an edit
between separate `prepare` and `start` commands remains a conflict even if the
game later re-saves it. Proven game writes restore to the original bytes unless
owner input was seen. A killed, missing or unproven monitor counts as owner presence;
it cannot authorize rollback of an unknown rewrite.
Crash recovery uses the recorded writer lifetime rather than extending it to the
recovery command. An uncertain failed shutdown disables rewrite attribution;
exact-hash recovery continues and other rewrites become conflicts. Once sealed,
later edits cannot be attributed to the game.

A file already matching its original hash needs no write. Other files are changed
only while their live hash matches an allowed session hash, checked again just
before replacement or unlink. If owner input was seen, a game/owner rewrite is
treated as a conflict. That file is left as is, its original backup is retained
under `displaced/`, and stop/restore returns 2 with `restored-with-conflicts`.
The displacement intent commits before moving the backup, so an interruption
resumes safely and does not block unrelated files. Backup and displaced paths are
confined to their private session subdirectories.

Changed generated files inside the disposable ETS2 home are retained with result
`left_session_file`, without an owner-conflict warning. Untracked game outputs,
saves and empty directories are retained. Temporary replacements have exact
session/row names recorded before creation. Recovery removes only those manifest
entries whose content matches their recorded hashes or the saved write intent.
A shorter prefix or bounded zero-filled torn write is cleaned automatically after
a power cut. Other changed temporaries are preserved and reported for inspection;
untracked names are never removed. The temporary payload is dropped after recovery
finishes, leaving its hashes in the journal. Older journals without a payload need
manual inspection of a torn temporary: preserve it outside the target directory,
then retry restore. Never delete by filename pattern alone.

Restored bytes are verified twice. Read-only files are handled individually and
their original mode is restored; sharing failures retry briefly, report the exact
path, and leave `cleanup-required` so other safe rows can finish and a later command
can resume. After successful verification, known backup files and the empty private
backup directory are removed; manifests, hashes, receipts and conflict backups
remain. A corrupt backup or unresolved display step requires recovery before the
session can finish. Existing managed files are replaced atomically.

For a corrupt `state.json`, `status` reports whether the valid manifest was used;
`restore` automatically uses that manifest. If both records are corrupt or from
the older folder-backup format, automatic recovery refuses. Manual recovery:

1. Preserve the entire private session directory. Close only processes whose saved
   PID, creation time and executable path can still be verified as session-owned.
2. If display recovery cannot run, use Windows Display settings to make the physical
   display primary and extend the desktop, then disable only the recorded virtual
   adapter. Keep physical monitors enabled.
3. Inspect a valid manifest or recovered metadata copy. Each `backups` row maps one
   target to its original file under `backup/` or `displaced/`, original mode/hash,
   and session write hash. Compare SHA-256 values before copying an individual file.
   Leave any later owner change alone. Remove a created file only if it still
   matches the session write. Never replace a whole Documents, profile or userdata
   tree. Do not guess a destination when its mapping is unavailable.
4. Retain evidence until file and display verification succeeds. Version 1 folder
   backups must be inspected/restored selectively by hand, never rolled back as
   whole folders.

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
