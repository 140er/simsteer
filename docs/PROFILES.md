# Game profiles

A game is one TOML file. SimSteer reads the built-in profiles from
`simsteer/games/profiles/` and yours from `%LOCALAPPDATA%\SimSteer\games\`
(or `$SIMSTEER_DATA_DIR/games/`). A profile of yours with the same `id` as a
built-in one replaces it.

```
python -m simsteer --list-games
```

prints every profile. A file with a mistake is skipped, and the error names
the file and the key, for example
`games\mygame.toml: liveparams.gamepad.sed: unknown key`. The other
profiles still load.

`--game auto` selects a running game's profile. If none is running, it
uses the last game selected for a live session, then ETS2. Replays use
the clip's game unless `--game` overrides it, and do not change the
remembered game.

## AC and Forza setup

| Profile | Telemetry setup | Status |
|---|---|---|
| `ac` | Enter a driving session. AC publishes its physics page without a plugin. | Ported from the pilot adapter. |
| `acc` | Enter a driving session. Uses the same physics page as AC. | Untested. |
| `fh5` | Settings > HUD and Gameplay: enable Data Out, send to the local loopback address on port `7777`, and choose Dash if a packet format option is present. | Ported from the pilot adapter. |
| `fh4` | Use the same Data Out settings as FH5. | Untested. |

For Forza, `forza_port` in `settings.json` overrides the profile's
`telemetry.port`. Without that setting, the profile chooses the port.
Only one telemetry app can listen on a port at a time. The reader keeps
one socket through menus and pauses and resumes when packets return.

AC and Forza derive the road-wheel angle from yaw rate and speed. They
keep the pilot's loose fitting gates and steering seeds for both output
devices. ACC and FH4 still need a driving check before being marked
tested.

To exercise an adapter without a game, run a synthetic sender and replay
a clip using live telemetry:

```sh
python tools/fake_telemetry.py forza --port 7777
python -m simsteer --game fh5 --source clips/drive1 --live-telemetry --headless
```

The `ac` and `ets2` senders require Windows; the `forza` sender also
works on Linux. `--format alternate` switches between 324-byte Horizon
and 311-byte Dash packets. `--gap 10:20` pauses transmission for ten
seconds. Replay keeps device output disabled, including with live
telemetry. Without `--live-telemetry`, replay reads the clip's recorded
telemetry and stays deterministic.

## Add a game

1. Copy a built-in profile whose telemetry your game shares, or start from
   the minimal one below, into `%LOCALAPPDATA%\SimSteer\games\<id>.toml`.
2. Set `id`, `display_name` and `process_names`.
3. Run `SimSteer-debug.exe --list-games` and check your game is listed.
4. Start the game, then SimSteer with `--game <id>` or `--game auto`.

A game with a telemetry format SimSteer cannot read yet needs a new adapter
in `simsteer/runtime/telemetry/`, added to `KINDS` there. Everything else is
data.

## Minimal profile

```toml
id = "mygame"
display_name = "My Game"
process_names = ["MyGame.exe"]

[telemetry]
kind = "scs_shm"
```

## Full example

```toml
id = "ets2"
display_name = "Euro Truck Simulator 2"
process_names = ["eurotrucks2.exe"]
steam_app_id = 227300
tested = true

[telemetry]
kind = "scs_shm"

[controller]
wheelbase_m = 4.0
steer_authority = 1.0
lookahead_s = 0.3
curvature_anticipation_s = 0.0

[camera]
fov_h_deg = 90.0
hint = "Interior cab view"

[capabilities]
longitudinal = true
speed_source = "telemetry"
synth_wheel_angle = false
passive_fit_safe = false

[liveparams.gates]
profile = "strict"

[liveparams.gamepad]
seed = [3.5, 0.0015, 0.0]

[liveparams.wheel]
seed = [4.0, 0.0, 0.0]

[[setup]]
check = "scs_plugin"

[[setup]]
check = "camera_fov"
where = "Options > Gameplay > Camera"
```

## Keys

Only `id`, `display_name` and `telemetry.kind` are required. Every key not
listed here is an error.

### Top level

| Key | Type | Default | Meaning |
|---|---|---|---|
| `id` | string | required | Lowercase letters, digits, `-` and `_`. Names the state files (`calibration_<id>.json`). Not `auto` or `default`. |
| `display_name` | string | required | Shown in the tuner, the window title and `--list-games`. |
| `process_names` | list of strings | `[]` | Executable names. `--game auto` drives the first profile with one of them running. Case does not matter. |
| `tested` | bool | `false` | Someone has driven this game with SimSteer. Shown by `--list-games`. |
| `steam_app_id` | integer | none | The Steam app id, for finding the install folder. Not read yet. |

### `[telemetry]`

| Key | Type | Meaning |
|---|---|---|
| `kind` | string | The adapter: `scs_shm` (ETS2 and ATS), `ac_shm` (AC and ACC), `forza_udp` (Forza Data Out), or `none`. |
| `port` | integer | `forza_udp` only: UDP listen port, default `7777`. The `forza_port` setting overrides it. |
| `host` | string | `forza_udp` only: local interface to bind. Omit to listen on all interfaces. |

Other keys in this table go to the adapter, and only the ones it declares
are allowed.

### `[controller]`

Starting values for the controller config, used until SimSteer has saved
`controller_<id>.json` for the game. Each is optional and falls back to the
built-in default.

| Key | Meaning |
|---|---|
| `wheelbase_m` | Front to rear axle, metres. Turns curvature into wheel angle. |
| `steer_authority` | Multiplier on the planned curvature. |
| `steer_max` | Cap on the steering axis. |
| `lookahead_s` | Steering actuator delay, seconds. |
| `curvature_anticipation_s` | Extra plan lookahead, seconds. |

### `[camera]`

| Key | Type | Meaning |
|---|---|---|
| `fov_h_deg` | number | Horizontal FOV seed, 10 to 170, used until `calibration_<id>.json` exists. |
| `hint` | string | The camera view to drive in. Not read yet. |
| `[camera.fov_source]` | table | Where the game saves its FOV: `file`, `key` and `axis` (`horizontal` or `vertical`). Not read yet. |

### `[capabilities]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `passive_fit_safe` | bool | `false` | LiveParams may learn while disengaged. Otherwise only with `--passive-fit`. |
| `longitudinal` | bool | `true` | SimSteer drives throttle and brake. Not read yet. |
| `speed_source` | string | `telemetry` | `telemetry` or `model`. Not read yet. |
| `synth_wheel_angle` | bool | `false` | The adapter derives the wheel angle from yaw rate. Not read yet. |

### `[liveparams]`

| Key | Meaning |
|---|---|
| `[liveparams.gates] profile` | `strict` (games that report the wheel angle) or `loose` (games where it is derived from yaw rate). |
| `[liveparams.gamepad] seed`, `[liveparams.wheel] seed` | `[a, b, c]` in `axis = a*wheel + b*wheel*v^2 + c`, the starting steering fit for that device until `liveparams_<id>_<device>.json` exists. Without one, `[10, 0, 0]`. |

### `[[setup]]`

One table per setup check: `check` names it, any other keys are its
settings (strings, numbers or booleans). Not read yet.
