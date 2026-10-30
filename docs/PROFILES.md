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
| `tested` | bool | `false` | Someone has driven this game with SimSteer. `auto` falls back to the first tested profile when no known game runs. |
| `steam_app_id` | integer | none | The Steam app id, for finding the install folder. Not read yet. |

### `[telemetry]`

| Key | Type | Meaning |
|---|---|---|
| `kind` | string | The adapter: `scs_shm` (ETS2 and ATS shared memory through the scs-telemetry plugin) or `none` (no telemetry). |

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
