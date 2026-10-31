"""Game-specific disposable configuration. No account or driver operations."""

import configparser
import io
import math
import ntpath
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
import zipfile

from . import RigError
from .discovery import GAMES
from .state import fingerprint, reject_links


def component(value, label, *, empty=False):
    if (empty and value == "") or re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    raise RigError(f"Invalid {label}; use a single directory name")


def ini(path, changes, journal, *, seed=None, replace=False):
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    source = path if path.exists() else seed
    if not replace and source and source.exists():
        parser.read_string(source.read_text(encoding="utf-8-sig"))
    for section, key, value in changes:
        if not parser.has_section(section):
            parser.add_section(section)
        parser.set(section, key, str(value))
    stream = io.StringIO()
    parser.write(stream, space_around_delimiters=False)
    journal.write(path, stream.getvalue().encode("utf-8"))


def uset(path, values, journal):
    text = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    for key, value in values.items():
        pattern = rf"(?m)^uset {re.escape(key)} .*?$"
        line = f'uset {key} "{value}"'
        text = re.sub(pattern, lambda _: line, text) if re.search(pattern, text) else text + "\n" + line + "\n"
    journal.write(path, text.encode("utf-8"))


class AssettoCorsa:
    files = ("controls.ini", "video.ini", "camera_onboard.ini", "camera_manager.ini", "assists.ini", "race.ini")
    def validate(self, installed, options):
        track, car = component(options["track"], "track"), component(options["car"], "car")
        layout = component(options["layout"], "layout", empty=True)
        track_path = installed.game / "content/tracks" / track
        if not track_path.is_dir() or (layout and not (track_path / layout).is_dir()):
            raise RigError(f"AC track/layout is not installed: {track}/{layout}")
        car_path = installed.game / "content/cars" / car
        if not car_path.is_dir() or not (installed.game / "AssettoCorsa.exe").is_file():
            raise RigError("AC car or Steam launcher is not installed")

    def targets(self, installed, options):
        return [folder / name for folder in (installed.documents / "cfg", installed.game / "cfg")
                for name in self.files]

    def prepare(self, installed, directory, options, journal):
        cfg = installed.documents / "cfg"
        controls = [
            ("HEADER", "INPUT_METHOD", "X360"), ("STEER", "JOY", -1),
            ("STEER", "AXLE", -1), ("STEER", "FF_GAIN", 0),
            ("X360", "STEER_THUMB", "LEFT"), ("X360", "STEER_DEADZONE", 0),
            ("X360", "STEER_GAMMA", 1), ("X360", "STEER_FILTER", 0),
            ("X360", "SPEED_SENSITIVITY", 0), ("X360", "STEER_SPEED", 1),
            ("X360", "RUMBLE_INTENSITY", 0),
        ]
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(options["hfov"]) / 2) /
                                        (options["width"] / options["height"])))
        # AC reads some defaults from the install tree; both trees are backed up.
        for folder in (cfg, installed.game / "cfg"):
            seed = lambda name: installed.game / "cfg" / name
            ini(folder / "controls.ini", controls, journal, seed=seed("controls.ini"))
            ini(folder / "video.ini", [("VIDEO", "FULLSCREEN", 0),
                ("VIDEO", "WIDTH", options["width"]), ("VIDEO", "HEIGHT", options["height"]),
                ("VIDEO", "VSYNC", 1)], journal, seed=seed("video.ini"))
            ini(folder / "camera_onboard.ini", [("MODE", "FOV", vfov)], journal, seed=seed("camera_onboard.ini"))
            ini(folder / "camera_manager.ini", [("LAST_CAMERA", "DRIVABLE_MODE", 0)], journal, seed=seed("camera_manager.ini"))
            ini(folder / "assists.ini", [("ASSISTS", key, value) for key, value in
                (("AUTO_SHIFTER", 1), ("AUTO_CLUTCH", 1), ("AUTO_BLIP", 1),
                 ("AUTO_BRAKE", 0), ("DAMAGE", 0), ("FUEL_RATE", 0))], journal, seed=seed("assists.ini"))
        skins = installed.game / "content/cars" / options["car"] / "skins"
        choices = sorted(item.name for item in skins.iterdir() if item.is_dir()) if skins.exists() else []
        if not choices:
            raise RigError("AC car has no installed skin")
        race = {
            "HEADER": {"VERSION": 1},
            "RACE": {"TRACK": options["track"], "CONFIG_TRACK": options["layout"],
                     "MODEL": options["car"], "MODEL_CONFIG": "", "CARS": 1,
                     "AI_LEVEL": 98, "FIXED_SETUP": 0, "PENALTIES": 0},
            "CAR_0": {"MODEL": "-", "MODEL_CONFIG": "", "SKIN": choices[0],
                      "DRIVER_NAME": "SimSteer Test", "NATIONALITY": "", "NATION_CODE": ""},
            "SESSION_0": {"NAME": "Practice", "TYPE": 1, "DURATION_MINUTES": 0, "SPAWN_SET": "PIT"},
            "REMOTE": {"ACTIVE": 0}, "REPLAY": {"ACTIVE": 0, "FILENAME": ""},
            "BENCHMARK": {"ACTIVE": 0}, "GHOST_CAR": {"RECORDING": 0, "PLAYING": 0},
            "LIGHTING": {"SUN_ANGLE": -48, "TIME_MULT": 1},
            "TEMPERATURE": {"AMBIENT": 26, "ROAD": 34}, "WEATHER": {"NAME": "4_mid_clear"},
            "DYNAMIC_TRACK": {"SESSION_START": 100, "SESSION_TRANSFER": 100, "RANDOMNESS": 0},
        }
        # Replace, rather than merge, so a user's SESSION_1 or online parameters
        # cannot leak into this practice run.
        for folder in (cfg, installed.game / "cfg"):
            ini(folder / "race.ini", [(section, key, value) for section, values in race.items()
                                      for key, value in values.items()], journal, replace=True)

    def launch_uri(self, installed, directory):
        return "steam://rungameid/244210"

    def verify_process(self, process, installed, directory, backend):
        pass

    def scenario_ready(self, installed, directory):
        return True


def profile_text(source):
    try:
        text = (source / "profile.sii").read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise RigError("ETS2 needs a local offline seed with plaintext profile.sii; "
                       "export/decrypt the disposable seed outside this tool") from exc
    if not text.startswith("SiiNunit") or not re.search(r'(?m)^\s*profile_name\s*:', text):
        raise RigError("ETS2 seed profile.sii must be plaintext SiiNunit profile metadata")
    if not re.search(r'(?m)^\s*is_local\s*:\s*(?:true|1)\s*$', text):
        raise RigError("ETS2 seed must declare is_local: true; Steam Cloud profiles are refused")
    for key, value in re.findall(r'(?m)^\s*([\w]+)\s*:\s*([^\r\n]*)', text):
        if re.search(r"online|wotr|account|password|token|email", key, re.I):
            if value.strip() not in ('""', "0", "false", "nil"):
                raise RigError("ETS2 seed contains account/online data; provide a clean offline test profile")
        if key in ("steam_cloud", "cloud_enabled") and value.strip() not in ("false", "0"):
            raise RigError("ETS2 seed has Steam Cloud enabled; provide a local offline test profile")
    return text


class EuroTruckSimulator:
    name = "SimSteer Test"
    camera = Path(__file__).resolve().parents[2] / "comma_wide_cam.scs"

    def validate(self, installed, options):
        raw = options.get("profile_source")
        if not raw:
            raise RigError("ETS2 prepare requires --profile-source pointing to a disposable LOCAL offline "
                           "profile with a playable save and plaintext profile.sii")
        source = Path(raw)
        if any("steam_profiles" in part.lower() for part in source.parts):
            raise RigError("ETS2 cloud profiles are refused; supply a LOCAL offline test profile")
        fingerprint(source)
        profile_text(source)
        if not (source / "controls.sii").is_file() or not list(source.glob("save/*/game.sii")):
            raise RigError("ETS2 local seed needs controls.sii and a playable save/*/game.sii")
        with zipfile.ZipFile(self.camera) as archive:
            if b"mod_package : .package_name" not in archive.read("manifest.sii"):
                raise RigError("Repository camera archive has no valid package manifest")
        target = installed.game / "bin/win_x64/plugins/scs-telemetry.dll"
        plugin = options.get("telemetry_plugin")
        if plugin:
            if Path(plugin).suffix.lower() != ".dll" or not Path(plugin).is_file():
                raise RigError("--telemetry-plugin must name an existing trusted SCS telemetry DLL")
            reject_links(Path(plugin))
        elif not target.is_file():
            raise RigError("ETS2 SCS telemetry plugin is missing; supply a trusted --telemetry-plugin DLL")

    def targets(self, installed, options):
        targets = []
        if options.get("telemetry_plugin"):
            targets.append(installed.game / "bin/win_x64/plugins/scs-telemetry.dll")
        return targets

    def prepare(self, installed, directory, options, journal):
        home = directory / "ets2-home"
        if home.exists():
            raise RigError("ETS2 test home already exists; restore the interrupted session before preparing again")
        profile = home / "profiles" / self.name.encode("utf-8").hex().upper()
        source = Path(options["profile_source"])
        source_hashes = fingerprint(source)
        # Copy only game data from an explicit seed; never copy global Steam or
        # online configuration, cloud profiles, mods or credential stores.
        for file in sorted((source / "save").rglob("*")):
            if file.is_file():
                journal.write(profile / file.relative_to(source), file.read_bytes())
        for name in ("controls.sii", "config.cfg", "config_local.cfg"):
            if (source / name).is_file():
                journal.write(profile / name, (source / name).read_bytes())
        text = profile_text(source)
        text = re.sub(r'(?m)^(\s*profile_name\s*:)\s*.*$', rf'\1 "{self.name}"', text)
        text = re.sub(r'(?m)^\s*(?:active_mods|active_packages)(?:\[\d+\])?\s*:.*\n?', '', text)
        text = re.sub(r'(?m)^\s*(?:is_local|steam_cloud|cloud_enabled)\s*:.*\n?', '', text)
        end = text.rfind("}", 0, text.rfind("}"))
        if end < 0:
            raise RigError("ETS2 seed has invalid profile metadata")
        text = (text[:end] + '\n is_local: true\n steam_cloud: false\n active_mods: 1\n'
                ' active_mods[0]: "mod_package.comma_wide_cam|Comma Wide Cam (bumper 120deg/y+1m)"\n' + text[end:])
        journal.write(profile / "profile.sii", text.encode("utf-8"))
        controls = (profile / "controls.sii").read_text(encoding="utf-8-sig")
        changes = {"device joy": f'device joy `xinput.xinput_gamepad_{options["xinput_slot"] + 1}`',
                   "constant c_steer_dz": "constant c_steer_dz 0.000000",
                   "constant c_relatsteer": "constant c_relatsteer 0.000000",
                   "constant c_ff_enabled": "constant c_ff_enabled 0.000000",
                   "input j_steer": "input j_steer `joy.x`", "input j_throttle": "input j_throttle `joy.rt`",
                   "input j_brake": "input j_brake `joy.lt`"}
        for index in range(2, 7):
            changes[f"device joy{index}"] = f"device joy{index} ``"
        for key, value in changes.items():
            pattern = r'(?m)(config_lines\[\d+\]:\s*")' + re.escape(key) + r' [^"\r\n]*"'
            controls, count = re.subn(pattern, lambda match: match[1] + value + '"', controls)
            if count != 1:
                raise RigError(f"ETS2 seed controls must contain exactly one {key} entry")
        journal.write(profile / "controls.sii", controls.encode("utf-8"))
        uset(profile / "config.cfg", {"g_park_brake_init": 1}, journal)
        uset(profile / "config_local.cfg", {"g_trans": 1}, journal)
        uset(home / "config.cfg", {"r_fullscreen": 0, "r_mode_width": options["width"],
             "r_mode_height": options["height"], "r_vsync": 1, "g_start_in_truck": 1,
             "g_last_seen_profile": self.name, "g_save_format": 2, "g_online_loading_screens": 0}, journal)
        journal.write(home / "mod/comma_wide_cam.scs", self.camera.read_bytes())
        if options.get("telemetry_plugin"):
            plugin = installed.game / "bin/win_x64/plugins/scs-telemetry.dll"
            journal.write(plugin, Path(options["telemetry_plugin"]).read_bytes())
        if fingerprint(source) != source_hashes:
            raise RigError("ETS2 source profile changed while copying; preparation refused")

    def launch_uri(self, installed, directory):
        args = subprocess.list2cmdline(["-nointro", "-homedir", str(directory / "ets2-home"), "-profile", self.name])
        return "steam://rungameid/227300//" + quote(args, safe="") + "/"

    def verify_process(self, process, installed, directory, backend):
        actual = backend.argument(process, "-homedir")
        expected = str(directory / "ets2-home")
        if actual is None or ntpath.normcase(ntpath.normpath(actual)) != ntpath.normcase(ntpath.normpath(expected)):
            raise RigError("Steam did not forward the isolated ETS2 -homedir argument; stopping the launch. "
                           "No direct-executable fallback is attempted")

    def scenario_ready(self, installed, directory):
        log = directory / "ets2-home/game.log.txt"
        if not log.exists():
            return False
        text = log.read_text(encoding="utf-8", errors="replace")
        return (f"New profile selected: '{self.name}'" in text and
                bool(re.search(r"comma_wide_cam.*(?:Mounted|mounted)|(?:Mounted|mounted).*comma_wide_cam", text)))


ADAPTERS = {"ac": AssettoCorsa(), "ets2": EuroTruckSimulator()}


def expected_paths(game, installed):
    return [str(installed.game / item) for item in GAMES[game].executables]
