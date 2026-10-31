"""Steam KeyValues and path discovery; registry access stays in the backend."""

from dataclasses import dataclass
from pathlib import Path
import re

from . import RigError


def read_vdf(text: str) -> dict:
    # Steam emits quoted strings, braces and // comments. Do not use
    # unicode_escape: it corrupts non-ASCII library and redirected folder names.
    tokens = []
    pattern = re.compile(r'\s+|//[^\n]*|"((?:\\.|[^"\\])*)"|([{}])')
    pos = 0
    while pos < len(text):
        match = pattern.match(text, pos)
        if match is None:
            raise RigError(f"Invalid Steam VDF near character {pos}")
        if match.group(1) is not None:
            tokens.append(re.sub(r'\\([\\"])', r'\1', match.group(1)))
        elif match.group(2):
            tokens.append(match.group(2))
        pos = match.end()
    index = 0

    def block(nested=False):
        nonlocal index
        result = {}
        while index < len(tokens):
            key = tokens[index]
            index += 1
            if key == "}":
                if not nested:
                    raise RigError("Unexpected closing brace in Steam VDF")
                return result
            if key == "{" or index >= len(tokens):
                raise RigError("Missing key/value in Steam VDF")
            value = tokens[index]
            index += 1
            if value == "}":
                raise RigError("Missing value in Steam VDF")
            value = block(True) if value == "{" else value
            if key.casefold() in result:
                raise RigError(f"Duplicate Steam VDF key: {key}")
            result[key.casefold()] = value
        if nested:
            raise RigError("Unclosed brace in Steam VDF")
        return result

    return block()


def load_vdf(path: Path) -> dict:
    try:
        return read_vdf(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as exc:
        raise RigError(f"Cannot read Steam metadata {path}: {exc}") from exc


def libraries(steam: Path) -> list[Path]:
    paths = [steam]
    metadata = steam / "steamapps/libraryfolders.vdf"
    if metadata.exists():
        for key, value in load_vdf(metadata).get("libraryfolders", {}).items():
            if key.isdigit():
                raw = value.get("path") if isinstance(value, dict) else value
                if not isinstance(raw, str) or not raw:
                    raise RigError("Steam library entry has no path")
                path = Path(raw)
                if not path.is_absolute():
                    raise RigError("Steam library path is not absolute")
                if path not in paths:
                    paths.append(path)
    return paths


@dataclass(frozen=True)
class GameSpec:
    appid: str
    documents: str
    executables: tuple[str, ...]
    runner: str


# New games supply an adapter as well as a spec; there is no unsafe generic launch.
GAMES = {
    "ac": GameSpec("244210", "Assetto Corsa",
                   ("AssettoCorsa.exe", "acs.exe", "acs_x86.exe"), "acs.exe"),
    "ets2": GameSpec("227300", "Euro Truck Simulator 2",
                     ("bin/win_x64/eurotrucks2.exe",), "bin/win_x64/eurotrucks2.exe"),
}


@dataclass(frozen=True)
class Installation:
    steam: Path
    game: Path
    documents: Path
    userdata: Path

    def as_dict(self):
        return {key: str(value) for key, value in vars(self).items()}

    @classmethod
    def from_dict(cls, row):
        return cls(**{key: Path(row[key]) for key in ("steam", "game", "documents", "userdata")})


def discover(game: str, backend) -> Installation:
    spec = GAMES[game]
    roots = backend.steam_paths()
    if not roots:
        raise RigError("Steam installation not found in the Windows registry")
    for steam in roots:
        if not (steam / "steam.exe").is_file():
            continue
        for library in libraries(steam):
            manifest = library / "steamapps" / f"appmanifest_{spec.appid}.acf"
            if not manifest.exists():
                continue
            app = load_vdf(manifest).get("appstate", {})
            name = app.get("installdir", "")
            if (app.get("appid") != spec.appid or not name or
                    name in (".", "..") or re.search(r'[/\\:]', name)):
                raise RigError(f"Invalid install directory/app ID in {manifest}")
            installed = library / "steamapps/common" / name
            try:
                complete = bool(int(app.get("stateflags", "0")) & 4)
            except ValueError:
                complete = False
            if not complete or not (installed / spec.runner).is_file():
                raise RigError(f"{game}: Steam installation is incomplete; finish installation in Steam")
            user = backend.steam_user()
            if not user or not str(user).isdigit() or int(user) == 0:
                raise RigError("Steam has no active signed-in user; sign in manually before a test")
            return Installation(steam, installed, backend.documents() / spec.documents,
                                steam / "userdata" / str(user) / spec.appid)
    raise RigError(f"{game}: not installed or not owned by this Steam client (app {spec.appid}); "
                   "install and verify ownership in Steam manually")
