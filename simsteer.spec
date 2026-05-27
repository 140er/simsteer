# PyInstaller spec for SimSteer.
#
# Build:
#   .venv\Scripts\pyinstaller --noconfirm simsteer.spec
#
# Output: dist\SimSteer\ (onedir) — ship as a zip.
#
# Design notes:
# - Onedir (NOT onefile) because onefile re-extracts ~150 MB to %TEMP%
#   on every launch (cold-start hit) and is more prone to AV false
#   positives. Onedir = one folder + a shortcut.
# - Two exes from one Analysis: SimSteer.exe (windowed, release) and
#   SimSteer-debug.exe (console, shows tracebacks for support).
# - Models are bundled (~60 MB). Network download on first run is a
#   worse UX than the extra disk.
# - NO per-user JSON state is bundled — `pilot/paths.py:data_dir()`
#   points at %LOCALAPPDATA%\SimSteer\ when frozen and is empty on
#   first run by design (user calibrates from scratch).
# - SCS plugin DLL ships only if `prereqs/scs-telemetry.dll` is present
#   (it's built from the .scs-sdk-plugin/ submodule, not included in
#   the source tree). If missing, the preflight ETS2 install dialog
#   falls back to a download link.

# noinspection PyUnresolvedReferences
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import os

block_cipher = None

# Collect onnxruntime's native DLLs (incl. DirectML.dll for AMD/Intel
# GPU vision). The default hook covers most of these; we add explicit
# collection as a belt-and-suspenders measure.
ort_binaries = collect_dynamic_libs('onnxruntime')

# vgamepad ships ViGEmClient.dll inside its own package (under
# vgamepad/win/vigem/client/x64/). PyInstaller's default scan misses it
# because vgamepad loads it via ctypes.CDLL at runtime, not at import.
# `collect_data_files` walks the package and bundles non-Python files.
vgamepad_data = collect_data_files('vgamepad')

# Data files. Each tuple = (glob pattern relative to spec dir, dest dir
# inside the bundle). PyInstaller filters non-matching paths.
datas = [
    ('models/driving_vision.onnx',  'models'),
    ('models/driving_policy.onnx',  'models'),
    ('assets/.keep',                'assets'),
    ('prereqs/.keep',               'prereqs'),
    # Driver install/uninstall helpers shipped at the bundle root so an
    # end user can set up ViGEm/vJoy WITHOUT the source repo. The .bat
    # wrappers are the double-clickable entry points (a bare .ps1 opens
    # in Notepad); they find the .ps1 next to themselves in the bundle.
    ('tools/install_drivers.ps1',   '.'),
    ('tools/uninstall_drivers.ps1', '.'),
    ('install-drivers.bat',         '.'),
    ('uninstall-drivers.bat',       '.'),
]

# Only ship the SCS plugin DLL if it's been built locally first.
_scs_dll = os.path.join('prereqs', 'scs-telemetry.dll')
if os.path.exists(_scs_dll):
    datas.append((_scs_dll, 'prereqs'))

# Ship doc screenshots if present (referenced by preflight dialogs).
for _doc in ('forza-data-out.png', 'ets2-deadzone.png',
             'ac-content-manager.png'):
    _p = os.path.join('docs', _doc)
    if os.path.exists(_p):
        datas.append((_p, 'docs'))


a = Analysis(
    ['pilot/__main__.py'],
    pathex=['.'],
    binaries=ort_binaries,
    datas=datas + vgamepad_data,
    hiddenimports=[
        'vgamepad',
        'onnxruntime',
        'onnxruntime.capi._pybind_state',
        # Tk dialogs used by preflight.
        'tkinter', 'tkinter.messagebox',
        # winsound is a stdlib C module on Windows.
        'winsound',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Trim the bundle — we don't ship these.
        'matplotlib', 'pytest', 'IPython', 'jupyter', 'pandas',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Release exe — windowed (no console).
exe_release = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SimSteer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                         # UPX often trips AV; not worth it
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',         # uncomment once an icon is added
)

# Debug exe — console (tracebacks visible). Same payload, different
# entry stub.
exe_debug = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SimSteer-debug',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',
)

coll = COLLECT(
    exe_release,
    exe_debug,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='SimSteer',
)
