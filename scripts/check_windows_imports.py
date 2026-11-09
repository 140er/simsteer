"""Windows driver imports for scripts/check.ps1.

dxcam builds its DXGI output list at import time. In a session without a
desktop (ssh, services) DXGI raises COMError DXGI_ERROR_NOT_CURRENTLY_AVAILABLE,
so that case is reported, not failed. Any other error still fails the check.
"""
import importlib
from pathlib import Path
import pkgutil
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tools.testrig

    for module in pkgutil.walk_packages(tools.testrig.__path__, "tools.testrig."):
        importlib.import_module(module.name)
    print("imported: tools.testrig (no Windows effects at import)")
    if sys.platform != "win32":
        print("Windows native driver imports skipped on this platform")
        return
    import vgamepad  # noqa: F401

    print("imported: vgamepad")
    try:
        import dxcam  # noqa: F401
    except Exception as e:
        if type(e).__name__ != "COMError":
            raise
        print(f"dxcam: needs a desktop session, not importable here ({e})")
    else:
        print("imported: dxcam")


if __name__ == "__main__":
    main()
