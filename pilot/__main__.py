"""Entrypoint for `python -m pilot` and for PyInstaller.

Keeps `pilot.main:main` callable as a regular function while giving
PyInstaller a stable module path to bundle (`pilot/__main__.py`).

Also installs a fallback log file when running as the windowed
PyInstaller exe (SimSteer.exe with console=False, where stdout is
routed to NUL by Windows). Without this, a startup crash or a fatal
preflight that abort()s before opening the cv2 window leaves the user
staring at nothing. The log lands at `%LOCALAPPDATA%\\SimSteer\\launch.log`
so we can read it after the fact.
"""

import sys
import traceback


class _Tee:
    """Write to two streams. Used to keep console output working in
    the debug exe while ALSO landing a copy in launch.log."""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except (OSError, ValueError):
                pass

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        for st in self._streams:
            try:
                if st.isatty():
                    return True
            except Exception:
                pass
        return False


def _install_log_fallback() -> None:
    # Only fires under PyInstaller. In dev mode we leave stdout alone.
    if not getattr(sys, "frozen", False):
        return
    try:
        from pilot.paths import data_dir
        log_path = data_dir() / "launch.log"
        f = open(log_path, "w", encoding="utf-8", errors="replace",
                 buffering=1)   # line-buffered so a crash leaves a usable log
        # Tee so the debug exe still shows live output AND we get a log.
        sys.stdout = _Tee(sys.stdout, f)
        sys.stderr = _Tee(sys.stderr, f)
        print(f"[launch] logging to {log_path}")
    except Exception:
        # Best-effort — don't crash on the way in.
        pass


def _main() -> int:
    _install_log_fallback()
    try:
        from pilot.main import main
        return main()
    except SystemExit:
        raise
    except BaseException:
        # Last-chance trap so the windowed exe doesn't disappear
        # silently. The traceback lands in launch.log (or on the
        # console in dev / debug exe).
        print("[launch] FATAL — uncaught exception:")
        traceback.print_exc()
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            from pilot.paths import data_dir
            messagebox.showerror(
                "SimSteer — crashed at startup",
                "Uncaught exception during launch.\n\n"
                f"Full traceback at:\n  {data_dir() / 'launch.log'}\n\n"
                "Re-run SimSteer-debug.exe (the console version) to "
                "see live output if you need to debug interactively.")
            root.destroy()
        except Exception:
            pass
        return 3


if __name__ == "__main__":
    raise SystemExit(_main())
