"""Entrypoint for `python -m simsteer` and for PyInstaller.

Mirrors `pilot/__main__.py`: gives PyInstaller a stable module path to
bundle, and installs a launch-log fallback + crash trap so the windowed
exe (SimSteer.exe, console=False, where Windows routes stdout to NUL)
never dies silently. A startup crash lands in
`%LOCALAPPDATA%\\SimSteer\\launch.log` and pops a Tk error dialog.
"""

import sys
import traceback


class _Tee:
    """Write to two streams — keeps console output in the debug exe
    while also landing a copy in launch.log."""

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
    if not getattr(sys, "frozen", False):
        return
    try:
        from simsteer.paths import data_dir
        log_path = data_dir() / "launch.log"
        f = open(log_path, "w", encoding="utf-8", errors="replace",
                 buffering=1)
        sys.stdout = _Tee(sys.stdout, f)
        sys.stderr = _Tee(sys.stderr, f)
        print(f"[launch] logging to {log_path}")
    except Exception:
        pass


def _main() -> int:
    _install_log_fallback()
    try:
        from simsteer.app.main import main
        return main()
    except SystemExit:
        raise
    except BaseException:
        print("[launch] FATAL - uncaught exception:")
        traceback.print_exc()
        try:
            import tkinter as tk
            from tkinter import messagebox
            from simsteer.paths import data_dir
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "SimSteer - crashed at startup",
                "Uncaught exception during launch.\n\n"
                f"Full traceback at:\n  {data_dir() / 'launch.log'}\n\n"
                "Run SimSteer-debug.exe (console) to see live output.")
            root.destroy()
        except Exception:
            pass
        return 3


if __name__ == "__main__":
    raise SystemExit(_main())
