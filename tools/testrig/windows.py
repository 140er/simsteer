"""Lazy Windows effects. The CLI and all modules also import on Linux."""

import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import uuid

from . import RigError
from .capture import capture_activity, output_log_active, steam_log_active


def require_windows():
    if os.name != "nt":
        raise RigError("Test-rig operations require Windows and an interactive desktop")


def dll(name):
    require_windows()
    return C.WinDLL(name, use_last_error=True)


def signature(library, name, args, result):
    function = getattr(library, name)
    function.argtypes, function.restype = args, result
    return function


def desktop_pixels():
    """Keep window/overlay coordinates physical on mixed-DPI desktops."""
    user = dll("user32")
    try:
        aware = signature(user, "SetThreadDpiAwarenessContext", [W.HANDLE], W.HANDLE)
        if not aware(W.HANDLE(-4)):  # PER_MONITOR_AWARE_V2
            raise RigError("Cannot select per-monitor DPI awareness for display placement")
    except AttributeError as exc:
        raise RigError("Display automation requires per-monitor DPI awareness support") from exc


def powershell(action="inventory", device=None):
    require_windows()
    argv = ["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
            str(Path(__file__).with_suffix(".ps1")), "-Action", action]
    if device:
        argv += ["-DeviceId", device]
    try:
        result = subprocess.run(argv, capture_output=True, encoding="utf-8", timeout=45,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RigError(f"Windows {action} failed: {exc}") from exc
    if result.returncode:
        raise RigError(f"Windows {action} failed: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout.lstrip("\ufeff"))
    except ValueError as exc:
        raise RigError(f"Windows {action} returned invalid JSON") from exc


class ProcessHandle:
    """Validate creation time AND path through the same handle used for termination."""
    def __init__(self, row, *, terminate=False):
        self.kernel = dll("kernel32")
        self.handle = None
        access = 0x100000 | 0x1000 | (1 if terminate else 0)  # sync/query/terminate
        opened = signature(self.kernel, "OpenProcess", [W.DWORD, W.BOOL, W.DWORD], W.HANDLE)
        self.handle = opened(access, False, row["pid"])
        if not self.handle:
            if C.get_last_error() == 87:  # process already gone
                return
            raise RigError(f"Cannot open owned process {row['pid']}: {C.WinError(C.get_last_error())}")
        signature(self.kernel, "CloseHandle", [W.HANDLE], W.BOOL)
        signature(self.kernel, "WaitForSingleObject", [W.HANDLE, W.DWORD], W.DWORD)
        signature(self.kernel, "TerminateProcess", [W.HANDLE, W.UINT], W.BOOL)
        times = signature(self.kernel, "GetProcessTimes", [W.HANDLE] + [C.POINTER(W.FILETIME)] * 4, W.BOOL)
        query = signature(self.kernel, "QueryFullProcessImageNameW",
                          [W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)], W.BOOL)
        created, exited, kernel, user = (W.FILETIME() for _ in range(4))
        path, size = C.create_unicode_buffer(32768), W.DWORD(32768)
        try:
            if not times(self.handle, C.byref(created), C.byref(exited), C.byref(kernel), C.byref(user)):
                raise RigError("Cannot verify process creation time; refusing termination")
            if not query(self.handle, 0, path, C.byref(size)):
                raise RigError("Cannot verify process path; refusing termination")
            ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
            # CIM CreationDate has microsecond precision; FILETIME has 100ns precision.
            if str(ticks // 10) != row.get("created") or os.path.normcase(path.value) != os.path.normcase(row.get("path") or ""):
                raise RigError(f"Process identity changed for PID {row['pid']}; refusing termination")
        except BaseException:
            self.close()
            raise

    def alive(self):
        if self.handle is None:
            return False
        status = self.kernel.WaitForSingleObject(self.handle, 0)
        if status not in (0, 258):
            raise RigError("Owned process wait failed")
        return status == 258

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class Windows:
    def __init__(self):
        require_windows()

    def steam_paths(self):
        import winreg
        roots = []
        for hive, key, value in (
            (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
        ):
            try:
                with winreg.OpenKey(hive, key) as handle:
                    path = Path(winreg.QueryValueEx(handle, value)[0])
                    if path not in roots:
                        roots.append(path)
            except FileNotFoundError:
                pass
        return roots

    def steam_user(self):
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam\ActiveProcess") as handle:
                return winreg.QueryValueEx(handle, "ActiveUser")[0]
        except FileNotFoundError:
            return None

    def documents(self):
        shell, ole = dll("shell32"), dll("ole32")
        # FOLDERID_Documents follows folder redirection, including cloud-backed Documents.
        guid = (C.c_ubyte * 16).from_buffer_copy(uuid.UUID("FDD39AD0-238F-46AF-ADB4-6C85480369C7").bytes_le)
        get_folder = signature(shell, "SHGetKnownFolderPath", [C.c_void_p, W.DWORD, W.HANDLE, C.POINTER(C.c_void_p)], C.c_int32)
        free = signature(ole, "CoTaskMemFree", [C.c_void_p], None)
        result = C.c_void_p()
        if get_folder(C.byref(guid), 0, None, C.byref(result)):
            raise RigError("Cannot resolve the redirected Documents known folder")
        try:
            return Path(C.wstring_at(result))
        finally:
            free(result)

    def snapshot(self):
        desktop_pixels()
        user, kernel = dll("user32"), dll("kernel32")
        session = W.DWORD()
        get_session = signature(kernel, "ProcessIdToSessionId", [W.DWORD, C.POINTER(W.DWORD)], W.BOOL)
        console = signature(kernel, "WTSGetActiveConsoleSessionId", [], W.DWORD)
        if not get_session(os.getpid(), C.byref(session)) or session.value == 0 or session.value != console():
            raise RigError("Run in the logged-in interactive console desktop; service/remote desktops are refused")

        class LastInput(C.Structure):
            _fields_ = [("size", W.UINT), ("time", W.DWORD)]

        info = LastInput(C.sizeof(LastInput), 0)
        get_input = signature(user, "GetLastInputInfo", [C.POINTER(LastInput)], W.BOOL)
        tick = signature(kernel, "GetTickCount64", [], C.c_uint64)
        if not get_input(C.byref(info)):
            raise RigError("Cannot determine user activity; not starting")
        result = powershell()
        result["boot_uptime"] = tick() / 1000
        result["boot_observed_at"] = time.time()
        for process in result["processes"]:
            if process["created"]:
                process["created"] = str(int(process["created"]) // 10)
                process["born"] = int(process["created"]) / 1_000_000 - 11644473600
        result["idle_seconds"] = ((tick() & 0xffffffff) - info.time & 0xffffffff) / 1000
        windows = self.windows()
        foreground = signature(user, "GetForegroundWindow", [], W.HWND)()
        result["foreground"] = next((row for row in windows if row["hwnd"] == foreground), {})
        state = C.c_int()
        query = signature(dll("shell32"), "SHQueryUserNotificationState", [C.POINTER(C.c_int)], C.c_int32)
        if query(C.byref(state)):
            result.setdefault("warnings", []).append("Exclusive fullscreen state is not observable")
        else:
            result["foreground"]["exclusive"] = state.value == 3 and bool(result["foreground"].get("fullscreen"))
        steam_pids = {row["pid"] for row in result["processes"] if row["name"].lower() in ("steam.exe", "steamwebhelper.exe")}
        for row in windows:
            row["steam"] = row["pid"] in steam_pids
        result["activity"], warnings = self.activity(result["processes"], windows)
        result.setdefault("warnings", []).extend(warnings)
        return result

    def windows(self):
        user = dll("user32")
        callback_type = C.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
        signature(user, "EnumWindows", [callback_type, W.LPARAM], W.BOOL)
        signature(user, "EnumChildWindows", [W.HWND, callback_type, W.LPARAM], W.BOOL)
        signature(user, "IsWindowVisible", [W.HWND], W.BOOL)
        signature(user, "GetWindowThreadProcessId", [W.HWND, C.POINTER(W.DWORD)], W.DWORD)
        signature(user, "GetWindowTextW", [W.HWND, W.LPWSTR, C.c_int], C.c_int)
        signature(user, "GetClassNameW", [W.HWND, W.LPWSTR, C.c_int], C.c_int)
        signature(user, "GetWindow", [W.HWND, W.UINT], W.HWND)
        signature(user, "IsZoomed", [W.HWND], W.BOOL)
        signature(user, "GetWindowLongPtrW", [W.HWND, C.c_int], C.c_ssize_t)
        signature(user, "GetWindowRect", [W.HWND, C.POINTER(W.RECT)], W.BOOL)
        signature(user, "MonitorFromWindow", [W.HWND, W.DWORD], W.HANDLE)
        class MonitorInfo(C.Structure):
            _fields_ = [("size", W.DWORD), ("monitor", W.RECT), ("work", W.RECT), ("flags", W.DWORD)]
        signature(user, "GetMonitorInfoW", [W.HANDLE, C.POINTER(MonitorInfo)], W.BOOL)
        send = signature(user, "SendMessageTimeoutW",
                         [W.HWND, W.UINT, W.WPARAM, W.LPARAM, W.UINT, W.UINT, C.POINTER(C.c_size_t)], W.LPARAM)
        rows = []

        @callback_type
        def window(hwnd, _):
            if not user.IsWindowVisible(hwnd):
                return True
            text, pid = C.create_unicode_buffer(2048), W.DWORD()
            user.GetWindowTextW(hwnd, text, len(text))
            user.GetWindowThreadProcessId(hwnd, C.byref(pid))
            kind, rect, info = C.create_unicode_buffer(256), W.RECT(), MonitorInfo()
            user.GetClassNameW(hwnd, kind, len(kind))
            info.size = C.sizeof(info)
            fullscreen = False
            if user.GetWindowRect(hwnd, C.byref(rect)) and user.GetMonitorInfoW(user.MonitorFromWindow(hwnd, 2), C.byref(info)):
                fullscreen = (rect.left <= info.monitor.left and rect.top <= info.monitor.top and
                              rect.right >= info.monitor.right and rect.bottom >= info.monitor.bottom)
            row = {"hwnd": int(hwnd), "pid": pid.value, "title": text.value, "text": "", "class": kind.value,
                   "fullscreen": fullscreen, "owner": int(user.GetWindow(hwnd, 4) or 0),
                   "maximized": bool(user.IsZoomed(hwnd)),
                   "caption": bool(user.GetWindowLongPtrW(hwnd, -16) & 0x00c00000)}
            child_text = []

            @callback_type
            def child(handle, _):
                if not user.IsWindowVisible(handle):
                    return True
                if len(child_text) >= 40:
                    return False
                buf, value = C.create_unicode_buffer(1024), C.c_size_t()
                if send(handle, 13, len(buf), C.addressof(buf), 2, 50, C.byref(value)):
                    child_text.append(buf.value)
                return True

            # Read modal controls, not browser content or arbitrary owner windows.
            if any(word in text.value.lower() for word in ("steam", "assetto", "truck", "security", "firewall", "license", "eula", "obs", "discord", "teams", "streamlabs", "sunshine", "parsec", "nvidia")):
                user.EnumChildWindows(hwnd, child, 0)
            row["text"] = " ".join(child_text)
            rows.append(row)
            return True

        if not user.EnumWindows(window, 0):
            raise RigError("Cannot enumerate desktop windows")
        return rows

    def activity(self, processes, windows):
        """Read visible output controls and current-session logs, never credentials."""
        records, warnings = capture_activity(processes, windows, tail)
        for process in processes:
            name, pid = process["name"].lower(), process["pid"]
            text = " ".join(row["title"] + " " + row["text"] for row in windows if row["pid"] == pid).lower()
            if name == "steam.exe":
                indicators = any(value in text for value in ("stop streaming", "stop remote play"))
                helpers = any(row["name"].lower() in ("streaming_client.exe", "steam_streaming.exe") for row in processes)
                for root in self.steam_paths():
                    log = root / "logs/streaming_log.txt"
                    if log.exists():
                        records.append({"pid": pid, "active": steam_log_active(log, process.get("born"),
                                        corroborated=indicators or helpers), "kind": "Steam Remote Play streaming"})
                if indicators:
                    records.append({"pid": pid, "active": True, "kind": "Steam Remote Play streaming controls"})
            elif name in ("streaming_client.exe", "steam_streaming.exe"):
                records.append({"pid": pid, "active": True, "kind": "Steam Remote Play streaming"})
            elif name in ("sunshine.exe", "parsecd.exe"):
                active = any(value in text for value in ("stop recording", "stop streaming", "stop broadcasting", "stop hosting"))
                logs = []
                if name == "sunshine.exe" and process.get("path"):
                    logs = [Path(process["path"]).parent / "config/sunshine.log"]
                elif name == "parsecd.exe":
                    logs = [Path(value) / "Parsec/log.txt" for key in ("APPDATA", "PROGRAMDATA")
                            if (value := os.environ.get(key))]
                for log in logs:
                    if log.exists() and log.stat().st_mtime >= process.get("born", 0):
                        contents = tail(log)
                        active |= (connection_log_active(contents) if name == "sunshine.exe" else
                                   output_log_active(contents, "streaming"))
                records.append({"pid": pid, "active": active, "kind": "desktop recording/streaming"})
        return records, warnings

    def launch_uri(self, uri):
        try:
            os.startfile(uri)
        except OSError as exc:
            raise RigError(f"Steam URI launch failed: {exc}") from exc

    def launch_ac(self, executable):
        # The practice configuration is in Documents/cfg/race.ini. Steam has
        # already launched its registered launcher before the simulation starts.
        environment = os.environ.copy()
        environment.update(SteamAppId="244210", SteamGameId="244210")
        return subprocess.Popen([str(executable)], cwd=executable.parent, env=environment).pid

    def argument(self, process, name):
        shell, kernel = dll("shell32"), dll("kernel32")
        parse = signature(shell, "CommandLineToArgvW", [W.LPCWSTR, C.POINTER(C.c_int)], C.POINTER(W.LPWSTR))
        free = signature(kernel, "LocalFree", [C.c_void_p], C.c_void_p)
        count = C.c_int()
        args = parse(process.get("command") or "", C.byref(count))
        if not args:
            raise RigError("Cannot verify game launch arguments")
        try:
            values = [args[i] for i in range(count.value)]
            matches = [values[i + 1] for i in range(len(values) - 1) if values[i].lower() == name]
            return matches[0] if len(matches) == 1 else None
        finally:
            free(args)

    def stop_process(self, process):
        handle = ProcessHandle(process, terminate=True)
        try:
            if not handle.alive():
                return
            user = dll("user32")
            post = signature(user, "PostMessageW", [W.HWND, W.UINT, W.WPARAM, W.LPARAM], W.BOOL)
            for window in self.windows():
                if window["pid"] == process["pid"]:
                    post(window["hwnd"], 0x10, 0, 0)  # WM_CLOSE
            if handle.kernel.WaitForSingleObject(handle.handle, 10000) == 258:
                if not handle.kernel.TerminateProcess(handle.handle, 1):
                    raise RigError(f"Owned process {process['pid']} would not stop")
                if handle.kernel.WaitForSingleObject(handle.handle, 10000) != 0:
                    raise RigError(f"Owned process {process['pid']} is still running; restore refused")
        finally:
            handle.close()

    def place_window(self, window, process, width, height):
        desktop_pixels()
        handle = ProcessHandle(process)
        try:
            if not handle.alive():
                raise RigError("Game exited before VDD window placement")
            user = dll("user32")
            get_pid = signature(user, "GetWindowThreadProcessId", [W.HWND, C.POINTER(W.DWORD)], W.DWORD)
            pid = W.DWORD()
            get_pid(window["hwnd"], C.byref(pid))
            if pid.value != process["pid"]:
                raise RigError("Game window ownership changed before VDD placement")
            get_style = signature(user, "GetWindowLongPtrW", [W.HWND, C.c_int], C.c_ssize_t)
            set_style = signature(user, "SetWindowLongPtrW", [W.HWND, C.c_int, C.c_ssize_t], C.c_ssize_t)
            position = signature(user, "SetWindowPos", [W.HWND, W.HWND, C.c_int, C.c_int, C.c_int, C.c_int, W.UINT], W.BOOL)
            rectangle = signature(user, "GetWindowRect", [W.HWND, C.POINTER(W.RECT)], W.BOOL)
            set_style(window["hwnd"], -16, get_style(window["hwnd"], -16) & ~0x00cf0000)
            if not position(window["hwnd"], None, 0, 0, width, height, 0x0034):
                raise RigError("Cannot position the borderless game on the VDD primary display")
            rect = W.RECT()
            if not rectangle(window["hwnd"], C.byref(rect)) or (rect.left, rect.top, rect.right, rect.bottom) != (0, 0, width, height):
                raise RigError("Borderless VDD game placement did not verify")
        finally:
            handle.close()

    def telemetry(self, game):
        kernel = dll("kernel32")
        opened = signature(kernel, "OpenFileMappingW", [W.DWORD, W.BOOL, W.LPCWSTR], W.HANDLE)
        mapped = signature(kernel, "MapViewOfFile", [W.HANDLE, W.DWORD, W.DWORD, W.DWORD, C.c_size_t], C.c_void_p)
        unmap = signature(kernel, "UnmapViewOfFile", [C.c_void_p], W.BOOL)
        close = signature(kernel, "CloseHandle", [W.HANDLE], W.BOOL)

        def read(name, size):
            # Open ONLY: mmap(-1, ..., tagname) can create a bogus zero-filled map.
            handle = opened(4, False, name)
            if not handle:
                return None
            view = None
            try:
                view = mapped(handle, 4, 0, 0, size)
                return C.string_at(view, size) if view else None
            finally:
                if view:
                    unmap(view)
                close(handle)

        if game == "ac":
            physics, graphics = read("acpmf_physics", 4), read("acpmf_graphics", 8)
            if physics and graphics and struct.unpack_from("<i", graphics, 4)[0] == 2:
                return [struct.unpack_from("<i", physics)[0], struct.unpack_from("<i", graphics)[0]]
        else:
            data = read(r"Local\SCSTelemetry", 56)
            if data and data[0] and not data[4] and struct.unpack_from("<I", data, 52)[0] == 1:
                tick = struct.unpack_from("<Q", data, 8)[0]
                if tick and struct.unpack_from("<I", data, 40)[0]:
                    return [tick]
        return None

    def start_display(self, journal, options):
        from .display import start_worker
        start_worker(journal, options, self)

    def stop_display(self, journal):
        from .display import stop_worker
        stop_worker(journal, self)

    sleep = staticmethod(time.sleep)
    monotonic = staticmethod(time.monotonic)
    wall_time = staticmethod(time.time)


def tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 256 * 1024))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def connection_log_active(text):
    clients = 0
    for line in text.lower().splitlines():
        if "sunshine version" in line:
            clients = 0
        elif "client connected" in line:
            clients += 1
        elif "client disconnected" in line:
            clients = max(0, clients - 1)
    return clients > 0
