"""Windows display topology API. Never changes monitor power or installs devices."""

import base64
import ctypes as C
from ctypes import wintypes as W
import struct

from . import RigError
from .windows import desktop_pixels, dll, powershell, signature


def encoded(buffer):
    return base64.b64encode(bytes(buffer)).decode("ascii")


class NativeDisplay:
    def __init__(self):
        desktop_pixels()
        self.user = dll("user32")
        signature(self.user, "GetDisplayConfigBufferSizes", [W.UINT, C.POINTER(W.UINT), C.POINTER(W.UINT)], C.c_int32)
        signature(self.user, "QueryDisplayConfig", [W.UINT, C.POINTER(W.UINT), C.c_void_p,
                                                    C.POINTER(W.UINT), C.c_void_p, C.c_void_p], C.c_int32)
        signature(self.user, "SetDisplayConfig", [W.UINT, C.c_void_p, W.UINT, C.c_void_p, W.UINT], C.c_int32)
        signature(self.user, "EnumDisplaySettingsExW", [W.LPCWSTR, W.DWORD, C.c_void_p, W.DWORD], W.BOOL)
        signature(self.user, "ChangeDisplaySettingsExW", [W.LPCWSTR, C.c_void_p, W.HWND, W.DWORD, C.c_void_p], C.c_int32)

    def devices(self):
        class Device(C.Structure):
            _fields_ = [("size", W.DWORD), ("name", W.WCHAR * 32), ("description", W.WCHAR * 128),
                        ("flags", W.DWORD), ("instance", W.WCHAR * 128), ("key", W.WCHAR * 128)]

        enum = signature(self.user, "EnumDisplayDevicesW", [W.LPCWSTR, W.DWORD, C.POINTER(Device), W.DWORD], W.BOOL)
        result = []
        for index in range(64):
            item = Device()
            item.size = C.sizeof(item)
            if not enum(None, index, C.byref(item), 0):
                break
            if not item.flags & 8:  # exclude mirroring pseudo drivers
                result.append({"name": item.name, "instance": item.instance,
                               "active": bool(item.flags & 1), "primary": bool(item.flags & 4)})
        return result

    def mode(self, name, index=0xffffffff):
        # DEVMODEW's fixed layout is 220 bytes. Driver-private bytes are neither
        # requested nor changed. Raw bytes also retain refresh/orientation fields.
        mode = C.create_string_buffer(220)
        struct.pack_into("<H", mode, 68, 220)  # dmSize
        if not self.user.EnumDisplaySettingsExW(name, index, mode, 0):
            return None
        return mode

    def active(self):
        result = []
        for item in self.devices():
            if item["active"]:
                mode = self.mode(item["name"])
                if mode is None:
                    raise RigError("Cannot read an active display mode")
                result.append({**item, "mode": encoded(mode),
                               "x": struct.unpack_from("<i", mode, 76)[0],
                               "y": struct.unpack_from("<i", mode, 80)[0],
                               "width": struct.unpack_from("<I", mode, 172)[0],
                               "height": struct.unpack_from("<I", mode, 176)[0]})
        return result

    def capture(self):
        for _ in range(3):  # topology can change between size and query
            paths, modes = W.UINT(), W.UINT()
            if self.user.GetDisplayConfigBufferSizes(2, C.byref(paths), C.byref(modes)):
                raise RigError("Cannot size the original display topology")
            if not 0 < paths.value <= 64 or not 0 < modes.value <= 256:
                raise RigError("Invalid/empty original display topology")
            # DISPLAYCONFIG_PATH_INFO=72, DISPLAYCONFIG_MODE_INFO=64 on Windows.
            path_data, mode_data = C.create_string_buffer(paths.value * 72), C.create_string_buffer(modes.value * 64)
            status = self.user.QueryDisplayConfig(2, C.byref(paths), path_data, C.byref(modes), mode_data, None)
            if status == 122:  # ERROR_INSUFFICIENT_BUFFER
                continue
            if status:
                raise RigError(f"Cannot record display topology (Windows error {status})")
            displays = self.active()
            primary = [item["name"] for item in displays if item["primary"]]
            if len(primary) != 1:
                raise RigError("Cannot identify the original physical primary display")
            return {"path_count": paths.value, "mode_count": modes.value,
                    "paths": encoded(bytes(path_data)[:paths.value * 72]),
                    "modes": encoded(bytes(mode_data)[:modes.value * 64]),
                    "displays": displays, "primary": primary[0]}
        raise RigError("Display topology kept changing; not starting")

    def device_state(self, instance):
        return powershell("device-state", instance)["problem"]

    def enable(self, instance):
        if powershell("device-enable", instance)["problem"] != 0:
            raise RigError("Virtual display did not become enabled")

    def disable(self, instance):
        if self.device_state(instance) == 22:
            return
        if powershell("device-disable", instance)["problem"] != 22:
            raise RigError("Virtual display disabled state could not be verified")

    def change(self, name, mode, flags):
        status = self.user.ChangeDisplaySettingsExW(name, mode, None, flags, None)
        if status:
            raise RigError(f"Display mode change failed for {name} (result {status})")

    def extend_primary(self, instance, original, width, height):
        matches = [item for item in self.devices() if item["instance"].casefold() == instance.casefold()]
        if len(matches) != 1:
            raise RigError("Cannot match the selected VDD to one Windows display adapter")
        virtual = matches[0]["name"]
        if virtual in {item["name"] for item in original["displays"]}:
            raise RigError("The VDD was already active in the original topology")
        selected = None
        for index in range(512):
            mode = self.mode(virtual, index)
            if mode is None:
                break
            if struct.unpack_from("<II", mode, 172) == (width, height):
                selected = mode
                break
        if selected is None:
            raise RigError(f"VDD has no {width}x{height} mode; choose a supported test resolution")
        offset = max(item["x"] + item["width"] for item in original["displays"])
        # Translate every physical display together. Their modes remain enabled
        # and their arrangement is preserved to the left of the new primary.
        for item in original["displays"]:
            raw = base64.b64decode(item["mode"])
            mode = C.create_string_buffer(raw, len(raw))
            struct.pack_into("<I", mode, 72, 0x20)  # DM_POSITION only
            struct.pack_into("<ii", mode, 76, item["x"] - offset, item["y"])
            self.change(item["name"], mode, 0x10000001)  # UPDATEREGISTRY | NORESET
        struct.pack_into("<I", selected, 72, 0x20 | 0x40000 | 0x80000 | 0x100000 | 0x400000)
        struct.pack_into("<ii", selected, 76, 0, 0)
        self.change(virtual, selected, 0x10000011)  # also CDS_SET_PRIMARY
        self.change(None, None, 0)
        current = self.active()
        by_name = {item["name"]: item for item in current}
        if not by_name.get(virtual, {}).get("primary"):
            raise RigError("VDD primary display could not be verified")
        for item in original["displays"]:
            actual = by_name.get(item["name"])
            if not actual or actual["primary"] or (actual["width"], actual["height"]) != (item["width"], item["height"]):
                raise RigError("A physical display did not remain enabled as a secondary")
        return {"name": virtual, "width": width, "height": height}

    def physical_rectangles(self, original):
        wanted = {item["name"] for item in original["displays"]}
        current = [item for item in self.active() if item["name"] in wanted]
        if len(current) != len(wanted):
            raise RigError("A physical display disappeared before overlay creation")
        return [(item["x"], item["y"], item["width"], item["height"]) for item in current]

    def restore(self, original):
        paths, modes = base64.b64decode(original["paths"]), base64.b64decode(original["modes"])
        if len(paths) != original["path_count"] * 72 or len(modes) != original["mode_count"] * 64:
            raise RigError("Saved display topology is invalid; retain state for recovery")
        path_data, mode_data = C.create_string_buffer(paths, len(paths)), C.create_string_buffer(modes, len(modes))
        status = self.user.SetDisplayConfig(original["path_count"], path_data, original["mode_count"], mode_data,
                                           0x20 | 0x80 | 0x200)  # supplied | apply | save database
        if status:
            raise RigError(f"Cannot restore original display topology (Windows error {status})")
        current = self.active()
        expected = {(item["name"], item["x"], item["y"], item["width"], item["height"], item["primary"])
                    for item in original["displays"]}
        actual = {(item["name"], item["x"], item["y"], item["width"], item["height"], item["primary"])
                  for item in current}
        if actual != expected:
            raise RigError("Original topology/physical primary restoration did not verify")

    def physical_primary(self, original):
        """Best-effort primary recovery if exact SetDisplayConfig failed."""
        current = self.active()
        primary = next((item for item in current if item["name"] == original["primary"]), None)
        if primary is None:
            raise RigError("Original physical primary is no longer connected")
        if primary["primary"]:
            return
        for item in current:
            raw = base64.b64decode(item["mode"])
            mode = C.create_string_buffer(raw, len(raw))
            struct.pack_into("<I", mode, 72, 0x20)
            struct.pack_into("<ii", mode, 76, item["x"] - primary["x"], item["y"] - primary["y"])
            self.change(item["name"], mode, 0x10000001 | (0x10 if item["name"] == primary["name"] else 0))
        self.change(None, None, 0)
        if not any(item["name"] == primary["name"] and item["primary"] for item in self.active()):
            raise RigError("Physical primary recovery did not verify")


class BlackOverlay:
    def __init__(self):
        self.roots = []
        self.kernel = None

    def keep_awake(self):
        self.kernel = dll("kernel32")
        keep_awake = signature(self.kernel, "SetThreadExecutionState", [W.DWORD], W.DWORD)
        if not keep_awake(0x80000003):
            raise RigError("Cannot keep the physical display on during the session")

    def start(self, rectangles):
        import sys
        if sys.getwindowsversion().build < 19041:
            raise RigError("Capture-excluded overlay requires Windows 10 build 19041 or later")
        import tkinter as tk
        if not self.kernel:
            self.keep_awake()
        self.user = dll("user32")
        signature(self.user, "GetAncestor", [W.HWND, W.UINT], W.HWND)
        signature(self.user, "GetWindowLongPtrW", [W.HWND, C.c_int], C.c_ssize_t)
        signature(self.user, "SetWindowLongPtrW", [W.HWND, C.c_int, C.c_ssize_t], C.c_ssize_t)
        signature(self.user, "SetWindowPos", [W.HWND, W.HWND, C.c_int, C.c_int, C.c_int, C.c_int, W.UINT], W.BOOL)
        signature(self.user, "SetLayeredWindowAttributes", [W.HWND, W.DWORD, C.c_ubyte, W.DWORD], W.BOOL)
        signature(self.user, "SetWindowDisplayAffinity", [W.HWND, W.DWORD], W.BOOL)
        signature(self.user, "GetWindowDisplayAffinity", [W.HWND, C.POINTER(W.DWORD)], W.BOOL)
        self.windows = []
        for x, y, width, height in rectangles:
            root = tk.Tk() if not self.roots else tk.Toplevel(self.roots[0])
            self.roots.append(root)
            root.withdraw()
            root.overrideredirect(True)
            root.configure(background="black")
            root.update_idletasks()
            hwnd = self.user.GetAncestor(root.winfo_id(), 2)
            flags = 0x080800A0  # NOACTIVATE | LAYERED | TRANSPARENT | TOOLWINDOW
            self.user.SetWindowLongPtrW(hwnd, -20, self.user.GetWindowLongPtrW(hwnd, -20) | flags)
            if self.user.GetWindowLongPtrW(hwnd, -20) & flags != flags:
                raise RigError("Cannot set nonactivating overlay styles")
            if not self.user.SetLayeredWindowAttributes(hwnd, 0, 255, 2):
                raise RigError("Cannot make physical-display overlay opaque")
            if not self.user.SetWindowDisplayAffinity(hwnd, 0x11):
                raise RigError("WDA_EXCLUDEFROMCAPTURE could not be applied")
            affinity = W.DWORD()
            if not self.user.GetWindowDisplayAffinity(hwnd, C.byref(affinity)) or affinity.value != 0x11:
                raise RigError("Overlay capture exclusion did not verify")
            # Show without activating; Tk's deiconify can otherwise steal focus.
            if not self.user.SetWindowPos(hwnd, W.HWND(-1), x, y, width, height, 0x0050):
                raise RigError("Cannot cover the physical display")
            self.windows.append(hwnd)
            root.update()

    def pump(self):
        for root, hwnd in zip(self.roots, self.windows):
            root.update()
            if not self.user.SetWindowPos(hwnd, W.HWND(-1), 0, 0, 0, 0, 0x13):
                raise RigError("Physical-display overlay lost topmost placement")

    def close(self):
        try:
            for root in reversed(self.roots):
                root.destroy()
            self.roots.clear()
        finally:
            if self.kernel:
                self.kernel.SetThreadExecutionState(0x80000000)
