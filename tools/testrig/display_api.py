"""Windows display topology API. Never changes monitor power or installs devices."""

import base64
import ctypes as C
from ctypes import wintypes as W
import json
import struct
import time

from . import RigError
from .windows import Windows, desktop_pixels, dll, powershell, signature


def encoded(buffer):
    return base64.b64encode(bytes(buffer)).decode("ascii")


def device_instance(path):
    """Convert a CCD adapter interface path to its PnP instance, not hardware ID."""
    value = path.removeprefix("\\\\?\\").removeprefix("\\??\\")
    return value.split("#{", 1)[0].replace("#", "\\").casefold()


class ActivationTimeout(RigError):
    """The requested source remained absent from the active desktop."""


class NativeDisplay:
    def __init__(self):
        desktop_pixels()
        self.user = dll("user32")
        self.action_time = Windows.action_time
        signature(self.user, "GetDisplayConfigBufferSizes", [W.UINT, C.POINTER(W.UINT), C.POINTER(W.UINT)], C.c_int32)
        signature(self.user, "QueryDisplayConfig", [W.UINT, C.POINTER(W.UINT), C.c_void_p,
                                                    C.POINTER(W.UINT), C.c_void_p, C.c_void_p], C.c_int32)
        signature(self.user, "SetDisplayConfig", [W.UINT, C.c_void_p, W.UINT, C.c_void_p, W.UINT], C.c_int32)
        signature(self.user, "DisplayConfigGetDeviceInfo", [C.c_void_p], C.c_int32)
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
                result.append({"name": item.name, "hardware_id": item.instance,
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
                    self._result("EnumDisplaySettingsExW", 0, "Cannot read an active display mode",
                                 device=item["name"], index="0xffffffff", flags="0x0")
                result.append({**item, "mode": encoded(mode),
                               "x": struct.unpack_from("<i", mode, 76)[0],
                               "y": struct.unpack_from("<i", mode, 80)[0],
                               "width": struct.unpack_from("<I", mode, 172)[0],
                               "height": struct.unpack_from("<I", mode, 176)[0]})
        return result

    def query(self, flags):
        for _ in range(3):  # topology can change between size and query
            paths, modes = W.UINT(), W.UINT()
            status = self.user.GetDisplayConfigBufferSizes(flags, C.byref(paths), C.byref(modes))
            if status:
                self._result("GetDisplayConfigBufferSizes", status, "Cannot size the original display topology", flags=hex(flags))
            if not paths.value or not modes.value:
                if flags == 2:
                    raise RigError("Invalid/empty original display topology")
                raise RigError(f"Empty display inventory ({paths.value} paths, {modes.value} modes)")
            # DISPLAYCONFIG_PATH_INFO=72, DISPLAYCONFIG_MODE_INFO=64 on Windows.
            if paths.value * 72 + modes.value * 64 > 1024 * 1024:
                raise RigError(f"Display inventory exceeds 1 MiB ({paths.value} paths, {modes.value} modes)")
            path_data, mode_data = C.create_string_buffer(paths.value * 72), C.create_string_buffer(modes.value * 64)
            status = self.user.QueryDisplayConfig(flags, C.byref(paths), path_data, C.byref(modes), mode_data, None)
            if status == 122:  # ERROR_INSUFFICIENT_BUFFER
                continue
            if status:
                self._result("QueryDisplayConfig", status, "Cannot record display topology", flags=hex(flags),
                             path_count=paths.value, mode_count=modes.value)
            return {"path_count": paths.value, "mode_count": modes.value,
                    "paths": encoded(bytes(path_data)[:paths.value * 72]),
                    "modes": encoded(bytes(mode_data)[:modes.value * 64])}
        raise RigError("Display topology kept changing; not starting")

    def device_info(self, kind, luid, identifier, size, offset, *, record=True):
        data = C.create_string_buffer(size)
        struct.pack_into("<II", data, 0, kind, size)
        data[8:16] = luid
        struct.pack_into("<I", data, 16, identifier)
        status = self.user.DisplayConfigGetDeviceInfo(data)
        if status:
            self._result("DisplayConfigGetDeviceInfo", status, "Cannot identify CCD display device",
                         record=record, kind=kind, identifier=identifier, adapter=bytes(luid).hex(), size=size)
        return bytes(data)[offset:].decode("utf-16-le").split("\0", 1)[0]

    def sources(self, *, active_only=False):
        query = self.query(2 if active_only else 1)
        paths = base64.b64decode(query["paths"])
        rows = {}
        for index in range(query["path_count"]):
            path = paths[index * 72:(index + 1) * 72]
            source_id, target_id = struct.unpack_from("<I", path, 8)[0], struct.unpack_from("<I", path, 28)[0]
            try:
                name = self.device_info(1, path[:8], source_id, 84, 20, record=active_only)
                adapter = self.device_info(4, path[:8], 0, 276, 20, record=active_only)
                target = self.device_info(2, path[20:28], target_id, 420, 164, record=active_only)
            except RigError:
                if active_only:
                    raise
                continue  # disconnected combinations need not have a GDI source
            row = rows.setdefault((name, adapter), {"name": name, "adapter_path": adapter,
                                  "source_id": source_id, "target_paths": []})
            if target and target not in row["target_paths"]:
                row["target_paths"].append(target)
        return list(rows.values())

    def capture(self):
        query = self.query(2)
        sources = self.sources(active_only=True)
        displays = self.active()
        for display in displays:
            matches = [row for row in sources if row["name"] == display["name"]]
            if len(matches) != 1:
                raise RigError("Cannot record stable physical display identity")
            display.update(matches[0])
        primary = [item["name"] for item in displays if item["primary"]]
        if len(primary) != 1:
            raise RigError("Cannot identify the original physical primary display")
        return {**query, "displays": displays, "primary": primary[0]}

    def physical_displays(self, original):
        sources = self.sources(active_only=True)
        result = []
        for saved in original["displays"]:
            targets = {value.casefold() for value in saved.get("target_paths", [])}
            matches = [row for row in sources if targets & {value.casefold() for value in row["target_paths"]}]
            if not matches and saved.get("adapter_path"):
                matches = [row for row in sources if device_instance(row["adapter_path"]) == device_instance(saved["adapter_path"])]
                if len(matches) > 1:
                    matches = [row for row in matches if row["source_id"] == saved["source_id"]]
            if len(matches) != 1:
                raise RigError("Cannot uniquely recover physical display identity; use Windows Display settings")
            result.append({**saved, "name": matches[0]["name"]})
        return result

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
        inputs = {"device": name, "flags": hex(flags)}
        if mode is not None:
            inputs.update(dmFields=hex(struct.unpack_from("<I", mode, 72)[0]),
                          position=list(struct.unpack_from("<ii", mode, 76)),
                          size=list(struct.unpack_from("<II", mode, 172)),
                          bits_per_pixel=struct.unpack_from("<I", mode, 168)[0],
                          frequency=struct.unpack_from("<I", mode, 184)[0], devmode=encoded(mode))
        self._result("ChangeDisplaySettingsExW", status, "CDS stage" if name else "CDS apply", **inputs)

    def _result(self, api, status, operation, *, record=True, **inputs):
        names = ({0: "DISP_CHANGE_SUCCESSFUL", 1: "DISP_CHANGE_RESTART", -1: "DISP_CHANGE_FAILED",
                  -2: "DISP_CHANGE_BADMODE", -3: "DISP_CHANGE_NOTUPDATED", -4: "DISP_CHANGE_BADFLAGS",
                  -5: "DISP_CHANGE_BADPARAM", -6: "DISP_CHANGE_BADDUALVIEW"} if api == "ChangeDisplaySettingsExW" else
                 {0: "ERROR_SUCCESS", 5: "ERROR_ACCESS_DENIED", 31: "ERROR_GEN_FAILURE", 50: "ERROR_NOT_SUPPORTED",
                  87: "ERROR_INVALID_PARAMETER", 122: "ERROR_INSUFFICIENT_BUFFER", 1610: "ERROR_BAD_CONFIGURATION"})
        symbol = names.get(status, "UNKNOWN_RETURN_CODE")
        failed = bool(status)
        code = "result" if api == "ChangeDisplaySettingsExW" else "Windows error"
        if api == "EnumDisplaySettingsExW":
            symbol, failed, code = ("TRUE" if status else "FALSE"), not status, "BOOL result"
        row = {"api": api, "return_code": status, "return_name": symbol, "operation": operation, "inputs": inputs}
        if hasattr(self, "action_time"):
            row.update(self.action_time())
        if record:
            self.__dict__.setdefault("display_calls", []).append(row)
        if failed:
            summary = {key: value for key, value in inputs.items() if key not in ("paths", "modes", "devmode")}
            raise RigError(f"{operation}: {api} {symbol} ({code} {status}); inputs={json.dumps(summary, sort_keys=True)}")

    def _set_config(self, config, flags, operation):
        config = config or {"path_count": 0, "mode_count": 0, "paths": "", "modes": "", "layout": {}}
        paths, modes = base64.b64decode(config["paths"]), base64.b64decode(config["modes"])
        path_data = C.create_string_buffer(paths, len(paths)) if paths else None
        mode_data = C.create_string_buffer(modes, len(modes)) if modes else None
        status = self.user.SetDisplayConfig(config["path_count"], path_data, config["mode_count"], mode_data, flags)
        self._result("SetDisplayConfig", status, operation, flags=hex(flags), **config)

    def wait_active_virtual(self, instance, *, clock=time.monotonic, pause=time.sleep, flags="0x84"):
        deadline, detail = clock() + 10, "VDD absent from QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS)"
        while True:
            try:
                matches = {row["name"] for row in self.sources(active_only=True)
                           if device_instance(row["adapter_path"]) == instance.casefold()}
            except RigError as exc:
                detail, matches = str(exc), set()
            if len(matches) > 1:
                raise RigError("QueryDisplayConfig(active) returned multiple active VDD sources; refusing primary switch")
            if matches:
                return next(iter(matches))
            remaining = deadline - clock()
            if remaining <= 0:
                raise ActivationTimeout(f"VDD activation timed out after 10s; flags={flags}, device={instance}; {detail}")
            pause(min(0.2, remaining))

    def plan_primary(self, instance, original, width, height):
        """Query a supplied configuration; never apply or enable a device here."""
        physical = {item["name"]: item for item in self.physical_displays(original)}
        offset = max(item["x"] + item["width"] for item in physical.values())
        query = self.query(2)  # fresh active paths, including every physical target
        raw = base64.b64decode(query["paths"])
        paths = [bytearray(raw[index * 72:(index + 1) * 72]) for index in range(query["path_count"])]
        modes = bytearray(base64.b64decode(query["modes"]))

        def identity(path, *, record=True):
            source = struct.unpack_from("<I", path, 8)[0]
            return (self.device_info(1, path[:8], source, 84, 20, record=record),
                    device_instance(self.device_info(4, path[:8], 0, 276, 20, record=record)))

        if not any(identity(path)[1] == instance.casefold() for path in paths):
            inventory = self.query(1)
            all_paths = base64.b64decode(inventory["paths"])
            candidates = {}
            for index in range(inventory["path_count"]):
                path = bytearray(all_paths[index * 72:(index + 1) * 72])
                try:
                    virtual = identity(path, record=False)[1] == instance.casefold()
                except RigError:
                    continue  # disconnected source combinations need not resolve
                if virtual and struct.unpack_from("<I", path, 60)[0]:
                    candidates[bytes(path[:12] + path[20:32])] = path
            if len(candidates) != 1:
                raise RigError(f"QueryDisplayConfig(all, flags=0x1, return 0): expected one available VDD path for {instance}; "
                               f"found {len(candidates)}")
            path = next(iter(candidates.values()))
            mode = bytearray(64)
            struct.pack_into("<II", mode, 0, 1, struct.unpack_from("<I", path, 8)[0])
            mode[8:16] = path[:8]
            struct.pack_into("<IIIii", mode, 16, width, height, 4, 0, 0)  # SOURCE, 32 BPP
            struct.pack_into("<I", path, 12, len(modes) // 64)
            struct.pack_into("<I", path, 68, struct.unpack_from("<I", path, 68)[0] | 1)
            modes.extend(mode)
            paths.append(path)
        layout, virtual_paths = {}, []
        for path in paths:
            name, adapter = identity(path)
            index = struct.unpack_from("<I", path, 12)[0]
            start = index * 64
            if (start + 64 > len(modes) or struct.unpack_from("<II", modes, start) !=
                    (1, struct.unpack_from("<I", path, 8)[0]) or modes[start + 8:start + 16] != path[:8]):
                raise RigError(f"QueryDisplayConfig(active, return 0): invalid source mode {index} for {name}")
            if adapter == instance.casefold():
                struct.pack_into("<II", modes, start + 16, width, height)
                x, y = 0, 0
                # Supply the source size; let best-mode logic choose missing
                # target timing, without allowing changes to supplied modes.
                struct.pack_into("<I", path, 32, 0xffffffff)
                struct.pack_into("<III", path, 48, 0, 0, 0)
                virtual_paths.append(path)
            else:
                if name not in physical:
                    raise RigError(f"QueryDisplayConfig(active, return 0): unexpected active source {name}; primary plan refused")
                saved = physical[name]
                x, y = saved["x"] - offset, saved["y"]
            struct.pack_into("<ii", modes, start + 28, x, y)
            layout[name] = [x, y, *struct.unpack_from("<II", modes, start + 16)]
        ordered = virtual_paths + [path for path in paths if path not in virtual_paths]
        if len(ordered) * 72 + len(modes) > 1024 * 1024:
            raise RigError(f"Supplied display inventory exceeds 1 MiB ({len(ordered)} paths, {len(modes) // 64} modes)")
        # Identity lookup and the fresh topology query can race a disconnected
        # target. Refuse before validate/apply, including clones of one source.
        active_targets = {}
        for path in ordered:
            if struct.unpack_from("<I", path, 68)[0] & 1:
                name, adapter = identity(path)
                if adapter != instance.casefold():
                    target = self.device_info(2, path[20:28], struct.unpack_from("<I", path, 28)[0], 420, 164)
                    active_targets.setdefault(name, set()).add(target.casefold())
        preserved = []
        for name, saved in physical.items():
            targets = saved.get("target_paths", [])
            for target in targets or [name]:
                if not targets or name not in layout or target.casefold() not in active_targets.get(name, set()):
                    raise RigError(f"fallback would deactivate physical display {target}; primary plan refused")
                preserved.append({"name": name, "target": target, "active": True})
        return {"path_count": len(ordered), "mode_count": len(modes) // 64,
                "paths": encoded(b"".join(ordered)), "modes": encoded(modes), "layout": layout,
                "physical_targets": preserved}

    def _supplied_primary(self, instance, original, width, height, reason):
        try:
            config = self.plan_primary(instance, original, width, height)
            self._set_config(config, 0x60, "supplied primary validate")
            self._set_config(config, 0xa0, "supplied primary apply")
            return self.wait_active_virtual(instance, flags="0xa0")
        except RigError as exc:
            raise RigError(f"{reason}; supplied primary fallback: {exc}") from exc

    def wait_virtual(self, instance, *, clock=time.monotonic, pause=time.sleep):
        deadline = clock() + 10
        detail = "output is unavailable"
        while True:
            try:
                matches = {row["name"] for row in self.sources() if device_instance(row["adapter_path"]) == instance.casefold()}
            except RigError as exc:
                detail, matches = str(exc), set()
            if len(matches) > 1:
                raise RigError("Cannot match the VDD PnP instance to one CCD/GDI output; multiple outputs are refused")
            if matches:
                return next(iter(matches))
            remaining = deadline - clock()
            if remaining <= 0:
                raise RigError("VDD CCD/GDI output timed out after 10s; " + detail)
            pause(min(0.2, remaining))

    def extend_primary(self, instance, original, width, height):
        virtual = self.wait_virtual(instance)
        physical = self.physical_displays(original)
        if virtual in {item["name"] for item in physical}:
            raise RigError("The VDD was already active in the original topology")
        self._set_config(None, 0x84, "extend activation")  # no SUPPLIED or SAVE
        try:
            virtual = self.wait_active_virtual(instance)
        except ActivationTimeout as exc:
            virtual = self._supplied_primary(instance, original, width, height, str(exc))
            self._verify_primary(virtual, self.physical_displays(original), width, height)
            return {"name": virtual, "width": width, "height": height}
        physical = self.physical_displays(original)
        selected = None
        for index in range(512):
            mode = self.mode(virtual, index)
            if mode is None:
                break
            if struct.unpack_from("<II", mode, 172) == (width, height):
                selected = mode
                break
        if selected is None:
            raise RigError(f"EnumDisplaySettingsExW: no VDD {width}x{height} mode; device={virtual}, "
                           f"index={index}, return={int(mode is not None)}; choose a supported test resolution")
        offset = max(item["x"] + item["width"] for item in physical)
        # Translate every physical display together. Their modes remain enabled
        # and their arrangement is preserved to the left of the new primary.
        for item in physical:
            raw = base64.b64decode(item["mode"])
            mode = C.create_string_buffer(raw, len(raw))
            struct.pack_into("<I", mode, 72, 0x20)  # DM_POSITION only
            struct.pack_into("<ii", mode, 76, item["x"] - offset, item["y"])
            self.change(item["name"], mode, 0x10000001)  # UPDATEREGISTRY | NORESET
        struct.pack_into("<I", selected, 72, 0x20 | 0x40000 | 0x80000 | 0x100000 | 0x400000)
        struct.pack_into("<ii", selected, 76, 0, 0)
        try:
            self.change(virtual, selected, 0x10000011)  # also CDS_SET_PRIMARY
        except RigError as exc:
            virtual = self._supplied_primary(instance, original, width, height, str(exc))
            physical = self.physical_displays(original)
        else:
            self.change(None, None, 0)
        self._verify_primary(virtual, physical, width, height)
        return {"name": virtual, "width": width, "height": height}

    def _verify_primary(self, virtual, physical, width, height):
        current = self.active()
        by_name = {item["name"]: item for item in current}
        expected = {virtual: (0, 0, width, height, True)}
        offset = max(item["x"] + item["width"] for item in physical)
        for item in physical:
            expected[item["name"]] = (item["x"] - offset, item["y"], item["width"], item["height"], False)
        for name, wanted in expected.items():
            actual = by_name.get(name, {})
            observed = tuple(actual.get(key) for key in ("x", "y", "width", "height", "primary"))
            if observed != wanted:
                last = self.display_calls[-1]
                code = "result" if last["api"] == "ChangeDisplaySettingsExW" else "Windows error"
                raise RigError(f"{last['api']} {last['return_name']} ({code} {last['return_code']}): layout/position verification "
                               f"failed for {name}; expected={wanted}, actual={observed}; inputs={last['inputs'].get('flags')}")

    def physical_rectangles(self, original):
        wanted = {item["name"] for item in self.physical_displays(original)}
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
        self._result("SetDisplayConfig", status, "Cannot restore original display topology", flags="0x2a0",
                     path_count=original["path_count"], mode_count=original["mode_count"],
                     paths=original["paths"], modes=original["modes"])
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
        saved_primary = next(item for item in self.physical_displays(original) if item["primary"])
        primary = next((item for item in current if item["name"] == saved_primary["name"]), None)
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

    def restore_physical(self, original):
        """Recover saved physical modes via stable paths after LUID/name changes."""
        physical = self.physical_displays(original)
        for item in physical:
            raw = base64.b64decode(item["mode"])
            mode = C.create_string_buffer(raw, len(raw))
            struct.pack_into("<I", mode, 72, 0x20 | 0x80 | 0x40000 | 0x80000 | 0x100000 | 0x400000)
            self.change(item["name"], mode, 0x10000001 | (0x10 if item["primary"] else 0))
        self.change(None, None, 0)
        by_name = {item["name"]: item for item in self.active()}
        for item in physical:
            actual = by_name.get(item["name"])
            if not actual or any(actual[key] != item[key] for key in ("x", "y", "width", "height", "primary")):
                raise RigError("Physical layout fallback did not verify; use Windows Display settings")
            for offset in (84, 168, 184):  # orientation, bit depth, refresh rate
                if struct.unpack_from("<I", base64.b64decode(actual["mode"]), offset) != struct.unpack_from("<I", base64.b64decode(item["mode"]), offset):
                    raise RigError("Physical display mode fallback did not verify")


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
