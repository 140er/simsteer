"""Inactive-source activation, primary switching and exact display rollback."""

import base64
from copy import deepcopy
import ctypes
import json
import struct

import pytest

from tools.testrig import RigError
from tools.testrig.cli import main
from tools.testrig.display import DisplayLifecycle
from tools.testrig.display_api import NativeDisplay, encoded


INSTANCE = r"ROOT\DISPLAY\0000"


@pytest.fixture(autouse=True)
def forbid_display_commands(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("DisplaySwitch or another external display command was invoked")
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("tools.testrig.display_api.powershell", forbidden)


def devmode(width, height, x=0, y=0):
    data = ctypes.create_string_buffer(220)
    struct.pack_into("<H", data, 68, 220)
    struct.pack_into("<ii", data, 76, x, y)
    struct.pack_into("<III", data, 168, 32, width, height)
    struct.pack_into("<I", data, 184, 60)
    return data


def source_mode(adapter, width, height, x=0, y=0):
    data = bytearray(64)
    struct.pack_into("<IIQIIIii", data, 0, 1, 0, adapter, width, height, 4, x, y)
    return data


class Desktop:
    def __init__(self):
        self.events, self.configs, self.staged, self.restored = [], [], {}, []
        self.enabled = self.attached = self.extending = False
        self.attach_after, self.polls, self.now = 1, 0, 0.0
        self.extend_rc = self.primary_rc = self.physical_rc = self.null_rc = 0
        self.validate_rc = self.supplied_rc = 0
        self.wrong_physical = self.wrong_physical_y = self.wrong_virtual = False
        self.layout = {"physical": self.display("physical", 2560, 1440, 0, 0, True)}

    @staticmethod
    def display(name, width, height, x, y, primary):
        return {"name": name, "width": width, "height": height, "x": x, "y": y,
                "primary": primary, "active": True, "mode": encoded(devmode(width, height, x, y))}

    def pause(self, seconds):
        self.now += seconds

    def query(self, flags):
        self.events.append("query-active" if flags == 2 else "query-all")
        if flags == 2 and self.extending and not self.attached and self.attach_after is not None:
            self.polls += 1
            if self.polls >= self.attach_after:
                self.attached = True
                self.layout["virtual"] = self.display("virtual", 1920, 1080, 2560, 0, False)
                self.events.append("vdd-active")
        modes = [source_mode(1, 2560, 1440, self.layout["physical"]["x"], self.layout["physical"]["y"]),
                 bytearray(64)]
        struct.pack_into("<IIQ", modes[1], 0, 2, 10, 1)
        paths = []
        for adapter in ([1, 2] if flags == 1 or self.attached else [1]):
            path = bytearray(72)
            struct.pack_into("<QIIIQII", path, 0, adapter, 0, 0 if adapter == 1 else 2, 1,
                             adapter, 10, 1 if adapter == 1 else 3)
            struct.pack_into("<IIIIIIII", path, 36, 5, 1, 1, 60, 1, 1, 1, 1)
            struct.pack_into("<I", path, 68, int(adapter == 1 or self.attached))
            if adapter == 2:
                if self.attached:
                    v = self.layout["virtual"]
                    modes += [source_mode(2, v["width"], v["height"], v["x"], v["y"]), bytearray(64)]
                    struct.pack_into("<IIQ", modes[3], 0, 2, 10, 2)
                else:
                    struct.pack_into("<I", path, 12, 0xffffffff)
                    struct.pack_into("<I", path, 32, 0xffffffff)
            paths.append(path)
        return {"path_count": len(paths), "mode_count": len(modes),
                "paths": encoded(b"".join(paths)), "modes": encoded(b"".join(modes))}

    def DisplayConfigGetDeviceInfo(self, data):
        kind = struct.unpack_from("<I", data)[0]
        virtual = struct.unpack_from("<Q", data, 8)[0] == 2
        adapter = r"\\?\ROOT#DISPLAY#0000#{00000000-0000-0000-0000-000000000000}" if virtual else "physical-adapter"
        offset, value = {1: (20, "virtual" if virtual else "physical"), 4: (20, adapter),
                         2: (164, "virtual-monitor" if virtual else "physical-monitor")}[kind]
        content = (value + "\0").encode("utf-16-le")
        data[offset:offset + len(content)] = content
        return 0

    def SetDisplayConfig(self, paths, path_data, modes, mode_data, flags):
        self.events.append({0x84: "extend", 0x60: "validate", 0xa0: "supplied-apply", 0x2a0: "restore"}[flags])
        raw_paths = bytes(path_data) if path_data is not None else b""
        raw_modes = bytes(mode_data) if mode_data is not None else b""
        self.configs.append((flags, paths, raw_paths, modes, raw_modes))
        if flags == 0x84:
            assert paths == modes == 0 and path_data is mode_data is None
            self.extending = not self.extend_rc
            return self.extend_rc
        if flags == 0x60:
            return self.validate_rc
        if flags == 0x2a0:
            self.restored.append((paths, raw_paths, modes, raw_modes))
            self.attached = self.extending = False
            self.layout = {"physical": self.display("physical", 2560, 1440, 0, 0, True)}
            return 0
        if self.supplied_rc:
            return self.supplied_rc
        assert paths == 2  # the physical target must never be dropped
        for index in range(paths):
            path = raw_paths[index * 72:(index + 1) * 72]
            adapter = struct.unpack_from("<Q", path)[0]
            assert struct.unpack_from("<I", path, 68)[0] & 1
            mode = struct.unpack_from("<I", path, 12)[0] * 64
            assert struct.unpack_from("<IIQ", raw_modes, mode) == (1, 0, adapter)
            width, height, pixel, x, y = struct.unpack_from("<IIIii", raw_modes, mode + 16)
            assert pixel == 4
            name = "physical" if adapter == 1 else "virtual"
            self.layout[name] = self.display(name, width, height, x, y, (x, y) == (0, 0))
        self.attached = True
        return 0

    def ChangeDisplaySettingsExW(self, name, mode, hwnd, flags, reserved):
        assert hwnd is reserved is None
        self.events.append("null-apply" if name is None else name + "-stage")
        if name == "virtual" and not self.attached:
            return -1  # the formerly permissive fake missed this real condition
        if name:
            assert flags == (0x10000011 if name == "virtual" else 0x10000001)
            rc = self.primary_rc if name == "virtual" else self.physical_rc
            if rc:
                return rc
            self.staged[name] = (bytes(mode), flags)
        else:
            assert mode is None and flags == 0
            if self.null_rc:
                return self.null_rc
            for display, (raw, staged_flags) in self.staged.items():
                width, height = struct.unpack_from("<II", raw, 172)
                x, y = struct.unpack_from("<ii", raw, 76)
                if display == "physical" and self.wrong_physical:
                    x = 0
                if display == "physical" and self.wrong_physical_y:
                    y = 1
                if display == "virtual" and self.wrong_virtual:
                    y = 1
                self.layout[display] = self.display(display, width, height, x, y, bool(staged_flags & 0x10))
        return 0


def make_desktop():
    user = Desktop()
    api = NativeDisplay.__new__(NativeDisplay)
    api.user, api.query = user, user.query
    api.active = lambda: deepcopy(list(user.layout.values()))
    api.mode = lambda name, index=0xffffffff: devmode(1920, 1080) if name == "virtual" else devmode(2560, 1440)
    api.device_state = lambda _: 0 if user.enabled else 22
    api.enable = lambda _: setattr(user, "enabled", True)
    api.disable = lambda _: setattr(user, "enabled", False)
    if hasattr(api, "wait_active_virtual"):
        wait = api.wait_active_virtual
        api.wait_active_virtual = lambda instance, **kwargs: wait(instance, clock=lambda: user.now, pause=user.pause, **kwargs)
    return api, user


class Overlay:
    def keep_awake(self):
        pass

    def close(self):
        pass


def lifecycle(api):
    return DisplayLifecycle(api, Overlay(), lambda _: None,
        {"display": "vdd", "overlay": False, "vdd_device": INSTANCE, "width": 1920, "height": 1080}, "test")


def assert_rollback(worker, user):
    original = worker.row["original"]
    assert user.restored == [(original["path_count"], base64.b64decode(original["paths"]),
                              original["mode_count"], base64.b64decode(original["modes"]))]
    assert worker.row["topology_restored"] and worker.row["vdd_disabled"]
    assert not user.enabled and not worker.row["ready"]


def test_extend_activation_precedes_staging_and_keeps_physical_active():
    api, user = make_desktop()
    original = api.capture()
    api.extend_primary(INSTANCE, original, 1920, 1080)
    order = [event for event in user.events if event in
             ("extend", "vdd-active", "physical-stage", "virtual-stage", "null-apply")]
    assert order == ["extend", "vdd-active", "physical-stage", "virtual-stage", "null-apply"]
    raw, flags = user.staged["virtual"]
    assert flags == 0x10000011 and struct.unpack_from("<I", raw, 72)[0] == 0x5c0020
    assert struct.unpack_from("<ii", raw, 76) == (0, 0)
    assert struct.unpack_from("<II", raw, 172) == (1920, 1080)
    assert user.layout["physical"]["x"] == -2560 and user.layout["physical"]["y"] == 0
    assert user.layout["physical"]["active"] and not user.layout["physical"]["primary"]
    assert [config[0] for config in user.configs] == [0x84]
    assert all(not config[0] & 0xb for config in user.configs)  # no INTERNAL/CLONE/EXTERNAL


def test_activation_polls_before_any_cds_call():
    api, user = make_desktop()
    user.attach_after = 3
    api.extend_primary(INSTANCE, api.capture(), 1920, 1080)
    assert user.now == 0.4
    assert user.events.index("vdd-active") < user.events.index("physical-stage")


def test_inactive_source_fake_returns_named_minus_one():
    api, _ = make_desktop()
    with pytest.raises(RigError, match="ChangeDisplaySettingsExW.*DISP_CHANGE_FAILED.*-1"):
        api.change("virtual", devmode(1920, 1080), 0x10000011)


@pytest.mark.parametrize("code,symbol", [(-1, "FAILED"), (-2, "BADMODE"), (-3, "NOTUPDATED"),
                                       (-4, "BADFLAGS"), (-5, "BADPARAM"), (-6, "BADDUALVIEW"), (1, "RESTART")])
def test_cds_errors_include_symbol_code_and_mode_inputs(code, symbol):
    api, user = make_desktop()
    user.attached, user.primary_rc = True, code
    mode = devmode(1920, 1080)
    struct.pack_into("<I", mode, 72, 0x5c0020)
    with pytest.raises(RigError, match=rf"ChangeDisplaySettingsExW.*DISP_CHANGE_{symbol}.*{code}"):
        api.change("virtual", mode, 0x10000011)
    inputs = api.display_calls[-1]["inputs"]
    assert inputs["position"] == [0, 0] and inputs["size"] == [1920, 1080]
    assert inputs["frequency"] == 60 and inputs["bits_per_pixel"] == 32
    assert inputs["flags"] == "0x10000011" and inputs["dmFields"] == "0x5c0020"


def test_attach_timeout_has_no_cds_and_rolls_back_identically():
    api, user = make_desktop()
    user.attach_after, user.validate_rc = None, 87
    worker = lifecycle(api)
    with pytest.raises(RigError, match="VDD activation timed out.*SetDisplayConfig.*0x84"):
        worker.start()
    assert not any(event.endswith("-stage") or event == "null-apply" for event in user.events)
    assert user.now == 10
    assert_rollback(worker, user)


@pytest.mark.parametrize("inactive", [False, True])
def test_supplied_fallback_validates_before_apply_and_preserves_targets(inactive):
    api, user = make_desktop()
    user.primary_rc = -1
    if inactive:
        user.attach_after = None
    worker = lifecycle(api)
    worker.start()
    flags = [config[0] for config in user.configs]
    assert flags == [0x84, 0x60, 0xa0]
    assert all(not flag & (0x200 | 0x400) for flag in flags)
    validate, apply = user.configs[-2:]
    assert validate[1:] == apply[1:]
    physical = next(validate[2][i:i + 72] for i in range(0, len(validate[2]), 72)
                    if struct.unpack_from("<Q", validate[2], i)[0] == 1)
    assert struct.unpack_from("<QI", physical, 20) == (1, 10)
    original = worker.row["original"]
    assert physical == base64.b64decode(original["paths"])
    saved_modes = base64.b64decode(original["modes"])
    assert validate[4][:28] == saved_modes[:28] and validate[4][36:128] == saved_modes[36:128]
    assert user.layout["virtual"]["primary"] and user.layout["virtual"]["x"] == user.layout["virtual"]["y"] == 0
    assert user.layout["physical"]["active"] and user.layout["physical"]["x"] == -2560
    assert "query-active" in user.events[user.events.index("extend") + 1:user.events.index("validate")]
    if not inactive:
        call = next(row for row in worker.row["display_calls"] if row["return_code"] == -1)
        assert call["api"] == "ChangeDisplaySettingsExW" and call["inputs"]["device"] == "virtual"


def test_supplied_validate_failure_never_applies_and_is_receipted():
    api, user = make_desktop()
    user.primary_rc, user.validate_rc = -1, 87
    worker = lifecycle(api)
    with pytest.raises(RigError, match="SetDisplayConfig.*ERROR_INVALID_PARAMETER.*87"):
        worker.start()
    assert "supplied-apply" not in user.events
    error = next(row for row in worker.row["display_calls"] if row["return_code"] == 87)
    assert error["inputs"]["flags"] == "0x60" and error["inputs"]["path_count"] == 2
    assert any("DISP_CHANGE_FAILED" in message for message in worker.row["errors"])
    assert_rollback(worker, user)


@pytest.mark.parametrize("failure,symbol", [("extend_rc", "ERROR_ACCESS_DENIED"), ("physical_rc", "DISP_CHANGE_BADMODE"),
                                            ("null_rc", "DISP_CHANGE_NOTUPDATED"), ("supplied_rc", "ERROR_GEN_FAILURE")])
def test_api_failures_have_named_inputs_and_exact_rollback(failure, symbol):
    api, user = make_desktop()
    setattr(user, failure, {"extend_rc": 5, "physical_rc": -2, "null_rc": -3, "supplied_rc": 31}[failure])
    if failure == "supplied_rc":
        user.primary_rc = -1
    worker = lifecycle(api)
    with pytest.raises(RigError, match=symbol):
        worker.start()
    assert any(row["return_name"] == symbol and row["inputs"] for row in worker.row["display_calls"])
    if failure != "supplied_rc":
        assert "validate" not in user.events
    assert_rollback(worker, user)


@pytest.mark.parametrize("field", ["wrong_physical", "wrong_physical_y", "wrong_virtual"])
def test_wrong_final_positions_refuse_and_roll_back(field):
    api, user = make_desktop()
    setattr(user, field, True)
    worker = lifecycle(api)
    with pytest.raises(RigError, match="position|layout"):
        worker.start()
    assert_rollback(worker, user)


def test_plan_primary_queries_without_mutating():
    api, user = make_desktop()
    before = deepcopy(user.layout)
    plan = api.plan_primary(INSTANCE, api.capture(), 1920, 1080)
    assert plan["path_count"] == 2 and plan["mode_count"] >= 3
    assert plan["layout"]["virtual"] == [0, 0, 1920, 1080]
    assert plan["layout"]["physical"] == [-2560, 0, 2560, 1440]
    assert not user.configs and not user.staged and user.layout == before


def test_plan_primary_cli_has_no_mutation_or_session_recovery(monkeypatch, capsys):
    api, user = make_desktop()
    monkeypatch.setattr("tools.testrig.cli.NativeDisplay", lambda: api, raising=False)
    assert main(["display", "plan-primary", "--vdd-device", INSTANCE]) == 0
    assert json.loads(capsys.readouterr().out)["layout"]["virtual"] == [0, 0, 1920, 1080]
    assert not user.configs and not user.enabled
