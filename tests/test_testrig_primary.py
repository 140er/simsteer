"""Inactive-source activation, primary switching and exact display rollback."""

import base64
from copy import deepcopy
import ctypes
import json
import os
import struct
from types import SimpleNamespace

import pytest

from tools.testrig import RigError
from tools.testrig.cli import main
from tools.testrig.display import DisplayLifecycle, recover_leftover_displays, worker
from tools.testrig.display_api import NativeDisplay, encoded
from tools.testrig.state import lock, read_json


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
        self.visible = True
        self.attach_after, self.polls, self.now = 1, 0, 0.0
        self.extend_rc = self.primary_rc = self.physical_rc = self.null_rc = 0
        self.validate_rc = self.supplied_rc = 0
        self.restore_rc = 0
        self.enable_error = self.disable_error = False
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
        if flags == 2 and self.enabled and self.extending and not self.attached and self.attach_after is not None:
            self.polls += 1
            if self.polls >= self.attach_after:
                self.attached = True
                self.layout["virtual"] = self.display("virtual", 1920, 1080, 2560, 0, False)
                self.events.append("vdd-active")
        modes = [source_mode(1, 2560, 1440, self.layout["physical"]["x"], self.layout["physical"]["y"]),
                 bytearray(64)]
        struct.pack_into("<IIQ", modes[1], 0, 2, 10, 1)
        paths = []
        for adapter in ([1, 2] if self.enabled and self.visible and (flags == 1 or self.attached) else [1]):
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
            if self.restore_rc:
                return self.restore_rc
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
    def enable(_):
        user.enabled = True
        user.events.append("enable")
        if user.enable_error:
            raise RigError("Enable-PnpDevice failed after enable")
    def disable(_):
        user.events.append("disable")
        if user.disable_error:
            raise RigError("Disable-PnpDevice failed")
        user.enabled = False
    api.enable, api.disable = enable, disable
    presence = api.wait_virtual
    api.wait_virtual = lambda instance: presence(instance, clock=lambda: user.now, pause=user.pause)
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
    api.enable(INSTANCE)
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
    api.enable(INSTANCE)
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
    with pytest.raises(RigError, match="VDD activation timed out.*flags=0x84"):
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
    api.enable(INSTANCE)
    before = deepcopy(user.layout)
    plan = api.plan_primary(INSTANCE, api.capture(), 1920, 1080)
    assert plan["path_count"] == 2 and plan["mode_count"] >= 3
    assert plan["layout"]["virtual"] == [0, 0, 1920, 1080]
    assert plan["layout"]["physical"] == [-2560, 0, 2560, 1440]
    assert not user.configs and not user.staged and user.layout == before


@pytest.fixture
def dry_run(tmp_path, monkeypatch):
    api, user = make_desktop()
    caller = {"pid": os.getpid(), "created": "plan-caller", "path": "python.exe", "born": 1}
    snapshot = {"boot_id": "counter:10", "processes": [caller], "errors": []}
    backend = SimpleNamespace(snapshot=lambda: deepcopy(snapshot))
    monkeypatch.setattr("tools.testrig.cli.NativeDisplay", lambda: api)
    monkeypatch.setattr("tools.testrig.display.NativeDisplay", lambda: api)
    monkeypatch.setattr("tools.testrig.display.BlackOverlay", Overlay)
    root = tmp_path / "state"
    args = ["display", "plan-primary", "--vdd-device", INSTANCE, "--state-dir", str(root)]
    return SimpleNamespace(api=api, user=user, root=root, args=args, backend=backend, snapshot=snapshot,
                           receipt=root / "plan-primary/default/display-state.json")


def test_disabled_vdd_is_absent_and_requires_explicit_enable(dry_run, capsys):
    rig = dry_run
    with pytest.raises(RigError, match="found 0"):
        rig.api.plan_primary(INSTANCE, rig.api.capture(), 1920, 1080)
    assert main(rig.args, backend=rig.backend) == 2
    output = capsys.readouterr()
    assert "VDD is disabled; rerun with --enable-vdd to enable it for this dry run" in output.err
    assert not output.out and "enable" not in rig.user.events and not rig.user.configs


def test_enabled_read_only_plan_does_not_disable_another_session(dry_run, capsys):
    rig = dry_run
    rig.user.enabled = True
    assert main(rig.args, backend=rig.backend) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["layout"]["virtual"] == [0, 0, 1920, 1080]
    assert rig.user.enabled and not rig.user.configs
    assert "disable" not in rig.user.events


def test_temporary_enable_plan_restores_bytes_before_disable(dry_run, capsys):
    rig = dry_run
    original = rig.api.capture()
    saved = rig.api.enable
    def enable(instance):
        receipt = read_json(rig.receipt)
        assert receipt["original"] == original
        assert receipt["enable_intent"] and receipt["topology_intent"]
        saved(instance)
    rig.api.enable = enable
    plan = rig.api.plan_primary
    def compute(*args):
        rig.user.events.append("plan")
        return plan(*args)
    rig.api.plan_primary = compute
    assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["physical_targets"] == [{"name": "physical", "target": "physical-monitor", "active": True}]
    receipt = read_json(rig.receipt)
    assert receipt["plan"] == {key: value for key, value in output.items() if key not in ("dry_run", "activation_flags")}
    assert_rollback(SimpleNamespace(row=receipt), rig.user)
    assert [event for event in rig.user.events if event in ("enable", "plan", "restore", "disable")] == [
        "enable", "plan", "restore", "disable"]
    assert [row[0] for row in rig.user.configs] == [0x2a0]  # restore only; no extend/apply/validate


@pytest.mark.parametrize("failure", ["capture", "enable", "interrupt", "presence", "plan", "restore", "disable"])
def test_temporary_plan_failure_matrix(dry_run, monkeypatch, capsys, failure):
    rig = dry_run
    if failure == "capture":
        monkeypatch.setattr(rig.api, "capture", lambda: (_ for _ in ()).throw(RigError("capture failed")))
    elif failure == "enable":
        rig.user.enable_error = True
    elif failure == "interrupt":
        enable = rig.api.enable
        def interrupted(instance):
            enable(instance)
            raise KeyboardInterrupt("interrupted enable")
        monkeypatch.setattr(rig.api, "enable", interrupted)
    elif failure == "presence":
        rig.user.visible = False
    elif failure == "plan":
        monkeypatch.setattr(rig.api, "plan_primary", lambda *a: (_ for _ in ()).throw(RigError("plan failed")))
    elif failure == "restore":
        rig.user.restore_rc = 31
    elif failure == "disable":
        rig.user.disable_error = True
    assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    output = capsys.readouterr()
    assert not output.out
    expected = {"capture": ["capture failed"], "enable": ["Enable-PnpDevice"], "interrupt": ["interrupted enable"],
                "presence": ["timed out"], "plan": ["plan failed"],
                "restore": ["SetDisplayConfig", "ERROR_GEN_FAILURE", "31"], "disable": ["Disable-PnpDevice"]}
    assert all(text in output.err for text in expected[failure])
    if failure != "capture":
        assert "restore" in rig.user.events and "disable" in rig.user.events
        assert rig.user.events.index("restore") < rig.user.events.index("disable")
    assert rig.user.enabled == (failure == "disable")
    receipt = read_json(rig.receipt)
    assert not receipt["ready"]
    if failure not in ("capture", "restore", "disable"):
        assert_rollback(SimpleNamespace(row=receipt), rig.user)


def test_enable_plan_refuses_live_device_and_command_lock(dry_run, capsys):
    rig = dry_run
    rig.user.enabled = True
    assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    assert "already enabled" in capsys.readouterr().err
    assert rig.user.enabled and "disable" not in rig.user.events
    rig.user.enabled = False
    with lock(rig.root):
        assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    assert "Another test-rig command is running" in capsys.readouterr().err
    assert "enable" not in rig.user.events


@pytest.mark.parametrize("state", [0, 28])
def test_enable_plan_requires_problem_code_22(dry_run, monkeypatch, capsys, state):
    rig = dry_run
    monkeypatch.setattr(rig.api, "device_state", lambda _: state)
    assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    assert "VDD must be disabled" in capsys.readouterr().err
    assert "enable" not in rig.user.events and "disable" not in rig.user.events


@pytest.mark.parametrize("errors", [None, ["inventory denied"]])
def test_enable_plan_requires_readable_supervisor_inventory(dry_run, capsys, errors):
    rig = dry_run
    rig.snapshot["errors"] = errors
    assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    assert "process inventory is unreadable" in capsys.readouterr().err
    assert "enable" not in rig.user.events


@pytest.mark.parametrize("boot", ["counter:10", "counter:11"])
def test_killed_plan_receipt_recovers_on_same_or_next_boot(dry_run, monkeypatch, capsys, boot):
    rig = dry_run
    # Emulate process death: the OS releases the lock but does not execute finally.
    with monkeypatch.context() as killed:
        killed.setattr(DisplayLifecycle, "close", lambda _: None)
        assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    assert not capsys.readouterr().out and rig.user.enabled
    rig.snapshot["processes"] = []
    rig.snapshot["boot_id"] = boot
    assert recover_leftover_displays(rig.root, rig.backend, rig.backend.snapshot()) == [str(rig.receipt.parent)]
    assert_rollback(SimpleNamespace(row=read_json(rig.receipt)), rig.user)
    before = list(rig.user.events)
    assert recover_leftover_displays(rig.root, rig.backend, rig.backend.snapshot()) == []
    assert rig.user.events == before


def test_plan_recovery_refuses_live_caller_or_pending_enable_child(dry_run, monkeypatch, capsys):
    rig = dry_run
    with monkeypatch.context() as killed:
        killed.setattr(DisplayLifecycle, "close", lambda _: None)
        assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    capsys.readouterr()
    assert recover_leftover_displays(rig.root, rig.backend, rig.backend.snapshot()) == []
    caller = rig.snapshot["processes"][0]
    rig.snapshot["processes"] = [{"pid": 900, "parent": caller["pid"], "born": 2,
                                   "created": "child", "path": "powershell.exe", "name": "powershell.exe"}]
    with pytest.raises(RigError, match="PowerShell.*still running"):
        recover_leftover_displays(rig.root, rig.backend, rig.backend.snapshot())
    assert rig.user.enabled and "restore" not in rig.user.events


def test_plan_recovery_also_seals_a_crash_before_enable_intent(dry_run, monkeypatch, capsys):
    rig = dry_run
    calls = 0
    def state(_):
        nonlocal calls
        calls += 1
        if calls == 2:  # original saved, before the enable intent
            raise KeyboardInterrupt("killed before intent")
        return 22
    with monkeypatch.context() as killed:
        killed.setattr(rig.api, "device_state", state)
        killed.setattr(DisplayLifecycle, "close", lambda _: None)
        assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    capsys.readouterr()
    assert not read_json(rig.receipt)["enable_intent"]
    rig.snapshot["processes"] = []
    assert recover_leftover_displays(rig.root, rig.backend, rig.backend.snapshot())
    assert "enable" not in rig.user.events and "disable" not in rig.user.events
    assert all(read_json(rig.receipt)[key] for key in ("topology_restored", "vdd_disabled", "overlay_closed"))


def test_next_plan_command_recovers_its_killed_predecessor(dry_run, monkeypatch, capsys):
    rig = dry_run
    with monkeypatch.context() as killed:
        killed.setattr(DisplayLifecycle, "close", lambda _: None)
        assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    capsys.readouterr()
    rig.snapshot["processes"] = []
    assert main(rig.args, backend=rig.backend) == 2  # recovery then explicit-enable requirement
    output = capsys.readouterr()
    assert "recovered display layout" in output.err and "rerun with --enable-vdd" in output.err
    assert not output.out
    assert_rollback(SimpleNamespace(row=read_json(rig.receipt)), rig.user)


def test_plan_recovery_rechecks_liveness_and_refuses_unreadable_inventory(dry_run, monkeypatch, capsys):
    rig = dry_run
    with monkeypatch.context() as killed:
        killed.setattr(DisplayLifecycle, "close", lambda _: None)
        assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    capsys.readouterr()
    stale = {**rig.backend.snapshot(), "processes": []}
    assert recover_leftover_displays(rig.root, rig.backend, stale) == []  # caller is live under lock
    rig.snapshot["errors"] = ["process inventory denied"]
    with pytest.raises(RigError, match="process inventory is unreadable"):
        recover_leftover_displays(rig.root, rig.backend, stale)
    assert rig.user.enabled and "disable" not in rig.user.events


def test_failed_plan_disable_is_recoverable_without_repeating_restore(dry_run, capsys):
    rig = dry_run
    rig.user.disable_error = True
    assert main(rig.args + ["--enable-vdd"], backend=rig.backend) == 2
    assert not capsys.readouterr().out
    rig.user.disable_error = False
    rig.snapshot["processes"] = []
    assert recover_leftover_displays(rig.root, rig.backend, rig.backend.snapshot())
    assert_rollback(SimpleNamespace(row=read_json(rig.receipt)), rig.user)
    assert rig.user.events.count("restore") == 1 and rig.user.events.count("disable") == 2


@pytest.mark.parametrize("inactive", [False, True])
@pytest.mark.parametrize("change", ["missing", "inactive", "replaced-target"])
def test_fallback_refuses_physical_target_loss_before_validate(monkeypatch, inactive, change):
    api, user = make_desktop()
    user.primary_rc = -1
    if inactive:
        user.attach_after = None
    plan = api.plan_primary
    query = api.query
    def unsafe_plan(*args):
        queries = 0
        def changed(flags):
            nonlocal queries
            data = query(flags)
            if flags != 2:
                return data
            queries += 1
            if queries != 2:  # physical identity lookup, then the fresh topology
                return data
            paths = bytearray(base64.b64decode(data["paths"]))
            if change == "missing":
                paths = paths[72:]
                data["path_count"] -= 1
            elif change == "inactive":
                struct.pack_into("<I", paths, 68, 0)
            else:
                struct.pack_into("<I", paths, 28, 99)
            return {**data, "paths": encoded(paths)}
        with monkeypatch.context() as changing:
            changing.setattr(api, "query", changed)
            info = user.DisplayConfigGetDeviceInfo
            def renamed(data):
                if struct.unpack_from("<I", data)[0] == 2 and struct.unpack_from("<I", data, 16)[0] == 99:
                    value = "other-monitor\0".encode("utf-16-le")
                    data[164:164 + len(value)] = value
                    return 0
                return info(data)
            changing.setattr(user, "DisplayConfigGetDeviceInfo", renamed)
            return plan(*args)
    monkeypatch.setattr(api, "plan_primary", unsafe_plan)
    monitor = lifecycle(api)
    with pytest.raises(RigError, match="fallback would deactivate physical display physical-monitor"):
        monitor.start()
    assert "validate" not in user.events and "supplied-apply" not in user.events
    assert_rollback(monitor, user)


def test_swallowed_all_path_errors_do_not_accumulate_receipt_rows(monkeypatch):
    api, user = make_desktop()
    monkeypatch.setattr(user, "DisplayConfigGetDeviceInfo", lambda _: 87)
    for _ in range(52):
        assert api.sources() == []
    assert not getattr(api, "display_calls", [])
    with pytest.raises(RigError, match="DisplayConfigGetDeviceInfo.*87"):
        api.sources(active_only=True)
    assert len(api.display_calls) == 1


def test_swallowed_plan_candidate_error_is_not_recorded(monkeypatch):
    api, user = make_desktop()
    original = api.capture()
    api.enable(INSTANCE)
    info = user.DisplayConfigGetDeviceInfo
    monkeypatch.setattr(user, "DisplayConfigGetDeviceInfo", lambda data:
        87 if struct.unpack_from("<Q", data, 8)[0] == 2 else info(data))
    with pytest.raises(RigError, match="found 0"):
        api.plan_primary(INSTANCE, original, 1920, 1080)
    assert not getattr(api, "display_calls", [])


@pytest.mark.parametrize("restoring", [False, True])
def test_active_mode_failure_names_enum_api_device_and_bool_result(restoring):
    api, user = make_desktop()
    original = api.capture()
    api.active = lambda: NativeDisplay.active(api)
    api.devices = lambda: deepcopy(original["displays"])
    api.mode = lambda *a: None
    with pytest.raises(RigError, match="EnumDisplaySettingsExW.*FALSE.*0.*physical"):
        api.restore(original) if restoring else api.active()
    row = api.display_calls[-1]
    assert row["api"] == "EnumDisplaySettingsExW" and row["inputs"]["device"] == "physical"


def test_activation_timeout_does_not_invent_a_set_config_success_row():
    api, user = make_desktop()
    user.attach_after = None
    api.enable(INSTANCE)
    api._set_config(None, 0x84, "extend activation")
    with pytest.raises(RigError, match="VDD activation timed out") as error:
        api.wait_active_virtual(INSTANCE)
    assert "ERROR_SUCCESS" not in str(error.value)
    assert len(api.display_calls) == 1 and api.display_calls[0]["operation"] == "extend activation"


def test_set_config_layout_failure_labels_the_windows_error():
    api, _ = make_desktop()
    original = api.capture()
    api._set_config(None, 0x84, "extend activation")
    api.active = lambda: []
    with pytest.raises(RigError, match=r"SetDisplayConfig ERROR_SUCCESS \(Windows error 0\)"):
        api._verify_primary("virtual", original["displays"], 1920, 1080)


def test_worker_records_startup_exception_only_once(tmp_path, monkeypatch):
    api, _ = make_desktop()
    (tmp_path / "display-go").touch()
    state = {"display_parent": {}, "options": {"display": "vdd", "overlay": False}, "id": "test"}
    monkeypatch.setattr("tools.testrig.display.Journal", lambda _: SimpleNamespace(data=state))
    monkeypatch.setattr("tools.testrig.display.Windows", lambda: SimpleNamespace(wall_time=lambda: 20))
    monkeypatch.setattr("tools.testrig.display.ProcessHandle", lambda _: SimpleNamespace(alive=lambda: True, close=lambda: None))
    monkeypatch.setattr("tools.testrig.display.NativeDisplay", lambda: api)
    monkeypatch.setattr("tools.testrig.display.BlackOverlay", Overlay)
    monkeypatch.setattr("tools.testrig.activity.OwnerInput", lambda: SimpleNamespace(active=lambda: False, close=lambda: None))
    monkeypatch.setattr(api, "capture", lambda: (_ for _ in ()).throw(RigError("startup failure")))
    assert worker(tmp_path) == 1
    assert read_json(tmp_path / "display-state.json")["errors"] == ["startup failure"]
