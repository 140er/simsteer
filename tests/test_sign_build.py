from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
THUMB = "0123456789ABCDEF0123456789ABCDEF01234567"


@pytest.fixture(scope="module")
def sign():
    spec = importlib.util.spec_from_file_location("sign_build", ROOT / "tools" / "sign_build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def signtool(tmp_path):
    path = tmp_path / "signtool.exe"
    path.write_bytes(b"")
    return path


def test_unset_skips_signing(sign, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("SIMSTEER_SIGN_CERT_SHA1", raising=False)
    assert sign.load_config({"SIMSTEER_SIGN_CERT_SHA1": "  "}) is None
    assert sign.main(["--dist", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip().splitlines() == [
        "sign_build: SIMSTEER_SIGN_CERT_SHA1 not set, signing skipped."]


def test_thumbprint_copied_from_the_certificate_dialog(sign):
    pasted = "\u200e01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef 01 23 45 67"
    assert sign.clean_thumbprint(pasted) == THUMB


@pytest.mark.parametrize("bad", ["abc", THUMB + "0", "Z" * 40, "C:\\certs\\simsteer.pfx"])
def test_bad_thumbprint_is_refused(sign, bad):
    with pytest.raises(sign.SignError, match="40-character"):
        sign.clean_thumbprint(bad)


def test_config_defaults_and_overrides(sign, signtool):
    env = {"SIMSTEER_SIGN_CERT_SHA1": THUMB, "SIMSTEER_SIGN_TOOL": f'"{signtool}"'}
    cfg = sign.load_config(env)
    assert cfg == sign.SignConfig(THUMB, signtool, sign.DEFAULT_TIMESTAMP_URL)

    env["SIMSTEER_SIGN_TIMESTAMP_URL"] = "http://ts.example.test"
    assert sign.load_config(env).timestamp_url == "http://ts.example.test"

    env["SIMSTEER_SIGN_TIMESTAMP_URL"] = "ts.example.test"
    with pytest.raises(sign.SignError, match="http"):
        sign.load_config(env)


def test_missing_signtool_is_named(sign, tmp_path):
    env = {"SIMSTEER_SIGN_CERT_SHA1": THUMB, "SIMSTEER_SIGN_TOOL": str(tmp_path / "nope.exe")}
    with pytest.raises(sign.SignError, match="does not exist"):
        sign.load_config(env)


def test_newest_sdk_signtool_wins(sign, tmp_path, monkeypatch):
    monkeypatch.setattr(sign.shutil, "which", lambda name: None)
    for ver in ("10.0.9600.0", "10.0.22621.0", "10.0.19041.0"):
        tool = tmp_path / "Windows Kits" / "10" / "bin" / ver / "x64" / "signtool.exe"
        tool.parent.mkdir(parents=True)
        tool.write_bytes(b"")
    found = sign.find_signtool({"ProgramFiles(x86)": str(tmp_path)})
    assert found.parent.parent.name == "10.0.22621.0"

    with pytest.raises(sign.SignError, match="Windows SDK"):
        sign.find_signtool({"ProgramFiles(x86)": str(tmp_path / "empty")})


def test_commands_sign_both_exes_with_sha256_and_timestamp(sign, tmp_path, signtool):
    for name in sign.EXES:
        (tmp_path / name).write_bytes(b"MZ")
    cfg = sign.SignConfig(THUMB, signtool, "http://ts.example.test")
    files = sign.targets(tmp_path)
    assert [f.name for f in files] == ["SimSteer.exe", "SimSteer-debug.exe"]

    cmd = sign.sign_command(cfg, files)
    assert cmd[:2] == [str(signtool), "sign"]
    for flag, value in (("/sha1", THUMB), ("/fd", "SHA256"), ("/tr", "http://ts.example.test"),
                        ("/td", "SHA256")):
        assert cmd[cmd.index(flag) + 1] == value
    assert cmd[-2:] == [str(f) for f in files]
    assert sign.verify_command(cfg, files) == [str(signtool), "verify", "/pa", *map(str, files)]


def test_empty_dist_is_an_error(sign, tmp_path):
    with pytest.raises(sign.SignError, match="Run PyInstaller first"):
        sign.targets(tmp_path)


@pytest.mark.parametrize("results,steps,expected", [
    ([7], ["sign"], 7),
    ([0, 9], ["sign", "verify"], 9),
    ([0, 0], ["sign", "verify"], 0),
])
def test_sign_and_verify_results_reach_the_build(sign, tmp_path, signtool, monkeypatch,
                                                results, steps, expected):
    monkeypatch.setenv("SIMSTEER_SIGN_CERT_SHA1", THUMB)
    monkeypatch.setenv("SIMSTEER_SIGN_TOOL", str(signtool))
    for name in sign.EXES:
        (tmp_path / name).write_bytes(b"MZ")
    called = []
    codes = iter(results)

    def call(cmd):
        called.append(cmd[1])
        return next(codes)

    monkeypatch.setattr(sign.subprocess, "call", call)
    assert sign.main(["--dist", str(tmp_path)]) == expected
    assert called == steps


def test_signtool_launch_error_is_clear(sign, tmp_path, signtool, monkeypatch, capsys):
    monkeypatch.setenv("SIMSTEER_SIGN_CERT_SHA1", THUMB)
    monkeypatch.setenv("SIMSTEER_SIGN_TOOL", str(signtool))
    (tmp_path / "SimSteer.exe").write_bytes(b"MZ")

    def call(cmd):
        raise OSError("permission denied")

    monkeypatch.setattr(sign.subprocess, "call", call)
    assert sign.main(["--dist", str(tmp_path)]) == 2
    assert "signtool sign could not start: permission denied" in capsys.readouterr().err
