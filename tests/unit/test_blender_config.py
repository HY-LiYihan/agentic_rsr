from pathlib import Path

import pytest

from real2sim import blender


@pytest.fixture
def executable(tmp_path, monkeypatch):
    monkeypatch.delenv("BLENDER_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    monkeypatch.setattr(blender.sys, "platform", "linux")
    monkeypatch.setattr(blender, "MACOS_BLENDER", tmp_path / "macos/Blender")

    def create(*parts, mode=0o755):
        path = tmp_path.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(mode)
        return path

    return create


def test_explicit_argument_precedes_environment_and_path(executable, monkeypatch):
    selected = executable("explicit", "Blender")
    environment = executable("environment", "Blender")
    executable("path", "blender")
    monkeypatch.setenv("BLENDER_BIN", str(environment))
    assert blender.resolve_blender(selected) == selected


def test_environment_precedes_path(executable, monkeypatch):
    selected = executable("environment", "Blender")
    executable("path", "blender")
    monkeypatch.setenv("BLENDER_BIN", str(selected))
    assert blender.resolve_blender() == selected


def test_default_discovers_linux_path(executable):
    selected = executable("path", "blender")
    assert blender.resolve_blender() == selected


def test_command_name_argument_is_resolved_on_path(executable):
    selected = executable("path", "blender")
    assert blender.resolve_blender("blender") == selected


@pytest.mark.parametrize("source", ["argument", "environment"])
def test_invalid_configuration_does_not_use_valid_default(executable, monkeypatch, source):
    executable("path", "blender")
    missing = "missing-blender-installation"
    if source == "environment":
        monkeypatch.setenv("BLENDER_BIN", missing)
        argument = None
    else:
        monkeypatch.setenv("BLENDER_BIN", str(executable("environment", "Blender")))
        argument = missing
    with pytest.raises(FileNotFoundError, match="Configured Blender executable"):
        blender.resolve_blender(argument)


@pytest.mark.parametrize("mode", [0o644, 0o600])
def test_non_executable_file_is_rejected(executable, mode):
    selected = executable("explicit", "Blender", mode=mode)
    with pytest.raises(FileNotFoundError, match="not executable"):
        blender.resolve_blender(selected)


def test_tilde_and_spaces_in_explicit_path(executable, tmp_path, monkeypatch):
    selected = executable("Blender Installation", "blender")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert blender.resolve_blender("~/Blender Installation/blender") == selected


def test_relative_explicit_path_is_made_absolute(executable, tmp_path, monkeypatch):
    selected = executable("tools", "blender")
    monkeypatch.chdir(tmp_path)
    assert blender.resolve_blender(Path("tools/blender")) == selected


def test_macos_fallback_when_path_is_empty(executable, monkeypatch):
    selected = executable("macos", "Blender")
    monkeypatch.setattr(blender.sys, "platform", "darwin")
    assert blender.resolve_blender() == selected


def test_path_precedes_macos_application(executable, monkeypatch):
    selected = executable("path", "blender")
    executable("macos", "Blender")
    monkeypatch.setattr(blender.sys, "platform", "darwin")
    assert blender.resolve_blender() == selected


def test_missing_linux_installation_reports_setup(executable):
    executable("macos", "Blender")
    with pytest.raises(FileNotFoundError, match="Install blender on PATH"):
        blender.resolve_blender()


def test_empty_environment_uses_path(executable, monkeypatch):
    selected = executable("path", "blender")
    monkeypatch.setenv("BLENDER_BIN", "")
    assert blender.resolve_blender() == selected


@pytest.mark.parametrize("command", ["inspect", "render", "render-rgb"])
def test_stage2_worker_cli_accepts_explicit_blender(executable, tmp_path, monkeypatch, command):
    from real2sim.stage2 import agent_runner

    selected = executable("Blender Installation", "blender")
    monkeypatch.setenv("BLENDER_BIN", "missing-environment-install")
    scene = tmp_path / "initial.blend"
    scene.write_bytes(b"scene")
    output = tmp_path / "output"
    calls = []

    def worker(path, source, action, destination):
        calls.append((path, source, action, destination))

    monkeypatch.setattr(agent_runner, "_run_blender", worker)
    assert agent_runner.main([command, "--scene", str(scene), "--output", str(output),
                              "--blender", str(selected)]) == 0
    assert calls == [(selected, scene, command, output)]
