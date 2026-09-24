"""The native host launcher must keep recording opt-in and fail closed on bad paths."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.install_youtube_extension import launcher_text


def test_launcher_quotes_record_path_and_clears_inherited_setting() -> None:
    workspace = Path(r"C:\work & media\repo")
    executable = workspace / ".venv/Scripts/python.exe"
    record = workspace / "runs/clicks & review"

    recording = launcher_text(workspace, executable, record).splitlines()
    assert 'set "FACT_CHECKER_RECORD_CLICKS="' in recording
    assert f'set "FACT_CHECKER_RECORD_CLICKS={record}"' in recording
    assert recording.index('set "FACT_CHECKER_RECORD_CLICKS="') < recording.index(
        f'set "FACT_CHECKER_RECORD_CLICKS={record}"'
    )
    assert f'cd /d "{workspace}" || exit /b 1' in recording
    assert launcher_text(workspace, executable, None).splitlines().count('set "FACT_CHECKER_RECORD_CLICKS="') == 1
    assert f'set "FACT_CHECKER_RECORD_CLICKS={record}"' not in launcher_text(workspace, executable, None)


@pytest.mark.skipif(sys.platform != "win32", reason="executes a Windows cmd launcher")
def test_launcher_handles_ampersand_and_inherited_recording(tmp_path: Path) -> None:
    workspace = tmp_path / "repo & samples"
    workspace.mkdir()
    executable = workspace / "fake & python.cmd"
    executable.write_text('@echo off\necho invoked\necho cwd="%cd%"\necho record="%FACT_CHECKER_RECORD_CLICKS%"\n',
                          encoding="utf-8")
    launcher = tmp_path / "host.cmd"
    inherited = os.environ.copy()
    inherited["FACT_CHECKER_RECORD_CLICKS"] = "inherited-clicks"

    launcher.write_text(launcher_text(workspace, executable, None), encoding="utf-8")
    plain = subprocess.run(["cmd.exe", "/d", "/c", str(launcher)], cwd=tmp_path, env=inherited,
                           capture_output=True, text=True, check=False)
    assert plain.returncode == 0, plain.stderr
    assert "invoked" in plain.stdout
    assert f'cwd="{workspace}"' in plain.stdout
    assert 'record=""' in plain.stdout

    record = workspace / "clicks & review"
    launcher.write_text(launcher_text(workspace, executable, record), encoding="utf-8")
    recording = subprocess.run(["cmd.exe", "/d", "/c", str(launcher)], cwd=tmp_path, env=inherited,
                               capture_output=True, text=True, check=False)
    assert recording.returncode == 0, recording.stderr
    assert "invoked" in recording.stdout
    assert f'record="{record}"' in recording.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="executes a Windows cmd launcher")
def test_launcher_does_not_start_host_when_workspace_is_missing(tmp_path: Path) -> None:
    executable = tmp_path / "fake.cmd"
    executable.write_text('@echo off\necho invoked\n', encoding="utf-8")
    launcher = tmp_path / "host.cmd"
    launcher.write_text(launcher_text(tmp_path / "missing-workspace", executable, None), encoding="utf-8")

    result = subprocess.run(["cmd.exe", "/d", "/c", str(launcher)], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "invoked" not in result.stdout
