"""Replay CLI preserves ranges and rejects invalid pacing."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


def test_replay_cli_burst_preserves_original_indices_and_limit(tmp_path):
    path = tmp_path / "speech.txt"
    path.write_text("Jobs grew. Wages rose. Inflation fell.", encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "scripts.replay_transcript", str(path),
                             "--speed", "0", "--start-index", "1", "--limit", "1"],
                            capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert result.returncode == 0, result.stderr
    updates = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(updates) == 1
    assert updates[0]["id"] == "1" and updates[0]["text"] == "Wages rose."


@pytest.mark.parametrize("args", [["--speed", "-1"], ["--speed", "nan"], ["--limit", "-1"], ["--start-index", "-1"]])
def test_replay_cli_refuses_invalid_timing_or_ranges(tmp_path, args):
    path = tmp_path / "speech.txt"
    path.write_text("Jobs grew.", encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "scripts.replay_transcript", str(path), *args],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
