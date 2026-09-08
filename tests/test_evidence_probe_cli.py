"""Reject future sources before model loading and record missing exports."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.eval import evidence_probe as probe


def test_future_source_rejected_before_inference(tmp_path, monkeypatch):
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"as_of": "2016-01-12", "cases": [{"sources": [{"published": "2016-01-13", "url": "future"}]}]}))
    output = tmp_path / "results.json"
    monkeypatch.setattr("sys.argv", ["probe", str(manifest), str(output)])
    monkeypatch.setattr(probe, "VerdictRuntime", lambda _: pytest.fail("loaded future evidence"))
    with pytest.raises(ValueError, match="Future evidence"):
        probe.main()
    assert not output.exists()


def test_missing_exports_are_explicit(tmp_path, monkeypatch):
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"as_of": "2016-01-12", "cases": []}))
    output = tmp_path / "results.json"
    monkeypatch.setattr("sys.argv", ["probe", str(manifest), str(output)])
    monkeypatch.setattr(probe, "VerdictRuntime", lambda name: SimpleNamespace(root=tmp_path / name))
    probe.main()
    result = json.loads(output.read_text())
    assert set(result["models"]) == {"retrieved", "claim_only"}
    assert all(m["status"] == "unavailable" for m in result["models"].values())
    assert result["results"] == []
