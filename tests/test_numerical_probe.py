"""Numerical controls must retain all evidence before assigning relation labels."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.eval import numerical_probe as probe


@pytest.mark.parametrize("fault", ["future", "clipped", "dropped", "none"])
def test_numeric_inputs_and_relation_mapping(tmp_path, monkeypatch, fault):
    manifest = {"as_of": "2016-01-12", "sources": [{"published": "2016-01-13" if fault == "future" else "2016-01-12"}],
                "evidence_arms": {"source": [["title", 0, "word " * 65 if fault == "clipped" else "value"]]},
                "pairs": [{"id": "pair", "claims": {"a": "claim"}, "expected": {"source": {"a": "contradicted"}}}]}
    path, output = tmp_path / "cases.json", tmp_path / "results.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr("sys.argv", ["probe", str(path), str(output)])
    if fault in ("future", "clipped"):
        monkeypatch.setattr(probe, "VerdictRuntime", lambda _: pytest.fail("loaded invalid inputs"))
    else:
        (tmp_path / "onnx").mkdir()
        (tmp_path / "onnx/verdict.onnx").write_bytes(b"fixture")
        (tmp_path / "calibration.json").write_text(json.dumps({"calibrator": {"name": "uncalibrated"}}))
        (tmp_path / "contract.json").write_text("{}")
        monkeypatch.setattr(probe, "VerdictRuntime", lambda _: SimpleNamespace(root=tmp_path))
        monkeypatch.setattr(probe, "score_inputs", lambda *args: [{"wording": "a", "evidence_arm": "source",
                            "evidence_used": 0 if fault == "dropped" else 1, "evidence_offered": 1}])
    if fault == "none":
        probe.main()
        result = json.loads(output.read_text())
        assert result["results"][0]["arms"][0]["expected_evidence_relation"] == "contradicted"
        assert result["manifest"] == manifest
    else:
        with pytest.raises(ValueError, match={"future": "published", "clipped": "clipped", "dropped": "dropped"}[fault]):
            probe.main()
        assert not output.exists()
