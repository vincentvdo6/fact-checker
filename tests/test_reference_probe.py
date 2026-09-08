"""Fixed SOTU comparisons cannot silently omit cases or report unpacked title hits."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.eval import reference_probe as probe


@pytest.mark.parametrize("fault", ["source", "missing", "duplicate", "none"])
def test_fixed_transcript_and_packed_rank(tmp_path, monkeypatch, fault):
    rows = [{"index": i, "text": "Jobs grew."} for i in range(150)]
    rows[87]["text"] = "The Affordable Care Act changed."
    rows[91]["text"] = "Jobs grew since it became law."
    if fault == "missing":
        rows.pop()
    if fault == "duplicate":
        rows[-1] = rows[-2]
    transcript = {"transcript": "sotu-2016", "source": "wrong" if fault == "source" else probe.SOTU_SOURCE, "rows": rows}
    path, output = tmp_path / "cases.json", tmp_path / "results.json"
    path.write_text(json.dumps(transcript))
    monkeypatch.setattr("sys.argv", ["probe", str(path), str(output)])
    calls, closed = [], []
    if fault != "none":
        monkeypatch.setattr(probe, "Verifier", lambda: pytest.fail("retrieved invalid transcript"))
        with pytest.raises(ValueError, match="Expected"):
            probe.main()
        assert not output.exists()
        return
    evidence = [("Other", 0, "text"), (probe.TARGET_TITLE, 0, "text")]
    runtime = SimpleNamespace(contract=SimpleNamespace(max_length=1), measure=lambda *args: 0)
    verifier = SimpleNamespace(runtime=runtime, retrieve=lambda query: (evidence, [2.0, 1.0]), close=lambda: closed.append(True))
    monkeypatch.setattr(probe, "Verifier", lambda: verifier)
    monkeypatch.setattr(probe, "pack", lambda claim, *args: calls.append(claim) or 1)
    probe.main()
    result = json.loads(output.read_text())
    assert [row["index"] for row in result["results"]] == list(probe.CASES)
    for row in result["results"]:
        for arm in row["arms"].values():
            assert arm["packed_count"] == 1 and len(arm["packed"]) == 1
            assert arm["aca_first_packed_rank"] is None
            assert arm["query"]["claim"] == rows[row["index"]]["text"]
    law = next(r for r in result["results"] if r["index"] == 91)
    assert law["arms"]["previous_context"]["query"]["references"] == []
    assert law["arms"]["law_context"]["query"]["references"][0]["segment_id"] == "87"
    assert calls == [rows[i]["text"] for i in probe.CASES for _ in range(3)]
    assert closed == [True]
