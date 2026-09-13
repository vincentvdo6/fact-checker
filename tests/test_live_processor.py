"""Skipped and failed claims must not acquire a verdict through the live adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.pipeline.context import SpeechMetadata
from src.pipeline.live_processor import LiveProcessor
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.web_research import WebResearch
from src.retrieval.google_news_sources import GoogleNewsSources


def processor(worthy):
    instance = LiveProcessor.__new__(LiveProcessor)
    instance.metadata = SpeechMetadata(country="United States")
    instance.query_mode = "context"
    instance.detector = SimpleNamespace(decide=lambda _: SimpleNamespace(worthy=worthy, reason="rule", score=0.4))
    return instance


def context():
    return ClaimContext(TranscriptUpdate("a", 0, "Jobs increased.", 0, 1, True), ())


def test_filter_rejection_never_calls_verifier():
    instance = processor(False)
    instance.verifier = SimpleNamespace(verify=lambda *a, **k: pytest.fail("rejected claim was verified"))
    result = instance.process(context())
    assert result["status"] == "skipped"
    assert "verdict" not in result


def test_inference_failure_is_not_an_abstention():
    instance = processor(True)

    def broken(*args, **kwargs):
        raise RuntimeError("graph failed")

    instance.verifier = SimpleNamespace(verify=broken)
    with pytest.raises(RuntimeError, match="graph failed"):
        instance.process(context())


def test_live_result_keeps_original_claim_query_and_calibration_scope():
    instance = processor(True)
    calls = []
    judgement = SimpleNamespace(outcome="answered", verdict="supported", predicted="supported", band="strong",
                                confidence=0.9, sufficiency=0.6, probabilities=(0.9, 0.05, 0.05),
                                evidence=(("Jobs", 1, "Employment increased."),))
    instance.verifier = SimpleNamespace(verify=lambda claim, **kw: calls.append((claim, kw)) or judgement)
    result = instance.process(context())
    assert calls == [("Jobs increased.", {"query": "Jobs increased. United States"})]
    assert result["status"] == "verified"
    assert result["verdict"] == "supported"
    assert result["evidence"] == [["Jobs", 1, "Employment increased."]]
    assert result["calibration_scope"] == "FEVER only; live verdict accuracy is unmeasured"


def test_topic_search_abstains_without_running_model_when_only_distractors_return():
    instance = processor(True)
    instance.query_mode = "topic"
    instance.verifier = SimpleNamespace(
        retrieve=lambda _: ([("We_Don't_Care", 0, "We Don't Care is a song.")], [5.0]),
        judge=lambda *args: pytest.fail("irrelevant evidence reached verdict model"),
    )
    result = instance.process(ClaimContext(TranscriptUpdate("a", 0, "We don't have a labor shortage.", 0, 1, True), ()))
    assert result["outcome"] == "declined_no_relevant_evidence"
    assert result["verdict"] is None and result["evidence"] == []
    assert "confidence" not in result and "predicted" not in result


@pytest.mark.parametrize("boundary", [". We", ", we", "; we", ", but we"])
def test_compound_contrast_shows_background_without_claiming_the_model_settled_it(boundary):
    instance = processor(True)
    instance.query_mode = "topic"
    claim = "We don't have a labor shortage" + boundary + " have a good job shortage."
    evidence = [("Labor", 0, "Labor shortage refers to insufficient workers.")]
    instance.verifier = SimpleNamespace(
        retrieve=lambda _: (evidence, [4.0]),
        judge=lambda *args: pytest.fail("atomic model used to settle a compound contrast"),
    )
    result = instance.process(ClaimContext(TranscriptUpdate("a", 0, claim, 0, 1, True), ()))
    assert result["claim"] == claim
    assert result["outcome"] == "declined_compound_claim"
    assert result["verdict"] is None and result["evidence"] == [list(evidence[0])]
    assert "confidence" not in result and "predicted" not in result


def test_topic_filter_passes_original_claim_and_aligned_scores_to_judge():
    instance = processor(True)
    instance.query_mode = "topic"
    evidence = [("Song", 0, "We Don't Care is a song."), ("Labor", 1, "Jobs increased in 2010.")]
    calls = []
    judgement = SimpleNamespace(outcome="declined_both", verdict=None, predicted="supported", band=None,
                                confidence=0.4, sufficiency=0.3, probabilities=(0.4, 0.3, 0.3), evidence=(evidence[1],))
    instance.verifier = SimpleNamespace(
        retrieve=lambda _: (evidence, [5.0, 2.0]),
        judge=lambda *args: calls.append(args) or judgement,
    )
    claim = "Jobs increased in 2010."
    result = instance.process(ClaimContext(TranscriptUpdate("a", 0, claim, 0, 1, True), ()))
    assert calls == [(claim, [evidence[1]], [2.0])]
    assert result["verdict"] is None


def test_web_research_does_not_feed_web_scores_into_fever_calibration():
    instance = processor(True)
    instance.query_mode = "topic"
    instance.verifier = SimpleNamespace(
        retrieve=lambda _: ([], []),
        judge=lambda *args: pytest.fail("web source reached FEVER calibration"),
    )
    calls = []
    instance.web = SimpleNamespace(review=lambda *args: calls.append(args) or {"sources": [], "errors": []})
    claim = context()
    result = instance.process(claim)
    assert calls == [(claim, instance.metadata)]
    assert result["outcome"] == "declined_web_review" and result["verdict"] is None
    assert result["research"] == {"sources": [], "errors": []}
    assert "confidence" not in result and result["evidence"] == []


def test_news_mode_skips_discarded_offline_retrieval(monkeypatch):
    instance = processor(True)
    instance.query_mode = "topic"
    instance.verifier = SimpleNamespace(
        retrieve=lambda _: pytest.fail("News mode retrieved unused offline evidence"),
        judge=lambda *args: pytest.fail("old background settled a claim in news mode"),
    )
    sources = GoogleNewsSources()
    calls = []
    monkeypatch.setattr(sources, "search_before", lambda query, cutoff: calls.append(query) or [])
    instance.web = WebResearch(sources)
    result = instance.process(context())
    assert calls == ["Jobs increased. United States"]
    assert result["outcome"] == "declined_web_review" and result["verdict"] is None
    assert "confidence" not in result


def test_rejected_claim_does_not_load_offline_artifacts_and_can_close(monkeypatch):
    monkeypatch.setattr("src.pipeline.live_processor.DetectorFilter", lambda: processor(False).detector)
    monkeypatch.setattr("src.pipeline.live_processor.Verifier", lambda: pytest.fail("Skipped claim loaded artifacts"))
    instance = LiveProcessor(SpeechMetadata())
    assert instance.process(context())["status"] == "skipped"
    instance.close()


@pytest.mark.parametrize("mode", ["claim", "context", "topic"])
def test_offline_verifier_loads_once_on_first_use_and_closes(monkeypatch, mode):
    calls = []
    judgement = SimpleNamespace(outcome="declined_both", verdict=None, predicted="supported", band=None,
                                confidence=0.4, sufficiency=0.3, probabilities=(0.4, 0.3, 0.3), evidence=())

    def verifier():
        calls.append("load")
        return SimpleNamespace(
            retrieve=lambda _: ([("Jobs", 0, "Jobs increased.")], [2.0]),
            judge=lambda *args: judgement, verify=lambda *args, **kwargs: judgement,
            close=lambda: calls.append("close"),
        )

    monkeypatch.setattr("src.pipeline.live_processor.DetectorFilter", lambda: processor(True).detector)
    monkeypatch.setattr("src.pipeline.live_processor.Verifier", verifier)
    instance = LiveProcessor(SpeechMetadata(), query_mode=mode)
    assert calls == []
    for _ in range(2):
        assert instance.process(context())["outcome"] == "declined_both"
    assert calls == ["load"]
    instance.close()
    assert calls == ["load", "close"]


def test_offline_loading_failure_propagates_when_first_needed(monkeypatch):
    def broken():
        raise OSError("missing local model")

    monkeypatch.setattr("src.pipeline.live_processor.DetectorFilter", lambda: processor(True).detector)
    monkeypatch.setattr("src.pipeline.live_processor.Verifier", broken)
    instance = LiveProcessor(SpeechMetadata())
    with pytest.raises(OSError, match="missing local model"):
        instance.process(context())
    instance.close()


def test_decomposed_draft_is_attached_only_with_a_judge_and_never_as_a_verdict(monkeypatch):
    instance = processor(True)
    instance.query_mode = "topic"
    instance.verifier = SimpleNamespace(retrieve=lambda _: pytest.fail("offline retrieval ran"),
                                        judge=lambda *args: pytest.fail("offline judge ran"))
    sources = GoogleNewsSources()
    monkeypatch.setattr(sources, "search_before", lambda query, cutoff: [])
    instance.web = WebResearch(sources)
    assert "decomposed" not in instance.process(context())
    instance.judge = lambda assertions, units: []
    result = instance.process(context())
    assert result["verdict"] is None and result["outcome"] == "declined_web_review"
    draft = result["decomposed"]
    assert draft["status"].startswith("draft") and draft["verdict"]["relationship"] == "insufficient"
    assert draft["verdict"]["eligible"] == 0 and draft["text"].startswith("Claim:")
    assert "notes" not in result
    instance.judge, instance.notes = None, ["Sentence-level reading is off: the pair judge is not installed."]
    result = instance.process(context())
    assert "decomposed" not in result and result["notes"] == instance.notes
