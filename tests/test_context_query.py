"""Context may guide a search, but it must never become an invented assertion or evidence."""

from __future__ import annotations

import pytest

from src.pipeline.context import CONTEXT_WORDS, SpeechMetadata, build_query
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.verify import Verifier


def context(text="Our businesses added jobs since it became law.", previous="The Affordable Care Act became law."):
    return ClaimContext(
        TranscriptUpdate("b", 0, text, 2, 3, True),
        (TranscriptUpdate("a", 0, previous, 0, 1, True),),
    )


def test_default_search_is_byte_identical_to_the_claim():
    item = context()
    query = build_query(item, SpeechMetadata(country="United States", spoken_at="2016-01-12"))
    assert query.query == query.claim == item.claim.text
    assert query.context_ids == ()


def test_context_hints_are_auditable_without_resolving_ambiguous_references():
    item = context()
    query = build_query(item, SpeechMetadata(country="United States", spoken_at="2016-01-12"), mode="context")
    assert query.claim == item.claim.text
    assert query.query == item.claim.text + " United States 2016 The Affordable Care Act became law."
    assert query.context_ids == ("a",)
    assert "it became law" in query.claim


def test_self_contained_claim_does_not_inherit_unrelated_previous_speech():
    query = build_query(context("Unemployment fell in 2015."), SpeechMetadata(country="United States"), mode="context")
    assert query.query == "Unemployment fell in 2015. United States"
    assert query.context_ids == ()


def test_context_budget_keeps_recent_words_and_all_of_the_claim():
    previous = " ".join(f"word{i}" for i in range(100))
    query = build_query(context(previous=previous), SpeechMetadata(), mode="context")
    assert query.query.endswith(" ".join(previous.split()[-CONTEXT_WORDS:]))
    assert "word0 " not in query.query
    assert query.query.startswith(query.claim)


def test_source_url_is_provenance_and_never_a_query_hint():
    query = build_query(context(), SpeechMetadata(source="https://example.com/speech"), mode="context")
    assert "example.com" not in query.query


@pytest.mark.parametrize("kwargs", [{"spoken_at": "yesterday"}, {"country": None}, {"source": "a" * 2049}])
def test_invalid_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SpeechMetadata(**kwargs)


def test_unknown_query_mode_is_rejected():
    with pytest.raises(ValueError, match="query mode"):
        build_query(context(), SpeechMetadata(), mode="guess")


def test_query_changes_retrieval_but_not_the_model_claim(monkeypatch):
    verifier = Verifier.__new__(Verifier)
    calls = []
    evidence, scores = [("Page", 0, "Evidence")], [1.0]
    monkeypatch.setattr(verifier, "retrieve", lambda query: calls.append(query) or (evidence, scores))
    monkeypatch.setattr(verifier, "judge", lambda claim, rows, values: (claim, rows, values))
    assert verifier.verify("It passed.", query="It passed. Affordable Care Act") == ("It passed.", evidence, scores)
    assert calls == ["It passed. Affordable Care Act"]
    verifier.verify("Original claim")
    assert calls[-1] == "Original claim"
    with pytest.raises(ValueError, match="empty"):
        verifier.verify("Original claim", query=" ")
