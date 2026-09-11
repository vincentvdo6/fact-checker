"""Search hints may change retrieval, never the assertion or its evidentiary role."""

from __future__ import annotations

import pytest

from src.pipeline.context import SpeechMetadata, build_query
from src.pipeline.topic_query import CONTEXT_WORDS, content_terms
from src.pipeline.transcript import ClaimContext, TranscriptUpdate


def update(key, text, start=0, end=1, final=True):
    return TranscriptUpdate(key, 0, text, start, end, final)


def test_topic_query_uses_overlapping_prior_speech_without_rewriting_claim():
    claim = "We don't have a labor shortage. We have a good job shortage."
    context = ClaimContext(update("claim", claim, 4, 5), (
        update("unrelated", "A singer released an album."),
        update("labor", "People have fallen out of the labor market and work part-time."),
        update("future", "Labor productivity increased.", 5, 6),
        update("interim", "Labor migration doubled.", final=False),
    ))
    result = build_query(context, SpeechMetadata(source="https://example.com", country="Invented"), mode="topic")
    assert result.claim == claim
    assert result.query == "labor shortage good job people fallen market work part time"
    assert result.context_ids == ("labor",)
    assert "Invented" not in result.query and "example.com" not in result.query
    assert "don't" in result.claim


def test_caption_context_with_shared_excerpt_timings_is_ordered_by_caller():
    context = ClaimContext(update("claim", "Labor shortages rose.", 4, 5),
                           (update("prior", "The labor market shrank.", 4, 5),))
    assert build_query(context, SpeechMetadata(), mode="topic").context_ids == ("prior",)


def test_query_budget_keeps_recent_context_and_unique_terms():
    earlier = " ".join(f"word{i}" for i in range(CONTEXT_WORDS * 2)) + " labor"
    context = ClaimContext(update("c", "Labor labor labor shortage.", 4, 5), (update("p", earlier),))
    result = build_query(context, SpeechMetadata(), mode="topic")
    assert result.query.split().count("labor") == 1
    assert "word0" not in result.query.split()
    assert result.query.endswith(f"word{CONTEXT_WORDS * 2 - 1}")
    assert sum(len(text.split()) for text in result.context_text) == CONTEXT_WORDS
    assert all("word0" not in text.split() for text in result.context_text)
    assert content_terms("We don’t have it; we don't have it.") == []
    assert content_terms("Labor labor shortages shortages.") == ["labor", "shortages"]


def test_each_time_boundary_excludes_unheard_context_independently():
    context = ClaimContext(update("claim", "Labor shortages rose.", 4, 5), (
        update("later_start", "Labor demand rose.", 4.5, 4.9),
        update("later_end", "Labor wages rose.", 3, 6),
    ))
    assert build_query(context, SpeechMetadata(), mode="topic").context_ids == ()


def test_empty_content_query_falls_back_to_original_claim():
    claim = "We don't have it."
    result = build_query(ClaimContext(update("claim", claim), ()), SpeechMetadata(), mode="topic")
    assert result.claim == result.query == claim


@pytest.mark.parametrize("claim,expected", [
    ("Labor shortages rose.", "labor shortages rose"),
    ("Solar capacity increased.", "solar capacity increased"),
    ("Train delays increased.", "train delays increased"),
])
def test_truncated_context_must_still_contain_a_claim_anchor(claim, expected):
    prior = claim + " " + "A singer released another album. " * 25
    context = ClaimContext(update("claim", claim, 4, 5), (update("prior", prior),))
    result = build_query(context, SpeechMetadata(), mode="topic")
    assert result.query == expected
    assert result.context_ids == result.context_text == ()


@pytest.mark.parametrize("claim,prior,expected", [
    ("Labor shortages rose.", "Labor demand increased.", "labor shortages rose demand increased"),
    ("Solar capacity increased.", "Solar installations doubled.", "solar capacity increased installations doubled"),
    ("Train delays increased.", "Train services slowed.", "train delays increased services slowed"),
])
def test_rejected_context_tail_does_not_displace_earlier_relevant_speech(claim, prior, expected):
    context = ClaimContext(update("claim", claim, 4, 5), (
        update("relevant", prior),
        update("drifted", claim + " " + "A singer released another album. " * 25),
    ))
    result = build_query(context, SpeechMetadata(), mode="topic")
    assert result.context_ids == ("relevant",)
    assert result.context_text == (prior,)
    assert result.query == expected
