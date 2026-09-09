"""Caption punctuation must not turn a corrective contrast into separate verdicts."""

from __future__ import annotations

import pytest

from src.pipeline.caption_claims import caption_claims, contrast_parts, corrective_contrast


@pytest.mark.parametrize("text", [
    "We don't have a labor shortage. We have a good job shortage.",
    "We don’t have a labor shortage. We have a good job shortage.",
    "They do not have a water shortage. They have a storage shortage.",
    "This isn't a supply problem. This is a distribution problem.",
    "We have no housing shortage. We have an affordable housing shortage.",
])
def test_corrective_pair_keeps_exact_source_span(text):
    source = "Earlier claim. " + text + " Later claim."
    claims = caption_claims(source)
    assert [item.text for item in claims] == ["Earlier claim.", text, "Later claim."]
    assert [item.index for item in claims] == [0, 1, 2]
    assert all(source[item.start:item.end] == item.text for item in claims)
    assert corrective_contrast(text)


@pytest.mark.parametrize("text", [
    "We don't have a labor shortage, we have a good job shortage.",
    "We don't have a labor shortage; we have a good job shortage.",
    "We don't have a labor shortage, but we have a good job shortage.",
])
def test_existing_single_sentence_contrast_is_not_rewritten(text):
    assert [item.text for item in caption_claims(text)] == [text]
    assert corrective_contrast(text)


@pytest.mark.parametrize("text", [
    "We have a labor shortage. We have a housing shortage.",
    "We don't have a labor shortage. They have a good job shortage.",
    "We don't have a labor shortage. We have rising wages.",
    "We don't have a labor shortage? We have a good job shortage.",
    "We don't have a labor shortage. We have a labor shortage.",
    "We don't have a labor shortage. We don't have a good job shortage.",
])
def test_unrelated_or_uncertain_pairs_remain_separate(text):
    assert len(caption_claims(text)) == 2
    assert not corrective_contrast(text)


def test_contrasts_require_matching_predicates_and_bounded_clauses():
    mismatch = "We don't have a labor shortage. We are a good job shortage."
    long = "We don't have a " + "severe " * 24 + "shortage. We have a housing shortage."
    for text in (mismatch, long):
        assert len(caption_claims(text)) == 2
        assert not corrective_contrast(text)


def test_assertion_spans_preserve_punctuation_and_only_adjacent_pairs_merge():
    text = "We don't have a labor shortage, but we have a good job shortage."
    assert contrast_parts(text) == ("We don't have a labor shortage", "we have a good job shortage.")
    separated = "We don't have a labor shortage. Wages rose. We have a good job shortage."
    assert len(caption_claims(separated)) == 3 and contrast_parts(separated) == ()
    assert caption_claims("") == [] and contrast_parts("") == ()
