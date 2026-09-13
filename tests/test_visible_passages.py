"""Passage relevance comes from readable prose, not tokens hidden in link targets."""

from __future__ import annotations

import pytest

from src.pipeline.web_research import source_excerpts
from src.retrieval.visible_text import visible_text


@pytest.mark.parametrize("passage", [
    "](https://finance.example.org/news/functional-unemployment-and-labor-shortage-estimates-for-current-workers.html)",
    "An album was released by a singer last year. [Read the review](https://example.org/labor-shortage-functional-unemployment-2025).",
    "An album was released by a singer last year. More details: https://example.org/labor-shortage-functional-unemployment-2025",
    "An album was released by a singer last year. [Read the review](https://example.org/archive/(music)/labor-shortage).",
    "An album was released by a singer last year. [Read the review](https://example.org/archive/(music)/labor-shortage",
])
def test_urls_cannot_supply_topic_matches(passage):
    assert source_excerpts(passage, "labor shortage functional unemployment") == []


def test_long_link_destinations_do_not_discard_meaningful_source_prose():
    passage = ("The survey measures people in [poverty-wage jobs](https://example.org/" + "long-path/" * 35 + "). "
               "These workers are included in the functional unemployment measure along with those unable to find full-time work.")
    assert source_excerpts(passage, "functional unemployment poverty-wage jobs") == [passage]


def test_mostly_linked_recommendations_remain_excluded():
    passage = "[Functional unemployment and labor shortages: what every reader should understand about current economic statistics](https://example.org/report)"
    assert source_excerpts(passage, "functional unemployment labor shortages") == []


def test_url_tokens_cannot_out_rank_a_passage_with_more_readable_claim_terms():
    bad = ("The solar report describes storage. [Details](https://example.org/solar-energy-storage-capacity-growth) "
           "A description of unrelated accounting procedures follows this section.")
    good = "Solar energy storage capacity increased during the measured period, according to the report's stated observations."
    assert source_excerpts(bad + "\n\n" + good, "solar energy storage capacity increased") == [good]


@pytest.mark.parametrize("target", ["https://example.org/(nested)/x", "https://example.org/(a(b))/c",
                                   r"https://example.org/escaped\)still-hidden"])
def test_balanced_and_escaped_destinations_keep_only_their_visible_label(target):
    text, length = visible_text(f"Read [the source]({target}) for the original evidence.")
    assert text == "Read the source for the original evidence." and length == len("the source")


def test_truncated_teaser_with_parenthesized_link_is_not_an_intact_passage():
    passage = ("The report discusses labor shortage estimates and how they differ between industries... "
               "[Read more](https://example.org/archive/(industry)/story)")
    assert source_excerpts(passage, "labor shortage") == []
