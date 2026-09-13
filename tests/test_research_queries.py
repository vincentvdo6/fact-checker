"""Fallback search must broaden discovery without supplying new facts or extra budgets."""

from __future__ import annotations

import pytest

from src.pipeline.context import SpeechMetadata
from src.pipeline.research_queries import fallback_query
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.web_research import WebResearch
from src.retrieval.web_sources import SourceUnavailable


@pytest.mark.parametrize("claim,expected", [
    ("We don't have a labor shortage.", "labor shortage."),
    ("Solar energy storage capacity has increased.", "Solar energy storage capacity increased."),
    ("A rocket launched satellites this week.", "rocket launched satellites this week."),
    ("The central bank raised rates by 4.2 %.", "The central bank raised rates by 4.2 %."),
    ("It cost $25,000 before 2020.", "It cost $25,000 before 2020."),
    ("Temperatures rose from -2.5°C to 3°C.", "Temperatures rose from -2.5°C to 3°C."),
    ("The board measures 12 in by 2 m.", "The board measures 12 in by 2 m."),
    ("It finished after 5 s.", "It finished after 5 s."),
    ("The house occupies 25 square m.", "The house occupies 25 square m."),
    ("The rover travels at 5 m per s.", "The rover travels at 5 m per s."),
    ("The budget is US $5 million.", "The budget is US $5 million."),
    ("The budget is under $5 million.", "The budget is under $5 million."),
    ("It rose by more than five percent.", "It rose by more than five percent."),
    ("The rover travels at five m per s.", "The rover travels at five m per s."),
])
def test_compact_queries_use_literal_claim_words_numbers_and_supplied_scope(claim, expected):
    assert fallback_query(claim, "New Zealand", "2025-07-20") == expected + " New Zealand 2025"


def test_vague_and_oversized_fallbacks_are_not_sent():
    assert fallback_query("It increased.", "", "") == ""
    assert fallback_query("Solar capacity " * 100, "", "") == ""


class Sources:
    compact_search = True

    def __init__(self, primary=(), fallback=(), *, blocked=(), published="2020-01-01"):
        self.results = [primary, fallback]
        self.blocked = blocked
        self.published = published
        self.queries = []
        self.pages = []

    def search(self, query):
        self.queries.append(query)
        return [{"url": "https://example.org/" + name} for name in self.results[len(self.queries) - 1]]

    def scrape(self, url):
        name = url.rsplit("/", 1)[-1]
        self.pages.append(name)
        if name in self.blocked:
            raise SourceUnavailable("Page unavailable")
        passage = ("Solar energy storage capacity increased during the period covered by these measurements, with regional differences."
                   if name.startswith("good") else "The orchestra released an album featuring music performed at several concerts around the world.")
        return {"markdown": passage, "metadata": {"datePublished": self.published}}


def review(sources):
    claim = TranscriptUpdate("claim", 0, "Solar energy storage capacity has increased.", 0, 1, True)
    return WebResearch(sources).review(ClaimContext(claim, ()), SpeechMetadata(country="Germany", spoken_at="2020-02-01"))


def test_empty_primary_search_can_recover_without_rewriting_claim_or_scope():
    sources = Sources(fallback=["good"])
    result = review(sources)
    assert sources.queries == ["Solar energy storage capacity has increased. Germany 2020",
                               "Solar energy storage capacity increased. Germany 2020"]
    assertion = result["assertions"][0]
    assert assertion["text"] == "Solar energy storage capacity has increased."
    assert assertion["source_ids"] == ["source-1"] and assertion["status"] == "unresolved"
    assert [attempt["status"] for attempt in assertion["searches"]] == ["empty", "returned_links"]
    assert not {"verdict", "confidence"}.intersection(result)


def test_unusable_primary_pages_reserve_fallback_attempts_within_the_same_total_limit():
    sources = Sources(primary=["blocked", "blocked", "noise1", "noise2", "unused"],
                      fallback=["blocked", "noise1", "noise2", "good1", "good2", "unused2"], blocked=["blocked"])
    result = review(sources)
    assert sources.pages == ["blocked", "noise1", "noise2", "good1", "good2"]
    assert len(sources.queries) == 2
    assert len(result["assertions"][0]["source_ids"]) == 2
    assert result["status"] == "partial" and result["errors"] == ["Page unavailable"]


def test_successful_primary_search_does_not_spend_another_query():
    sources = Sources(primary=["good"])
    review(sources)
    assert len(sources.queries) == 1


def test_fallback_does_not_relax_the_publication_boundary():
    sources = Sources(primary=["good1"], fallback=["good2"], published="2026-09-09")
    result = review(sources)
    assert len(sources.queries) == 2
    assert all(query.endswith("Germany 2020") for query in sources.queries)
    assert result["assertions"][0]["source_ids"] == []
    assert all(source["temporal_status"] == "later_publication" for source in result["sources"])


def test_search_failure_is_not_retried_as_if_it_were_an_empty_result():
    sources = Sources()

    def fail(query):
        sources.queries.append(query)
        raise SourceUnavailable("Search unavailable")

    sources.search = fail
    result = review(sources)
    assert len(sources.queries) == 1 and sources.pages == []
    assert result["status"] == "unavailable" and result["errors"] == ["Search unavailable"]
