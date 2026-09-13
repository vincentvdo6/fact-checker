"""Following a source reference adds candidates and lineage, never independent proof."""

from __future__ import annotations

import copy
import time

import pytest

from src.pipeline.caption_concepts import concept_packet
from src.pipeline.context import SpeechMetadata
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.web_research import WebResearch, research_plan
from src.retrieval.web_sources import SourceUnavailable

PARENT = "https://example.org/overview"
CHILD = "https://reports.example.net/original"
WATER = "Water samples contained dissolved lead above the reporting threshold in 24.3% of 5,700 surveyed households."
NOTICE = "Published: March 2024"
NOTE = "Only the 2018–2021 household sample is represented; no population-wide estimate was made."


def context(claim="Water samples contained dissolved lead.", preceding=()):
    return ClaimContext(TranscriptUpdate("claim", 0, claim, 30, 35, True), preceding)


def page(text, url, published="2024-06-01"):
    return {"metadata": {"sourceURL": url, "publishedTime": published, "statusCode": 200}, "markdown": text}


def parent(text=WATER, destinations=(CHILD,)):
    references = ", ".join(f"[source {index}]({url})" for index, url in enumerate(destinations, 1))
    return text + f" The complete measurement methods are described in {references}."


class Sources:
    def __init__(self, parents, children, *, enabled=True):
        self.parents, self.children = parents, children
        self.scraped, self.read, self.searches = [], [], []
        self.resets, self.deadline = 0, 0.0
        if enabled:
            self.read_link = self.read_page

    def begin_check(self):
        self.resets += 1
        self.deadline = time.monotonic() + 45

    def search_before(self, query, cutoff):
        self.searches.append((query, cutoff))
        return [{"url": url} for url in self.parents]

    def scrape(self, url):
        self.scraped.append(url)
        return copy.deepcopy(self.parents[url])

    def read_page(self, url):
        self.read.append(url)
        if time.monotonic() >= self.deadline:
            raise SourceUnavailable("The existing check deadline expired.")
        value = self.children[url]
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def review(sources, claim=None, metadata=None, selection=None):
    web = WebResearch(sources)
    web.begin_check()
    deadline = sources.deadline
    result = web.review(claim or context(), metadata or SpeechMetadata(source_published_at="2025-07-20"),
                        concept_selection=selection)
    assert sources.deadline == deadline and sources.resets == 1
    return result


@pytest.mark.parametrize("claim,paragraph", [
    ("Water samples contained dissolved lead.", WATER),
    ("Battery storage capacity increased.", "Battery storage capacity increased in the measured installations during the reporting period."),
    ("Rental housing vacancy declined.", "Rental housing vacancy declined among surveyed properties during the stated sample period."),
])
def test_linked_candidates_keep_exact_parent_lineage_original_claim_and_source_provenance(claim, paragraph):
    original = parent(paragraph)
    sources = Sources({PARENT: page(original, PARENT)}, {CHILD: page(paragraph, CHILD)})
    claim_context = context(claim)
    metadata = SpeechMetadata(source_published_at="2025-07-20")
    plan = review(sources, claim_context, metadata)
    assert sources.scraped == [PARENT] and sources.read == [CHILD]
    assert len(plan["sources"]) == 2 and len({row["id"] for row in plan["sources"]}) == 2
    parent_source, child = plan["sources"]
    markup = f"[source 1]({CHILD})"
    start = original.index(markup)
    expected = [{"parent_source_id": parent_source["id"], "parent_url": PARENT,
        "parent_passage_id": parent_source["id"] + ":p1", "paragraph": original,
        "label": "source 1", "destination": CHILD, "start": start, "end": start + len(markup), "markup": markup}]
    assert child["linked_from"] == expected
    assert parent_source["excerpts"] == [original] and child["excerpts"] == [paragraph]
    assert [source["url"] for source in plan["sources"]] == [PARENT, CHILD]
    assert all(part["status"] == "unresolved" for part in plan["assertions"])
    unchanged = research_plan(claim_context, metadata)
    assert [part["text"] for part in plan["assertions"]] == [part["text"] for part in unchanged["assertions"]]
    assert plan["claim_scope"] == unchanged["claim_scope"] and plan["scope_limits"] == unchanged["scope_limits"]
    assert not {"verdict", "confidence", "probabilities"}.intersection(plan)


def test_two_distinct_attempts_include_failures_and_ignore_duplicate_and_extra_references():
    second, third = CHILD + "-two", CHILD + "-three"
    original = parent(destinations=(CHILD, CHILD, second, third))
    sources = Sources({PARENT: page(original, PARENT)}, {
        CHILD: SourceUnavailable("Web request failed (HTTP 403)"), second: page(WATER, second), third: page(WATER, third)})
    plan = review(sources)
    assert sources.read == [CHILD, second]
    assert [row["url"] for row in plan["sources"]] == [PARENT, second]
    assert plan["page_failures"] == [{"url": CHILD, "reason": "Web request failed (HTTP 403)"}]
    assert plan["errors"] == ["Web request failed (HTTP 403)"] and plan["status"] == "partial"


def test_children_cannot_authorize_a_second_hop_even_when_attempt_budget_remains():
    grandchild = CHILD + "-grandchild"
    sources = Sources({PARENT: page(parent(), PARENT)}, {CHILD: page(parent(destinations=(grandchild,)), CHILD)})
    plan = review(sources)
    assert sources.read == [CHILD]
    assert len(plan["sources"]) == 2


@pytest.mark.parametrize("child_text,has_error", [
    ("The orchestra performed at several unrelated music festivals during the recording period.", False),
    (WATER + "\n\n# Notes\n\n" + "The sample excludes other residents. " * 270, True),
])
def test_children_use_existing_relevance_and_complete_context_gates(child_text, has_error):
    sources = Sources({PARENT: page(parent(), PARENT)}, {CHILD: page(child_text, CHILD)})
    plan = review(sources)
    assert sources.read == [CHILD] and [row["url"] for row in plan["sources"]] == [PARENT]
    assert bool(plan["errors"]) is has_error
    assert plan["page_failures"] == []


def test_later_child_remains_dated_candidate_without_resolving_assertions():
    sources = Sources({PARENT: page(parent(), PARENT)}, {CHILD: page(WATER, CHILD, "2026-01-01")})
    plan = review(sources)
    child = next(row for row in plan["sources"] if row["url"] == CHILD)
    assert child["temporal_status"] == "later_publication" and child["linked_from"]
    assert child["published_at"] == "2026-01-01" and child["excerpts"] == [WATER]
    assert all(child["id"] not in part["source_ids"] and part["status"] == "unresolved" for part in plan["assertions"])
    assert plan["claim_scope"]["country"] == plan["claim_scope"]["spoken_at"] == ""


def test_partial_publication_notice_and_complete_sample_note_survive_reference_reading():
    markdown = NOTICE + "\n\n" + WATER + "\n\n# Notes\n\n" + NOTE
    sources = Sources({PARENT: page(parent(), PARENT)}, {CHILD: page(markdown, CHILD, "")})
    plan = review(sources)
    child = next(row for row in plan["sources"] if row["url"] == CHILD)
    assert child["published_at"] == ""
    assert NOTICE in child["reading_context"] and NOTE in child["reading_context"]
    assert child["excerpts"] == [WATER] and child["linked_from"][0]["paragraph"] == parent()


def test_shared_reference_is_read_once_and_retains_both_parent_sources():
    other = PARENT + "-two"
    sources = Sources({PARENT: page(parent(), PARENT), other: page(parent(WATER + " This is another account."), other)},
        {CHILD: page(WATER, CHILD)})
    plan = review(sources)
    assert sources.read == [CHILD] and len(plan["sources"]) == 3
    child = next(row for row in plan["sources"] if row["url"] == CHILD)
    assert {row["parent_source_id"] for row in child["linked_from"]} == {"source-1", "source-2"}
    assert len(child["linked_from"]) == 2


@pytest.mark.parametrize("parent_url", [PARENT, "https://example.org:443/overview"])
def test_child_redirect_to_existing_parent_does_not_duplicate_the_final_source(parent_url):
    sources = Sources({parent_url: page(parent(), parent_url)}, {CHILD: page(WATER, PARENT)})
    plan = review(sources)
    assert sources.read == [CHILD]
    assert len(plan["sources"]) == 1 and plan["sources"][0]["url"] == parent_url
    assert plan["sources"][0]["excerpts"] == [parent()]
    assert plan["sources"][0]["linked_from"][0]["destination"] == CHILD
    assert plan["sources"][0]["linked_from"][0]["parent_url"] == parent_url


def test_missing_reader_capability_preserves_ordinary_research_without_following():
    sources = Sources({PARENT: page(parent(), PARENT)}, {}, enabled=False)
    plan = review(sources)
    assert sources.read == [] and len(plan["sources"]) == 1
    assert "reference_research" not in plan and "linked_from" not in plan["sources"][0]


def test_existing_deadline_is_not_restarted_for_linked_research(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    sources = Sources({PARENT: page(parent(), PARENT)}, {CHILD: page(WATER, CHILD)})
    original_scrape = sources.scrape

    def scrape(url):
        clock[0] = 46
        return original_scrape(url)

    sources.scrape = scrape
    plan = review(sources)
    assert sources.deadline == 45 and sources.resets == 1
    assert [row["url"] for row in plan["sources"]] == [PARENT]
    assert plan["page_failures"] == [{"url": CHILD, "reason": "The existing check deadline expired."}]


def test_context_discovery_parent_can_follow_a_reference_without_resolving_original_assertions():
    claim = context("We do not lack energy. We lack access.", (
        TranscriptUpdate("prior", 0, "The report refers to energy poverty.", 20, 29, True),))
    candidate = next(row for row in concept_packet(claim)["context_candidates"] if row["text"] == "energy poverty")
    definition = "The index measures households receiving power for fewer than eight hours per day during the study."
    sources = Sources({PARENT: page("# Energy poverty\n\n" + parent(definition), PARENT)},
        {CHILD: page("# Energy poverty\n\n" + definition, CHILD)})
    plan = review(sources, claim, selection={"selected_context_ids": [candidate["id"]]})
    assert sources.read == [CHILD]
    assert all(part["source_ids"] == [] and part["status"] == "unresolved" for part in plan["assertions"])
    child = next(row for row in plan["sources"] if row["url"] == CHILD)
    assert child["excerpts"] == [] and child["reading_passages"] == [definition]
    assert child["linked_from"][0]["paragraph"] == parent(definition)
