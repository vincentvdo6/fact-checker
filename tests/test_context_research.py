"""Selected caption concepts broaden research without resolving the original assertions."""

from __future__ import annotations

import pytest

from src.pipeline.caption_concepts import concept_packet
from src.pipeline.context import SpeechMetadata
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.web_research import WebResearch
from src.retrieval.web_sources import SourceUnavailable


def context() -> ClaimContext:
    return ClaimContext(TranscriptUpdate("claim", 0, "We don't have an energy shortage. We have an access shortage.", 10, 15, True),
                        (TranscriptUpdate("prior", 0, "The report refers to energy poverty.", 0, 9, True),))


def selection() -> dict:
    candidate = next(item for item in concept_packet(context())["context_candidates"] if item["text"] == "energy poverty")
    return {"selected_context_ids": [candidate["id"]]}


DEFINITION = "The index includes households that receive power for fewer than eight hours per day."


class Sources:
    compact_search = True

    def __init__(self, published="2025-05-01"):
        self.calls, self.scraped = [], []
        self.published = published
        self.resets = 0

    def begin_check(self):
        self.resets += 1

    def search_before(self, query, cutoff):
        self.calls.append((query, cutoff))
        return [{"url": "https://example.org/report"}] if "energy poverty" in query else []

    def scrape(self, url):
        self.scraped.append(url)
        return {"metadata": {"sourceURL": url, "publishedTime": self.published},
                "markdown": "# Energy poverty\n\n" + DEFINITION}


def test_context_definition_is_discovered_but_does_not_resolve_either_assertion():
    sources = Sources()
    web = WebResearch(sources)
    web.begin_check()
    plan = web.review(context(), SpeechMetadata(country="France", source_published_at="2025-07-20"),
                      concept_selection=selection())
    assert sources.resets == 1
    assert sources.calls[0] == ("energy poverty France 2025", "2025-07-20")
    assert all(part["source_ids"] == [] and part["status"] == "unresolved" for part in plan["assertions"])
    assert [part["text"] for part in plan["assertions"]] == ["We don't have an energy shortage.", "We have an access shortage."]
    source = plan["sources"][0]
    assert source["excerpts"] == [] and source["reading_passages"] == [DEFINITION]
    assert plan["context_research"][0]["source_ids"] == [source["id"]]
    assert [concept["id"] for concept in source["context_concepts"]] == selection()["selected_context_ids"]
    assert source["context_concepts"][0]["text"] == "energy poverty"
    assert plan["claim_scope"] == {"country": "France", "spoken_at": "", "source_published_at": "2025-07-20",
                                   "country_basis": "supplied by the viewer", "spoken_at_basis": ""}


def test_context_source_after_boundary_is_not_assigned_to_context_research():
    plan = WebResearch(Sources("2026-02-01")).review(context(), SpeechMetadata(source_published_at="2025-07-20"),
                                                   concept_selection=selection())
    assert plan["sources"][0]["temporal_status"] == "later_publication"
    assert plan["context_research"][0]["source_ids"] == []
    assert all(part["source_ids"] == [] for part in plan["assertions"])


def test_unknown_selection_is_rejected_before_network_access():
    sources = Sources()
    with pytest.raises(ValueError, match="unknown"):
        WebResearch(sources).review(context(), SpeechMetadata(), concept_selection={"selected_context_ids": ["invented"]})
    assert sources.calls == [] and sources.scraped == []


def test_each_context_phrase_has_two_page_attempts_and_no_generated_fallback():
    class Unreadable(Sources):
        def search_before(self, query, cutoff):
            self.calls.append((query, cutoff))
            return ([{"url": f"https://example.org/{index}"} for index in range(20)]
                    if query == "energy poverty" else [])

        def scrape(self, url):
            self.scraped.append(url)
            raise SourceUnavailable("Publisher unavailable")

    sources = Unreadable()
    plan = WebResearch(sources).review(context(), SpeechMetadata(), concept_selection=selection())
    assert sources.scraped == ["https://example.org/0", "https://example.org/1"]
    assert plan["context_research"][0]["searches"] == [{"query": "energy poverty", "status": "returned_links"}]
    assert plan["errors"] == ["Publisher unavailable"] and plan["status"] == "unavailable"


def test_context_and_assertions_share_publisher_documents_without_sharing_evidence_relations():
    class SameArticle(Sources):
        def search_before(self, query, cutoff):
            return [{"url": "https://example.org/report"}]

    sources = SameArticle()
    plan = WebResearch(sources).review(context(), SpeechMetadata(), concept_selection=selection())
    assert sources.scraped == ["https://example.org/report"]
    assert len(plan["sources"]) == 1
    assert all(part["source_ids"] == [] for part in plan["assertions"])
    assert plan["sources"][0]["reading_passages"] == [DEFINITION]


def test_optional_selection_leaves_default_discovery_unchanged():
    sources = Sources()
    plan = WebResearch(sources).review(context(), SpeechMetadata())
    assert "context_research" not in plan
    assert plan["sources"] == []
    assert all(query != "energy poverty" for query, _ in sources.calls)


@pytest.mark.parametrize("claim", [
    "We don't have a labor shortage. We have a good job shortage.",
    "Households have a clean water shortage.",
    "Vaccination rates increased among children.",
    "Vaccination rates increased in 2025.",
])
@pytest.mark.parametrize("extra", ["", " [Background](https://example.org/labor-job-shortage-water-vaccination)"])
def test_incidental_context_phrase_cannot_admit_an_unrelated_page(claim, extra):
    speech = ClaimContext(TranscriptUpdate("claim", 0, claim, 10, 15, True),
                          (TranscriptUpdate("prior", 0, "We bring people together.", 0, 9, True),))
    concept = concept_packet(speech)["context_candidates"][0]

    class Incidental(Sources):
        def search_before(self, query, cutoff):
            return [{"url": "https://example.org/app"}] if "people together" in query else []

        def scrape(self, url):
            return {"metadata": {"sourceURL": url}, "markdown": "# A new app brings people together in 2025\n\n"
                    "The new app lets users invite friends and family to celebrations, share photo albums and plan events." + extra}

    result = WebResearch(Incidental()).review(speech, SpeechMetadata(),
        concept_selection={"selected_context_ids": [concept["id"]]})
    assert result["sources"] == []
    assert result["context_research"][0]["source_ids"] == []
    assert result["context_research"][0]["searches"][0]["status"] == "returned_links"


@pytest.mark.parametrize("claim,heading,definition", [
    ("Households lack clean water.", "Water quality", "The index measures contamination levels observed across the households enrolled in the national survey."),
    ("Vaccine coverage fell.", "Vaccine uptake", "The survey tracks the share of children who received all recommended doses during the observation period."),
])
def test_context_definition_can_connect_through_its_heading(claim, heading, definition):
    speech = ClaimContext(TranscriptUpdate("claim", 0, claim, 10, 15, True),
                          (TranscriptUpdate("prior", 0, f"The report refers to {heading}.", 0, 9, True),))
    concept = next(item for item in concept_packet(speech)["context_candidates"] if item["text"] == heading)

    class Definition(Sources):
        def search_before(self, query, cutoff):
            return [{"url": "https://example.org/report"}] if heading in query else []

        def scrape(self, url):
            return {"metadata": {"sourceURL": url}, "markdown": "# " + heading + "\n\n" + definition}

    result = WebResearch(Definition()).review(speech, SpeechMetadata(),
        concept_selection={"selected_context_ids": [concept["id"]]})
    assert result["sources"][0]["reading_passages"] == [definition]
    assert result["sources"][0]["excerpts"] == []
    assert all(not part["source_ids"] for part in result["assertions"])


def test_assertion_candidate_also_carries_section_dates_and_short_scope_notes():
    class Forecast(Sources):
        def search_before(self, query, cutoff):
            return [{"url": "https://example.org/report"}]

        def scrape(self, url):
            return {"metadata": {"sourceURL": url}, "markdown": "# Forecast for 2030\n\n"
                    "The energy shortage is expected to disappear under the assumptions used in this national model.\n\n"
                    "Adults only."}

    plan = WebResearch(Forecast()).review(context(), SpeechMetadata())
    assert plan["sources"][0]["reading_context"] == ["# Forecast for 2030", "Adults only."]


@pytest.mark.parametrize("use_context", [False, True])
@pytest.mark.parametrize("incomplete_notes", [
    "x" * 9000 + "\n\n# References", "x" * 160_000,
    "# Methods\n\n" + "x" * 160_000 + "\n\n# Notes\n\n1. The survey only covers observations before 2008.",
], ids=["oversized", "truncated", "later_section"])
def test_incomplete_source_notes_block_both_assertion_and_context_candidates(use_context, incomplete_notes):
    class Incomplete(Sources):
        def search_before(self, query, cutoff):
            return [{"url": "https://example.org/report"}]

        def scrape(self, url):
            return {"metadata": {"sourceURL": url}, "markdown": (
                "Energy poverty and an energy shortage are measured in this report using the same household panel."
                "\n\n# Notes\n\n" + incomplete_notes)}

    plan = WebResearch(Incomplete()).review(context(), SpeechMetadata(),
        concept_selection=selection() if use_context else None)
    assert plan["sources"] == []
    assert all(not part["source_ids"] for part in plan["assertions"])
    assert all(not part["source_ids"] for part in plan.get("context_research", []))
    assert plan["status"] == "unavailable" and len(plan["errors"]) == 1
    assert "reading limit" in plan["errors"][0]
    assert plan["page_failures"] == []


def test_research_preserves_distant_sample_note_without_changing_claim_scope():
    note = "1. The survey only covers households observed in 1998–2007; the sample excludes temporary residences."

    class DistantNotes(Sources):
        def scrape(self, url):
            background = "\n\n".join(["The authors discuss the report's historical background and administrative arrangements."] * 12)
            return {"metadata": {"sourceURL": url}, "markdown": (
                "# Energy poverty\n\n" + DEFINITION + "\n\n" + background + "\n\n# Notes\n\n" + note)}

    plan = WebResearch(DistantNotes()).review(context(), SpeechMetadata(), concept_selection=selection())
    assert note in plan["sources"][0]["reading_context"]
    assert plan["claim_scope"] == {"country": "", "spoken_at": "", "source_published_at": "", "country_basis": "", "spoken_at_basis": ""}
    assert plan["sources"][0]["published_at"] == "" and plan["sources"][0]["temporal_status"] == "date_unconfirmed"


@pytest.mark.parametrize("use_context", [False, True])
def test_partial_publication_notice_reaches_research_without_inventing_a_day(use_context):
    notice = "Materials Bulletin, Vol. 12, No. 3, March  2025"

    class Monthly(Sources):
        def search_before(self, query, cutoff):
            return [{"url": "https://example.org/report", "feed_published_at": "2025-12-13"}]

        def scrape(self, url):
            paragraph = "Energy poverty and an energy shortage are measured in this report using the same household panel."
            background = "\n\n".join(f"Section {index} discusses the report's methods and administrative arrangements."
                                      for index in range(8))
            return {"metadata": {"sourceURL": url}, "markdown": notice + "\n\n" + background + "\n\n" + paragraph}

    plan = WebResearch(Monthly()).review(context(), SpeechMetadata(source_published_at="2025-07-20"),
        concept_selection=selection() if use_context else None)
    source = plan["sources"][0]
    assert notice in source["reading_context"]
    assert source["published_at"] == "" and source["temporal_status"] == "date_unconfirmed"
    assert source["publication_basis"] == "Exact publication day unconfirmed; partial publication notice retained in source context."
    assert plan["claim_scope"] == {"country": "", "spoken_at": "", "source_published_at": "2025-07-20", "country_basis": "", "spoken_at_basis": ""}
