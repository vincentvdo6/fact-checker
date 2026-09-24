"""A shared page keeps the full reading window for each contrast assertion."""

from __future__ import annotations

import pytest

from src.pipeline.caption_concepts import concept_packet
from src.pipeline.context import SpeechMetadata
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.web_research import WebResearch
from src.verdict.chain import research_packet
from src.verdict.reading import reading_packet


@pytest.mark.parametrize("first,second", [("water", "bridge"), ("bridge", "water")])
def test_shared_page_keeps_each_assertions_qualifying_paragraphs(first: str, second: str) -> None:
    claim = f"We don't have a {first} shortage. We have a {second} shortage."
    first_passages = [
        f"The regional report describes a {first} shortage in district {number}, with too few available resources "
        "for residents and a survey of local conditions across the area."
        for number in range(9)
    ]
    definition = (f"The report defines a {second} shortage as a sustained gap between available resources and "
                  "the needs reported by residents in the surveyed districts.")
    finding = (f"Researchers found a {second} shortage across the surveyed districts, with reported needs "
               "exceeding available resources throughout the period covered by the report.")
    markdown = "\n\n".join([*first_passages, definition, finding])

    class SharedPage:
        def __init__(self) -> None:
            self.reads = 0

        def begin_check(self) -> None:
            pass

        def search(self, query: str) -> list[dict]:
            return [{"url": "https://example.org/shared-report"}]

        def scrape(self, url: str) -> dict:
            self.reads += 1
            return {"markdown": markdown, "metadata": {"sourceURL": url, "datePublished": "2025-01-01"}}

    pages = SharedPage()
    context = ClaimContext(TranscriptUpdate("claim", 0, claim, 30, 35, True), ())
    plan = WebResearch(pages).review(context, SpeechMetadata(source_published_at="2025-06-01"))
    assert pages.reads == 1
    assert len(plan["sources"]) == 1
    assert [part["source_ids"] for part in plan["assertions"]] == [["source-1"], ["source-1"]]

    packet = research_packet(claim, plan)
    excerpts = packet["sources"][0]["excerpts"]
    assert all(paragraph in excerpts for paragraph in [*first_passages, definition, finding])
    passages = reading_packet(packet)["passages"]
    assert all(passage["origin"] == "assertion" for passage in passages)


def test_shared_page_keeps_both_selected_concept_windows_as_context() -> None:
    claim = "Households lack reliable water."
    speech = ClaimContext(TranscriptUpdate("claim", 0, claim, 30, 35, True),
                          (TranscriptUpdate("prior", 0, "River access. Bridge access.", 20, 29, True),))
    candidates = concept_packet(speech)["context_candidates"]
    selected = [next(item["id"] for item in candidates if item["text"] == phrase)
                for phrase in ("River access", "Bridge access")]
    first_passages = [
        f"The survey of river access in district {number} describes how people obtain water from local sources "
        "during the observation period and records conditions across the area."
        for number in range(9)
    ]
    definition = ("The report defines bridge access by the availability of safe crossings to reach water sources "
                  "during the observation period in the surveyed districts.")
    finding = ("The survey found limited bridge access in several districts, making water sources difficult "
               "to reach for residents throughout the observation period.")
    markdown = "\n\n".join([*first_passages, definition, finding])

    class SharedPage:
        def __init__(self) -> None:
            self.reads = 0

        def begin_check(self) -> None:
            pass

        def search(self, query: str) -> list[dict]:
            return [{"url": "https://example.org/shared-report"}]

        def scrape(self, url: str) -> dict:
            self.reads += 1
            return {"markdown": markdown, "metadata": {"sourceURL": url, "datePublished": "2025-01-01"}}

    pages = SharedPage()
    plan = WebResearch(pages).review(speech, SpeechMetadata(source_published_at="2025-06-01"),
                                     concept_selection={"selected_context_ids": selected})
    assert pages.reads == 1
    assert len(plan["sources"]) == 1
    assert [part["source_ids"] for part in plan["context_research"]] == [["source-1"], ["source-1"]]
    assert all(not part["source_ids"] for part in plan["assertions"])

    packet = research_packet(claim, plan)
    assert packet["sources"][0]["excerpts"] == []
    assert all(paragraph in packet["sources"][0]["context_excerpts"]
               for paragraph in [*first_passages, definition, finding])
    assert all(passage["origin"] == "context" for passage in reading_packet(packet)["passages"])
