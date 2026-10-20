"""A selected source excerpt keeps its own bounded reading neighborhood."""

from __future__ import annotations

import pytest

from src.retrieval.research_passages import reading_window, source_excerpts


@pytest.mark.parametrize(("query", "detail", "qualification"), [
    ("solar panel electricity output", "calibrated sunlight",
     "The reported percentage uses the inverter as its measurement boundary and excludes losses upstream of that point."),
    ("library Saturday opening hours", "holiday schedule",
     "The listed schedule applies during the regular academic term and excludes public holidays and campus closures."),
])
def test_selected_later_excerpt_keeps_its_neighborhood(query, detail, qualification):
    early = f"The {query} appears in an overview of the service and its history."
    filler = [f"Background paragraph {index} describes the institution and its ordinary operations in the area."
              for index in range(8)]
    selected = f"The {query} was measured under the {detail}, with the source's stated scope and period."
    page = "\n\n".join([early, *filler, selected, qualification])
    excerpts = source_excerpts(page, query, context_query=f"{query} {detail}")
    assert excerpts == [selected]
    old = reading_window(page, query)
    assert selected not in old["passages"]

    window = reading_window(page, query, selected_excerpts=excerpts)
    assert selected in window["passages"]
    assert qualification in window["passages"]
    assert len(window["passages"]) <= 9
    assert all(text in page.split("\n\n") for text in [*window["passages"], *window["context"]])


def test_selected_excerpt_must_be_an_exact_citable_block():
    heading = "# Solar panel electricity output"
    body = "Solar panel electricity output depends on sunlight and the inverter's operating conditions."
    page = f"{heading}\n\n{body}"
    original = reading_window(page, "solar panel electricity output")
    assert original["passages"] == [body] and heading in original["context"]
    for invalid in ("Not on this page", heading, "[solar panel](https://example.org/output) " * 5):
        with pytest.raises(ValueError, match="exact citable source block"):
            reading_window(page, "solar panel electricity output", selected_excerpts=[invalid])
    note_body = "Solar panel electricity output was mentioned in this long citation note about a separate report."
    notes = page + "\n\n# Notes\n\n" + note_body
    with pytest.raises(ValueError, match="exact citable source block"):
        reading_window(notes, "solar panel electricity output", selected_excerpts=[note_body])
    navigation = "[Solar panel electricity output](https://example.org/navigation) " * 5
    with pytest.raises(ValueError, match="exact citable source block"):
        reading_window(page + "\n\n" + navigation, "solar panel electricity output",
                       selected_excerpts=[navigation.strip()])


def test_more_than_nine_distinct_selected_blocks_is_rejected():
    blocks = [f"Solar panel electricity output is measured in source block {index} under the same conditions."
              for index in range(10)]
    with pytest.raises(ValueError, match="nine-paragraph"):
        reading_window("\n\n".join(blocks), "solar panel electricity output", selected_excerpts=blocks)


@pytest.mark.parametrize("stride", [1, 3])
def test_nine_selected_blocks_respect_cap_and_page_order(stride):
    blocks = [f"Solar panel electricity output is measured in source block {index} under the same conditions."
              for index in range(11 * stride)]
    selected = blocks[stride:10 * stride:stride]
    window = reading_window("\n\n".join(blocks), "solar panel electricity output",
                            selected_excerpts=list(reversed(selected)))
    assert window["passages"] == selected
    assert len(window["passages"]) == 9


def test_explicit_selected_excerpt_keeps_neighbors_with_empty_query():
    selected = "The measured percentage covers completed responses in the surveyed group and the stated time period."
    neighbor = "The denominator includes every eligible respondent recorded during the study's original collection period."
    page = "\n\n".join([selected, neighbor])
    assert reading_window(page, "", selected_excerpts=[selected])["passages"] == [selected, neighbor]
    assert reading_window(page, "")["passages"] == []


def test_already_included_selected_excerpt_preserves_complete_original_window():
    blocks = [f"Solar panel electricity output is recorded for observation group {index} with a stated period."
              for index in range(14)]
    page = "# Study measures\n\n" + "\n\n".join(blocks)
    original = reading_window(page, "solar panel electricity output")
    assert blocks[8] in original["passages"] and blocks[9] not in original["passages"]
    assert reading_window(page, "solar panel electricity output", selected_excerpts=[blocks[8]]) == original


def test_repeated_excerpt_uses_first_occurrence_for_section_context():
    excerpt = "Library Saturday opening hours are 9 a.m. to 5 p.m. during the regular academic term."
    filler = [f"Archive entry {index} describes unrelated building repairs and furniture planning for the campus."
              for index in range(10)]
    page = "\n\n".join(["# Current hours", *filler[:4], excerpt, *filler[4:], "# Archive", excerpt])
    window = reading_window(page, "library Saturday opening hours", selected_excerpts=[excerpt])
    assert window["passages"].count(excerpt) == 1
    assert "# Current hours" in window["context"]
    assert "# Archive" not in window["context"]
