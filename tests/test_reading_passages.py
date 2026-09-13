"""Definitions and qualifications survive without being promoted to assertion evidence."""

from __future__ import annotations

import pytest

from src.retrieval.research_passages import reading_passages, reading_window, source_excerpts


def test_section_heading_can_locate_a_definition_without_becoming_evidence():
    paragraphs = ["This measure differs from the conventional indicator used in official reporting.",
                  "The conventional measure excludes some people who could benefit from improved access.",
                  "The index includes households that receive power for fewer than eight hours per day."]
    markdown = "# What is energy poverty?\n\n" + "\n\n".join(paragraphs)
    assert source_excerpts(markdown, "energy poverty") == []
    assert reading_passages(markdown, "energy poverty") == paragraphs


def test_nearby_short_qualification_is_not_lost_for_lacking_search_terms():
    match = "Available water supplies exceed current demand in the areas surveyed by the local agency."
    qualification = "This does not cover rural areas."
    assert reading_window(match + "\n\n" + qualification, "water supplies") == {"passages": [match], "context": [qualification]}


@pytest.mark.parametrize("anchor", [
    "[Details](https://example.org/water-supplies)\n\n",
    "# [Details](https://example.org/(report)/water-supplies)\n\n",
    "](https://example.org/water-supplies)\n\n",
])
def test_hidden_link_tokens_cannot_admit_unrelated_neighbors(anchor):
    assert reading_passages(anchor + "An unrelated singer discussed an album released during the previous year.",
                            "water supplies") == []


def test_reading_window_is_bounded_and_keeps_original_paragraphs():
    blocks = [f"Report {index} measures battery storage capacity in the region, with separate estimates by district."
              for index in range(20)]
    result = reading_passages("\n\n".join(blocks), "battery storage capacity")
    assert len(result) == 9 and all(text in blocks for text in result)
    assert result == blocks[:9]


def test_reader_omits_tables_teasers_navigation_and_cut_off_page_blocks():
    match = "The school funding formula accounts for enrollment and student needs across all districts."
    bad = ["| school funding | amount |", "Read about school funding and the proposed changes...",
           "[School funding details and enrollment changes for every local district in the report](https://example.org/report)"]
    assert reading_passages("\n\n".join([match, *bad]), "school funding") == [match]
    assert reading_passages("x" * 159_950 + "\n\n" + match, "school funding") == []


def test_source_numbers_and_qualifications_remain_literal():
    text = "Battery storage was projected to exceed 250 MW by 2030; this was not an observed increase in 2025."
    assert reading_passages(text, "battery storage") == [text]
    assert reading_passages(text, "") == []


@pytest.mark.parametrize("qualifier", ["Adults only.", "Before taxes.", "In 2020 only.", "Nominal values."])
def test_short_scope_qualifications_travel_as_source_context(qualifier):
    paragraph = "The average income was reported as $25,000 for individuals measured in the published household survey."
    window = reading_window(paragraph + "\n\n" + qualifier, "average income")
    assert window == {"passages": [paragraph], "context": [qualifier]}


def test_forecast_and_date_headings_survive_without_becoming_standalone_citations():
    headings = "# Forecast for 2030\n\n# Solar generation"
    paragraph = "Annual electricity production reaches 250 GWh for the region under the assumptions used in this report."
    window = reading_window(headings + "\n\n" + paragraph, "solar generation")
    assert window == {"passages": [paragraph], "context": ["# Forecast for 2030", "# Solar generation"]}


def test_source_context_that_cannot_fit_is_not_silently_dropped():
    headings = "\n\n".join("# Scope qualification " + "x" * 200 for _ in range(50))
    paragraph = "The average income was reported as $25,000 for individuals measured in the published household survey."
    assert reading_window(headings + "\n\n" + paragraph, "average income") == {
        "passages": [], "context": [], "unavailable_reason": "Complete source context exceeds the reading limit."}


def test_entire_adjacent_chain_of_short_scope_notes_survives():
    paragraph = "The average income was reported as $25,000 for individuals measured in the published household survey."
    scope = ["Adults only.", "Before taxes.", "In 2020 only.", "Nominal values.", "National survey."]
    assert reading_window("\n\n".join([paragraph, *scope]), "average income")["context"] == scope


def test_linked_recommendation_heading_and_short_feed_metadata_cannot_consume_reading_slots():
    paragraphs = [f"Definition paragraph {index} explains the metric's coverage and the exclusions that readers need to consider."
                  for index in range(1, 5)]
    main = "# Energy poverty\n\n" + "\n\n".join(paragraphs)
    related = ("# [Energy poverty estimates updated again](https://example.org/another-report)\n\n"
               "Example Newswire  20d ago\n\n"
               "# [Energy poverty estimates at record high](https://example.org/third-report)\n\n"
               "Example Newswire  3mo ago")
    assert reading_passages(main + "\n\n" + related, "energy poverty") == paragraphs


@pytest.mark.parametrize("topic,measurement,definition", [
    ("energy access", "The survey found that 18% of households received power for fewer than eight hours per day in 2024.",
     "The index counts households that receive less than eight hours of electricity per day, including rural households."),
    ("water availability", "The survey counted 350 households with unreliable service in 2023, including those with contaminated supplies.",
     "The measure includes homes with intermittent service or contaminated supplies, even where physical supplies are abundant."),
])
def test_reader_retains_measurement_before_definition_without_ranking_by_numbers(topic, measurement, definition):
    prior = [measurement,
             "The authors discussed the survey results in a statement describing the households included in the study.",
             "The findings describe access to essential services and may differ from other measures of infrastructure.",
             "The study does not establish the situation in countries or periods beyond the scope of the data collected."]
    after = [definition,
             "The definition covers several distinct barriers, which should not be interpreted as the same physical problem.",
             "Results depend on the stated population and the survey's definition; alternative measures can differ.",
             "This measure does not establish that every household experienced the same conditions at the same time."]
    distant = "A distant promotional section invites readers to subscribe for future articles about unrelated subjects."
    page = "\n\n".join([distant, *prior, "# " + topic, *after, distant])
    assert reading_passages(page, topic) == [*prior, *after]


@pytest.mark.parametrize("heading", ["# Notes", "# Footnotes", "# Endnotes"])
def test_distant_complete_notes_preserve_scope_without_matching_query_words(heading):
    paragraph = "The battery storage survey reports increased capacity at facilities included in the panel."
    distant = "\n\n".join(f"Section {index} discusses the report's background and its administrative arrangements."
                          for index in range(12))
    notes = ["1. The sample covers 1998–2007 only.", "2. Estimates exclude systems below 2.5 MW."]
    page = "\n\n".join([paragraph, distant, heading, *notes, "# References", "Do not include the bibliography."])
    window = reading_window(page, "battery storage")
    assert window["context"] == [heading, *notes]
    assert all(note not in window["passages"] for note in notes)


def test_notes_at_real_eof_are_complete_but_notes_cut_off_by_reader_are_unavailable():
    paragraph = "The school funding survey reports the amounts available to each participating district."
    prefix = paragraph + "\n\n# Notes\n\n"
    assert reading_window(prefix + "1. Rural districts only.", "school funding")["context"] == [
        "# Notes", "1. Rural districts only."]
    window = reading_window(prefix + "x" * 160_000, "school funding")
    assert window == {"passages": [], "context": [], "unavailable_reason": "Source notes extend beyond the reading limit."}


def test_notes_overflow_rejects_the_window_instead_of_selecting_a_convenient_subset():
    paragraph = "The water supply report describes the annual survey of the region's public utilities."
    notes = "\n\n".join(f"{index}. " + "scope qualification " * 90 for index in range(6))
    window = reading_window(paragraph + "\n\n# Notes\n\n" + notes + "\n\n# References", "water supply")
    assert window == {"passages": [], "context": [], "unavailable_reason": "Complete source context exceeds the reading limit."}


def test_linked_notes_heading_cannot_import_an_unrelated_article_section():
    paragraph = "The battery storage survey reports increased capacity at facilities included in the panel."
    distant = "\n\n".join(f"Section {index} discusses the report's background and its administrative arrangements."
                          for index in range(12))
    page = paragraph + "\n\n" + distant + "\n\n# [Notes](https://example.org/other)\n\nUnrelated short text."
    assert reading_window(page, "battery storage")["context"] == []


def test_long_matching_notes_remain_context_instead_of_becoming_standalone_evidence():
    paragraph = "The battery storage survey reports increased capacity at facilities included in the panel."
    note = ("1. Battery storage estimates cover 1998–2007 only, exclude installations below 2.5 MW, "
            "and do not describe all facilities operating today.")
    page = paragraph + "\n\n# Notes\n\n" + note
    assert source_excerpts(page, "battery storage") == [paragraph]
    assert reading_passages(page, "battery storage") == [paragraph]
    assert reading_window(page, "battery storage") == {"passages": [paragraph], "context": ["# Notes", note]}


def test_query_matching_only_a_note_cannot_admit_unrelated_article_passages():
    paragraph = "The agency's annual report describes the spending on administration at its headquarters."
    note = "1. Battery storage estimates cover 1998–2007 only and do not describe all facilities operating today."
    page = paragraph + "\n\n# Notes\n\n" + note
    assert source_excerpts(page, "battery storage") == []
    assert reading_window(page, "battery storage") == {"passages": [], "context": []}


@pytest.mark.parametrize("position", [159_992, 159_995, 160_100])
def test_notes_heading_crossing_or_beyond_reading_limit_cannot_silently_disappear(position):
    paragraph = "The battery storage survey reports increased capacity at facilities included in the panel."
    prefix = paragraph + "\n\n"
    page = prefix + "x" * (position - len(prefix) - 2) + "\n\n# Notes\n\n1. Facilities observed before 2008 only."
    assert page.index("# Notes") == position
    assert reading_window(page, "battery storage") == {
        "passages": [], "context": [], "unavailable_reason": "Source notes extend beyond the reading limit."}


def test_source_context_inspection_has_a_separate_bound_without_truncating_citations():
    paragraph = "The water supply report describes the annual survey of the region's public utilities."
    assert reading_window(paragraph + "\n\n" + "x" * 2_000_000, "water supply") == {
        "passages": [], "context": [], "unavailable_reason": "Source context exceeds the validation limit."}
