"""Publication eligibility requires an explicit date, never an observation or retrieval date."""

from __future__ import annotations

import pytest

from src.retrieval.source_dates import publication_date, publication_notices, temporal_scope


@pytest.mark.parametrize("metadata,body,expected", [
    ({"publishedTime": "2025-07-01T10:00:00Z"}, "", "2025-07-01"),
    ({}, "August 20, 2026\n\nFor Immediate Release\n\nUnemployment in July...", "2026-08-20"),
    ({}, "**Transmission of material in this news release is embargoed until** **8:30 a.m. (ET) Friday, September 4, 2026**", "2026-09-04"),
    ({}, "In July 2025 the unemployment rate changed.\n\nCopyright 2026", ""),
    ({"dateModified": "2026-07-01"}, "", ""),
    ({"datePublished": "2025-07-20Tnot-a-time"}, "", ""),
    ({"datePublished": "2025-07-01"}, "Published August 1, 2026", ""),
    ({}, "Published February 30, 2025", ""),
    ({}, "For release 10:00 a.m. (ET) Tuesday, July 1, 2025       USDL-25-1087", "2025-07-01"),
])
def test_publication_dates_require_unambiguous_explicit_notices(metadata, body, expected):
    published, basis = publication_date(metadata, body)
    assert published == expected and basis


def test_unknown_dates_never_acquire_temporal_applicability():
    assert temporal_scope("", "2025-07-20") == "date_unconfirmed"
    assert temporal_scope("2025-07-20", "") == "date_unconfirmed"


def test_publication_boundary_is_inclusive_without_asserting_observation_time():
    assert temporal_scope("2025-07-20", "2025-07-20") == "published_by_cutoff"
    assert temporal_scope("2025-07-21", "2025-07-20") == "later_publication"
    assert temporal_scope("2025-07-19", "2025-07-20") == "published_by_cutoff"


def test_notices_outside_the_bounded_header_do_not_supply_publication():
    assert publication_date({}, "x" * 12_000 + "\nPublished July 20, 2025")[0] == ""


def test_matching_notices_keep_exact_provenance():
    notice = "Published July 20, 2025"
    assert publication_date({"article:published_time": "2025-07-20T08:00:00-07:00"}, notice) == ("2025-07-20", notice)
    published, basis = publication_date({"datePublished": "2025-07-20"}, "")
    assert published == "2025-07-20" and basis == "Page metadata datePublished: 2025-07-20"


@pytest.mark.parametrize("raw,expected", [
    ("2025-07-20", "2025-07-20"),
    ("2025-07-20T08:00:00-07:00", "2025-07-20"),
    ("2025-02-30", ""), ("2025-07-20Tinvalid", ""), ("July 20, 2025", ""), (None, ""),
])
def test_structured_publication_dates_require_valid_dates_and_keep_exact_provenance(raw, expected):
    published, basis = publication_date({"jsonld:datePublished": raw}, "")
    assert published == expected
    assert basis == (f"Page metadata jsonld:datePublished: {raw}" if expected else "No explicit publication date found.")


@pytest.mark.parametrize("metadata,body", [
    ({"publication_conflict": True, "datePublished": "2025-07-20"}, ""),
    ({"publication_conflict": True}, "Published July 20, 2025"),
    ({"jsonld:datePublished": "2025-07-20", "datePublished": "2025-07-21"}, ""),
    ({"jsonld:datePublished": "2025-07-20"}, "Published July 21, 2025"),
])
def test_structured_publication_conflicts_cannot_be_resolved_by_another_notice(metadata, body):
    assert publication_date(metadata, body) == ("", "Conflicting publication notices.")


def test_matching_structured_publication_dates_remain_usable():
    metadata = {"datePublished": "2025-07-20", "jsonld:datePublished": "2025-07-20T08:00:00Z"}
    assert publication_date(metadata, "") == ("2025-07-20", "Page metadata jsonld:datePublished: 2025-07-20T08:00:00Z")


def test_embargo_date_can_be_on_the_immediately_following_line():
    notice = ("Transmission of material in this news release is embargoed until\t USDL-25-1089\r\n"
              "8:30 a.m. (ET) Thursday, July 3, 2025")
    assert publication_date({}, notice + "\n\nJune employment report.") == ("2025-07-03", notice)
    unrelated = "Transmission of material in this news release is embargoed until\n\nEmployment increased on July 3, 2025"
    assert publication_date({}, unrelated)[0] == ""
    assert publication_date({"datePublished": "2025-07-04"}, notice)[0] == ""


def test_wrapped_release_dates_require_a_release_continuation_and_complete_header_line():
    prefix = "Transmission of material in this news release is embargoed until"
    for continuation in ("Employment increased on July 3, 2025", "July 3, 2025 (release not confirmed)"):
        assert publication_date({}, prefix + "\n" + continuation)[0] == ""
    notice = prefix + "\nJuly 3, 2025"
    assert publication_date({}, notice) == ("2025-07-03", notice)
    padding = "x" * (12_000 - len(notice) - 2) + "\n\n"
    assert publication_date({}, padding + notice + " (release not confirmed)")[0] == ""
    assert publication_date({}, notice + "\n\n" + "x" * 12_000) == ("2025-07-03", notice)
    assert publication_date({}, prefix + "x" * 81 + "\nJuly 3, 2025")[0] == ""


@pytest.mark.parametrize("notice", [
    "Published: March  2025",
    "Published on September 2024",
    "Materials Bulletin, Vol. 12, No. 3, March  2025",
    "Museum Review, Issue 7, February 2021 [DOI](https://doi.org/10.1000/example)",
])
def test_partial_publication_notices_preserve_literal_spacing_without_supplying_a_day(notice):
    assert publication_notices(notice) == [notice]
    published, _ = publication_date({}, notice)
    assert published == "" and temporal_scope(published, "2025-07-20") == "date_unconfirmed"


@pytest.mark.parametrize("text", [
    "The sample ran in March 2025.", "Copyright March 2025", "Last modified: March 2025",
    "Retrieved: March 2025", "Updated March 2025", "# [Published: March 2025](https://example.org/other)",
    "See Materials Bulletin, No. 3, March 2025", "Published: March 2025 describes the sample.",
    "# References\n\nMaterials Bulletin, No. 3, March 2025",
    "Bibliography\n\nMaterials Bulletin, No. 3, March 2025",
    "# Related stories\n\nPublished: March 2025", "# Notes\n\nPublished: March 2025",
    "# References:\n\nMaterials Bulletin, No. 3, March 2025",
    "**References**\n\nMaterials Bulletin, No. 3, March 2025",
    "# References\nA citation on the next line.\n\nPublished: March 2025",
    "Updated Materials Bulletin, No. 3, March 2025",
    "Cf. Materials Bulletin, No. 3, March 2025",
])
def test_partial_notice_reader_excludes_body_dates_citations_and_other_articles(text):
    assert publication_notices(text) == []


def test_partial_notices_require_complete_blocks_within_both_limits():
    notice = "Published: March 2025"
    prefix = "x" * (12_000 - len(notice) - 5) + "\n\n"
    assert publication_notices(prefix + notice) == [notice]
    assert publication_notices(prefix + notice + " and a qualification beyond the header limit") == []
    assert publication_notices("x" * 12_000 + "\n\n" + notice) == []
    assert publication_notices("Published: March 2025 [" + "d" * 80 + "](https://example.org/" + "x" * 400 + ")") == []


def test_repeated_partial_notices_keep_only_one_verbatim_copy():
    notice = "Published: March  2025"
    assert publication_notices(notice + "\n\n" + notice) == [notice]
