"""Publication eligibility requires an explicit date, never an observation or retrieval date."""

from __future__ import annotations

import pytest

from src.retrieval.source_dates import publication_date, temporal_scope


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
