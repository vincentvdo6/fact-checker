"""A header dateline dates a page; a date inside prose, a revision line or two datelines never do."""

from __future__ import annotations

from src.retrieval.byline_dates import byline_date


def test_datelines_are_read_from_the_header_in_their_common_forms():
    shrm = ("# Share\n\n# January 2025 Labor Market Review: The Talent Shortage Persists\n\n"
            "January 13, 2025 | [  Justin Ladner  ](https://www.shrm.org/about/bio)\n\n# Employment Growth\n\n"
            "On January 6, 2021, rioters entered the Capitol.")
    assert byline_date(shrm) == ("2025-01-13", "Dateline: January 13, 2025 | [  Justin Ladner  ](https://www.shrm.org/about/bio)")
    assert byline_date("# Title\n\nMarch 3, 2025\n\nBody.")[0] == "2025-03-03"
    assert byline_date("Posted on March 3, 2025\n\nBody.")[0] == "2025-03-03"
    assert byline_date("Tuesday, March 4, 2025 · By Jane Doe\n\nBody.")[0] == "2025-03-04"
    assert byline_date("By Jane Doe | March 3, 2025\n\nBody.")[0] == "2025-03-03"


def test_prose_dates_revisions_conflicts_and_impossible_days_yield_nothing():
    assert byline_date("# Title\n\nOn January 6, 2021, rioters entered the Capitol. The report of March 3, 2025 says otherwise.") == ("", "")
    assert byline_date("Updated March 3, 2025\n\nBody.") == ("", ""), "a revision date is not a publication date"
    assert byline_date("March 3, 2025 | Updated 10:00 ET\n\nBody.") == ("", ""), "a dateline marked as a revision is not read"
    assert byline_date("March 3, 2025\n\nBody\n\nApril 1, 2025 | Editor") == ("", ""), "two datelines conflict"
    assert byline_date("February 30, 2025\n\nBody.") == ("", "")
    assert byline_date("x" * 12_000 + "\nMarch 3, 2025\n") == ("", ""), "only the header is read"
    assert byline_date("") == ("", "")
