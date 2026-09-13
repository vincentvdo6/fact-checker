"""A publisher's path can date a page; a partial path never invents a day and only ever excludes."""

from __future__ import annotations

from src.retrieval.url_dates import later_by_path, url_path_date


def test_path_dates_are_read_whole_partial_or_not_at_all():
    assert url_path_date("https://www.nytimes.com/2025/07/20/us/politics/story.html") == "2025-07-20"
    assert url_path_date("https://www.cnn.com/interactive/2026/02/politics/sotu/") == "2026-02"
    assert url_path_date("https://unstats.un.org/sdgs/report/2024/Goal-01/") == "2024"
    assert url_path_date("https://example.org/2025/02/30/story") == "", "an impossible day is not a date"
    assert url_path_date("https://example.org/story-2025-07-20") == "" and url_path_date("") == ""
    assert url_path_date("https://example.org/id/20250720/story") == "", "digits must be path segments"
    assert url_path_date("https://example.org/1850/story") == "" and url_path_date("https://example.org/2025/13/story") == "2025"


def test_partial_path_excludes_only_when_every_possible_day_is_past_the_cutoff():
    assert later_by_path("2026-02", "2025-07-20") and later_by_path("2026", "2025-07-20")
    assert not later_by_path("2025-07", "2025-07-20"), "the 1st to the 20th are inside the boundary"
    assert not later_by_path("2025", "2025-07-20") and not later_by_path("2024", "2025-07-20")
    assert not later_by_path("", "2025-07-20") and not later_by_path("2026", "")
