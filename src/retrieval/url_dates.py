"""Read a publication date from a publisher's own URL path when the page states none.

News publishers put the publication date in the path -- /2025/07/20/story, /2026/02/politics/
-- and a page without date metadata is otherwise admitted as "date unconfirmed" whatever its
path says. A full year/month/day path gives a day with its own provenance string. A partial
path (year, or year and month) never invents a day: it changes nothing except when every day
it could denote is already after the search boundary, where the page is excluded as a later
publication, because that conclusion holds for any day in the period. Search-filter dates and
landing pages remain what they were: not article dates.
"""

from __future__ import annotations

import re
from datetime import date
from urllib.parse import urlsplit

_PATH_DATE = re.compile(r"/((?:19|20)\d{2})(?:/(0[1-9]|1[0-2])(?:/(0[1-9]|[12]\d|3[01]))?)?(?=/|$)")


def url_path_date(url: str) -> str:
    """The first plausible path date as YYYY, YYYY-MM or YYYY-MM-DD; empty when the path has none."""
    path = urlsplit(url or "").path
    for match in _PATH_DATE.finditer(path):
        year, month, day = match.groups()
        if day:
            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError:
                continue
        return f"{year}-{month}" if month else year
    return ""


def later_by_path(path_date: str, cutoff: str) -> bool:
    """Whether every day a partial path date could denote lies after the inclusive cutoff."""
    if not path_date or not cutoff:
        return False
    return path_date > cutoff[:len(path_date)]
