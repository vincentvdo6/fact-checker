"""Read a page's dateline when its metadata carries no publication date.

The commonest way a news page states its date is a header line of its own -- "January 13,
2025 | Justin Ladner", "Posted March 3, 2025", a date alone -- which `publication_date`'s
notice patterns (press-release embargoes, "Published:" lines) do not cover, so such a page
was admitted as "date unconfirmed" and could never corroborate. A dateline is read only when
the date opens a short line in the page header and is followed by nothing or a byline
separator; a date inside prose ("On January 6, 2021, ...") is an event date and never
qualifies, and an "Updated" or "Modified" line is a revision, not a publication. Two
different datelines in one header are a conflict and yield nothing.
"""

from __future__ import annotations

import re
from datetime import datetime

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATELINE = re.compile(
    rf"^(?:By\s+[^|·•—–]{{1,60}}?\s*[|·•—–]\s*)?(?:(?:Published|Posted|Date)(?:\s+on)?:?\s+)?"
    rf"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+)?"
    rf"((?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}})(?:\s*(?:\||·|•|—|–|-|,)\s*.{{0,160}})?$", re.I)
_REVISION = re.compile(r"\b(?:updated|modified|revised|corrected|retrieved|accessed)\b", re.I)
HEADER = 12_000


def byline_date(markdown: str) -> tuple[str, str]:
    """An ISO day and the dateline it came from, or empty strings."""
    found: dict[str, str] = {}
    for raw in markdown[:HEADER].splitlines():
        line = raw.strip().lstrip("#*_ ").strip()
        if not line or len(line) > 200 or _REVISION.search(line):
            continue
        match = _DATELINE.match(line)
        if not match:
            continue
        try:
            day = datetime.strptime(match[1], "%B %d, %Y").date().isoformat()
        except ValueError:
            continue
        found.setdefault(day, line)
    if len(found) != 1:
        return "", ""
    day, line = next(iter(found.items()))
    return day, f"Dateline: {line[:120]}"
