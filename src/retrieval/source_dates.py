"""Read explicit publication notices without treating observation dates as publication.

Each parsed date carries its exact field or notice. Conflicting notices remain unknown;
article-body dates, copyright years and PDF creation timestamps cannot supply the date.
"""

from __future__ import annotations

import re
from datetime import date, datetime

_MONTH_DATE = r"(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}"


def publication_date(metadata: dict, markdown: str) -> tuple[str, str]:
    """Return an ISO day and provenance, or an explicit unknown/conflict reason."""
    found: dict[str, str] = {}
    for key in ("publishedTime", "article:published_time", "datePublished"):
        raw = metadata.get(key)
        if not isinstance(raw, str) or not re.match(r"^\d{4}-\d{2}-\d{2}(?:T|$)", raw):
            continue
        try:
            if "T" in raw:
                datetime.fromisoformat(raw)
            found[date.fromisoformat(raw[:10]).isoformat()] = f"Page metadata {key}: {raw[:100]}"
        except ValueError:
            continue
    header = markdown[:12_000]
    patterns = (
        rf"(?m)^({_MONTH_DATE})\s*\n\s*For Immediate Release\b",
        rf"(?im)^Published(?: on|:)?\s+({_MONTH_DATE})\s*$",
        rf"Transmission of material in this news release is embargoed until[^\n]{{0,120}}?({_MONTH_DATE})",
        rf"(?m)^For release [^\n]{{0,80}}?({_MONTH_DATE})\s+USDL-\d{{2}}-\d+",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, header):
            try:
                found[datetime.strptime(match[1], "%B %d, %Y").date().isoformat()] = match[0]
            except ValueError:
                continue
    if len(found) == 1:
        return next(iter(found.items()))
    return "", "Conflicting publication notices." if found else "No explicit publication date found."


def temporal_scope(published: str, cutoff: str) -> str:
    """An earlier publication is only eligible for review, never proof of the right period."""
    if not cutoff or not published:
        return "date_unconfirmed"
    return "later_publication" if published > cutoff else "published_by_cutoff"
