"""Read explicit publication notices without treating observation dates as publication.

Each parsed date carries its exact field or notice. Conflicting notices remain unknown;
article-body dates, copyright years and PDF creation timestamps cannot supply the date.
"""

from __future__ import annotations

import re
from datetime import date, datetime

_MONTH_DATE = r"(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}"
_MONTH_YEAR = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)[ \t]+\d{4}"
_NOTICE_LINK = r"(?:[ \t]+\[[^\[\]\n]{1,80}\]\(https?://[^\s()]+\))?"
_PARTIAL_NOTICE = re.compile(rf"Published(?:[ \t]+on)?[ \t]*:?[ \t]+{_MONTH_YEAR}{_NOTICE_LINK}", re.I)
_SERIAL_NOTICE = re.compile(
    rf"[A-Za-z][^\n,\[\]<>:;]{{1,160}},[ \t]*"
    rf"(?:(?:Vol\.?|No\.?|Issue)[ \t]+\d+(?:[.-]\d+)?[ \t]*,[ \t]*)+{_MONTH_YEAR}{_NOTICE_LINK}", re.I)


def publication_notices(markdown: str) -> list[str]:
    """Preserve coarse publication notices verbatim without inventing an ISO day.

    Only complete short header blocks with explicit publication or serial-issue
    syntax qualify. Bibliographies, related-story sections and article-body dates
    do not establish the publication notice of this page.
    """
    blocks = re.split(r"\n\s*\n", markdown[:12_000])
    if len(markdown) > 12_000:
        blocks.pop()
    notices = []
    for raw in blocks:
        block = raw.strip()
        heading = re.sub(r"^#{1,6}\s+", "", block.split("\n", 1)[0]).strip(" \t\r:*_").casefold()
        if heading in {"references", "bibliography", "notes", "footnotes", "endnotes", "related stories", "related articles"}:
            break
        if not 1 <= len(block) <= 400 or re.search(r"\b(?:copyright|retrieved|updated|modified)\b", block, re.I):
            continue
        if re.match(r"(?:See|Cf\.?|Compare|Cited)[ \t]+", block, re.I):
            continue
        if (_PARTIAL_NOTICE.fullmatch(block) or _SERIAL_NOTICE.fullmatch(block)) and block not in notices:
            notices.append(block)
    return notices


def publication_date(metadata: dict, markdown: str) -> tuple[str, str]:
    """Return an ISO day and provenance, or an explicit unknown/conflict reason."""
    if metadata.get("publication_conflict"):
        return "", "Conflicting publication notices."
    found: dict[str, str] = {}
    for key in ("publishedTime", "article:published_time", "datePublished", "jsonld:datePublished"):
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
    if len(markdown) > 12_000:
        header = header.rsplit("\n", 1)[0] if "\n" in header else ""
    patterns = (
        rf"(?m)^({_MONTH_DATE})\s*\n\s*For Immediate Release\b",
        rf"(?im)^Published(?: on|:)?\s+({_MONTH_DATE})\s*$",
        rf"Transmission of material in this news release is embargoed until[^\n]{{0,120}}?({_MONTH_DATE})",
        rf"(?m)^Transmission of material in this news release is embargoed until[^\n]{{0,80}}\n"
        rf"(?:\d{{1,2}}:\d{{2}}[ \t]+(?:a\.m\.|p\.m\.)[ \t]+(?:\([A-Z]{{2,4}}\)[ \t]+)?)?"
        rf"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),[ \t]+)?({_MONTH_DATE})[ \t]*\r?$",
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
