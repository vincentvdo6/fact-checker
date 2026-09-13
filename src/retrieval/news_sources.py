"""Parse RSS discovery links without promoting feed metadata to evidence.

RSS is discovery only: feed descriptions never become evidence, and feed dates never
become page publication dates. Search coverage and endpoint availability are unmeasured.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from src.retrieval.web_sources import SourceUnavailable, public_url


def news_results(body: bytes, *, provider: str = "News RSS") -> list[dict]:
    try:
        text = body.decode("utf-8-sig")
    except UnicodeError:
        raise SourceUnavailable("The news feed encoding could not be read.") from None
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise SourceUnavailable("The news feed contains unsupported XML declarations.")
    try:
        root = ET.fromstring(text)
    except (ET.ParseError, ValueError):
        raise SourceUnavailable("The news search returned an invalid feed.") from None
    channel = root.find("channel") if root.tag == "rss" else None
    if channel is None:
        raise SourceUnavailable("The news search returned an invalid feed.")
    found: dict[str, dict] = {}
    for item in channel.findall("item")[:50]:
        url = public_url(item.findtext("link") or "")
        if not url or url in found:
            continue
        raw_date = (item.findtext("pubDate") or "")[:100]
        reported = ""
        try:
            parsed_date = parsedate_to_datetime(raw_date)
            if parsed_date.tzinfo is not None:
                reported = parsed_date.isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
        found[url] = {"url": url, "title": (item.findtext("title") or "")[:300],
                      "feed_published_at": reported, "feed_date_basis": f"{provider} pubDate: {raw_date}"}
        if len(found) == 20:
            break
    return list(found.values())
