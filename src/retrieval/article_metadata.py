"""Use article-scoped JSON-LD dates, never dates from related stories or organizations."""

from __future__ import annotations

import json
from datetime import datetime
from urllib.parse import urljoin

from src.retrieval.web_sources import public_url

_ARTICLES = {"Article", "NewsArticle", "Report", "BlogPosting", "ScholarlyArticle"}


def article_publication(blocks: list[str], url: str) -> dict[str, object]:
    candidates = []
    for block in blocks:
        try:
            data = json.loads(block)
        except (ValueError, RecursionError):
            continue
        nodes = data if isinstance(data, list) else [data]
        for node in nodes[:100]:
            if isinstance(node, dict) and isinstance(node.get("@graph"), list):
                candidates.extend(node["@graph"][:100])
            else:
                candidates.append(node)
    articles = []
    for node in candidates:
        if not isinstance(node, dict):
            continue
        types = node.get("@type", [])
        types = [types] if isinstance(types, str) else types if isinstance(types, list) else []
        if any(isinstance(kind, str) and kind.rsplit("/", 1)[-1] in _ARTICLES for kind in types):
            articles.append(node)
    dates = set()
    for node in articles:
        identities = []
        for key in ("url", "mainEntityOfPage", "@id"):
            value = node.get(key)
            if isinstance(value, dict):
                value = value.get("@id")
            if value:
                try:
                    identities.append(public_url(urljoin(url, value)) if isinstance(value, str) else "")
                except ValueError:
                    identities.append("")
        if not identities or any(identity != public_url(url) for identity in identities):
            continue
        raw = node.get("datePublished")
        if isinstance(raw, str) and len(raw) <= 100:
            dates.add(raw)
    if len(dates) > 1:
        try:
            instants = [datetime.fromisoformat(raw) for raw in dates]
            agree = (all(value.tzinfo is not None for value in instants)
                     and len(set(instants)) == 1 and len({value.date() for value in instants}) == 1)
        except ValueError:
            agree = False
        if not agree:
            return {"publication_conflict": True}
    usable = [raw for raw in dates if len(raw) == 10 or raw[10:11] == "T"]
    return {"jsonld:datePublished": min(usable)} if usable else {}
