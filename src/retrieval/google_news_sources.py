"""Experimental historical discovery through public Google News RSS.

Resolve article links before reading publisher HTML. The news landing page, headline,
and feed timestamp never become evidence. Public RSS and its link resolver are not a
supported API contract; changes or challenges fail closed within the shared deadline.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from html.parser import HTMLParser
from urllib.parse import urlencode, urlsplit

from src.retrieval.http_pages import decode_html, fetch_page
from src.retrieval.news_sources import news_results
from src.retrieval.page_sources import PageSources
from src.retrieval.search_contract import valid_query
from src.retrieval.web_sources import SourceUnavailable, public_url

_ORIGIN = "https://news.google.com"


def article_id(url: str) -> str:
    parsed = urlsplit(public_url(url))
    match = re.fullmatch(r"/rss/articles/([A-Za-z0-9_-]{1,2048})", parsed.path)
    if parsed.hostname != "news.google.com" or not match:
        raise SourceUnavailable("The Google News article link could not be read.")
    return match[1]


class LinkParameters(HTMLParser):
    """Accept one resolver parameter set bound to the requested article."""

    def __init__(self, identity: str) -> None:
        super().__init__()
        self.identity = identity
        self.values: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "div" and values.get("data-n-a-id") == self.identity:
            timestamp, signature = values.get("data-n-a-ts", ""), values.get("data-n-a-sg", "")
            if (isinstance(timestamp, str) and re.fullmatch(r"[0-9]{1,12}", timestamp)
                    and isinstance(signature, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,256}", signature)):
                self.values.append((timestamp, signature))


def publisher_link(body: bytes) -> str:
    """Read exactly one matching RPC result, validating the destination independently."""
    try:
        text = body.decode("utf-8")
        if not text.startswith(")]}'\n"):
            raise ValueError
        rows = json.loads(text[5:])
        if not isinstance(rows, list):
            raise ValueError
        matches = [row for row in rows if isinstance(row, list) and len(row) >= 3
                   and row[:2] == ["wrb.fr", "Fbv4je"]]
        if len(matches) != 1 or not isinstance(matches[0][2], str):
            raise ValueError
        result = json.loads(matches[0][2])
        if not isinstance(result, list) or len(result) < 2 or result[0] != "garturlres":
            raise ValueError
        url = public_url(result[1])
        if not url or urlsplit(url).hostname == "news.google.com":
            raise ValueError
        return url
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise SourceUnavailable("The Google News publisher link could not be resolved.") from None


class GoogleNewsSources(PageSources):
    always_search = True
    compact_search = True
    scope_note = ("Experimental Google News RSS discovery using the English US edition; "
                  "this does not establish the claim's location. Coverage is unmeasured. "
                  "Search date filters do not establish an article's publication or event date.")

    def search(self, query: str) -> list[dict]:
        return self.search_before(query, "")

    def search_before(self, query: str, cutoff: str) -> list[dict]:
        if not valid_query(query):
            raise SourceUnavailable("The news query exceeds the input limits.")
        if cutoff:
            try:
                # before is exclusive; keep articles on the supplied boundary day.
                exclusive = date.fromisoformat(cutoff) + timedelta(days=1)
            except (ValueError, OverflowError):
                raise SourceUnavailable("The news search boundary could not be read.") from None
            query = f"{query} before:{exclusive.isoformat()}"
        url = _ORIGIN + "/rss/search?" + urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        response = fetch_page(url, deadline=self.deadline, limit=1_000_000, redirects=0,
                              headers={"Accept": "application/rss+xml, application/xml, text/xml"})
        return news_results(response.body, provider="Google News RSS")

    def scrape(self, url: str) -> dict:
        if url in self._pages:
            return self._pages[url]
        identity = article_id(url)
        landing = fetch_page(_ORIGIN + "/rss/articles/" + identity + "?hl=en-US&gl=US&ceid=US:en",
                             deadline=self.deadline, redirects=0)
        parser = LinkParameters(identity)
        parser.feed(decode_html(landing))
        parser.close()
        if len(parser.values) != 1:
            raise SourceUnavailable("The Google News article resolver is unavailable.")
        timestamp, signature = parser.values[0]
        edition = ["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1,
                   None, None, None, None, None, 0, 1]
        context = [edition, "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0]
        request = ["garturlreq", context, identity, int(timestamp), signature]
        body = urlencode({"f.req": json.dumps([[["Fbv4je", json.dumps(request)]]])}).encode()
        response = fetch_page(_ORIGIN + "/_/DotsSplashUi/data/batchexecute", deadline=self.deadline,
                              body=body, redirects=0, limit=65_536,
                              headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"})
        self._pages[url] = super().scrape(publisher_link(response.body))
        return self._pages[url]
