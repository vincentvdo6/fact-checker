"""Read discovered pages locally with a shared per-click deadline and transient cache."""

from __future__ import annotations

import time

from src.retrieval.html_page import HTMLPage
from src.retrieval.http_pages import decode_html, fetch_page


class PageSources:
    def __init__(self) -> None:
        self.deadline = 0.0
        self._pages: dict[str, dict] = {}

    def begin_check(self) -> None:
        self.deadline = time.monotonic() + 45
        self._pages.clear()

    def scrape(self, url: str) -> dict:
        return self.read_link(url)

    def read_link(self, url: str) -> dict:
        """Read publisher HTML directly using this check's deadline and page cache."""
        if url not in self._pages:
            response = fetch_page(url, deadline=self.deadline, headers={"Accept": "text/html"})
            parser = HTMLPage(response.url)
            parser.feed(decode_html(response))
            parser.close()
            self._pages[url] = parser.page()
        return self._pages[url]
