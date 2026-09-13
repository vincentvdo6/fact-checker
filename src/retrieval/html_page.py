"""Extract page text and links without executing scripts or generating evidence.

Block boundaries preserve publication notices and paragraph quotations. Hidden content,
navigation and scripts are omitted; this is a bounded HTML reader, not a browser.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin

from src.retrieval.article_metadata import article_publication
from src.retrieval.web_sources import SourceUnavailable

_VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
_BLOCKS = frozenset("main article section div p li ul ol h1 h2 h3 h4 h5 h6 pre blockquote tr".split())
_OMIT = frozenset("script style noscript nav header footer aside figcaption svg form button table".split())
_ANNOTATIONS = {"sup": ("^(", ")"), "sub": ("_(", ")"),
                "del": ("[deleted: ", "]"), "ins": ("[inserted: ", "]")}


class HTMLPage(HTMLParser):
    """Keep literal visible text with absolute links and explicit date metadata."""

    def __init__(self, url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.url = url
        self.stack: list[tuple[str, bool, str]] = []
        self.parts: list[str] = []
        self.metadata: dict[str, object] = {"sourceURL": url}
        self.title: list[str] = []
        self.jsonld: list[str] = []
        self.jsonld_current: list[str] | None = None
        self.active_regions: dict[int, tuple[str, int]] = {}
        self.regions: dict[str, list[tuple[int, int]]] = {"body": [], "article": [], "main": []}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if (tag == "script" and (values.get("type") or "").lower() == "application/ld+json"
                and not (self.stack and self.stack[-1][1])):
            self.jsonld_current = []
        hidden = (bool(self.stack and self.stack[-1][1]) or tag in _OMIT
                  or "hidden" in values or values.get("aria-hidden") == "true"
                  or bool(re.search(r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\s*(?:!important\s*)?(?:;|$)",
                                    values.get("style") or "", re.I)))
        if tag == "meta" and not hidden:
            name = values.get("property") or values.get("name")
            if name in {"datePublished", "article:published_time", "publishedTime"} and values.get("content"):
                if name in self.metadata and self.metadata[name] != values["content"]:
                    self.metadata["publication_conflict"] = True
                self.metadata[name] = values["content"]
        href = urljoin(self.url, values.get("href") or "") if tag == "a" else ""
        if tag not in _VOID:
            if len(self.stack) >= 128:
                raise SourceUnavailable("The source HTML nesting exceeded the parsing limit.")
            self.stack.append((tag, hidden, href))
        if not hidden:
            if tag not in _VOID and "articleBody" in (values.get("itemprop") or "").split():
                self.active_regions[len(self.stack) - 1] = ("body", len(self.parts))
            elif tag in {"article", "main"}:
                self.active_regions[len(self.stack) - 1] = (tag, len(self.parts))
            if tag in _BLOCKS:
                self.parts.append("\n\n")
                if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                    self.parts.append("# ")
            elif tag == "br":
                self.parts.append("\n")
            elif tag == "a":
                self.parts.append("[")
            elif tag in _ANNOTATIONS:
                self.parts.append(_ANNOTATIONS[tag][0])

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.jsonld_current is not None:
            self.jsonld.append("".join(self.jsonld_current))
            self.jsonld_current = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                _, hidden, href = self.stack[index]
                del self.stack[index:]
                if not hidden:
                    if tag == "a":
                        self.parts.append(f"]({href})")
                    elif tag in _BLOCKS:
                        self.parts.append("\n\n")
                    elif tag in _ANNOTATIONS:
                        self.parts.append(_ANNOTATIONS[tag][1])
                for depth in list(self.active_regions):
                    if depth >= index:
                        kind, start = self.active_regions.pop(depth)
                        self.regions[kind].append((start, len(self.parts)))
                break

    def handle_data(self, data: str) -> None:
        if self.jsonld_current is not None:
            self.jsonld_current.append(data)
        if self.stack and self.stack[-1][1]:
            return
        if any(tag == "title" for tag, _, _ in self.stack):
            self.title.append(data)
        elif not any(tag == "head" for tag, _, _ in self.stack):
            self.parts.append(data if any(tag == "pre" for tag, _, _ in self.stack)
                              else re.sub(r"\s+", " ", data))

    def page(self) -> dict:
        """Return source text and provenance in the research adapter's format."""
        start, end = 0, len(self.parts)
        for kind in ("main", "body", "article"):
            regions = self.regions[kind]
            if len(regions) == 1:
                start, end = regions[0]
                break
        parts = self.parts[start:end]
        text = re.sub(r"[ \t]+\n", "\n", "".join(parts))
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return {"markdown": text, "metadata": self.metadata | article_publication(self.jsonld, self.url) | {
            "title": "".join(self.title).strip()}}
