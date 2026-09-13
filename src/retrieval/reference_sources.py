"""Follow a bounded set of literal references without treating them as independent proof.

Only accepted parent passages authorize discovery. The injected reader owns network
safety and the existing deadline; this adapter cannot extend either or follow children.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit

from src.retrieval.visible_text import inline_links
from src.retrieval.web_sources import SourceUnavailable, public_url


def reference_url(value: object) -> str:
    """Normalize public transport aliases without changing literal reference metadata."""
    if not isinstance(value, str) or any(char.isspace() or ord(char) < 32 or char == "\\" for char in value):
        return ""
    validated = public_url(value)
    if not validated:
        return ""
    parsed = urlsplit(validated)
    host = (parsed.hostname or "").removesuffix(".")
    if host.endswith("."):
        return ""
    authority = f"[{host}]" if ":" in host else host
    return urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, ""))


def reference_candidates(parent_sources: Iterable[dict], excluded_hosts: Iterable[str]) -> list[dict]:
    """Keep the first twenty unique destinations and every matching literal lineage."""
    parents = list(parent_sources)
    existing = {reference_url(source.get("url")) for source in parents}
    excluded = {host.lower().removeprefix("www.").rstrip(".") for host in excluded_hosts}
    candidates: dict[str, dict] = {}
    for source in parents:
        parent_url = reference_url(source.get("url"))
        identity = source.get("id")
        if not parent_url or not isinstance(identity, str) or not identity or source.get("temporal_status") == "later_publication":
            continue
        paragraphs = list(dict.fromkeys(text for text in [*source.get("excerpts", []), *source.get("reading_passages", [])]
                                        if isinstance(text, str) and text.strip()))
        for index, paragraph in enumerate(paragraphs, 1):
            for link in inline_links(paragraph):
                url = reference_url(link["destination"])
                host = (urlsplit(url).hostname or "").removeprefix("www.").rstrip(".")
                if not url or url in existing or any(host == item or host.endswith("." + item) for item in excluded):
                    continue
                if url not in candidates:
                    if len(candidates) >= 20:
                        continue
                    candidates[url] = {"url": url, "title": link["label"], "lineage": []}
                lineage = {"parent_source_id": identity, "parent_url": source["url"],
                           "parent_passage_id": f"{identity}:p{index}", "paragraph": paragraph,
                           **link, "markup": paragraph[link["start"]:link["end"]]}
                if lineage not in candidates[url]["lineage"]:
                    candidates[url]["lineage"].append(lineage)
    return list(candidates.values())


class ReferenceSources:
    """Reuse successful and failed reads within one immutable two-call allowance."""

    def __init__(self, candidates: list[dict], reader_callable: Callable[[str], dict]) -> None:
        self._candidates = {item["url"]: deepcopy(item) for item in candidates}
        if (len(candidates) > 20 or len(self._candidates) != len(candidates)
                or any(not url or reference_url(url) != url for url in self._candidates)):
            raise ValueError("Reference candidates must have unique public transport URLs.")
        self._reader = reader_callable
        self._cache: dict[str, dict | Exception] = {}
        self._attempts: dict[str, dict] = {}
        self._resolved_urls: dict[str, str] = {}

    @property
    def attempts(self) -> dict[str, dict]:
        return deepcopy(self._attempts)

    @property
    def resolved_urls(self) -> dict[str, str]:
        return dict(self._resolved_urls)

    @property
    def lineage(self) -> dict[str, list[dict]]:
        return {url: deepcopy(candidate["lineage"]) for url, candidate in self._candidates.items()}

    def begin_check(self) -> None:
        """The parent check already owns the deadline and the adapter's lifetime."""

    def search(self, query: str) -> list[dict]:
        remaining = 2 - len(self._attempts)
        found = []
        for url, candidate in self._candidates.items():
            if url in self._attempts:
                found.append(deepcopy(candidate))
            elif remaining > 0:
                found.append(deepcopy(candidate))
                remaining -= 1
        return found

    def search_before(self, query: str, cutoff: str) -> list[dict]:
        """A search boundary does not establish the referenced page's publication date."""
        return self.search(query)

    def scrape(self, url: str) -> dict:
        if not isinstance(url, str) or url not in self._candidates:
            raise SourceUnavailable("The page is not an accepted parent reference.")
        if url not in self._cache:
            if url in self._attempts:
                raise SourceUnavailable("The referenced page read is already in progress.")
            if len(self._attempts) >= 2:
                raise SourceUnavailable("The reference page budget was reached.")
            self._attempts[url] = {"status": "attempted"}
            try:
                page = self._reader(url)
                if not isinstance(page, dict) or not isinstance(page.get("metadata"), dict):
                    raise SourceUnavailable("The referenced page has no valid provenance.")
                info = page["metadata"]
                final_url = reference_url(info.get("sourceURL", url))
                if not final_url or not isinstance(page.get("markdown"), str) or info.get("error") or info.get("statusCode", 200) != 200:
                    raise SourceUnavailable("The referenced page could not be extracted from a public URL.")
                self._resolved_urls[url] = final_url
                self._cache[url] = deepcopy(page)
                self._attempts[url] = {"status": "read", "final_url": final_url}
            except Exception as error:
                self._cache[url] = error
                detail = {"reason": str(error)[:300]} if isinstance(error, SourceUnavailable) else {"error_type": type(error).__name__[:100]}
                self._attempts[url] = {"status": "failed", **detail}
                raise
        cached = self._cache[url]
        if isinstance(cached, Exception):
            raise cached
        return deepcopy(cached)
