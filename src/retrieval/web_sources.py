"""Provider-independent source contract and validation for research candidates."""

from __future__ import annotations

import ipaddress
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit


class SourceUnavailable(RuntimeError):
    """An operational failure, distinct from an empty search or insufficient evidence."""


def public_url(value: object) -> str:
    """Reject credentials, local addresses and non-web schemes before linking or scraping."""
    if not isinstance(value, str) or len(value) > 2048:
        return ""
    try:
        url = urlsplit(value)
        host = url.hostname or ""
        if (url.scheme != "https" or url.username or url.password or url.port not in (None, 443)
                or "." not in host or host.endswith((".local", ".localhost"))):
            return ""
        try:
            if not ipaddress.ip_address(host).is_global:
                return ""
        except ValueError:
            pass
        return urlunsplit(("https", url.netloc.lower(), url.path or "/", url.query, ""))
    except ValueError:
        return ""


class WebSources(Protocol):
    """Explicit source adapter contract; no provider or network client is configured."""

    def begin_check(self) -> None:
        """Reset the adapter's per-check resources."""
        ...

    def search(self, query: str) -> list[dict]:
        """Discover candidate pages; snippets are not evidence."""
        ...

    def scrape(self, url: str) -> dict:
        """Return page text and provenance for a discovered public URL."""
        ...
