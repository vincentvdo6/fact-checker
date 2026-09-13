"""Read public HTTPS pages with bounded bodies and no ambient cookies or credentials.

Resolve and validate every redirect destination, then connect to the validated address.
TLS still checks the original hostname. A search result must never access local services.
"""

from __future__ import annotations

import http.client
import ipaddress
import queue
import re
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from email.message import Message
from urllib.parse import urljoin, urlsplit

from src.retrieval.web_sources import SourceUnavailable, public_url

_READERS = threading.BoundedSemaphore(2)


def remaining_time(deadline: float) -> float:
    """Never start another network operation after the lookup budget expires."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SourceUnavailable("The web lookup time budget was reached.")
    return min(10.0, remaining)


@dataclass(frozen=True)
class PageResponse:
    url: str
    body: bytes
    headers: Message


def connect_public(host: str, timeout: float) -> http.client.HTTPSConnection:
    """Pin connections to public resolved addresses without bypassing TLS validation."""
    deadline = time.monotonic() + timeout
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    timeout = remaining_time(deadline)
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise SourceUnavailable("The source address is not public.")
    family, kind, protocol, _, address = addresses[0]
    connection = http.client.HTTPSConnection(host, timeout=timeout)
    raw = socket.socket(family, kind, protocol)
    try:
        raw.settimeout(timeout)
        raw.connect(address)
        raw.settimeout(remaining_time(deadline))
        connection.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        return connection
    except BaseException:
        raw.close()
        raise


def _fetch_page(url: str, *, deadline: float, headers: dict[str, str] | None = None,
               redirects: int = 3, limit: int = 2_000_000, body: bytes | None = None) -> PageResponse:
    """Follow bounded public redirects; custom request headers never cross a redirect."""
    for attempt in range(redirects + 1):
        url = public_url(url)
        if not url:
            raise SourceUnavailable("The source URL is not public HTTPS.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SourceUnavailable("The web lookup time budget was reached.")
        parsed = urlsplit(url)
        connection = None
        try:
            connection = connect_public(parsed.hostname, min(10.0, remaining))
            if connection.sock is not None:
                connection.sock.settimeout(remaining_time(deadline))
            else:
                remaining_time(deadline)
            request_headers = {"User-Agent": "fact-checker/0.1 (source research)", "Accept-Encoding": "identity"}
            if attempt == 0:
                request_headers.update(headers or {})
            path = parsed.path + ("?" + parsed.query if parsed.query else "")
            if body is None:
                connection.request("GET", path, headers=request_headers)
            else:
                connection.request("POST", path, body=body, headers=request_headers)
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if body is not None or not location or attempt == redirects:
                    raise SourceUnavailable("The source redirect could not be followed.")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise SourceUnavailable(f"Web request failed (HTTP {response.status}).")
            if response.getheader("Content-Encoding", "identity") != "identity":
                raise SourceUnavailable("The source returned an unsupported content encoding.")
            body = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SourceUnavailable("The web lookup time budget was reached.")
                if connection.sock is not None:
                    connection.sock.settimeout(min(10.0, remaining))
                chunk = response.read1(min(65_536, limit + 1 - len(body)))
                if time.monotonic() >= deadline:
                    raise SourceUnavailable("The web lookup time budget was reached.")
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > limit:
                    raise SourceUnavailable("The source response exceeded the size limit.")
            return PageResponse(url, bytes(body), response.headers)
        except (OSError, ValueError, http.client.HTTPException):
            raise SourceUnavailable("The web request could not be completed.") from None
        finally:
            if connection is not None:
                connection.close()
    raise SourceUnavailable("The source redirect limit was reached.")


def fetch_page(url: str, *, deadline: float, headers: dict[str, str] | None = None,
               redirects: int = 3, limit: int = 2_000_000, body: bytes | None = None) -> PageResponse:
    """Bound the caller's wait even when DNS or response headers stall.

At most two daemon readers may remain in OS calls. Late results are discarded; a late
DNS answer cannot initiate a connection after its setup deadline. Busy readers fail closed.
"""
    remaining_time(deadline)
    if not _READERS.acquire(blocking=False):
        raise SourceUnavailable("Previous web reads are still finishing. Try again shortly.")
    result: queue.Queue[PageResponse | BaseException] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            result.put(_fetch_page(url, deadline=deadline, headers=headers, redirects=redirects, limit=limit, body=body))
        except BaseException as error:
            result.put(error)
        finally:
            _READERS.release()

    try:
        threading.Thread(target=read, daemon=True, name="source-reader").start()
    except RuntimeError:
        _READERS.release()
        raise SourceUnavailable("The web reader could not be started.") from None
    try:
        value = result.get(timeout=max(0.0, deadline - time.monotonic()))
    except queue.Empty:
        raise SourceUnavailable("The web lookup time budget was reached.") from None
    if isinstance(value, BaseException):
        raise value
    if time.monotonic() >= deadline:
        raise SourceUnavailable("The web lookup time budget was reached.")
    return value


def decode_html(page: PageResponse) -> str:
    """Honor declared encoding; preserve undeclared UTF-8 or legacy HTML text."""
    if page.headers.get_content_type() != "text/html":
        raise SourceUnavailable("The source is not an HTML page.")
    charset = page.headers.get_content_charset()
    if not charset:
        match = re.search(br'charset\s*=\s*["\x27]?([a-zA-Z0-9_-]+)', page.body[:4096])
        charset = match[1].decode("ascii") if match else ""
    if not charset:
        try:
            return page.body.decode("utf-8")
        except UnicodeDecodeError:
            charset = "cp1252"
    try:
        return page.body.decode("cp1252" if charset.lower() in {"iso-8859-1", "latin1", "latin-1"} else charset)
    except (UnicodeError, LookupError):
        raise SourceUnavailable("The source text encoding could not be read.") from None
