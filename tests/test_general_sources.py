"""Local page reads preserve source text within public-network and resource limits."""

from __future__ import annotations

import io
import socket
import threading
import time
from email.message import Message
from types import SimpleNamespace

import pytest

from src.retrieval.html_page import HTMLPage
from src.retrieval.http_pages import PageResponse, connect_public, decode_html, fetch_page
from src.retrieval.page_sources import PageSources
from src.retrieval.source_dates import publication_date
from src.retrieval.web_sources import SourceUnavailable


def headers(**values):
    result = Message()
    for name, value in values.items():
        result[name.replace("_", "-")] = value
    return result


def test_source_reads_are_cached_and_only_request_html(monkeypatch):
    calls = []

    def fetch(url, **kwargs):
        calls.append(kwargs)
        return PageResponse(url, b'<title>Report</title><p>Quoted source text.</p>', headers(Content_Type="text/html"))

    monkeypatch.setattr("src.retrieval.page_sources.fetch_page", fetch)
    sources = PageSources()
    sources.begin_check()
    assert sources.scrape("https://example.org/report")["markdown"] == "Quoted source text."
    sources.scrape("https://example.org/report")
    assert len(calls) == 1 and calls[0]["headers"] == {"Accept": "text/html"}
    sources.begin_check()
    sources.scrape("https://example.org/report")
    assert len(calls) == 2


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1"])
def test_private_dns_addresses_are_rejected_before_connection(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (address, 443))])
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("Private socket opened"))
    with pytest.raises(SourceUnavailable, match="not public"):
        connect_public("example.org", 1)


class Response(io.BytesIO):
    def __init__(self, body=b"<p>Source text</p>", status=200, **fields):
        super().__init__(body)
        self.status = status
        self.headers = headers(Content_Type="text/html", **fields)

    def getheader(self, name, default=None):
        return self.headers.get(name, default)


def connections(monkeypatch, responses):
    calls = []

    def connect(host, timeout):
        response = responses.pop(0)
        return SimpleNamespace(sock=None, request=lambda method, path, **kwargs: calls.append((host, method, path, kwargs)),
                               getresponse=lambda: response, close=lambda: response.close())

    monkeypatch.setattr("src.retrieval.http_pages.connect_public", connect)
    return calls


def test_redirects_revalidate_urls_and_do_not_forward_secret_headers(monkeypatch):
    calls = connections(monkeypatch, [Response(status=302, Location="https://other.org/report"), Response()])
    result = fetch_page("https://example.org/", deadline=time.monotonic() + 10, headers={"X-Subscription-Token": "secret"})
    assert result.url == "https://other.org/report" and len(calls) == 2
    assert "X-Subscription-Token" in calls[0][3]["headers"]
    assert "X-Subscription-Token" not in calls[1][3]["headers"]
    calls = connections(monkeypatch, [Response(status=302, Location="https://127.0.0.1/private")])
    with pytest.raises(SourceUnavailable, match="public HTTPS"):
        fetch_page("https://example.org/", deadline=time.monotonic() + 10)
    assert len(calls) == 1


def test_posted_claims_never_follow_a_service_redirect(monkeypatch):
    calls = connections(monkeypatch, [Response(status=307, Location="https://other.org/collect")])
    body = b'{"query":"Private spoken claim"}'
    with pytest.raises(SourceUnavailable, match="redirect"):
        fetch_page("https://example.org/v1/search", deadline=time.monotonic() + 10, body=body)
    assert len(calls) == 1
    assert calls[0][1:3] == ("POST", "/v1/search")
    assert calls[0][3]["body"] == body


@pytest.mark.parametrize("response,limit,error", [
    (Response(b"x" * 11), 10, "size limit"),
    (Response(status=429), 100, "HTTP 429"),
    (Response(status=202), 100, "HTTP 202"),
    (Response(status=302, Location="https://example.org/again"), 100, "redirect"),
    (Response(Content_Encoding="gzip"), 100, "content encoding"),
])
def test_http_failures_are_explicit(monkeypatch, response, limit, error):
    connections(monkeypatch, [response])
    with pytest.raises(SourceUnavailable, match=error):
        fetch_page("https://example.org/", deadline=time.monotonic() + 10, redirects=0, limit=limit)


def test_html_preserves_literal_text_and_conflicting_dates_without_site_rules():
    parser = HTMLPage("https://example.org/report")
    parser.feed('''<head><title>Source &amp; facts</title>
    <meta property="article:published_time" content="2020-01-01">
    <meta property="article:published_time" content="2021-01-01"></head>
    <p>Quoted <strong>source</strong> &amp; qualifications.</p>
    <p hidden>Hidden facts</p><p style="display:none">Other hidden facts</p>
    <table><tr><td>Joined</td><td>123</td></tr></table><script>fake text</script>
    <a href="/methodology">Methodology</a>''')
    page = parser.page()
    assert page["metadata"]["title"] == "Source & facts"
    assert publication_date(page["metadata"], page["markdown"])[0] == ""
    assert page["markdown"] == "Quoted source & qualifications.\n\n[Methodology](https://example.org/methodology)"


def test_html_encoding_is_explicit_and_non_html_is_rejected():
    assert decode_html(PageResponse("", b"<p>old \x95 report</p>", headers(Content_Type="text/html"))) == "<p>old • report</p>"
    with pytest.raises(SourceUnavailable, match="HTML"):
        decode_html(PageResponse("", b"%PDF", headers(Content_Type="application/pdf")))


def test_numeric_annotations_and_revisions_cannot_concatenate_into_new_numbers():
    parser = HTMLPage("https://example.org/")
    parser.feed('<p>4.2<sup>1</sup>% and H<sub>2</sub>O; <del>50</del><ins>40</ins> units.</p>')
    assert parser.page()["markdown"] == "4.2^(1)% and H_(2)O; [deleted: 50][inserted: 40] units."


def test_malformed_html_depth_is_bounded():
    with pytest.raises(SourceUnavailable, match="nesting"):
        HTMLPage("https://example.org/").feed("<div>" * 129)


def test_late_dns_cannot_open_a_socket(monkeypatch):
    clock = [0.0]

    def resolve(*args, **kwargs):
        clock[0] = 2.0
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.215.14", 443))]

    monkeypatch.setattr("src.retrieval.http_pages.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("Late connection"))
    with pytest.raises(SourceUnavailable, match="time budget"):
        connect_public("example.org", 1.0)


def test_public_connection_pins_address_and_validates_original_tls_hostname(monkeypatch):
    calls = []
    address = ("93.184.215.14", 443)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", address)])
    raw = SimpleNamespace(settimeout=lambda _: None, connect=lambda value: calls.append(value), close=lambda: None)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: raw)

    def wrap(value, *, server_hostname):
        assert value is raw
        calls.append(server_hostname)
        return raw

    monkeypatch.setattr("src.retrieval.http_pages.ssl.create_default_context", lambda: SimpleNamespace(wrap_socket=wrap))
    assert connect_public("example.org", 1).sock is raw
    assert calls == [address, "example.org"]


def test_stalled_readers_have_bounded_wait_and_concurrency(monkeypatch):
    release = threading.Event()
    finished = [threading.Event(), threading.Event()]
    slots = threading.BoundedSemaphore(2)
    monkeypatch.setattr("src.retrieval.http_pages._READERS", slots)
    started = []

    def stall(url, **kwargs):
        index = len(started)
        started.append(url)
        try:
            assert release.wait(2)
            return PageResponse(url, b"late result", Message())
        finally:
            finished[index].set()

    monkeypatch.setattr("src.retrieval.http_pages._fetch_page", stall)
    try:
        for _ in range(2):
            with pytest.raises(SourceUnavailable, match="time budget"):
                fetch_page("https://example.org/", deadline=time.monotonic() + 0.05)
        with pytest.raises(SourceUnavailable, match="still finishing"):
            fetch_page("https://example.org/", deadline=time.monotonic() + 1)
        assert len(started) == 2
    finally:
        release.set()
        assert all(event.wait(2) for event in finished)
        assert slots.acquire(timeout=2) and slots.acquire(timeout=2)
        slots.release()
        slots.release()
