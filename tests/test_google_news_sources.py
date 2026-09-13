"""Historical discovery must resolve real pages without promoting feed metadata."""

from __future__ import annotations

import json
from email.message import Message
from urllib.parse import parse_qs, urlsplit

import pytest

from src.pipeline.context import SpeechMetadata
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.web_research import WebResearch
from src.retrieval.google_news_sources import GoogleNewsSources, article_id, publisher_link
from src.retrieval.http_pages import PageResponse
from src.retrieval.page_sources import PageSources
from src.retrieval.web_sources import SourceUnavailable

LINK = "https://news.google.com/rss/articles/ABC_123?oc=5"
PUBLISHER = "https://example.org/report"


def response(url, text):
    headers = Message()
    headers["Content-Type"] = "text/html; charset=utf-8"
    return PageResponse(url, text.encode(), headers)


def rpc(url=PUBLISHER):
    return ")]}'\n\n" + json.dumps([["wrb.fr", "Fbv4je", json.dumps(["garturlres", url, 1])], ["di", 12]])


@pytest.mark.parametrize("cutoff,expected", [("", ""), ("2025-07-20", " before:2025-07-21"),
                                           ("2024-02-28", " before:2024-02-29"),
                                           ("2025-12-31", " before:2026-01-01")])
def test_search_boundary_is_inclusive_and_feed_metadata_is_separate(monkeypatch, cutoff, expected):
    calls = []

    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return response(url, f'<rss><channel><item><link>{LINK}</link><title>Headline</title>'
                        '<pubDate>Tue, 08 Sep 2026 05:00:00 GMT</pubDate>'
                        '<description>NEVER EVIDENCE</description></item></channel></rss>')

    monkeypatch.setattr("src.retrieval.google_news_sources.fetch_page", fetch)
    sources = GoogleNewsSources()
    sources.begin_check()
    hits = sources.search_before("The bridge opened in 1932.", cutoff)
    assert parse_qs(urlsplit(calls[0][0]).query)["q"] == ["The bridge opened in 1932." + expected]
    assert calls[0][1]["deadline"] == sources.deadline and calls[0][1]["redirects"] == 0
    assert hits[0]["feed_date_basis"].startswith("Google News RSS pubDate:")
    assert hits[0]["feed_published_at"] == "2026-09-08T05:00:00+00:00"
    assert "published_at" not in hits[0] and "NEVER EVIDENCE" not in str(hits)


@pytest.mark.parametrize("query,cutoff", [("", ""), ("x" * 601, ""), ("bridge", "2025-02-30"),
                                        ("bridge", "9999-12-31")])
def test_invalid_search_inputs_never_make_a_request(monkeypatch, query, cutoff):
    monkeypatch.setattr("src.retrieval.google_news_sources.fetch_page", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(SourceUnavailable):
        GoogleNewsSources().search_before(query, cutoff)


@pytest.mark.parametrize("url", ["https://news.google.com.evil.org/rss/articles/ABC", "http://news.google.com/rss/articles/ABC",
                                 "https://news.google.com/rss/articles/a/b", "https://news.google.com/rss/articles/%2f",
                                 "https://127.0.0.1/rss/articles/ABC", "https://[broken"])
def test_resolver_only_accepts_exact_public_article_links(url):
    with pytest.raises(SourceUnavailable):
        article_id(url)


@pytest.mark.parametrize("body", [b'{}', b'<html>Challenge</html>', b")]}'\n{}", b")]}'\n[]",
                                  b")]}'\n[[\"wrong\",\"Fbv4je\",\"[]\"]]", b")]}'\n[[\"wrb.fr\",\"Fbv4je\",null]]"])
def test_malformed_resolver_responses_fail_closed(body):
    with pytest.raises(SourceUnavailable):
        publisher_link(body)


@pytest.mark.parametrize("url", ["http://example.org/report", "https://127.0.0.1/private", "https://news.google.com/article",
                                 "https://user:password@example.org/report", None, {"url": PUBLISHER}])
def test_resolved_destinations_are_validated(url):
    with pytest.raises(SourceUnavailable):
        publisher_link(rpc(url).encode())


def test_duplicate_rpc_results_are_not_chosen_arbitrarily():
    row = ["wrb.fr", "Fbv4je", json.dumps(["garturlres", PUBLISHER, 1])]
    with pytest.raises(SourceUnavailable):
        publisher_link((")]}'\n" + json.dumps([row, row])).encode())


def test_resolution_fetches_publisher_and_caches_under_one_deadline(monkeypatch):
    calls, read = [], []

    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        if "batchexecute" in url:
            encoded = parse_qs(kwargs["body"].decode())["f.req"][0]
            request = json.loads(json.loads(encoded)[0][0][1])
            assert request[2:] == ["ABC_123", 123456, "signature"]
            return response(url, rpc())
        return response(url, '<div data-n-a-id="ABC_123" data-n-a-ts="123456" data-n-a-sg="signature"></div>')

    def scrape(self, url):
        read.append(url)
        return {"metadata": {"sourceURL": url, "datePublished": "2020-01-02"}, "markdown": "Original article."}

    monkeypatch.setattr("src.retrieval.google_news_sources.fetch_page", fetch)
    monkeypatch.setattr(PageSources, "scrape", scrape)
    sources = GoogleNewsSources()
    sources.begin_check()
    first = sources.scrape(LINK)
    assert sources.scrape(LINK) == first
    assert read == [PUBLISHER] and len(calls) == 2
    assert all(options["deadline"] == sources.deadline and options["redirects"] == 0 for _, options in calls)
    sources.begin_check()
    sources.scrape(LINK)
    assert read == [PUBLISHER, PUBLISHER]


@pytest.mark.parametrize("html", ['<div data-n-a-id="OTHER" data-n-a-ts="123" data-n-a-sg="sig"></div>',
                                  '<div data-n-a-id="ABC_123" data-n-a-ts="bad" data-n-a-sg="sig"></div>',
                                  '<div data-n-a-id="ABC_123" data-n-a-ts="123" data-n-a-sg="sig"></div>' * 2,
                                  '<html>Consent or challenge</html>'])
def test_unbound_or_ambiguous_parameters_do_not_start_rpc_or_publisher_reads(monkeypatch, html):
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        return response(url, html)

    monkeypatch.setattr("src.retrieval.google_news_sources.fetch_page", fetch)
    with pytest.raises(SourceUnavailable):
        GoogleNewsSources().scrape(LINK)
    assert len(calls) == 1 and "batchexecute" not in calls[0]


def test_search_filter_does_not_override_the_pages_later_publication(monkeypatch):
    sources = GoogleNewsSources()
    calls = []

    def search(query, cutoff):
        calls.append(cutoff)
        return [{"url": LINK, "feed_published_at": "2020-01-01T00:00:00Z"}]

    monkeypatch.setattr(sources, "search_before", search)
    monkeypatch.setattr(sources, "scrape", lambda _: {
        "metadata": {"sourceURL": PUBLISHER, "datePublished": "2026-01-01"},
        "markdown": "Solar energy storage increased during the period described by this report, according to the measurements."})
    context = ClaimContext(TranscriptUpdate("claim", 0, "Solar energy storage increased.", 0, 1, True), ())
    result = WebResearch(sources).review(context, SpeechMetadata(source_published_at="2025-07-20"))
    assert calls and set(calls) == {"2025-07-20"}
    assert result["assertions"][0]["source_ids"] == []
    assert result["sources"][0]["published_at"] == "2026-01-01"
    assert result["sources"][0]["temporal_status"] == "later_publication"
