"""News feeds discover pages; their text and timestamps cannot stand in for evidence."""

from __future__ import annotations

import json
from email.message import Message
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape

import pytest

from src.pipeline.context import SpeechMetadata
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.web_research import WebResearch, source_excerpts
from src.retrieval.google_news_sources import GoogleNewsSources
from src.retrieval.html_page import HTMLPage
from src.retrieval.http_pages import PageResponse
from src.retrieval.news_sources import news_results
from src.retrieval.source_dates import publication_date
from src.retrieval.web_sources import SourceUnavailable


def rss(url="https://example.org/report", date="Tue, 08 Sep 2026 05:00:00 GMT"):
    return (f'<rss><channel><item><title>Feed headline</title><link>{escape(url)}</link>'
            f'<pubDate>{date}</pubDate><description>SNIPPET IS NOT EVIDENCE</description>'
            '</item></channel></rss>').encode()


@pytest.mark.parametrize("claim", ["Solar energy storage increased.", "The satellite launched yesterday.",
                                   "The central bank raised interest rates.", "We don't have a labor shortage."])
def test_news_search_preserves_claims_across_topics_and_discards_feed_descriptions(monkeypatch, claim):
    calls = []

    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return PageResponse(url, rss(), Message())

    monkeypatch.setattr("src.retrieval.google_news_sources.fetch_page", fetch)
    sources = GoogleNewsSources()
    sources.begin_check()
    results = sources.search(claim)
    url, options = calls[0]
    assert urlsplit(url).hostname == "news.google.com"
    assert parse_qs(urlsplit(url).query) == {"q": [claim], "hl": ["en-US"], "gl": ["US"], "ceid": ["US:en"]}
    assert options["redirects"] == 0 and options["deadline"] == sources.deadline
    assert results[0]["feed_published_at"] == "2026-09-08T05:00:00+00:00"
    assert "SNIPPET" not in str(results) and "published_at" not in results[0]


@pytest.mark.parametrize("url", ["https://127.0.0.1/private", "javascript:alert(1)", "http://example.org/report"])
def test_rss_discards_nonpublic_or_non_https_links(url):
    assert news_results(rss(url=url)) == []


@pytest.mark.parametrize("body", [b'<html>Challenge</html>', b'<rss/>', b'<rss>',
                                  b'<!DOCTYPE rss [<!ENTITY x "fake">]><rss><channel>&x;</channel></rss>',
                                  '<rss><channel/></rss>'.encode('utf-16')])
def test_invalid_feeds_are_operational_errors(body):
    with pytest.raises(SourceUnavailable):
        news_results(body)
    assert news_results(b'<rss><channel/></rss>') == []


@pytest.mark.parametrize("date", ["yesterday", "Tue, 08 Sep 2026 05:00:00", ""])
def test_ambiguous_feed_dates_stay_unknown(date):
    assert news_results(rss(date=date))[0]["feed_published_at"] == ""


def test_feed_date_cannot_supply_or_override_article_publication(monkeypatch):
    sources = GoogleNewsSources()
    monkeypatch.setattr(sources, "search_before", lambda *args: news_results(rss()))
    text = "Solar energy storage increased during the period described in this report, according to its stated measurements."
    monkeypatch.setattr(sources, "scrape", lambda _: {"markdown": text, "metadata": {"datePublished": "2020-01-01"}})
    context = ClaimContext(TranscriptUpdate("1", 0, "Solar energy storage increased.", 0, 1, True), ())
    result = WebResearch(sources).review(context, SpeechMetadata(spoken_at="2020-02-01"))
    source = result["sources"][0]
    assert source["published_at"] == "2020-01-01" and source["feed_published_at"].startswith("2026")
    assert source["temporal_status"] == "published_by_cutoff"
    monkeypatch.setattr(sources, "scrape", lambda _: {"markdown": text, "metadata": {}})
    source = WebResearch(sources).review(context, SpeechMetadata(spoken_at="2020-02-01"))["sources"][0]
    assert source["published_at"] == "" and source["temporal_status"] == "date_unconfirmed"


def parse(html):
    parser = HTMLPage("https://example.org/report")
    parser.feed(html)
    parser.close()
    return parser.page()


def test_article_body_excludes_recommendations_and_headlines_from_passage_selection():
    paragraph = "Solar energy storage increased during the period described in this report, according to its stated measurements."
    page = parse('<h2>Solar energy storage unrelated recommendation</h2><article><h1>Headline</h1>'
                 f'<div itemprop="articleBody"><p>{paragraph}</p></div>'
                 '<aside>Solar energy storage recommendations.</aside></article><p>Other content.</p>')
    assert page["markdown"] == paragraph
    assert source_excerpts(page["markdown"], "solar energy storage") == [paragraph]
    page = parse('<h3>' + paragraph + '</h3>')
    assert source_excerpts(page["markdown"], "solar energy storage") == []


@pytest.mark.parametrize("node,expected", [
    ({"@type": "NewsArticle", "datePublished": "2026-09-08", "url": "https://example.org/report"}, "2026-09-08"),
    ({"@type": "NewsArticle", "datePublished": "2026-09-08", "mainEntityOfPage": {"@id": "https://example.org/report"}}, "2026-09-08"),
    ({"@type": "NewsArticle", "datePublished": "2026-09-08", "@id": "https://example.org/other"}, ""),
    ({"@type": "NewsArticle", "dateModified": "2026-09-08"}, ""),
    ({"@type": "Organization", "datePublished": "2026-09-08"}, ""),
    ({"@type": "NewsArticle", "datePublished": "2026-09-08", "url": "https://example.org/other"}, ""),
    ({"@graph": [{"@type": "NewsArticle", "datePublished": "2026-09-08"}]}, ""),
    ({"@type": "NewsArticle", "datePublished": "2025-01-01"}, ""),
    ({"@type": "NewsArticle", "datePublished": "2026-09-08", "url": "https://[broken"}, ""),
])
def test_only_the_current_articles_structured_publication_date_is_used(node, expected):
    page = parse('<script type="application/ld+json">' + json.dumps(node) + '</script><p>Visible text.</p>')
    assert page["markdown"] == "Visible text."
    assert publication_date(page["metadata"], page["markdown"])[0] == expected


def test_conflicting_article_dates_and_ambiguous_unlinked_articles_stay_unknown():
    nodes = [{"@type": "NewsArticle", "datePublished": day} for day in ("2020-01-01", "2026-09-08")]
    page = parse('<script type="application/ld+json">' + json.dumps(nodes) + '</script>')
    assert publication_date(page["metadata"], page["markdown"])[0] == ""
    node = nodes[1] | {"url": "https://example.org/report"}
    page = parse('<meta property="article:published_time" content="2020-01-01">'
                 '<script type="application/ld+json">' + json.dumps(node) + '</script>')
    assert publication_date(page["metadata"], page["markdown"])[0] == ""


@pytest.mark.parametrize("second,expected", [
    ("2026-04-29T15:24:21-05:00", "2026-04-29"),
    ("2026-04-29T20:24:21Z", "2026-04-29"),
    ("2026-04-29 20:24:21+00:00", "2026-04-29"),
    ("2026-04-29T15:24:21+00:00", ""),
    ("2026-04-29T20:24:21", ""),
    ("2026-04-30T01:24:21+05:00", ""),
    ("not a date", ""),
])
def test_equivalent_zoned_publication_notices_agree_without_hiding_real_conflicts(second, expected):
    dates = ["2026-04-29T20:24:21+00:00", second]
    nodes = [{"@type": "NewsArticle", "url": "https://example.org/report", "datePublished": raw} for raw in dates]
    page = parse('<script type="application/ld+json">' + json.dumps(nodes) + '</script>')
    assert publication_date(page["metadata"], page["markdown"])[0] == expected


@pytest.mark.parametrize("related_tag", ['article', 'div itemprop="articleBody"'])
def test_main_report_takes_precedence_over_an_unrelated_article_region(related_tag):
    report = "Solar energy storage increased by 4 percent during the period, according to the measurements in this report."
    related = "Solar energy storage decreased by 8 percent during the period, according to a separate, unrelated report."
    closing_tag = related_tag.split()[0]
    page = parse(f'<main><p>{report}</p></main><section><{related_tag}><p>{related}</p></{closing_tag}></section>')
    assert page["markdown"] == report
    assert source_excerpts(page["markdown"], "solar energy storage") == [report]


def test_nested_article_card_cannot_discard_the_main_report():
    report = "Solar energy storage increased by 4 percent during the period, according to the measurements in this report."
    related = "A separate report describes a different period with an 8 percent decrease."
    page = parse(f'<main><p>{report}</p><section><article><p>{related}</p></article></section></main>')
    assert report in page["markdown"]
    assert source_excerpts(page["markdown"], "solar energy storage") == [report]


def test_photo_caption_of_an_older_event_is_not_selected_as_the_article_passage():
    report = "SpaceX scheduled two satellite launches for this week, with both missions awaiting final weather approval."
    caption = "A SpaceX Falcon 9 rocket launched satellites on March 8, months before this week's report was published."
    page = parse(f'<main><p>{report}</p><figure><img src="launch.jpg"><figcaption>{caption}</figcaption></figure></main>')
    assert page["markdown"] == report
    assert source_excerpts(page["markdown"], "SpaceX rocket launched satellites this week") == [report]
