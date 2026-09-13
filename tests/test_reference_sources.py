"""Literal references authorize bounded reads, never additional evidence claims."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.retrieval.reference_sources import ReferenceSources, reference_candidates, reference_url
from src.retrieval.web_sources import SourceUnavailable


def parent(text: str, **fields: object) -> dict:
    return {"id": "source-1", "url": "https://publisher.example/article", "excerpts": [text],
            "temporal_status": "published_by_cutoff", **fields}


def page(url: str, text: str = "The survey covers 24.3% of the sampled households.") -> dict:
    return {"markdown": text, "metadata": {"sourceURL": url, "title": "Survey", "publishedTime": "2024-03-08"}}


def candidates(count: int = 3) -> list[dict]:
    return reference_candidates([parent(" ".join(f"[Report {i}](https://source{i}.example/report)" for i in range(count)))], set())


def test_parent_paragraph_labels_destinations_numbers_and_spans_are_literal():
    paragraph = '  The [24.3% survey](https://DATA.example:443/study#results) covered 5,700 households in 2018–2021.  '
    original = parent(paragraph, reading_passages=[paragraph, "A [definition](https://data.example/meaning) follows."])
    before = deepcopy(original)
    found = reference_candidates([original], set())
    assert original == before
    assert [row["url"] for row in found] == ["https://data.example/study", "https://data.example/meaning"]
    lineage = found[0]["lineage"][0]
    start, end = paragraph.index("["), paragraph.index(")") + 1
    assert lineage == {"parent_source_id": "source-1", "parent_url": original["url"],
        "parent_passage_id": "source-1:p1", "paragraph": paragraph, "label": "24.3% survey",
        "destination": "https://DATA.example:443/study#results", "start": start, "end": end,
        "markup": paragraph[start:end]}
    assert found[1]["lineage"][0]["parent_passage_id"] == "source-1:p2"


def test_repeated_links_accumulate_lineage_in_source_passage_order_without_ranking():
    first = "See [one](https://data.example/report#a) and [two](https://data.example/report#b)."
    second = "The [original](https://data.example/report) is referenced again."
    sources = [parent(first), parent(second, id="source-2", url="https://second.example/article")]
    found = reference_candidates(sources, set())
    assert len(found) == 1
    assert [(row["parent_source_id"], row["label"]) for row in found[0]["lineage"]] == [
        ("source-1", "one"), ("source-1", "two"), ("source-2", "original")]


@pytest.mark.parametrize("authority", ["PUBLISHER.example:443", "PUBLISHER.example:0443", "PUBLISHER.example."])
def test_parent_identity_stays_literal_while_transport_aliases_share_one_comparison(authority):
    original_url = f"https://{authority}/article#section"
    text = "[self](https://publisher.example/article#other) [report](https://DATA.example:443/report#table)"
    found = reference_candidates([parent(text, url=original_url)], set())
    assert [candidate["url"] for candidate in found] == ["https://data.example/report"]
    assert found[0]["lineage"][0]["parent_url"] == original_url
    assert found[0]["lineage"][0]["destination"] == "https://DATA.example:443/report#table"
    assert reference_url(original_url) == "https://publisher.example/article"
    final_url = f"https://{authority.replace('PUBLISHER', 'FINAL')}/report#section"
    assert reference_url(final_url) == reference_url("https://final.example/report")


def test_twenty_candidate_bound_still_preserves_later_lineage_for_retained_urls():
    sources = [parent(" ".join(f"[Item {i}](https://source.example/{i})" for i in range(22))),
               parent("[Repeated](https://source.example/0#later)", id="source-2")]
    found = reference_candidates(sources, set())
    assert [row["url"] for row in found] == [f"https://source.example/{i}" for i in range(20)]
    assert len(found[0]["lineage"]) == 2


@pytest.mark.parametrize("destination", ["http://data.example/report", "https://127.0.0.1/report",
    "https://user:password@data.example/report", "https://data.example:8443/report", "https://service.local/report",
    "https://data.example/a b", "https://data.example/line\nbreak", "https://data.example/a\\b",
    "https://data.example../report", "../report"])
def test_nonpublic_or_ambiguous_destinations_never_authorize_transport(destination):
    assert reference_candidates([parent(f"Read [report]({destination}).")], set()) == []


def test_only_eligible_parent_passages_authorize_links():
    accepted = parent("[self](https://publisher.example/article#part) [excluded](https://news.blocked.example/a) "
                      "[existing](https://other.example/page#part) [new](https://source.example/a)",
                      markdown="[not accepted](https://source.example/fullpage)",
                      reading_context=["[heading](https://source.example/heading)"])
    sources = [accepted, parent("[later](https://source.example/later)", temporal_status="later_publication",
                               url="https://other.example/page"),
               parent("[private parent](https://source.example/private)", url="https://127.0.0.1/"),
               parent("[missing identity](https://source.example/missing)", id="")]
    assert [row["url"] for row in reference_candidates(sources, {"blocked.example"})] == ["https://source.example/a"]


def test_images_escaped_nested_or_broken_links_do_not_authorize_transport():
    for text in ("![image](https://source.example/a)", r"\[label](https://source.example/a)",
                 "[outer [inner]](https://source.example/a)", "[broken](https://source.example/a"):
        assert reference_candidates([parent(text)], set()) == []


def test_search_does_not_rewrite_queries_set_dates_or_reset_reader_deadline():
    class Reader:
        deadline = 17.5
        calls: list[str] = []

        def read(self, url: str) -> dict:
            self.calls.append(url)
            return page(url)

    reader = Reader()
    provider = ReferenceSources(candidates(), reader.read)
    provider.begin_check()
    assert reader.deadline == 17.5 and not reader.calls
    first = provider.search("Preserve 5,700 households and the whole contrast")
    assert first == provider.search_before("Unrelated subject", "1900-01-01")
    assert first == candidates()[:2]
    assert not hasattr(provider, "read_link") and not hasattr(provider, "deadline")
    first[0]["lineage"].clear()
    assert provider.lineage[candidates()[0]["url"]]


def test_two_reader_calls_include_cached_failures_and_survive_begin_check():
    called = []
    urls = [row["url"] for row in candidates()]

    def reader(url: str) -> dict:
        called.append(url)
        if url == urls[0]:
            raise SourceUnavailable("Web request failed (HTTP 403).")
        return page(url)

    provider = ReferenceSources(candidates(), reader)
    for _ in range(2):
        with pytest.raises(SourceUnavailable, match="HTTP 403"):
            provider.scrape(urls[0])
    assert provider.search("query") == candidates()[:2]
    assert provider.scrape(urls[1]) == page(urls[1])
    provider.begin_check()
    assert provider.scrape(urls[1]) == page(urls[1])
    with pytest.raises(SourceUnavailable, match="budget"):
        provider.scrape(urls[2])
    assert called == urls[:2]
    assert list(provider.attempts) == urls[:2]
    assert provider.attempts[urls[0]] == {"status": "failed", "reason": "Web request failed (HTTP 403)."}
    assert provider.attempts[urls[1]] == {"status": "read", "final_url": urls[1]}
    assert provider.search_before("another query", "2020-02-01") == candidates()[:2]


def test_redirect_aliases_keep_lineage_and_original_page_metadata():
    original = candidates(2)
    final = "https://FINAL.example:443/report#table"
    supplied = page(final)
    called = []

    def reader(url: str) -> dict:
        called.append(url)
        return supplied

    provider = ReferenceSources(original, reader)
    original[0]["lineage"].clear()
    for candidate in provider.search("query"):
        received = provider.scrape(candidate["url"])
        assert received == supplied
        received["metadata"]["sourceURL"] = "https://changed.example/"
    assert list(provider.resolved_urls.values()) == ["https://final.example/report"] * 2
    assert reference_url(supplied["metadata"]["sourceURL"]) in provider.resolved_urls.values()
    assert supplied["metadata"]["sourceURL"] == final
    assert provider.lineage[called[0]]
    assert provider.scrape(called[0]) == supplied
    aliases = provider.resolved_urls
    aliases.clear()
    assert len(provider.resolved_urls) == 2


@pytest.mark.parametrize("bad_page", [None, {"markdown": "text"}, page("https://127.0.0.1/"),
    {"markdown": 17, "metadata": {}}, {"markdown": "text", "metadata": {"statusCode": 403}}])
def test_malformed_or_private_redirect_pages_fail_once(bad_page):
    called = []

    def reader(url: str) -> dict:
        called.append(url)
        return bad_page

    provider = ReferenceSources(candidates(), reader)
    url = candidates()[0]["url"]
    for _ in range(2):
        with pytest.raises(SourceUnavailable):
            provider.scrape(url)
    assert called == [url] and not provider.resolved_urls
    assert provider.attempts[url]["status"] == "failed"


def test_unknown_requests_and_child_links_cannot_authorize_more_reads():
    called = []

    def reader(url: str) -> dict:
        called.append(url)
        return page(url, "[child](https://child.example/report)")

    provider = ReferenceSources(candidates(), reader)
    with pytest.raises(SourceUnavailable, match="parent reference"):
        provider.scrape("https://private.example/not-selected")
    provider.scrape(candidates()[0]["url"])
    with pytest.raises(SourceUnavailable, match="parent reference"):
        provider.scrape("https://child.example/report")
    assert called == [candidates()[0]["url"]]


def test_unexpected_reader_failure_propagates_its_type_and_is_cached_with_type_only_detail():
    called = []

    def reader(url: str) -> dict:
        called.append(url)
        raise RuntimeError("Private transport detail")

    provider = ReferenceSources(candidates(), reader)
    url = candidates()[0]["url"]
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Private transport detail") as caught:
            provider.scrape(url)
        assert type(caught.value) is RuntimeError
    assert called == [url]
    assert provider.attempts[url] == {"status": "failed", "error_type": "RuntimeError"}
    assert "Private transport detail" not in str(provider.attempts)


def test_reentrant_read_cannot_repeat_the_same_uncached_request():
    called = []

    def reader(url: str) -> dict:
        called.append(url)
        return provider.scrape(url)

    provider = ReferenceSources(candidates(), reader)
    url = candidates()[0]["url"]
    for _ in range(2):
        with pytest.raises(SourceUnavailable, match="already in progress"):
            provider.scrape(url)
    assert called == [url] and provider.attempts[url]["status"] == "failed"


@pytest.mark.parametrize("invalid", [[{"url": f"https://source.example/{i}", "lineage": []} for i in range(21)],
    [candidates()[0]] * 2,
    [{"url": "http://unsafe.example/", "lineage": []}],
    [{"url": "https://data.example/a#fragment", "lineage": []}]])
def test_adapter_rejects_invalid_candidate_sets(invalid):
    with pytest.raises(ValueError, match="unique public transport URLs"):
        ReferenceSources(invalid, lambda url: page(url))
