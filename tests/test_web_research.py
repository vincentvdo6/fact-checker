"""Web discovery must preserve provenance without manufacturing a supported verdict."""

from __future__ import annotations

import pytest

from src.pipeline.context import SpeechMetadata
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.web_research import WebResearch, publication_note, research_plan, source_excerpts
from src.retrieval.search_contract import valid_query
from src.retrieval.web_sources import SourceUnavailable, public_url


def context(text="We don't have a labor shortage. We have a good job shortage.", preceding=()):
    return ClaimContext(TranscriptUpdate("claim", 0, text, 30, 35, True), preceding)


def test_research_keeps_assertions_and_unknown_scope_without_adopting_future_context():
    prior = TranscriptUpdate("prior", 0, "They call this functional unemployment.", 20, 29, True)
    future = TranscriptUpdate("future", 0, "This means United States in 2025.", 36, 40, True)
    plan = research_plan(context(preceding=(prior, future)), SpeechMetadata())
    assert [part["text"] for part in plan["assertions"]] == [
        "We don't have a labor shortage.", "We have a good job shortage."]
    assert all(part["query"] == context().claim.text for part in plan["assertions"])
    assert "United States" not in str(plan) and "2025" not in str(plan)
    assert len(plan["gaps"]) == 2


def test_unrelated_claims_use_their_own_terms_and_explicit_scope():
    plan = research_plan(context("Solar panels produce electricity."),
                         SpeechMetadata(country="Germany", spoken_at="2020-01-02"))
    assert len(plan["assertions"]) == 1
    assert plan["assertions"][0]["query"] == "Solar panels produce electricity. Germany 2020"
    assert plan["gaps"] == []


def test_context_is_bounded_and_unfinalized_speech_is_excluded():
    prior = TranscriptUpdate("prior", 0, "old " * 300 + "new", 0, 29, True)
    interim = TranscriptUpdate("interim", 0, "secret interim", 0, 29, False)
    plan = research_plan(context(preceding=(prior, interim)), SpeechMetadata())
    assert len(plan["context"].split()) == 192
    assert plan["context"].endswith("new") and "interim" not in plan["context"]


@pytest.mark.parametrize("published,spoken,note", [
    ("2025-09-03T10:00:00Z", "2025-07-20", "Published after"),
    ("2025-06-01", "2025-07-20", "period covered still needs checking"),
    ("yesterday", "2025-07-20", "Publication date unknown"),
    ("2025-06-01", "", "Claim date unknown"),
])
def test_publication_is_not_observation_or_claim_time(published, spoken, note):
    assert note in publication_note(published, spoken)


def test_excerpts_are_intact_page_blocks_and_reject_irrelevant_text():
    good = "The job openings survey measures demand for labor, while unemployed workers are measured by a different survey."
    assert source_excerpts("A song about dancing.\n\n" + good, "labor job openings unemployed statistics") == [good]
    assert source_excerpts("We Don't Care is a song. " * 5, "labor shortage") == []
    assert source_excerpts("## " + good, "labor job openings") == []
    assert source_excerpts("Created with Highcharts: " + good, "labor job openings") == []
    links = "[labor](https://example.org/labor) [openings](https://example.org/job-openings) [unemployed](https://example.org/unemployed)"
    assert source_excerpts(links, "labor job openings unemployed") == []


def test_wrapped_unemployment_figures_beat_a_related_population_paragraph():
    other = "People outside the labor force who want jobs are not counted as unemployed if they are not actively seeking work."
    chosen = "Both the unemployment\r\nrate and the number of unemployed people changed little, according to this monthly household survey."
    assert source_excerpts(other + "\n\n" + chosen, "unemployment rate number of unemployed") == [chosen]


def test_page_budget_never_returns_a_cut_off_qualification():
    block = "Job openings and unemployed workers can indicate labor shortages. " * 5
    block += "However, this does not establish a shortage across all occupations."
    page = "x" * 159_798 + "\n\n" + block
    assert source_excerpts(page, "labor job openings unemployed") == []


class Sources:
    def __init__(self):
        self.scraped = []

    def search(self, query):
        return [{"url": "https://example.org/report", "title": "Search title", "description": "SNIPPET ONLY"}]

    def scrape(self, url):
        self.scraped.append(url)
        return {"markdown": "A labor shortage concerns worker availability. A good job shortage concerns job quality, including wages and hours of employment.",
                "metadata": {"title": "Page title", "publishedTime": "2025-09-03", "statusCode": 200}}


def test_wrapped_links_to_one_publisher_do_not_count_as_independent_sources():
    class Wrapped(Sources):
        def search(self, query):
            return [{"url": f"https://search.example.org/{i}"} for i in range(3)]

        def scrape(self, url):
            page = super().scrape(url)
            page["metadata"]["sourceURL"] = "https://example.org/other" if url.endswith("/2") else "https://example.org/report"
            return page

    result = WebResearch(Wrapped()).review(context(), SpeechMetadata(spoken_at="2025-10-20"))
    assert [source["url"] for source in result["sources"]] == ["https://example.org/report", "https://example.org/other"]
    assert all(part["source_ids"] == ["source-1", "source-2"] for part in result["assertions"])


@pytest.mark.parametrize("host", ["www.youtube.com", "reddit.com", "www.x.com", "m.youtube.com", "news.reddit.com",
                                 "youtube.com.", "news.google.com", "news.google.com."])
def test_wrapped_links_do_not_bypass_publisher_exclusions(host):
    class Wrapped(Sources):
        def scrape(self, url):
            page = super().scrape(url)
            page["metadata"]["sourceURL"] = f"https://{host}/post"
            return page

    result = WebResearch(Wrapped()).review(context(), SpeechMetadata(spoken_at="2025-10-20"))
    assert result["sources"] == [] and all(part["source_ids"] == [] for part in result["assertions"])


def test_shared_sources_are_fetched_once_and_mapped_to_both_unresolved_assertions():
    sources = Sources()
    result = WebResearch(sources).review(context(), SpeechMetadata(spoken_at="2025-10-20"))
    assert len(sources.scraped) == 1 and len(result["sources"]) == 1
    assert all(part["source_ids"] == ["source-1"] and part["status"] == "unresolved" for part in result["assertions"])
    source = result["sources"][0]
    assert source["title"] == "Page title" and source["temporal_status"] == "published_by_cutoff"
    assert "SNIPPET ONLY" not in str(result) and "markdown" not in source
    assert not {"verdict", "confidence", "probabilities"}.intersection(result)


def test_video_publication_is_a_search_boundary_not_the_speech_date():
    metadata = SpeechMetadata(source_published_at="2025-07-20")
    result = WebResearch(Sources()).review(context(), metadata)
    assert result["search_cutoff"] == "2025-07-20"
    assert "speech may be older" in result["cutoff_basis"]
    assert any("date the claim describes has not been established" in gap for gap in result["gaps"])
    assert all(not part["source_ids"] for part in result["assertions"])
    assert result["sources"][0]["temporal_status"] == "later_publication"
    assert metadata.spoken_at == ""


@pytest.mark.parametrize("reason", ["The web source timed out.", "Web request failed (HTTP 403)."])
def test_scrape_failure_is_operational_and_snippet_cannot_become_evidence(reason):
    sources = Sources()

    def fail(url):
        raise SourceUnavailable(reason)

    sources.scrape = fail
    result = WebResearch(sources).review(context(), SpeechMetadata())
    assert result["status"] == "unavailable" and result["sources"] == []
    assert result["errors"] == [reason]
    assert result["page_failures"] == [{"url": "https://example.org/report", "reason": reason}]
    assert "SNIPPET ONLY" not in str(result)


@pytest.mark.parametrize("page", [
    {"metadata": {"error": "<html>Private error details</html>"}, "markdown": "Unusable page"},
    {"metadata": {"statusCode": 403}, "markdown": "Unusable page"},
    {"metadata": {}, "markdown": None},
])
def test_extraction_failure_records_attempted_url_without_response_content(page):
    sources = Sources()
    sources.scrape = lambda _: page
    result = WebResearch(sources).review(context(), SpeechMetadata())
    reason = "A discovered page could not be extracted."
    assert result["page_failures"] == [{"url": "https://example.org/report", "reason": reason}]
    assert result["errors"] == [reason] and result["status"] == "unavailable"
    assert result["sources"] == [] and "Private error details" not in str(result)


def test_structured_failure_reason_is_bounded_without_changing_existing_errors():
    sources = Sources()
    reason = "Source unavailable. " * 30

    def fail(url):
        raise SourceUnavailable(reason)

    sources.scrape = fail
    result = WebResearch(sources).review(context(), SpeechMetadata())
    assert result["page_failures"] == [{"url": "https://example.org/report", "reason": reason[:300]}]
    assert result["errors"] == [reason] and result["status"] == "unavailable"


def test_empty_search_is_distinct_from_provider_failure():
    sources = Sources()
    sources.search = lambda _: []
    result = WebResearch(sources).review(context(), SpeechMetadata())
    assert result["status"] == "complete" and result["errors"] == [] and result["sources"] == []
    assert result["page_failures"] == []


@pytest.mark.parametrize("url", ["javascript:alert(1)", "http://example.org", "https://127.0.0.1/", "https://10.0.0.1/",
                                  "https://localhost/", "https://x.local/", "https://user:pass@example.org/",
                                  "https://example.org:8000/", "https://[::1]/", None])
def test_source_links_reject_unsafe_urls(url):
    assert public_url(url) == ""


@pytest.mark.parametrize("claim,passage", [
    ("Solar panels produce electricity.", "Solar panels produce electricity when light reaches their cells. Output depends on illumination and the equipment in use."),
    ("The bridge opened in 1932.", "The bridge opened in 1932 according to the municipal archive, which records the construction period and opening ceremony."),
    ("Rent increased in the city.", "Rent increased in the city during the period covered by this housing survey, with different changes across neighborhoods."),
    ("Water boils at a lower temperature at high altitude.", "Water boils at a lower temperature at high altitude because atmospheric pressure is lower, under the stated conditions."),
    ("We don't have a labor shortage.", "The labor shortage described in this report affected several occupations, with its location and observation period stated explicitly."),
])
def test_all_topics_use_the_same_discovery_and_passage_pipeline(claim, passage):
    calls = []

    class GeneralSources:
        def search(self, query):
            calls.append(query)
            return [{"url": "https://example.org/report", "description": "Search snippets are not evidence."}]

        def scrape(self, url):
            return {"markdown": "Unrelated music album details. " * 4 + "\n\n" + passage,
                    "metadata": {"datePublished": "2020-01-01"}}

    result = WebResearch(GeneralSources()).review(context(claim), SpeechMetadata(spoken_at="2020-02-01"))
    assert calls == [claim + " 2020"]
    assert result["sources"][0]["excerpts"] == [passage]
    assert result["assertions"][0]["source_ids"] == ["source-1"]
    assert result["assertions"][0]["status"] == "unresolved"
    assert not {"verdict", "confidence"}.intersection(result)


def test_queries_preserve_negation_and_do_not_invent_subject_specific_expansions():
    claim = "Vaccines do not contain microchips."
    prior = TranscriptUpdate("prior", 0, "Vaccines were discussed by the committee.", 20, 29, True)
    result = research_plan(context(claim, preceding=(prior,)), SpeechMetadata())
    assert result["assertions"][0]["query"] == claim + " discussed committee"
    assert result["assertions"][0]["text"] == claim


def test_numbers_do_not_receive_a_special_passage_ranking_bonus():
    precise = "Solar panels produce electricity through their cells, with the output depending on the available illumination."
    numeric = "Solar panels account for 10 million entries in this directory. " * 3
    assert source_excerpts(numeric + "\n\n" + precise, "Solar panels produce electricity.") == [precise]


def test_context_only_matches_cannot_become_claim_evidence():
    prior = TranscriptUpdate("prior", 0, "The bridge was discussed alongside tourism, hotels and regional investment.", 20, 29, True)
    sources = Sources()
    sources.scrape = lambda _: {"markdown": "Tourism, hotels and regional investment have dominated this year's planning discussions throughout the city.",
                                "metadata": {}}
    result = WebResearch(sources).review(context("The bridge opened in 1932.", preceding=(prior,)), SpeechMetadata())
    assert result["sources"] == [] and result["assertions"][0]["source_ids"] == []
    assert result["page_failures"] == [] and result["errors"] == [] and result["status"] == "complete"


def test_context_breaks_ties_only_between_passages_matching_the_claim():
    first = "The bridge opened with a ceremony attended by residents from several neighborhoods and other local communities."
    second = "The bridge opened to support tourism and regional investment, according to contemporary council meeting records."
    assert source_excerpts(first + "\n\n" + second, "The bridge opened.",
                           context_query="bridge tourism regional investment") == [second]


@pytest.mark.parametrize("ending", ["[...]", "[…]", "…", "...", "... Read more", "… Continue reading",
                                    "... [Read more](https://example.org/report)"])
def test_visibly_truncated_teasers_cannot_be_selected_as_intact_evidence(ending):
    teaser = "Labor unions are discussing a labor shortage, seeking protections, training and shared benef" + ending
    assert source_excerpts(teaser, "labor shortage") == []


def test_search_continues_past_duplicates_failed_and_irrelevant_pages_with_a_bound():
    calls = []
    passage = "Solar panels produce electricity through their cells, with the output depending on the available illumination."

    class FallbackSources:
        def search(self, query):
            return [None, {"url": "https://localhost/"}, *[
                {"url": f"https://example.org/{path}"}
                for path in ("failed", "failed", "unrelated", "good", "other", "unused")]]

        def scrape(self, url):
            calls.append(url.rsplit("/", 1)[-1])
            if calls[-1] == "failed":
                raise SourceUnavailable("Source blocked")
            return {"markdown": "Music album details. " * 5 if calls[-1] == "unrelated" else passage,
                    "metadata": {"datePublished": "2020-01-01"}}

    result = WebResearch(FallbackSources()).review(context("Solar panels produce electricity."), SpeechMetadata())
    assert calls == ["failed", "unrelated", "good", "other"]
    assert len(result["assertions"][0]["source_ids"]) == 2
    assert result["status"] == "partial" and result["errors"] == ["Source blocked"]
    assert result["page_failures"] == [{"url": "https://example.org/failed", "reason": "Source blocked"}]


def test_failed_pages_are_not_retried_across_assertions_and_attempts_are_bounded():
    calls = []

    class BlockedSources:
        def search(self, query):
            return [{"url": f"https://example.org/{index}"} for index in range(30)]

        def scrape(self, url):
            calls.append(url)
            raise SourceUnavailable("Source blocked")

    result = WebResearch(BlockedSources()).review(context(), SpeechMetadata())
    assert len(calls) == 5 and len(set(calls)) == 5
    assert result["status"] == "unavailable" and not result["sources"]
    assert result["page_failures"] == [{"url": url, "reason": "Source blocked"} for url in calls]
    assert result["errors"] == ["Source blocked"]


@pytest.mark.parametrize("claim", [
    "We don't have a labor shortage. We have a good job shortage.",
    "We don't have a water shortage. We have a clean water shortage.",
    "We don't have a $25,000 funding shortage. We have a $50,000 equipment shortage.",
])
def test_contrast_discovery_keeps_the_whole_claim_and_supplied_scope(claim):
    prior = TranscriptUpdate("prior", 0, "A shortage was discussed alongside unrelated speculation.", 20, 29, True)
    plan = research_plan(context(claim, preceding=(prior,)),
                         SpeechMetadata(country="New Zealand", source_published_at="2025-07-20"))
    assert len(plan["assertions"]) == 2
    assert all(part["query"] == claim + " New Zealand 2025" for part in plan["assertions"])
    assert plan["context"] == prior.text
    assert "speech may be older" in plan["cutoff_basis"]


def test_shared_contrast_search_is_cached_only_for_the_current_review():
    class Shared(Sources):
        def __init__(self):
            super().__init__()
            self.queries = []

        def search(self, query):
            self.queries.append(query)
            return super().search(query)

    sources = Shared()
    research = WebResearch(sources)
    for _ in range(2):
        result = research.review(context(), SpeechMetadata())
        assert all(part["source_ids"] == ["source-1"] for part in result["assertions"])
        assert all(part["searches"] == [{"query": context().claim.text, "status": "returned_links"}]
                   for part in result["assertions"])
    assert sources.queries == [context().claim.text] * 2


def test_shared_search_failure_is_not_retried_for_the_other_assertion():
    calls = []

    class Failed(Sources):
        compact_search = True

        def search(self, query):
            calls.append(query)
            raise SourceUnavailable("Search unavailable")

    result = WebResearch(Failed()).review(context(), SpeechMetadata())
    assert calls == [context().claim.text]
    assert result["status"] == "unavailable" and result["errors"] == ["Search unavailable"]
    assert result["page_failures"] == []
    assert all(part["searches"][0]["status"] == "unavailable" for part in result["assertions"])


def test_shared_empty_search_still_allows_distinct_assertion_fallbacks():
    calls = []

    class Empty(Sources):
        compact_search = True

        def search(self, query):
            calls.append(query)
            return []

    result = WebResearch(Empty()).review(context(), SpeechMetadata())
    assert calls == [context().claim.text, "labor shortage.", "good job shortage."]
    assert all(len(part["searches"]) == 2 for part in result["assertions"])
    assert result["status"] == "complete" and result["sources"] == []


@pytest.mark.parametrize("claim,prior,referent", [
    ("We don't have a water shortage. We have a clean water shortage.",
     "The water supply in Eastford is under discussion.", "Eastford"),
    ("We don't have a housing shortage. We have an affordable housing shortage.",
     "The housing supply in Greenbank is under discussion.", "Greenbank"),
])
def test_context_fallback_keeps_a_grounded_referent_when_shared_discovery_is_empty(claim, prior, referent):
    calls = []

    class Local(Sources):
        compact_search = True

        def search(self, query):
            calls.append(query)
            return [{"url": "https://example.org/local"}] if referent in query else []

        def scrape(self, url):
            return {"markdown": claim + " The local report discusses this contrast, its definitions, and the surveyed conditions.",
                    "metadata": {"datePublished": "2020-01-01"}}

    preceding = (TranscriptUpdate("prior", 0, prior, 20, 29, True),)
    result = WebResearch(Local()).review(context(claim, preceding), SpeechMetadata(spoken_at="2020-02-01"))
    assert calls[0] == claim + " 2020" and len(calls) == 3
    assert all(prior in query for query in calls[1:])
    assert all(part["source_ids"] == ["source-1"] for part in result["assertions"])
    assert all(len(part["searches"]) == 2 for part in result["assertions"])
    assert result["claim_scope"]["country"] == ""


def test_context_fallback_never_adds_future_interim_or_cut_off_numeric_context():
    claim = "We don't have a water shortage. We have a clean water shortage."
    prior = TranscriptUpdate("prior", 0, "Eastford water use fell from 5 m per s to 3 m per s.", 20, 29, True)
    future = TranscriptUpdate("future", 0, "Futuretown water supply costs $250,000.", 36, 40, True)
    interim = TranscriptUpdate("interim", 0, "Secretville water supply costs five million dollars.", 20, 29, False)
    plan = research_plan(context(claim, (prior, future, interim)), SpeechMetadata(spoken_at="2020-02-01"))
    assert all(prior.text in part["context_query"] for part in plan["assertions"])
    assert "Futuretown" not in str(plan) and "Secretville" not in str(plan)
    oversized = TranscriptUpdate("prior", 0, "water supply " + "expensive " * 80 + "$250,000.", 20, 29, True)
    plan = research_plan(context(claim, (oversized,)), SpeechMetadata())
    assert all("context_query" not in part for part in plan["assertions"])


def test_three_discovery_queries_share_five_page_attempts_per_assertion():
    queries, pages = [], []

    class Bounded(Sources):
        compact_search = True

        def search(self, query):
            queries.append(query)
            return [{"url": f"https://example.org/query-{len(queries)}-page-{i}"} for i in range(10)]

        def scrape(self, url):
            pages.append(url.rsplit("/", 1)[-1])
            return {"markdown": "The orchestra performed classical music at several concert venues, with new recordings released later.",
                    "metadata": {}}

    claim = "We don't have a water shortage. We have a clean water shortage."
    prior = TranscriptUpdate("prior", 0, "The water supply in Eastford is under discussion.", 20, 29, True)
    result = WebResearch(Bounded()).review(context(claim, (prior,)), SpeechMetadata())
    assert len(queries) == 5
    assert pages == ["query-1-page-0", "query-1-page-1", "query-1-page-2",
                     "query-2-page-0", "query-3-page-0", "query-4-page-0", "query-5-page-0"]
    assert all(len(part["searches"]) == 3 for part in result["assertions"])
    assert result["sources"] == []


def test_context_search_failure_stays_operational_and_does_not_trigger_compact_retry():
    calls = []

    class Failed(Sources):
        compact_search = True

        def search(self, query):
            calls.append(query)
            if "Eastford" in query:
                raise SourceUnavailable("Context search unavailable")
            return []

    claim = "We don't have a water shortage. We have a clean water shortage."
    prior = TranscriptUpdate("prior", 0, "The water supply in Eastford is under discussion.", 20, 29, True)
    result = WebResearch(Failed()).review(context(claim, (prior,)), SpeechMetadata())
    assert len(calls) == 3 and all("Eastford" in query for query in calls[1:])
    assert all([attempt["status"] for attempt in part["searches"]] == ["empty", "unavailable"]
               for part in result["assertions"])
    assert result["status"] == "unavailable" and result["errors"] == ["Context search unavailable"]


@pytest.mark.parametrize("prefix", ["5 million dollars", "12 m per s", "Eastford water use at 5 million liters"])
def test_long_context_cannot_drop_a_numeric_prefix_or_block_compact_discovery(prefix):
    calls = []

    class Validated(Sources):
        compact_search = True

        def search(self, query):
            assert valid_query(query)
            calls.append(query)
            return []

    claim = "We don't have a water shortage. We have a clean water shortage."
    prior = TranscriptUpdate("prior", 0, prefix + " " + "and " * 92 + "water shortage", 20, 29, True)
    result = WebResearch(Validated()).review(context(claim, (prior,)), SpeechMetadata())
    assert all("context_query" not in part for part in result["assertions"])
    assert calls == [claim, "water shortage.", "clean water shortage."]
    assert result["status"] == "complete" and result["errors"] == []


def test_context_query_uses_the_actual_provider_word_limit_even_below_character_limit():
    claim = "We don't have a water shortage. We have a clean water shortage."
    prior = TranscriptUpdate("prior", 0, "Eastford water supply " + "and " * 70 + "5 million liters", 20, 29, True)
    assert len(prior.text) < 400
    plan = research_plan(context(claim, (prior,)), SpeechMetadata())
    assert all("context_query" not in part for part in plan["assertions"])


def test_valid_context_query_keeps_the_complete_original_numeric_sentence():
    claim = "We don't have a water shortage. We have a clean water shortage."
    prior = TranscriptUpdate("prior", 0, "5 million liters of Eastford water supply remain under review.", 20, 29, True)
    plan = research_plan(context(claim, (prior,)), SpeechMetadata(country="Germany", spoken_at="2020-02-01"))
    for part in plan["assertions"]:
        assert part["context_query"] == part["text"] + " " + prior.text + " Germany 2020"
        assert valid_query(part["context_query"])


@pytest.mark.parametrize("url, published, status, basis", [
    ("https://example.org/2025/07/01/report", "2025-07-01", "published_by_cutoff", "URL path date: 2025-07-01"),
    ("https://example.org/2026/02/politics/report", "", "later_publication", "URL path dates the page to 2026-02"),
    ("https://example.org/report/2024/goal-01/", "", "date_unconfirmed", "No explicit publication date found."),
    ("https://example.org/2025/07/report", "", "date_unconfirmed", "No explicit publication date found."),
    ("https://example.org/news/report", "", "date_unconfirmed", "No explicit publication date found."),
])
def test_publisher_path_dates_admit_or_exclude_pages_without_metadata_dates(url, published, status, basis):
    class Undated(Sources):
        def search(self, query):
            return [{"url": url, "title": "Search title"}]

        def scrape(self, requested):
            page = super().scrape(requested)
            page["metadata"] = {"title": "Page title", "statusCode": 200}
            return page

    result = WebResearch(Undated()).review(context(), SpeechMetadata(source_published_at="2025-07-20"))
    source = result["sources"][0]
    assert source["published_at"] == published and source["temporal_status"] == status
    assert source["publication_basis"].startswith(basis)
    assert (source["id"] in result["assertions"][0]["source_ids"]) == (status != "later_publication")


def test_a_declared_country_is_recorded_as_supplied_by_the_viewer_and_nothing_else_sets_one():
    declared = research_plan(context(), SpeechMetadata(country="United States", source_published_at="2025-07-20"))
    assert declared["claim_scope"]["country"] == "United States"
    assert declared["claim_scope"]["country_basis"] == "supplied by the viewer"
    assert not any("country" in gap for gap in declared["gaps"])
    unset = research_plan(context(), SpeechMetadata(source_published_at="2025-07-20"))
    assert unset["claim_scope"]["country"] == "" and unset["claim_scope"]["country_basis"] == ""
    assert any("country or jurisdiction has not been established" in gap for gap in unset["gaps"])


def test_a_header_dateline_dates_an_otherwise_undated_page_and_prose_dates_do_not():
    class Dated(Sources):
        def __init__(self, markdown):
            super().__init__()
            self.markdown = markdown

        def scrape(self, requested):
            return {"markdown": self.markdown, "metadata": {"title": "Page title", "statusCode": 200}}

    body = ("A labor shortage concerns worker availability. A good job shortage concerns job quality, including wages "
            "and hours of employment.")
    review = f"# Review\n\nJanuary 13, 2025 | [Justin Ladner](https://example.org/bio)\n\n{body}"
    source = WebResearch(Dated(review)).review(context(), SpeechMetadata(source_published_at="2025-07-20"))["sources"][0]
    assert source["published_at"] == "2025-01-13" and source["temporal_status"] == "published_by_cutoff"
    assert source["publication_basis"].startswith("Dateline: January 13, 2025")
    prose = f"# Review\n\nOn January 13, 2025, the agency reported the following.\n\n{body}"
    source = WebResearch(Dated(prose)).review(context(), SpeechMetadata(source_published_at="2025-07-20"))["sources"][0]
    assert source["published_at"] == "" and source["temporal_status"] == "date_unconfirmed"


def test_assertion_pages_carry_their_reading_passages_for_the_sentence_level_stage():
    class Long(Sources):
        def scrape(self, url):
            page = super().scrape(url)
            page["markdown"] = ("# Understanding the labor shortage\n\nNearly every state is facing an unprecedented challenge finding "
                                "workers to fill its open jobs this year.\n\nThere are too many open jobs without people to fill them, a "
                                "labor shortage by any measure.\n\nJane Doe is a director on the communications team. "
                                "Her work on the labor shortage has been cited widely.\n\nUnrelated paragraph about weather.")
            return page

    result = WebResearch(Long()).review(context("We don't have a labor shortage."), SpeechMetadata(spoken_at="2025-10-20"))
    source = result["sources"][0]
    assert len(source["excerpts"]) == 1, "the research card keeps its one strongest excerpt"
    passages = source["assertion_passages"]
    assert "Nearly every state is facing an unprecedented challenge finding workers to fill its open jobs this year." in passages
    assert "There are too many open jobs without people to fill them, a labor shortage by any measure." in passages
    assert source["excerpts"][0] in passages and len(passages) <= 9
    assert "context_concepts" not in source and "reading_passages" not in source


def test_a_declared_speech_date_is_the_boundary_and_is_recorded_as_supplied():
    plan = research_plan(context(), SpeechMetadata(spoken_at="2025-07-15", source_published_at="2025-07-20"))
    assert plan["search_cutoff"] == "2025-07-15" and plan["cutoff_basis"] == "Supplied speech date"
    assert plan["claim_scope"]["spoken_at"] == "2025-07-15" and plan["claim_scope"]["spoken_at_basis"] == "supplied by the viewer"
    assert not any("date the claim describes" in gap for gap in plan["gaps"])
