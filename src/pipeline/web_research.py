"""Research unresolved claims without applying FEVER confidence to web passages.

Each assertion retains its original wording. Domain hints describe evidence to seek,
not a truth label. Lexical passage matches establish neither entailment nor temporal
applicability; these limitations travel with the source and the combined claim.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import date, datetime, timezone
from urllib.parse import urlsplit

from src.pipeline.caption_claims import contrast_parts
from src.pipeline.caption_concepts import concept_packet, select_concepts
from src.pipeline.context import SpeechMetadata
from src.pipeline.research_queries import fallback_query
from src.pipeline.topic_query import content_terms, topic_query
from src.pipeline.transcript import ClaimContext
from src.retrieval.byline_dates import byline_date
from src.retrieval.reference_sources import ReferenceSources, reference_candidates, reference_url
from src.retrieval.research_passages import reading_window
from src.retrieval.research_passages import source_excerpts as source_excerpts
from src.retrieval.search_contract import valid_query
from src.retrieval.source_dates import publication_date, publication_notices, temporal_scope
from src.retrieval.url_dates import later_by_path, url_path_date
from src.retrieval.visible_text import visible_text
from src.retrieval.web_sources import SourceUnavailable, WebSources, public_url

_EXCLUDED_HOSTS = {"facebook.com", "instagram.com", "reddit.com", "youtube.com", "x.com"}
_DISCOVERY_HOSTS = {"news.google.com"}


def excluded_host(host: str, *, publisher: bool = False) -> bool:
    """A redirect or subdomain cannot turn an excluded page into publisher evidence."""
    host = host.lower().rstrip(".")
    excluded_hosts = _EXCLUDED_HOSTS | _DISCOVERY_HOSTS if publisher else _EXCLUDED_HOSTS
    return any(host == excluded or host.endswith("." + excluded) for excluded in excluded_hosts)


def research_plan(context: ClaimContext, metadata: SpeechMetadata) -> dict:
    """Use only finalized earlier speech, with visible unknown geography and time."""
    earlier = [item.text for item in context.preceding if item.final
               and item.start <= context.claim.start and item.end <= context.claim.end]
    nearby = " ".join(" ".join(earlier).split()[-192:])
    parts = contrast_parts(context.claim.text) or (context.claim.text,)
    speech_day = date.fromisoformat(metadata.spoken_at).isoformat() if metadata.spoken_at else ""
    publication_day = date.fromisoformat(metadata.source_published_at).isoformat() if metadata.source_published_at else ""
    cutoff = min(filter(None, (speech_day, publication_day)), default="")
    combined_query = " ".join(filter(None, (context.claim.text, metadata.country, cutoff[:4])))
    shared_query = combined_query if len(parts) > 1 and len(combined_query) <= 500 and valid_query(combined_query) else ""
    assertions = []
    for part in parts:
        part_context = ClaimContext(replace(context.claim, text=part), context.preceding)
        _, context_ids, context_text = topic_query(part_context)
        hints = [term for term in content_terms(" ".join(context_text)) if term not in content_terms(part)][:24]
        query = shared_query or " ".join(filter(None, (part, " ".join(hints), metadata.country, cutoff[:4])))[:500]
        needs = "Sources addressing this assertion, with matching place, period and definitions."
        assertion = {"text": part, "query": query, "needs": needs, "status": "unresolved", "source_ids": []}
        complete_context = " ".join(item.text for item in context.preceding if item.id in context_ids and item.final
                                    and item.start <= context.claim.start and item.end <= context.claim.end)
        context_query = " ".join(filter(None, (part, complete_context, metadata.country, cutoff[:4])))
        if (shared_query and hints and len(context_query) <= 500
                and valid_query(context_query) and context_query != query):
            assertion["context_query"] = context_query
        assertions.append(assertion)
    missing = []
    if not metadata.country:
        missing.append("The claim's country or jurisdiction has not been established.")
    if not metadata.spoken_at:
        missing.append("The date the claim describes has not been established; upload date is not speech date.")
    return {"assertions": assertions, "context": nearby, "gaps": missing, "scope_limits": list(missing),
            "claim_scope": {"country": metadata.country, "spoken_at": metadata.spoken_at,
                            "source_published_at": metadata.source_published_at,
                            # A country or speech date only ever arrives by declaration; nothing here infers one.
                            "country_basis": "supplied by the viewer" if metadata.country else "",
                            "spoken_at_basis": "supplied by the viewer" if metadata.spoken_at else ""},
            "search_cutoff": cutoff,
            "cutoff_basis": "Supplied speech date" if cutoff and cutoff == speech_day
            else "Video publication date; the speech may be older" if cutoff else "Unknown"}


def publication_note(published: str, spoken_at: str) -> str:
    """Only compare explicit ISO dates; retrieval time never supplies publication time."""
    if not spoken_at:
        return "Claim date unknown; temporal applicability has not been checked."
    try:
        publication = date.fromisoformat(published[:10])
    except ValueError:
        return "Publication date unknown; temporal applicability has not been checked."
    if publication > date.fromisoformat(spoken_at):
        return "Published after the supplied claim date; contemporaneous applicability is unestablished."
    return "Published by the supplied claim date; the period covered still needs checking."


class WebResearch:
    def __init__(self, sources: WebSources) -> None:
        self.sources = sources
        self.always_search = getattr(sources, "always_search", False)

    def begin_check(self) -> None:
        self.sources.begin_check()

    def reference_research(self, context: ClaimContext, metadata: SpeechMetadata, sources: list[dict],
                           concept_selection: dict | None, identity_offset: int) -> dict | None:
        """Reuse source admission for one bounded hop, retaining the parent reference."""
        reader = getattr(self.sources, "read_link", None)
        if not callable(reader):
            return None
        candidates = reference_candidates(sources, _EXCLUDED_HOSTS | _DISCOVERY_HOSTS)
        if not candidates:
            return None
        references = ReferenceSources(candidates, reader)
        linked = WebResearch(references).review(context, metadata, concept_selection=concept_selection)
        existing = {reference_url(source["url"]): source for source in sources}
        resolved, ancestry = references.resolved_urls, references.lineage
        identities = []
        for index, child in enumerate(linked["sources"], 1):
            url = reference_url(child["url"])
            lineage = [entry for requested, final in resolved.items() if reference_url(final) == url
                       for entry in ancestry[requested]]
            if not lineage:
                continue
            source = existing.get(url)
            if source is None:
                source = child | {"id": f"source-{identity_offset + index}"}
                sources.append(source)
                existing[url] = source
            for entry in lineage:
                if entry not in source.setdefault("linked_from", []):
                    source["linked_from"].append(entry)
            identities.append(source["id"])
        return {"attempts": references.attempts, "source_ids": list(dict.fromkeys(identities)), "status": linked["status"],
                "errors": linked["errors"], "page_failures": linked["page_failures"]}

    def search_hits(self, assertion: dict, metadata: SpeechMetadata, cutoff: str,
                    seen: set[str], errors: list[str],
                    search_cache: dict[str, list[dict] | SourceUnavailable]) -> Iterator[dict]:
        """Share five page attempts across primary, contextual and compact discovery."""
        queries = [assertion["query"]]
        if context_query := assertion.get("context_query"):
            queries.append(context_query)
        if not assertion.get("concept") and getattr(self.sources, "compact_search", False):
            fallback = fallback_query(assertion["text"], metadata.country, cutoff)
            if fallback and fallback not in queries:
                queries.append(fallback)
        assertion["searches"] = []
        for index, query in enumerate(queries):
            if index and assertion["source_ids"]:
                break
            attempt = {"query": query, "status": "unavailable"}
            assertion["searches"].append(attempt)
            try:
                if query not in search_cache:
                    try:
                        dated_search = getattr(self.sources, "search_before", None)
                        hits = dated_search(query, cutoff) if dated_search else self.sources.search(query)
                        search_cache[query] = hits[:20]
                    except SourceUnavailable as error:
                        search_cache[query] = error
                hits = search_cache[query]
                if isinstance(hits, SourceUnavailable):
                    raise hits
            except SourceUnavailable as error:
                errors.append(str(error))
                break
            attempt["status"] = "returned_links" if hits else "empty"
            page_limit = 3 if not index and len(queries) > 1 else 5 - (len(queries) - index - 1)
            for hit in hits[:20]:
                if len(seen) >= page_limit and not assertion["source_ids"]:
                    break
                yield hit

    def review(self, context: ClaimContext, metadata: SpeechMetadata, *, concept_selection: dict | None = None) -> dict:
        plan = research_plan(context, metadata)
        context_research = []
        if concept_selection is not None:
            for concept in select_concepts(concept_packet(context), concept_selection):
                query = " ".join(filter(None, (concept["text"], metadata.country, plan["search_cutoff"][:4])))
                context_research.append({"concept": concept, "text": concept["text"], "query": query,
                                         "source_ids": [], "status": "unresolved"})
            plan["context_research"] = context_research
        if note := getattr(self.sources, "scope_note", ""):
            plan["gaps"].append(note)
        documents: dict[str, dict] = {}
        aliases: dict[str, str] = {}
        failed: set[str] = set()
        search_cache: dict[str, list[dict] | SourceUnavailable] = {}
        errors = []
        page_failures = []
        for assertion in [*context_research, *plan["assertions"]]:
            # Context gets two page attempts per selected phrase within the same deadline.
            is_context = "concept" in assertion
            page_limit = 2 if is_context else 5
            source_limit = 1 if is_context else 2
            seen: set[str] = set()
            for hit in self.search_hits(assertion, metadata, plan["search_cutoff"], seen, errors, search_cache):
                if len(seen) >= page_limit or len(assertion["source_ids"]) >= source_limit:
                    break
                if not isinstance(hit, dict):
                    continue
                url = public_url(hit.get("url"))
                host = (urlsplit(url).hostname or "").removeprefix("www.")
                if not url or url in seen or excluded_host(host):
                    continue
                seen.add(url)
                if url in failed:
                    continue
                if url not in aliases:
                    try:
                        page = self.sources.scrape(url)
                    except SourceUnavailable as error:
                        errors.append(str(error))
                        page_failures.append({"url": url, "reason": str(error)[:300]})
                        failed.add(url)
                        continue
                    info = page.get("metadata") or {}
                    markdown = page.get("markdown")
                    if (not isinstance(info, dict) or not isinstance(markdown, str)
                            or info.get("error") or info.get("statusCode", 200) != 200):
                        errors.append("A discovered page could not be extracted.")
                        page_failures.append({"url": url, "reason": "A discovered page could not be extracted."})
                        failed.add(url)
                        continue
                    final_url = public_url(info.get("sourceURL", url))
                    final_host = (urlsplit(final_url).hostname or "").removeprefix("www.")
                    if not final_url or excluded_host(final_host, publisher=True):
                        failed.add(url)
                        continue
                    published, date_basis = publication_date(info, markdown)
                    if date_basis == "No explicit publication date found." and publication_notices(markdown):
                        date_basis = "Exact publication day unconfirmed; partial publication notice retained in source context."
                    applicability = temporal_scope(published, plan["search_cutoff"])
                    if not published:
                        # A dateline in the page header is the publisher's own statement of the day.
                        published, dateline = byline_date(markdown)
                        if published:
                            date_basis = dateline
                            applicability = temporal_scope(published, plan["search_cutoff"])
                    if not published and (path_date := url_path_date(final_url)):
                        # The publisher's own path dates the page; a partial one only ever excludes.
                        if len(path_date) == 10:
                            published, date_basis = path_date, f"URL path date: {path_date}"
                            applicability = temporal_scope(published, plan["search_cutoff"])
                        elif later_by_path(path_date, plan["search_cutoff"]):
                            applicability = "later_publication"
                            date_basis = f"URL path dates the page to {path_date}, after the search boundary; exact day unconfirmed."
                    title = info.get("title") or hit.get("title") or final_url
                    aliases[url] = final_url
                    documents.setdefault(final_url, {
                        "id": f"source-{len(documents) + 1}", "url": final_url,
                        "title": str(title)[:300], "publisher": urlsplit(final_url).hostname,
                        "published_at": published, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "publication_basis": date_basis, "temporal_status": applicability,
                        "feed_published_at": hit.get("feed_published_at", ""),
                        "feed_date_basis": hit.get("feed_date_basis", ""),
                        "temporal_note": (
                            "Published after the search boundary. Its relevance to the earlier claim needs separate review."
                            if applicability == "later_publication" else
                            publication_note(published, metadata.spoken_at)),
                        "role": "Research candidate; relevance and applicability require review.",
                        "source_context": info.get("subheading", "") if isinstance(info.get("subheading", ""), str) else "",
                        "excerpts": [], "markdown": markdown,
                    })
                source = documents[aliases[url]]
                window = reading_window(source["markdown"], assertion["text"])
                if reason := window.get("unavailable_reason"):
                    errors.append(reason)
                    continue
                if is_context:
                    # A literal caption phrase can be incidental speech. Its page must also
                    # connect to the claim; URLs and distant page boilerplate cannot supply that link.
                    visible, _ = visible_text("\n\n".join([*window["context"], *window["passages"]]))
                    anchors = {term for term in content_terms(context.claim.text) if term.isalpha()}
                    if not anchors.intersection(content_terms(visible)):
                        continue
                excerpts = (window["passages"] if is_context else
                            source_excerpts(source["markdown"], assertion["text"],
                                            context_query=assertion.get("context_query", assertion["query"])))
                if excerpts:
                    if window["context"]:
                        source["reading_context"] = list(dict.fromkeys([*source.get("reading_context", []), *window["context"]]))
                    if is_context:
                        # Each selected phrase keeps its bounded window even when phrases share one page.
                        source["reading_passages"] = list(dict.fromkeys(
                            [*source.get("reading_passages", []), *excerpts]))[:9 * len(context_research)]
                        source.setdefault("context_concepts", []).append(assertion["concept"])
                    else:
                        source["excerpts"] = list(dict.fromkeys([*source["excerpts"], *excerpts]))[:2]
                        # The sentence-level reading gets the page's paragraphs around every match, the same
                        # reading concept pages get; each assertion retains its own nine-paragraph allowance.
                        source["assertion_passages"] = list(dict.fromkeys(
                            [*source.get("assertion_passages", []), *window["passages"]]))[:9 * len(plan["assertions"])]
                    if source["temporal_status"] != "later_publication" and source["id"] not in assertion["source_ids"]:
                        assertion["source_ids"].append(source["id"])
        sources = [{key: value for key, value in source.items() if key != "markdown"}
                   for source in documents.values() if source["excerpts"] or source.get("reading_passages")]
        references = self.reference_research(context, metadata, sources, concept_selection, len(documents))
        if references is not None:
            plan["reference_research"] = references
            errors.extend(references["errors"])
            page_failures.extend(references["page_failures"])
        plan["gaps"].append("Matching passages do not establish either assertion or the combined contrast."
                            if len(plan["assertions"]) == 2 else "Matching passages have not established this assertion.")
        return plan | {"sources": sources, "errors": list(dict.fromkeys(errors)), "page_failures": page_failures,
                       "status": "partial" if errors and sources else "unavailable" if errors else "complete"}
