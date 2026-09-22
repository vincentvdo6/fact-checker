"""Assemble the decomposed check: read, gate, judge each pair, compose by rule, render from spans.

The judge is the only model-facing step and it is injected, so the same chain runs
over saved role annotations with a stub, the local NLI cross-encoder, or a language
model behind the pair contract. Everything a judge returns passes through
`validate_pair_judgment` here, whatever produced it: a chain cannot be handed a
judgment that quotes text outside its sentence or names a withheld one.
"""

from __future__ import annotations

from collections.abc import Callable

from src.verdict.assertions import claim_assertions
from src.verdict.composition import compose
from src.verdict.eligibility import gate_units
from src.verdict.pair_judgment import MAX_QUALIFIERS, validate_pair_judgment
from src.verdict.reading import reading_packet, validate_roles
from src.verdict.role_rules import annotate
from src.verdict.scope import group_qualifiers, place_qualifiers
from src.verdict.span_render import render

Judge = Callable[[list[dict], list[dict]], list[dict]]
SOURCE_FIELDS = ("id", "url", "published_at", "temporal_status", "publication_basis", "temporal_note", "reading_context",
                 "linked_from")      # lineage: a page reached from another is that page's source, not a second one


def research_packet(claim: str, plan: dict) -> dict:
    """The packet shape the chain reads, from a live research plan: usable sources and exact excerpts only."""
    from src.retrieval.web_sources import public_url  # the live research stack; the chain itself needs none of it

    sources = []
    for source in plan.get("sources", []):
        # A page admitted for the assertion is read around every match, not only at its one excerpt.
        excerpts = list(dict.fromkeys([*source.get("excerpts", []), *source.get("assertion_passages", [])]))
        # Caption-concept research reads pages for the claim's terms, never for its assertions.
        context = [text for text in dict.fromkeys(source.get("reading_passages", [])) if text not in excerpts]
        if source.get("temporal_status") == "later_publication" or not public_url(source.get("url")) or not (excerpts or context):
            continue
        sources.append({key: source.get(key, "") for key in SOURCE_FIELDS if key in source}
                       | {"excerpts": excerpts, "context_excerpts": context})
    return {"claim": claim, "claim_context": plan.get("context", ""), "sources": sources,
            "scope_limits": list(plan.get("scope_limits", plan.get("gaps", []))),
            "claim_scope": dict(plan.get("claim_scope", {})), "search_cutoff": plan.get("search_cutoff", ""),
            "cutoff_basis": plan.get("cutoff_basis", "Unknown")}


def check_claim(claim: str, plan: dict, judge: Judge) -> dict:
    """Live path: rule-typed roles over a research plan, then the chain. The result is a draft."""
    packet = research_packet(claim, plan)
    result = run_chain(packet, annotate(reading_packet(packet)), judge)
    return result | {"roles": "closed-class rules, measured against labels/roles-*.json",
                     "status": "draft: rule-composed from an unmeasured judge; not a verified verdict"}


def run_chain(packet: dict, annotations: list[dict], judge: Judge) -> dict:
    """Every stage's output kept, so a wrong verdict can be traced to the stage that produced it."""
    reading = reading_packet(packet)
    roles = validate_roles(reading, annotations)
    gate = gate_units(reading, roles)
    assertions = claim_assertions(packet["claim"])
    units = {unit["id"]: unit for unit in gate["units"]}
    eligible = [units[identity] for identity in gate["eligible_ids"]]
    by_assertion = {assertion["id"]: assertion for assertion in assertions}
    country = str((packet.get("claim_scope") or {}).get("country") or "")
    judgments = []
    for raw in judge(assertions, eligible):
        assertion, unit = by_assertion[raw["assertion_id"]], units[raw["unit_id"]]
        # A subdivision of the declared country named in the sentence narrows it, whatever the judge said. One
        # qualifier already makes the row qualified, so a sentence listing many places keeps the first few.
        qualifiers = list(dict.fromkeys([*raw["qualifiers"], *place_qualifiers(unit["text"], assertion["text"], country),
                                         *group_qualifiers(unit["text"], assertion["text"])]))
        qualifiers = qualifiers[:MAX_QUALIFIERS]
        checked = validate_pair_judgment(assertion, unit, {"relation": raw["relation"], "span": raw["span"], "qualifiers": qualifiers})
        judgments.append(checked | {key: value for key, value in raw.items() if key not in checked})
    verdict = compose(assertions, gate, judgments, packet["sources"], packet.get("claim_scope"),
                      judge_measurement=getattr(judge, "measurement", None),
                      direction_measurement=getattr(judge, "direction_measurement", None))
    if review := getattr(judge, "review_composed", None):
        verdict = review(verdict, assertions, eligible)
    if order := getattr(judge, "order_context", None):
        verdict = order(verdict, assertions, eligible)
    return {"claim": packet["claim"], "assertions": assertions, "gate": gate, "judgments": judgments,
            "verdict": verdict, "text": render(verdict, packet["claim"])}
