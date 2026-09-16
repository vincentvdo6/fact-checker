"""Compose per-sentence relations into a verdict by fixed rules, with abstention as the default.

No model writes the conclusion. Only `states` and `states_negation` judgments on
gate-eligible observation sentences can move an assertion; `bears_on` findings are
carried for display and count for nothing. A qualifier on a judgment (a place, a
population, a period) can only weaken it: the assertion becomes qualified rather than
established, because whether that scope matches the claim's is not known here. An
assertion is established by an unqualified count from one independent source whose
publication date is confirmed, and the page names that source as its basis; an undated page
cannot be placed against the claim's period, so it can qualify but never establish. A page
reached by following a link from another, or a second page on the same publisher, is the same
source, so the count of independent sources shown is never inflated by repetition. Until
2026-09-14 two independent sources were required, a stopgap while the judge was unmeasured on
news; the rule was lifted on the measurement the viewer actually experiences -- on 153 unseen
claims the composed reading pointed a direction 47 times and was right 47 times (in-sample for
that decision, `scripts/measure_directions.py`), 29 of them held at "not established" by the
two-source rule alone -- while the judge's per-sentence counting error stayed near a half,
because a wrong count usually lands on a page that is about the claim being true or false.
When sources point both ways the assertion is contested, not averaged. A contrast is
established only when every one of its assertions is; otherwise the whole claim is reported as
partially established or unresolved, with each half's status kept apart.

The output is a draft: it proves which exact sentences were allowed to count and how
they were counted, not that the judgments were right.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from src.verdict.scope import different_period

RELATIONSHIPS = ("supported", "contradicted", "qualified", "insufficient")

STATUSES = ("supported", "contradicted", "qualified_support", "qualified_contradiction", "contested", "insufficient")
SUMMARIES = ("established", "partially_established", "partially_contradicted", "qualified_support",
             "qualified_contradiction", "contested", "unresolved")
FOR, AGAINST = ("supported", "qualified_support"), ("contradicted", "qualified_contradiction")


def _index_judgments(assertions: list[dict], gate: dict, judgments: list[dict]) -> dict[str, list[dict]]:
    eligible = set(gate["eligible_ids"])
    known = {assertion["id"] for assertion in assertions}
    seen, by_assertion = set(), {assertion["id"]: [] for assertion in assertions}
    for judgment in judgments:
        key = (judgment["assertion_id"], judgment["unit_id"])
        if judgment["unit_id"] not in eligible:
            raise ValueError("A judgment refers to a sentence the gate withheld from evidence.")
        if judgment["assertion_id"] not in known or key in seen:
            raise ValueError("A judgment refers to an unknown assertion or repeats a pair.")
        seen.add(key)
        by_assertion[judgment["assertion_id"]].append(judgment)
    return by_assertion


CONTEXT_NOTE = "found by caption-concept research; shown, never counted"
CONFLICT_NOTE = "this page is read both as stating and as denying the assertion; a misreading, not a dispute"


def _evidence_row(judgment: dict, unit: dict, source: dict, independent: str, assertion: dict) -> dict:
    row = {"unit_id": unit["id"], "passage_id": unit["passage_id"], "source_id": unit["source_id"],
           "independent_source": independent, "origin": unit.get("origin", "assertion"),
           "relation": judgment["relation"], "span": judgment["span"], "text": unit["text"],
           "qualifiers": list(judgment["qualifiers"]), "definitions": [dict(item) for item in unit["definitions"]],
           "url": source.get("url", ""), "published_at": source.get("published_at", ""),
           "publication_basis": source.get("publication_basis", "")}
    if "confidence" in judgment:
        row["confidence"] = float(judgment["confidence"])       # display order only; never a rule input
    if row["origin"] == "context" and row["relation"] in ("states", "states_negation"):
        # Concept research explains the claim's terms; its pages were never searched for the assertion.
        row["relation"], row["not_counted"] = "bears_on", f"{judgment['relation']}: {CONTEXT_NOTE}"
    if row["relation"] in ("states", "states_negation") and (dated := different_period(assertion["text"], unit["text"])):
        # A sentence dated wholly outside the claim's stated period can bear on it, never state or deny it.
        row["period"] = dated
        row["relation"], row["not_counted"] = "bears_on", f"{judgment['relation']}: states a different period ({', '.join(dated)})"
    return row


def _demote_self_conflicts(rows: list[dict]) -> None:
    """A page read as both stating and denying one assertion has been misread; none of its counts stand.

    Disagreement between sources is a dispute and stays contested. Disagreement inside one page is
    the judge contradicting itself, and the page's counted sentences are shown, never counted.
    """
    directions: dict[str, set[str]] = {}
    for row in rows:
        if row["relation"] in ("states", "states_negation"):
            directions.setdefault(row["source_id"], set()).add(row["relation"])
    for row in rows:
        if len(directions.get(row["source_id"], ())) == 2 and row["relation"] in ("states", "states_negation"):
            row["relation"], row["not_counted"] = "bears_on", f"{row['relation']}: {CONFLICT_NOTE}"


def _lineage_root(source_id: str, by_source: dict[str, dict]) -> str:
    seen: set[str] = set()
    while source_id not in seen:
        seen.add(source_id)
        parents = [entry.get("parent_source_id") for entry in by_source.get(source_id, {}).get("linked_from", [])
                   if isinstance(entry, dict)]
        parents = [parent for parent in parents if isinstance(parent, str) and parent in by_source]
        if not parents:
            break
        source_id = parents[0]
    return source_id


def independent_source(source_id: str, by_source: dict[str, dict]) -> str:
    """The key two counted sentences must differ on to be corroboration rather than repetition.

    A page reached by following a link from an accepted page is that page's lineage, and two
    pages on one publisher are one publisher; either is a single source however many
    sentences it yields.
    """
    root = _lineage_root(source_id, by_source)
    host = (urlsplit(by_source.get(root, {}).get("url", "")).hostname or "").removeprefix("www.")
    return host or root


def _corroborating(rows: list[dict]) -> list[dict]:
    """Counted sentences that can establish: unqualified, from a source whose publication date is confirmed."""
    return [row for row in rows if not row["qualifiers"] and row["published_at"]]


def _limits(rows: list[dict], scope_known: bool) -> list[str]:
    """Why counted sentences stop short of establishing: every reason that applies, in rule order."""
    limits = []
    if not scope_known:
        limits.append("the claim's country is not established")
    if all(row["qualifiers"] for row in rows):
        limits.append("every counted sentence states a narrower scope")
    elif not _corroborating(rows):
        limits.append("no counted sentence comes from a source with a confirmed publication date")
    return limits


def _basis(rows: list[dict]) -> str:
    """The independent dated sources an established assertion rests on, named so a single one is never hidden."""
    dated: dict[str, str] = {}
    for row in _corroborating(rows):
        dated.setdefault(row["independent_source"], row["published_at"])
    if not dated:
        return ""
    named = ", ".join(f"{source} (published {date})" for source, date in dated.items())
    return f"one source: {named}" if len(dated) == 1 else f"{len(dated)} independent sources: {named}"


def _assertion_status(rows: list[dict], scope_known: bool) -> tuple[str, str, list[str]]:
    support = [row for row in rows if row["relation"] == "states"]
    against = [row for row in rows if row["relation"] == "states_negation"]
    if support and against:
        return "contested", "mixed", []
    if support:
        limits = _limits(support, scope_known)
        return ("qualified_support" if limits else "supported"), "for", limits
    if against:
        limits = _limits(against, scope_known)
        return ("qualified_contradiction" if limits else "contradicted"), "against", limits
    return "insufficient", "none", []


def _summary(statuses: list[str]) -> tuple[str, str]:
    """Whole-claim relationship from the assertion statuses, never stronger than the weakest.

    A partial summary carries its direction: counts against one half and nothing for any half
    is "partially contradicted", never "partially established", which a reader would take as
    partly true. "Partially" is said only when some assertion is established; when every count
    is qualified the summary says so -- "supported, not established" -- because a single
    qualified count on a one-assertion claim is not partly anything. Both directions across
    the halves is contested.
    """
    if all(status == "supported" for status in statuses):
        return "supported", "established"
    if all(status == "contradicted" for status in statuses):
        return "contradicted", "established"
    if all(status == "insufficient" for status in statuses):
        return "insufficient", "unresolved"
    forward, against = any(status in FOR for status in statuses), any(status in AGAINST for status in statuses)
    if any(status == "contested" for status in statuses):
        # A half the sources dispute makes the claim contested whatever the other half does.
        return ("qualified" if forward or against else "insufficient"), "contested"
    if forward and against:
        return "qualified", "contested"
    if any(status in ("supported", "contradicted") for status in statuses):
        return "qualified", "partially_contradicted" if against else "partially_established"
    return "qualified", "qualified_contradiction" if against else "qualified_support"


def measurement_note(measurement: dict | None, direction: dict | None = None) -> str:
    """What the reading did on claims nobody looked at while building, said beside every count.

    The record that matches what a viewer sees is the direction record (`measure_directions`):
    of the readings that pointed a direction on unseen claims, how many pointed the right way.
    Without one, the judge's per-sentence counting record stands in, and it says the opposite
    thing about the same system -- a count is one sentence's reading, wrong about half the time
    on news -- so the page tells the reader to weigh the sentence, not the label.
    """
    if direction:
        return (f"When this reading pointed a direction on {direction['claims']} unseen claims "
                f"({direction['measured_on']}), it pointed the right way {direction['right']} of "
                f"{direction['directional']} times; the sentences it rests on are quoted below.")
    if not measurement:
        return ""
    return (f"When this judge counted a sentence on {measurement['claims']} unseen claims "
            f"({measurement['measured_on']}), it was wrong {measurement['wrong']} of {measurement['counted']} times; "
            "a counted sentence is a lead to read, not a finding.")


def compose(assertions: list[dict], gate: dict, judgments: list[dict], sources: list[dict],
            claim_scope: dict | None = None, judge_measurement: dict | None = None,
            direction_measurement: dict | None = None) -> dict:
    """Fixed rules over validated judgments; every counted sentence is quoted in full."""
    if not assertions or len({assertion["id"] for assertion in assertions}) != len(assertions):
        raise ValueError("Assertions must be nonempty and uniquely identified.")
    units = {unit["id"]: unit for unit in gate["units"]}
    by_source = {source["id"]: source for source in sources}
    if any(source.get("temporal_status") == "later_publication" for source in sources):
        raise ValueError("A source published after the search boundary cannot enter composition.")
    by_assertion = _index_judgments(assertions, gate, judgments)
    scope = dict(claim_scope or {})
    # A claim whose country was supplied as unknown cannot be established by a source about a known one.
    scope_known = not ("country" in scope and not scope["country"])
    results = []
    for assertion in assertions:
        rows = [_evidence_row(judgment, units[judgment["unit_id"]], by_source.get(units[judgment["unit_id"]]["source_id"], {}),
                              independent_source(units[judgment["unit_id"]]["source_id"], by_source), assertion)
                for judgment in by_assertion[assertion["id"]]]
        _demote_self_conflicts(rows)
        status, direction, limits = _assertion_status(rows, scope_known)
        counted = [row for row in rows if row["relation"] in ("states", "states_negation")]
        results.append({
            "id": assertion["id"], "text": assertion["text"], "negated": assertion["negated"],
            "status": status, "direction": direction, "limits": limits, "evidence": counted,
            # An established assertion names the independent dated sources it rests on; one is said to be one.
            "basis": _basis(counted) if status in ("supported", "contradicted") else "",
            # Numeric proximity cannot establish a shared measure or improve relevance.
            "relevant": sorted([row for row in rows if row["relation"] == "bears_on"],
                               key=lambda row: (row["origin"] != "assertion", -row.get("confidence", 0.0))),
            "sources_for": len({row["independent_source"] for row in counted if row["relation"] == "states"}),
            "sources_against": len({row["independent_source"] for row in counted if row["relation"] == "states_negation"}),
            "judged": len(rows), "eligible": len(gate["eligible_ids"])})
    relationship, summary = _summary([row["status"] for row in results])
    assert relationship in RELATIONSHIPS and summary in SUMMARIES
    limits = [f"The claim's {name} has not been established." for name, key in
              (("country", "country"), ("speech date", "spoken_at")) if key in scope and not scope[key]]
    notes = ([f"Claim country: {scope['country']} ({scope.get('country_basis') or 'basis not recorded'})."]
             if scope.get("country") else [])
    if scope.get("spoken_at"):
        notes.append(f"Speech date: {scope['spoken_at']} ({scope.get('spoken_at_basis') or 'basis not recorded'}).")
    counted_any = any(row["evidence"] for row in results)
    return {"status": "draft", "relationship": relationship, "summary": summary, "assertions": results,
            "withheld": dict(gate["withheld"]), "eligible": len(gate["eligible_ids"]), "scope_limits": limits,
            "scope_notes": notes,
            # Only a page with a count needs the judge's record; a page that counts nothing made no such reading.
            "judge_note": measurement_note(judge_measurement, direction_measurement) if counted_any else "",
            "rules": ["Only reported observations passing the eligibility gate can count.",
                      "Only states and states_negation relations count; bears_on is shown, never counted.",
                      "Numerical similarity alone does not confirm a claim's measure, population or period.",
                      "A sentence found by caption-concept research is shown, never counted.",
                      "A sentence dated wholly outside the claim's stated period is shown, never counted.",
                      "A qualifier on a counted sentence makes its assertion qualified, never established.",
                      "An assertion is established by an unqualified count from one independent source with a "
                      "confirmed publication date, which the page names; a linked page or a second page on one "
                      "publisher is the same source.",
                      "An unknown claim country makes every counted sentence qualifying, never establishing.",
                      "Sources pointing both ways make an assertion contested, not averaged.",
                      "One page read both ways is a misreading, not a dispute; none of its sentences count.",
                      "A claim is established only when every assertion is."]}
