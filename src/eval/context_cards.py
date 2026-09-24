"""Gate saved-card changes to contextual withholding and display order.

This checks record preservation and explicit reviewer expectations, not label accuracy.
Both inputs must come from the same saved click and frozen research material.
"""

from __future__ import annotations

from typing import Any

AUDIT_FIELDS = {"context_review", "context_relevance_score"}
VERDICT_AUDIT_FIELDS = {"context_ordering"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pair(row_index: int, assertion_id: str, value: dict) -> tuple[int, str, str]:
    unit_id = value.get("unit_id")
    require(isinstance(unit_id, str) and bool(unit_id), f"Missing unit ID in row {row_index}, {assertion_id}")
    return row_index, assertion_id, unit_id


def context_map(assertion: dict, row_index: int) -> tuple[dict, dict]:
    relevant: dict[tuple[int, str, str], dict] = {}
    withheld: dict[tuple[int, str, str], dict] = {}
    assertion_id = assertion.get("id")
    require(isinstance(assertion_id, str) and bool(assertion_id), f"Missing assertion ID in row {row_index}")
    for field, target in (("relevant", relevant), ("withheld_context", withheld)):
        values = assertion.get(field, [])
        require(isinstance(values, list), f"Malformed {field} in row {row_index}, {assertion_id}")
        for value in values:
            require(isinstance(value, dict), f"Malformed {field} entry in row {row_index}, {assertion_id}")
            require(value.get("relation") == ("bears_on" if field == "relevant" else "unrelated"),
                    f"Wrong {field} relation in row {row_index}, {assertion_id}")
            identity = pair(row_index, assertion_id, value)
            require(identity not in relevant and identity not in withheld, f"Duplicate context pair: {identity}")
            target[identity] = value
    return relevant, withheld


def immutable_context(value: dict, *, withheld: bool = False) -> dict:
    copy = {key: item for key, item in value.items() if key not in AUDIT_FIELDS}
    if withheld:
        require(copy.get("relation") == "unrelated" and copy.get("span") == "" and copy.get("qualifiers") == [],
                "Withheld context must use the unrelated relation and clear span/qualifiers")
        for field in ("relation", "span", "qualifiers"):
            copy.pop(field)
    return copy


def compare_context(before: dict, after: dict, moved: bool, identity: tuple[int, str, str]) -> None:
    require(before.get("relation") == "bears_on" or not moved, f"Cannot newly withhold a noncontext pair: {identity}")
    if moved:
        review = after.get("context_review", {})
        require(isinstance(review, dict) and review.get("prior_relation") == "bears_on"
                and review.get("relation") == "unrelated",
                f"Withheld context lacks an unrelated secondary reading: {identity}")
    original = immutable_context(before, withheld=before.get("relation") == "unrelated")
    current = immutable_context(after, withheld=moved or before.get("relation") == "unrelated")
    if moved:
        for field in ("relation", "span", "qualifiers"):
            original.pop(field, None)
    require(original == current, f"Context source or primary reading changed: {identity}")


def keyed_assertions(verdict: dict, row_index: int) -> dict[str, dict]:
    assertions = verdict.get("assertions")
    require(isinstance(assertions, list), f"Missing assertions in row {row_index}")
    result = {}
    for assertion in assertions:
        require(isinstance(assertion, dict) and isinstance(assertion.get("id"), str),
                f"Malformed assertion in row {row_index}")
        require(assertion["id"] not in result, f"Duplicate assertion in row {row_index}: {assertion['id']}")
        result[assertion["id"]] = assertion
    return result


def keyed_judgments(values: list, row_index: int) -> dict[tuple[int, str, str], dict]:
    result = {}
    require(isinstance(values, list), f"Malformed judgments in row {row_index}")
    for value in values:
        require(isinstance(value, dict) and isinstance(value.get("assertion_id"), str),
                f"Malformed judgment in row {row_index}")
        identity = pair(row_index, value["assertion_id"], value)
        require(identity not in result, f"Duplicate judgment: {identity}")
        result[identity] = value
    return result


def compare_judgments(before: list, after: list, row_index: int, removed: set) -> None:
    originals, candidates = keyed_judgments(before, row_index), keyed_judgments(after, row_index)
    require(originals.keys() == candidates.keys(), f"Judgment identities changed in row {row_index}")
    for identity, original in originals.items():
        candidate = candidates[identity]
        left = {key: value for key, value in original.items() if key != "context_review"}
        right = {key: value for key, value in candidate.items() if key != "context_review"}
        if identity in removed and left.get("relation") == "bears_on" and right.get("relation") == "unrelated":
            review = candidate.get("context_review", {})
            require(isinstance(review, dict) and review.get("prior_relation") == "bears_on"
                    and review.get("relation") == "unrelated",
                    f"Removed context lacks an unrelated secondary reading: {identity}")
            require(right.get("span") == "" and right.get("qualifiers") == [],
                    f"Removed judgment has unexpected fields: {identity}")
            for field in ("relation", "span", "qualifiers"):
                left.pop(field, None)
                right.pop(field, None)
        require(left == right, f"Primary judgment changed: {identity}")


def compare_records(baseline: dict, candidate: dict, expectations: dict) -> dict[str, Any]:
    require(isinstance(baseline, dict) and isinstance(candidate, dict), "Both records must be objects")
    require(baseline.get("payload") == candidate.get("payload") and isinstance(baseline.get("payload"), dict),
            "Saved click payloads differ")
    original_result, current_result = baseline.get("result"), candidate.get("result")
    require(isinstance(original_result, dict) and isinstance(current_result, dict), "Missing result object")
    original_rows, current_rows = original_result.get("rows"), current_result.get("rows")
    require(isinstance(original_rows, list) and isinstance(current_rows, list)
            and len(original_rows) == len(current_rows), "Claim rows differ")
    require({key: value for key, value in original_result.items() if key != "rows"} ==
            {key: value for key, value in current_result.items() if key != "rows"},
            "Result metadata or original excerpt changed")
    baseline_relevant: set[tuple[int, str, str]] = set()
    candidate_relevant: set[tuple[int, str, str]] = set()
    moved: set[tuple[int, str, str]] = set()
    audited_drops: set[tuple[int, str, str]] = set()
    for index, (old, new) in enumerate(zip(original_rows, current_rows, strict=True)):
        require(isinstance(old, dict) and isinstance(new, dict) and old.get("text") == new.get("text"),
                f"Original claim sentence changed in row {index}")
        require({key: value for key, value in old.items() if key != "result"} ==
                {key: value for key, value in new.items() if key != "result"}, f"Claim row metadata changed in row {index}")
        old_result, new_result = old.get("result"), new.get("result")
        require(isinstance(old_result, dict) and isinstance(new_result, dict), f"Missing claim result in row {index}")
        require({key: value for key, value in old_result.items() if key != "decomposed"} ==
                {key: value for key, value in new_result.items() if key != "decomposed"},
                f"Research or primary verdict changed in row {index}")
        original, current = old_result.get("decomposed"), new_result.get("decomposed")
        require((original is None) == (current is None), f"Sentence reading presence changed in row {index}")
        if original is None:
            continue
        require(isinstance(original, dict) and isinstance(current, dict), f"Malformed reading in row {index}")
        source_assertions, gate = original.get("assertions"), original.get("gate")
        require(isinstance(source_assertions, list) and isinstance(gate, dict)
                and isinstance(gate.get("units"), list) and isinstance(gate.get("eligible_ids"), list),
                f"Missing original assertions or source units in row {index}")
        source_ids = [item.get("id") for item in source_assertions if isinstance(item, dict)]
        unit_ids = [item.get("id") for item in gate["units"] if isinstance(item, dict)]
        require(len(source_ids) == len(source_assertions) == len(set(source_ids))
                and all(isinstance(item, str) and item for item in source_ids)
                and len(unit_ids) == len(gate["units"]) == len(set(unit_ids))
                and all(isinstance(item, str) and item for item in unit_ids),
                f"Malformed or duplicate original assertion/source identities in row {index}")
        by_unit = {unit["id"]: unit for unit in gate["units"]}
        eligible_ids = gate["eligible_ids"]
        require(all(isinstance(item, str) and item in by_unit for item in eligible_ids)
                and len(eligible_ids) == len(set(eligible_ids)),
                f"Malformed eligible source IDs in row {index}")
        research = old_result.get("research")
        require(isinstance(research, dict) and isinstance(research.get("sources"), list),
                f"Missing saved research sources in row {index}")
        sources = research["sources"]
        source_ids_saved = [source.get("id") for source in sources if isinstance(source, dict)]
        require(len(source_ids_saved) == len(sources) == len(set(source_ids_saved))
                and all(isinstance(item, str) and item for item in source_ids_saved),
                f"Malformed or duplicate research source in row {index}")
        by_source = {source["id"]: source for source in sources}
        omitted = {"verdict", "judgments", "text"}
        require({key: value for key, value in original.items() if key not in omitted} ==
                {key: value for key, value in current.items() if key not in omitted},
                f"Original assertions, gate or source units changed in row {index}")
        old_verdict, new_verdict = original.get("verdict"), current.get("verdict")
        require(isinstance(old_verdict, dict) and isinstance(new_verdict, dict), f"Missing verdict in row {index}")
        require({key: value for key, value in old_verdict.items() if key not in VERDICT_AUDIT_FIELDS | {"assertions"}} ==
                {key: value for key, value in new_verdict.items() if key not in VERDICT_AUDIT_FIELDS | {"assertions"}},
                f"Verdict status, summary, scope or counts changed in row {index}")
        old_assertions, new_assertions = keyed_assertions(old_verdict, index), keyed_assertions(new_verdict, index)
        require(old_assertions.keys() == new_assertions.keys(), f"Assertion identities changed in row {index}")
        require(old_assertions.keys() == set(source_ids), f"Verdict assertion differs from original in row {index}")
        source_assertions_by_id = {assertion["id"]: assertion for assertion in source_assertions}
        for assertion_id, assertion in old_assertions.items():
            source_assertion = source_assertions_by_id[assertion_id]
            for field in ("text", "negated"):
                if field in source_assertion:
                    require(assertion.get(field) == source_assertion[field],
                            f"Verdict assertion {field} differs from original in row {index}, {assertion_id}")
        old_judgments = keyed_judgments(original.get("judgments"), index)
        require(all(identity[1] in source_ids and identity[2] in gate["eligible_ids"]
                    for identity in old_judgments), f"Foreign original judgment in row {index}")
        row_moved: set[tuple[int, str, str]] = set()
        row_dropped: set[tuple[int, str, str]] = set()
        for assertion_id, old_assertion in old_assertions.items():
            new_assertion = new_assertions[assertion_id]
            omit = {"relevant", "withheld_context"}
            require({key: value for key, value in old_assertion.items() if key not in omit} ==
                    {key: value for key, value in new_assertion.items() if key not in omit},
                    f"Assertion evidence, status or count changed in row {index}, {assertion_id}")
            old_relevant, old_withheld = context_map(old_assertion, index)
            new_relevant, new_withheld = context_map(new_assertion, index)
            for identity, context in (old_relevant | old_withheld).items():
                unit = by_unit.get(identity[2])
                require(unit is not None and identity[2] in gate["eligible_ids"] and identity in old_judgments,
                        f"Context pair has no eligible source unit or original judgment: {identity}")
                for field in ("text", "definitions", "source_id", "passage_id", "origin"):
                    if field in unit:
                        require(context.get(field) == unit[field], f"Context {field} differs from source unit: {identity}")
                source = by_source.get(context.get("source_id"))
                require(source is not None, f"Context source missing from saved research: {identity}")
                for field in ("url", "published_at", "publication_basis"):
                    if field in source:
                        require(context.get(field) == source[field],
                                f"Context {field} differs from saved research source: {identity}")
            old_pairs = old_relevant.keys() | old_withheld.keys()
            new_pairs = new_relevant.keys() | new_withheld.keys()
            require(new_pairs <= old_pairs, f"New context pair appeared in row {index}, {assertion_id}")
            require(not (old_withheld.keys() & new_relevant.keys()), f"Previously withheld pair promoted in {assertion_id}")
            for identity, value in old_relevant.items():
                if identity not in new_pairs:
                    row_dropped.add(identity)
                    continue
                is_moved = identity in new_withheld
                compare_context(value, new_withheld[identity] if is_moved else new_relevant[identity], is_moved, identity)
                if is_moved:
                    row_moved.add(identity)
            for identity, value in old_withheld.items():
                require(identity in new_withheld, f"Previously withheld pair disappeared: {identity}")
                compare_context(value, new_withheld[identity], False, identity)
            baseline_relevant.update(old_relevant)
            candidate_relevant.update(new_relevant)
        candidate_judgments = keyed_judgments(current.get("judgments", []), index)
        for identity in row_dropped:
            review = candidate_judgments.get(identity, {}).get("context_review", {})
            require(candidate_judgments.get(identity, {}).get("relation") == "unrelated"
                    and isinstance(review, dict) and review.get("prior_relation") == "bears_on"
                    and review.get("relation") == "unrelated",
                    f"Context pair vanished without a secondary unrelated reading: {identity}")
        compare_judgments(original.get("judgments", []), current.get("judgments", []), index,
                          row_moved | row_dropped)
        moved.update(row_moved)
        audited_drops.update(row_dropped)
    require(isinstance(expectations, dict) and set(expectations) == {"must_retain", "must_drop"},
            "Expectations need must_retain and must_drop lists")
    expected = {}
    for name in ("must_retain", "must_drop"):
        values = expectations[name]
        require(isinstance(values, list), f"{name} must be a list")
        selected = set()
        for value in values:
            require(isinstance(value, dict) and set(value) == {"row", "assertion_id", "unit_id"}
                    and type(value["row"]) is int and value["row"] >= 0
                    and all(isinstance(value[field], str) and value[field] for field in ("assertion_id", "unit_id")),
                    f"Malformed {name} pair ID")
            identity = value["row"], value["assertion_id"], value["unit_id"]
            require(identity not in selected and identity in baseline_relevant, f"Duplicate or unknown {name}: {identity}")
            selected.add(identity)
        expected[name] = selected
    require(expected["must_retain"], "At least one useful context pair must be named")
    require(not expected["must_retain"] & expected["must_drop"], "Contradictory expectations")
    require(expected["must_retain"] <= candidate_relevant, "A required useful pair was withheld")
    require(expected["must_drop"] <= moved | audited_drops, "A required distraction remains contextual")
    return {"ok": True, "claim_rows": len(original_rows), "baseline_context": len(baseline_relevant),
            "candidate_context": len(candidate_relevant), "withheld": len(moved),
            "audited_drops": len(audited_drops),
            "must_retain": len(expected["must_retain"]), "must_drop": len(expected["must_drop"]),
            "scope": "Saved-card preservation and explicit expectations only; not an accuracy evaluation."}
