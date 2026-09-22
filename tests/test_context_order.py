"""Ordering preserves every verdict decision and literal source record."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.verdict.chain import check_claim
from src.verdict.context_order import ContextOrderedJudge, ContextOrderingUnavailable
from src.verdict.context_review import ContextReviewedJudge


class Primary:
    measurement = {"counted": 7}
    direction_measurement = {"directional": 3}

    def __call__(self, assertions, units):
        return [assertions, units]

    def review_composed(self, verdict, assertions, units):
        return verdict | {"reviewed": True}


def fixture():
    assertions = [{"id": "a1", "text": "The water is not safe."},
                  {"id": "a2", "text": "The water contains a contaminant."}]
    units = [{"id": name, "text": f"{name} [reported](https://example.org/{name}) a result.",
              "definitions": [{"text": "The [rate](https://example.org/rate) measures contamination.", "id": "d1"}]}
             for name in ["first", "second", "third", "concept"]]
    rows = [{"unit_id": unit["id"], "text": unit["text"], "span": unit["text"], "definitions": unit["definitions"],
             "relation": "bears_on", "confidence": 0.99 - index / 10,
             "origin": "context" if unit["id"] == "concept" else "assertion",
             "qualifiers": ["the city"], "source_id": "s1", "url": "https://example.org/report",
             "published_at": "2024-01-01", "context_review": {"relation": "bears_on"}}
            for index, unit in enumerate(units)]
    verdict = {"summary": "unresolved", "relationship": "insufficient", "rules": ["existing rule"],
               "assertions": [assertions[0] | {"relevant": rows, "evidence": [{"unit_id": "counted"}],
                                              "sources_for": 1, "sources_against": 0,
                                              "withheld_context": [{"unit_id": "withheld"}]},
                              assertions[1] | {"relevant": [deepcopy(rows[0])], "evidence": [], "sources_for": 0}]}
    return assertions, units, verdict


def test_delegates_both_judge_stages_and_measurements():
    primary = Primary()
    judge = ContextOrderedJudge(primary, lambda rows: [], model="fixture")
    assertions, units, verdict = fixture()
    assert judge.measurement is primary.measurement
    assert judge.direction_measurement is primary.direction_measurement
    assert judge(assertions, units) == [assertions, units]
    assert judge.review_composed(verdict, assertions, units) == verdict | {"reviewed": True}
    without_review = ContextOrderedJudge(lambda a, u: [], lambda rows: [], model="fixture")
    assert without_review.review_composed(verdict, assertions, units) is verdict


def test_orders_only_context_with_origin_priority_and_stable_ties():
    assertions, units, verdict = fixture()
    original = deepcopy(verdict)
    seen = []

    def score(rows):
        seen.extend(rows)
        return [0.1, 0.8, 0.8, 1.0, 0.4]

    ordered = ContextOrderedJudge(Primary(), score, model="frozen-artifact").order_context(verdict, assertions, units)
    assert [r["unit_id"] for r in ordered["assertions"][0]["relevant"]] == ["second", "third", "first", "concept"]
    assert [row["hypothesis"] for row in seen] == [assertions[0]["text"]] * 4 + [assertions[1]["text"]]
    assert seen[0] == {"hypothesis": "The water is not safe.", "visible_sentence": "first reported a result.",
                       "visible_definitions": ["The rate measures contamination."]}
    assert ordered["assertions"][1]["relevant"][0]["context_relevance_score"] == 0.4
    assert ordered["context_ordering"]["model"] == "frozen-artifact"
    assert verdict == original, "the input audit must remain intact"
    restored = deepcopy(ordered)
    restored.pop("context_ordering")
    for before, after in zip(original["assertions"], restored["assertions"], strict=True):
        by_id = {row["unit_id"]: row for row in after["relevant"]}
        after["relevant"] = [by_id[row["unit_id"]] for row in before["relevant"]]
        for row in after["relevant"]:
            row.pop("context_relevance_score")
    assert restored == original, "removing ordering/audit additions must restore every original field"


def test_empty_context_never_loads_scorer_and_errors_remain_visible():
    assertions, units, verdict = fixture()

    def fail(rows):
        raise RuntimeError("artifact unavailable")

    judge = ContextOrderedJudge(Primary(), fail, model="fixture")
    with pytest.raises(RuntimeError, match="artifact unavailable"):
        judge.order_context(verdict, assertions, units)
    for assertion in verdict["assertions"]:
        assertion["relevant"] = []
    assert judge.order_context(verdict, assertions, units) is verdict


def test_chain_orders_after_review_without_changing_counted_or_withheld_rows():
    fact = "Solar adoption reached 25% in 2024."
    context = "Solar adoption reached 24% in 2024."
    earlier = "In 2023, solar adoption reached 22%."
    noise = "The rail operator carried six million passengers."
    plan = {"claim_scope": {"country": "US"}, "sources": [
        {"id": "s1", "url": "https://example.org/current", "published_at": "2024-06-01", "excerpts": [fact, context]},
        {"id": "s2", "url": "https://example.org/earlier", "published_at": "2024-06-01", "excerpts": [earlier]},
        {"id": "s3", "url": "https://example.org/trains", "published_at": "2024-06-01", "reading_passages": [noise]}]}

    class Judge:
        root = "fixture"
        measurement = direction_measurement = None

        def __init__(self, reviewer=False):
            self.reviewer = reviewer

        def __call__(self, assertions, units):
            return [{"assertion_id": a["id"], "unit_id": u["id"], "relation": relation,
                     "span": "" if relation == "unrelated" else u["text"], "qualifiers": [], "confidence": 0.9}
                    for a in assertions for u in units for relation in [
                        ("unrelated" if u["text"] == noise else "bears_on") if self.reviewer else
                        ("bears_on" if u["text"] == context else "states")]]

    reviewed = ContextReviewedJudge(Judge(), Judge(reviewer=True))
    before = check_claim(fact, plan, reviewed)
    old = before["verdict"]["assertions"][0]
    assert {r["text"] for r in old["relevant"]} == {context, earlier}

    class Scorer:
        providers = ["CPUExecutionProvider"]

        def __call__(self, rows):
            assert {r["visible_sentence"] for r in rows} == {context, earlier}
            return [float(r["visible_sentence"] == earlier) for r in rows]

    after = check_claim(fact, plan, ContextOrderedJudge(reviewed, Scorer(), model="fixture"))
    new = after["verdict"]["assertions"][0]
    assert [r["text"] for r in new["relevant"]] == [earlier, context]
    assert new["evidence"] == old["evidence"] and new["withheld_context"] == old["withheld_context"]
    assert before["verdict"]["summary"] == after["verdict"]["summary"]
    assert after["text"].index(earlier) < after["text"].index(context)
    assert after["verdict"]["context_ordering"]["providers"] == ["CPUExecutionProvider"]


@pytest.mark.parametrize("reason", ["Ranking artifacts unavailable.", "Complete input exceeds the token limit."])
def test_unavailable_ranking_preserves_all_context_in_original_order_with_note(reason):
    assertions, units, verdict = fixture()
    original = deepcopy(verdict)

    def unavailable(rows):
        raise ContextOrderingUnavailable(reason)

    result = ContextOrderedJudge(Primary(), unavailable, model="fixture").order_context(verdict, assertions, units)
    assert result.pop("context_ordering") == {"model": "fixture", "status": "not_applied", "note": reason}
    assert result == verdict == original


@pytest.mark.parametrize("scores", [[0.2], [0.2] * 6, [float("nan")] * 5, [float("inf")] * 5,
                                    [-0.1] * 5, [1.1] * 5])
def test_invalid_score_vector_does_not_mutate_verdict(scores):
    assertions, units, verdict = fixture()
    original = deepcopy(verdict)
    with pytest.raises(ValueError, match="one finite relevance score"):
        ContextOrderedJudge(Primary(), lambda rows: scores, model="fixture").order_context(verdict, assertions, units)
    assert verdict == original


@pytest.mark.parametrize("change", ["assertion", "sentence", "definition", "counted", "duplicate"])
def test_cannot_rank_a_different_pair_or_counted_evidence(change):
    assertions, units, verdict = fixture()
    verdict = deepcopy(verdict)
    row = verdict["assertions"][0]["relevant"][0]
    if change == "assertion":
        verdict["assertions"][0]["text"] = "A different claim."
    elif change == "sentence":
        row["text"] = "A different sentence."
    elif change == "definition":
        row["definitions"] = []
    elif change == "counted":
        row["relation"] = "states"
    else:
        verdict["assertions"][0]["relevant"].append(deepcopy(row))
    with pytest.raises(ValueError, match="Context ordering"):
        ContextOrderedJudge(Primary(), lambda rows: pytest.fail("invalid input reached model"),
                            model="fixture").order_context(verdict, assertions, units)
