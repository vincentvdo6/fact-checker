"""Context review can only withhold context, never alter the verdict's evidence."""

from __future__ import annotations

import pytest

from src.verdict.chain import check_claim
from src.verdict.context_review import ContextReviewedJudge


class Judge:
    def __init__(self, classify):
        self.classify, self.calls = classify, []
        self.root = "fixture"
        self.measurement = {"counted": 7}
        self.direction_measurement = {"directional": 3}

    def __call__(self, assertions, units):
        self.calls.append((assertions, units))
        return [{"assertion_id": assertion["id"], "unit_id": unit["id"],
                 "relation": relation, "raw_label": relation, "read_as": relation,
                 "span": unit["text"] if relation != "unrelated" else "", "qualifiers": [],
                 "confidence": 0.8, "status": "unverified"}
                for assertion in assertions for unit in units
                for relation in [self.classify(assertion, unit)]]


def test_review_only_removes_context_and_preserves_counts_and_original_inputs():
    assertions = [{"id": "a1", "text": "Population rose."}, {"id": "a2", "text": "Costs fell."}]
    units = [{"id": name, "text": name, "definitions": [{"text": "Same paragraph definition."}]}
             for name in ["support", "contradiction", "context", "noise"]]
    primary = Judge(lambda a, u: {"support": "states", "contradiction": "states_negation",
                                 "context": "bears_on", "noise": "unrelated"}[u["id"]])
    reviewer = Judge(lambda a, u: "unrelated" if a["id"] == "a1" else "states")
    judge = ContextReviewedJudge(primary, reviewer)
    before = primary(assertions, units)
    rows = judge(assertions, units)
    assert judge.measurement is primary.measurement and judge.direction_measurement is primary.direction_measurement
    for old, row in zip(before, rows, strict=True):
        if old["relation"] != "bears_on":
            assert row == old, "all counted and already-unrelated pairs stay identical"
        elif row["assertion_id"] == "a1":
            assert row["relation"] == "unrelated" and row["span"] == "" and row["qualifiers"] == []
            assert row["raw_label"] == row["read_as"] == row["context_review"]["prior_relation"] == "bears_on"
        else:
            assert row["relation"] == "bears_on" and row["span"] == "context", "review support cannot promote context"
    assert len(reviewer.calls) == 2
    assert all(seen_units == [units[2]] for _, seen_units in reviewer.calls), "definitions and source text stay intact"
    assert [a[0]["id"] for a, _ in reviewer.calls] == ["a1", "a2"]


def test_no_context_does_not_load_or_call_the_second_model():
    primary = Judge(lambda a, u: "states")
    reviewer = Judge(lambda a, u: pytest.fail("unnecessary second-model call"))
    assert ContextReviewedJudge(primary, reviewer)([{"id": "a"}], [{"id": "u", "text": "A fact."}])[0]["relation"] == "states"
    assert reviewer.calls == []


def test_reviewer_failure_is_reported_instead_of_silently_removing_evidence():
    primary = Judge(lambda a, u: "bears_on")

    def fail(a, u):
        raise RuntimeError("Context model failed")

    with pytest.raises(RuntimeError, match="Context model failed"):
        ContextReviewedJudge(primary, Judge(fail))([{"id": "a"}], [{"id": "u", "text": "A fact."}])


def test_reviewed_context_leaves_the_card_but_counted_evidence_and_summary_stay():
    claim = "25% of households installed solar panels."
    fact = "The survey found that 25% of households installed solar panels."
    other = "The project reduced maintenance costs by 24%."
    plan = {"claim_scope": {"country": "US"}, "sources": [{"id": "s1", "url": "https://example.org/report",
            "published_at": "2024-06-01", "excerpts": [fact, other]}]}
    primary = Judge(lambda a, u: "states" if u["text"] == fact else "bears_on")
    primary.measurement = primary.direction_measurement = None
    baseline = check_claim(claim, plan, primary)
    checked = check_claim(claim, plan, ContextReviewedJudge(primary, Judge(lambda a, u: "unrelated")))
    assert other in baseline["text"] and other not in checked["text"]
    assert baseline["verdict"]["summary"] == checked["verdict"]["summary"] == "established"
    assert baseline["verdict"]["assertions"][0]["evidence"] == checked["verdict"]["assertions"][0]["evidence"]
    assert any(u["text"] == other for u in checked["gate"]["units"]), "withheld source text remains in the audit"


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong_assertion", "wrong_unit"])
def test_misaligned_reviewer_results_cannot_hide_another_sentence(mode):
    primary = Judge(lambda a, u: "bears_on")
    reviewer = Judge(lambda a, u: "unrelated")

    def broken(assertions, units):
        rows = reviewer(assertions, units)
        if mode == "missing":
            return []
        if mode == "duplicate":
            return rows * 2
        return [rows[0] | {"assertion_id" if mode == "wrong_assertion" else "unit_id": "other"}]

    class Broken:
        root = "fixture"
        __call__ = staticmethod(broken)

    with pytest.raises(ValueError, match="exactly"):
        ContextReviewedJudge(primary, Broken())([{"id": "a"}], [{"id": "u", "text": "A fact."}])
