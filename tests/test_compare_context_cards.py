"""Saved-card comparisons preserve the original reading while testing reviewed context."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.eval.context_cards import compare_records


def record() -> dict:
    def context(unit: str, text: str) -> dict:
        return {"unit_id": unit, "source_id": "source-1", "passage_id": "source-1:p1",
                "text": text, "definitions": [], "span": text, "qualifiers": [],
                "relation": "bears_on", "confidence": 0.8, "url": "https://example.org/one"}

    useful = context("source-1:p1:u1", "A definition of the claimed measure.")
    distraction = context("source-1:p1:u2", "A neighboring fact about another measure.")
    return {"payload": {"video_id": "abcdefghijk", "captions": [{"text": "Claim."}]},
            "result": {"video_id": "abcdefghijk", "excerpt": "Claim.", "rows": [{"text": "Claim.", "result": {
                "claim": "Claim.", "research": {"sources": [{"id": "source-1", "url": "https://example.org/one"}]},
                "verdict": {"status": "unverified"},
                "decomposed": {"claim": "Claim.",
                    "assertions": [{"id": "assertion-1", "text": "Claim."}],
                    "gate": {"units": [{"id": useful["unit_id"], "text": useful["text"]},
                                       {"id": distraction["unit_id"], "text": distraction["text"]}],
                             "eligible_ids": [useful["unit_id"], distraction["unit_id"]]},
                    "judgments": [{"assertion_id": "assertion-1", **useful},
                                  {"assertion_id": "assertion-1", **distraction}],
                    "text": "Claim: Claim.\nContext: both rows.",
                    "verdict": {"status": "draft", "summary": "unresolved", "relationship": "insufficient",
                                "assertions": [{"id": "assertion-1", "text": "Claim.", "status": "insufficient",
                                                "sources_for": 0, "sources_against": 0, "evidence": [],
                                                "relevant": [useful, distraction]}]}}}}]}}


def expected(*, drop: bool = False) -> dict:
    useful = {"row": 0, "assertion_id": "assertion-1", "unit_id": "source-1:p1:u1"}
    distraction = {"row": 0, "assertion_id": "assertion-1", "unit_id": "source-1:p1:u2"}
    return {"must_retain": [useful], "must_drop": [distraction] if drop else []}


def verdict(record: dict) -> dict:
    return record["result"]["rows"][0]["result"]["decomposed"]["verdict"]


def reading(record: dict) -> dict:
    return record["result"]["rows"][0]["result"]["decomposed"]


def test_self_comparison_and_context_reorder_pass() -> None:
    baseline = record()
    candidate = deepcopy(baseline)
    assert compare_records(baseline, candidate, expected())["ok"]
    assertion = verdict(candidate)["assertions"][0]
    assertion["relevant"].reverse()
    assertion["relevant"][0]["context_relevance_score"] = 0.2
    assertion["relevant"][1]["context_relevance_score"] = 0.9
    verdict(candidate)["context_ordering"] = {"status": "applied"}
    reading(candidate)["text"] = "Claim: Claim.\nContext: reordered rows."
    assert compare_records(baseline, candidate, expected())["candidate_context"] == 2


def test_postcomposition_withholding_and_explicit_expectation_pass() -> None:
    baseline = record()
    candidate = deepcopy(baseline)
    assertion = verdict(candidate)["assertions"][0]
    dropped = assertion["relevant"].pop()
    dropped.update(relation="unrelated", span="", qualifiers=[],
                   context_review={"prior_relation": "bears_on", "relation": "unrelated"})
    assertion["withheld_context"] = [dropped]
    report = compare_records(baseline, candidate, expected(drop=True))
    assert report["withheld"] == 1 and report["candidate_context"] == 1
    del dropped["context_review"]
    with pytest.raises(ValueError, match="lacks an unrelated secondary reading"):
        compare_records(baseline, candidate, expected(drop=True))


def test_precomposition_audited_drop_passes_but_silent_drop_fails() -> None:
    baseline = record()
    candidate = deepcopy(baseline)
    verdict(candidate)["assertions"][0]["relevant"].pop()
    judgment = reading(candidate)["judgments"][1]
    judgment.update(relation="unrelated", span="", qualifiers=[],
                    context_review={"prior_relation": "bears_on", "relation": "unrelated"})
    assert compare_records(baseline, candidate, expected(drop=True))["audited_drops"] == 1
    del judgment["context_review"]
    with pytest.raises(ValueError, match="without a secondary unrelated reading"):
        compare_records(baseline, candidate, expected(drop=True))


def test_useful_retention_and_distraction_drop_are_independent_gates() -> None:
    baseline = record()
    candidate = deepcopy(baseline)
    useful = verdict(candidate)["assertions"][0]["relevant"].pop(0)
    useful.update(relation="unrelated", span="", qualifiers=[],
                  context_review={"prior_relation": "bears_on", "relation": "unrelated"})
    verdict(candidate)["assertions"][0]["withheld_context"] = [useful]
    with pytest.raises(ValueError, match="required useful pair"):
        compare_records(baseline, candidate, expected())
    with pytest.raises(ValueError, match="required distraction"):
        compare_records(baseline, baseline, expected(drop=True))


@pytest.mark.parametrize("change", [
    lambda item: item["result"]["rows"][0]["result"]["decomposed"]["verdict"].update(summary="established"),
    lambda item: item["result"]["rows"][0]["result"]["decomposed"]["verdict"]["assertions"][0].update(sources_for=1),
    lambda item: item["result"]["rows"][0]["result"]["research"].update(sources=[]),
    lambda item: item["result"]["rows"][0]["result"]["decomposed"]["gate"]["units"][0].update(text="rewritten"),
    lambda item: item["result"]["rows"][0]["result"]["decomposed"]["verdict"]["assertions"][0]["relevant"][0].update(text="rewritten"),
    lambda item: item["payload"].update(time=12),
])
def test_count_source_text_and_payload_changes_fail(change) -> None:
    baseline, candidate = record(), record()
    change(candidate)
    with pytest.raises(ValueError):
        compare_records(baseline, candidate, expected())


def test_delete_everything_duplicate_and_missing_expectations_fail() -> None:
    baseline, candidate = record(), record()
    verdict(candidate)["assertions"][0]["relevant"] = []
    with pytest.raises(ValueError, match="without a secondary unrelated reading"):
        compare_records(baseline, candidate, expected())
    candidate = record()
    verdict(candidate)["assertions"][0]["relevant"].append(
        deepcopy(verdict(candidate)["assertions"][0]["relevant"][0]))
    with pytest.raises(ValueError, match="Duplicate context pair"):
        compare_records(baseline, candidate, expected())
    with pytest.raises(ValueError, match="Duplicate or unknown"):
        compare_records(baseline, baseline, {"must_retain": [expected()["must_retain"][0]] * 2,
                                             "must_drop": []})


def test_malformed_matching_records_do_not_pass() -> None:
    malformed = {"payload": {}, "result": {"rows": [{"text": "Claim.", "result": {
        "decomposed": {"verdict": {"assertions": [{"id": "a", "relevant": [{"unit_id": "u"}]}]}}}}]}}
    with pytest.raises(ValueError, match="Missing original assertions"):
        compare_records(malformed, malformed, {"must_retain": [{"row": 0, "assertion_id": "a", "unit_id": "u"}],
                                               "must_drop": []})


def test_matching_bogus_context_or_foreign_judgment_does_not_pass() -> None:
    bogus = record()
    verdict(bogus)["assertions"][0]["relevant"][0]["text"] = "Unsourced rewrite."
    with pytest.raises(ValueError, match="differs from source unit"):
        compare_records(bogus, bogus, expected())

    bogus = record()
    verdict(bogus)["assertions"][0]["relevant"][0]["url"] = "https://wrong.example/"
    with pytest.raises(ValueError, match="differs from saved research source"):
        compare_records(bogus, bogus, expected())

    bogus = record()
    reading(bogus)["judgments"].append({"assertion_id": "foreign", "unit_id": "source-1:p1:u1",
                                           "relation": "bears_on"})
    with pytest.raises(ValueError, match="Foreign original judgment"):
        compare_records(bogus, bogus, expected())

    bogus = record()
    verdict(bogus)["assertions"][0]["relevant"][0]["relation"] = "unrelated"
    with pytest.raises(ValueError, match="Wrong relevant relation"):
        compare_records(bogus, bogus, expected())

    bogus = record()
    reading(bogus)["judgments"] = []
    with pytest.raises(ValueError, match="no eligible source unit or original judgment"):
        compare_records(bogus, bogus, expected())

    bogus = record()
    verdict(bogus)["assertions"][0]["text"] = "Different assertion."
    with pytest.raises(ValueError, match="Verdict assertion text differs"):
        compare_records(bogus, bogus, expected())


def test_primary_judgment_changes_fail() -> None:
    baseline, candidate = record(), record()
    reading(candidate)["judgments"][0]["confidence"] = 0.1
    with pytest.raises(ValueError, match="Primary judgment changed"):
        compare_records(baseline, candidate, expected())


@pytest.mark.parametrize("bad_ids", [["source-1:p1:u1"] * 2, ["source-1:p1:u1", "foreign"]])
def test_matching_malformed_eligible_ids_fail(bad_ids) -> None:
    bogus = record()
    reading(bogus)["gate"]["eligible_ids"] = bad_ids
    with pytest.raises(ValueError, match="Malformed eligible source IDs"):
        compare_records(bogus, bogus, expected())
