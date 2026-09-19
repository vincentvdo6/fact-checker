"""Pair-judge scoring counts the errors that move verdicts, and the judge refuses a wrong contract."""

from __future__ import annotations

import json

import pytest

from src.eval.pairs import confusion, report
from src.verdict.pair_judge import PairJudge

LABELS = ["states", "states", "states_negation", "bears_on", "bears_on", "unrelated", "unrelated", "unrelated"]
PREDICTED = ["states", "unrelated", "states", "bears_on", "states_negation", "unrelated", "unrelated", "bears_on"]


def test_counting_error_counts_only_wrong_states_predictions():
    scored = report(LABELS, PREDICTED)
    assert scored["n"] == 8 and scored["accuracy"] == 4 / 8
    # three predictions would count (states, states, states_negation); two of them are wrong
    assert scored["counting_error"] == {"wrong": 2, "counted": 3, "rate": 2 / 3}
    # three labels are counted relations; one was found
    assert scored["counted_recall"] == {"found": 1, "actual": 3, "rate": 1 / 3}
    assert scored["per_relation"]["states"]["support"] == 2 and scored["per_relation"]["states"]["predicted"] == 2
    assert scored["per_relation"]["states"]["precision"] == 0.5 and scored["per_relation"]["states"]["recall"] == 0.5
    assert scored["confusion"]["bears_on"]["states_negation"] == 1
    assert scored["per_relation"]["states_negation"]["f1"] == 0.0


def test_scoring_rejects_length_mismatch_and_unknown_relations():
    with pytest.raises(ValueError):
        confusion(["states"], [])
    with pytest.raises(ValueError):
        confusion(["states"], ["proves"])
    empty = report([], [])
    assert empty["n"] == 0 and empty["accuracy"] is None and empty["macro_f1"] is None


def contract(tmp_path, **overrides):
    payload = {"labels": ["states", "states_negation", "bears_on", "unrelated"], "max_length": 192,
               "template_id": "premise_hypothesis_v1", "variant": "pair_judge"}
    payload.update(overrides)
    (tmp_path / "contract.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_pair_judge_refuses_a_permuted_label_order_or_other_template(tmp_path):
    judge = PairJudge(model_dir=contract(tmp_path))
    assert judge.max_length == 192 and judge.calibrator is None and judge.bands is None
    assert judge([], []) == []
    with pytest.raises(ValueError, match="label order"):
        PairJudge(model_dir=contract(tmp_path, labels=["unrelated", "bears_on", "states_negation", "states"]))
    with pytest.raises(ValueError, match="template"):
        PairJudge(model_dir=contract(tmp_path, template_id="sentence_v1"))


def test_hedge_probe_judges_the_sentence_figure_in_the_claims_place_and_records_it(tmp_path, monkeypatch):
    import numpy as np

    judge = PairJudge(model_dir=contract(tmp_path), probe_hedges=True, probe_negation=True)
    seen = []

    def probabilities(pairs):
        seen.extend(pairs)
        return np.array([[0.9, 0.05, 0.05, 0.0] if "24.3%" in hypothesis else [0.1, 0.1, 0.8, 0.0]
                         for _, hypothesis in pairs])

    monkeypatch.setattr(judge, "probabilities", probabilities)
    assertions = [{"id": "a1", "text": "That's around 25% of the population.", "negated": False, "contrast": False},
                  {"id": "a2", "text": "We don't have around 25% of the population out of work.", "negated": True, "contrast": False}]
    units = [{"id": "u1", "text": "The true rate of unemployment in April was 24.3%.", "definitions": []},
             {"id": "u2", "text": "The TRU for Black workers rose to 26.7%.", "definitions": []}]
    rows = judge(assertions, units)
    by_key = {(row["assertion_id"], row["unit_id"]): row for row in rows}
    read = by_key[("a1", "u1")]
    assert read["judged_text"] == "That's around 24.3% of the population." and read["relation"] == "states"
    assert read["figures_read"] == [{"hedge": "around", "relation": "near", "claimed": "25%", "read_as": "24.3%"}]
    assert by_key[("a1", "u2")]["judged_text"] == "That's around 25% of the population.", "26.7% is outside the hedge"
    assert "figures_read" not in by_key[("a1", "u2")] and by_key[("a1", "u2")]["relation"] == "bears_on"
    denial = by_key[("a2", "u1")]
    assert denial["judged_text"] == "We have around 24.3% of the population out of work.", "positive form, then the hedge"
    assert denial["relation"] == "states_negation" and denial["raw_label"] == "states"
    plain = PairJudge(model_dir=contract(tmp_path))
    monkeypatch.setattr(plain, "probabilities", probabilities)
    assert all("figures_read" not in row and row["judged_text"] == "That's around 25% of the population."
               for row in plain(assertions[:1], units))


def test_hedge_probe_preserves_population_claim_against_a_relative_risk(tmp_path, monkeypatch):
    import numpy as np

    judge = PairJudge(model_dir=contract(tmp_path), probe_hedges=True)
    claim = "That's around 25% of the population."
    sentence = "Formerly incarcerated people are 24% [less likely](https://example.org/report) to return to prison."
    seen = []

    def probabilities(pairs):
        seen.extend(pairs)
        return np.array([[0.1, 0.1, 0.8, 0.0] for _ in pairs])

    monkeypatch.setattr(judge, "probabilities", probabilities)
    row, = judge([{"id": "a1", "text": claim, "negated": False}],
                 [{"id": "u1", "text": sentence, "definitions": []}])
    assert seen == [(sentence, claim)]
    assert row["judged_text"] == claim and "figures_read" not in row
    assert row["span"] == sentence and row["relation"] == "bears_on", "quantity kinds do not override the model's relation"


def test_threshold_sweep_counts_the_judges_reading_not_its_banded_answer():
    from scripts.eval_pair_judge import threshold_sweep

    labels = ["states", "states_negation", "bears_on", "unrelated", "states"]
    rows = [{"read_as": "states", "confidence": 0.95}, {"read_as": "states_negation", "confidence": 0.7},
            {"read_as": "states", "confidence": 0.8}, {"read_as": "unrelated", "confidence": 0.9},
            {"read_as": "bears_on", "confidence": 0.99}]
    sweep = {row["threshold"]: row for row in threshold_sweep(labels, rows)}
    assert sweep[0.5] == {"threshold": 0.5, "counted": 3, "wrong": 1, "error": 1 / 3, "recall": 2 / 3}
    assert sweep[0.75] == {"threshold": 0.75, "counted": 2, "wrong": 1, "error": 0.5, "recall": 1 / 3}
    assert sweep[0.9]["counted"] == 1 and sweep[0.9]["wrong"] == 0 and sweep[0.9]["recall"] == 1 / 3
    assert threshold_sweep(labels, [{"relation": "states"}] * 5) == [], "no readings recorded, no sweep"
