"""
The transfer measurement, and the guard that stopped it becoming an overclaim.

This script exists because a lead was being carried as prose. The demo declines claims at FEVER's
rate while retrieving visibly unrelated evidence, and the tempting move is to write that up as
"the sufficiency gate does not transfer". The point estimate even supports it: AUC 0.5797 against
0.7485.

The interval does not. It spans both chance and the FEVER number, so at 40 claims the sample cannot
distinguish the two hypotheses -- and the script has to say so rather than reporting the point
estimate as a result. That is the property most worth guarding here, because the failure it prevents
is the one this whole repository is organised against: a number that reads like a finding and is
not.

What the labels do establish, without depending on the AUC at all, is that 35% of retrieval on this
transcript returns anything usable. That is a direct count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.selective import roc_auc

SCRIPT = Path("scripts/measure_sufficiency_transfer.py")
LABELS = Path("labels/relevance-sotu-2016.json")
ARTIFACT = Path("runs/sufficiency-transfer/metrics.json")


def truth() -> dict:
    return json.loads(LABELS.read_text(encoding="utf-8"))


# --- the labels ---------------------------------------------------------------------------------

def test_the_labels_carry_a_rubric_that_distinguishes_evidence_from_a_keyword_collision():
    """
    The whole judgement is "did retrieval find the subject", and the rubric has to draw that line
    explicitly -- shared vocabulary is exactly what BM25 produces when it fails here.
    """
    rubric = truth()["rubric"].lower()
    assert "keyword collision" in rubric
    assert "merely shares vocabulary is not relevant" in rubric


def test_the_sample_is_reproducible_from_its_description():
    """A sample nobody can rebuild is a sample nobody can check."""
    described = truth()["sample"]
    assert "random.Random(20260904)" in described
    assert "40" in described


def test_every_label_is_a_boolean_keyed_by_sentence_index():
    for key, value in truth()["labels"].items():
        assert key.isdigit()
        assert isinstance(value, bool)


def test_the_labels_are_not_all_one_class():
    """An AUC needs both classes; all-relevant or all-irrelevant would make it undefined."""
    values = list(truth()["labels"].values())
    assert 0 < sum(values) < len(values)


def test_the_stated_limitation_names_the_bias_it_could_not_avoid():
    """
    The scores were visible while labelling. That is a real bias risk pointing toward agreement,
    and a reader has to be told rather than left to assume a blind protocol.
    """
    limitation = truth()["limitation"].lower()
    assert "visible" in limitation
    assert "blind" in limitation or "second annotator" in limitation


def test_the_stated_limitation_does_not_claim_a_finding():
    """
    An earlier draft of this file said the separation was "near chance", which reads as a result.
    The interval spans chance and the FEVER number, so the honest word is inconclusive.
    """
    limitation = truth()["limitation"].lower()
    assert "inconclusive" in limitation


# --- the measurement ----------------------------------------------------------------------------

def test_the_script_reports_an_interval_and_not_only_a_point_estimate():
    """
    0.5797 alone reads as degradation. The interval is what turns it into a statement about what
    this sample can and cannot support.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "bootstrap_ci(" in source
    assert "interval.low" in source and "interval.high" in source


def test_the_verdict_is_decided_by_the_interval_rather_than_the_point_estimate():
    """
    The guard against the overclaim. A branch on `auc < 0.65` would print "does not transfer" for
    an estimate whose interval reaches 0.755.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "clears_chance = interval.low > 0.5" in source
    assert "below_fever = interval.high < FEVER_AUC" in source
    assert "auc < 0.65" not in source, "no verdict from a bare threshold on the point estimate"


def test_an_inconclusive_result_says_so_in_those_words():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "INCONCLUSIVE" in source
    assert "cannot distinguish" in source


def test_labels_that_do_not_match_the_run_are_a_hard_error():
    """
    Re-running the demo with a different filter changes which sentences are claims. Scoring old
    labels against new indices would produce a number for pairings nobody made.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "raise SystemExit(" in source
    assert "set(labelled) - set(scored)" in source


def test_the_fever_baseline_is_the_number_phase_04_actually_reported():
    from scripts.measure_sufficiency_transfer import FEVER_AUC

    assert FEVER_AUC == 0.7485


def test_a_bootstrap_draw_of_one_class_has_no_auc_to_report():
    """
    At 14 positives in 40, resampling lands an all-one-class draw often enough to matter. An AUC
    of 0.0 or 1.0 folded into the percentiles would widen or shift the interval for no reason.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "if drawn.all() or not drawn.any():" in source
    assert 'return float("nan")' in source


# --- against the artifact -------------------------------------------------------------------------

def test_the_reported_auc_matches_the_labels_and_the_run():
    if not ARTIFACT.exists():
        pytest.skip("transfer measurement not run")
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    demo = json.loads(Path("runs/demo/verdicts.json").read_text(encoding="utf-8"))
    scored = {r["index"]: r for r in demo["rows"] if r.get("outcome")}

    labelled = {int(i): v for i, v in truth()["labels"].items()}
    order = sorted(labelled)
    relevant = np.array([labelled[i] for i in order], dtype=bool)
    sufficiency = np.array([scored[i]["sufficiency"] for i in order], dtype=np.float64)

    assert report["auc"]["relevance_here"] == pytest.approx(roc_auc(sufficiency, relevant))
    assert report["relevant"] == int(relevant.sum())


def test_the_recorded_interval_still_spans_chance():
    """
    If a re-run ever narrows this below 0.5, the prose calling it inconclusive has to change with
    it. Pinning the relationship rather than the number is what keeps the two in step.
    """
    if not ARTIFACT.exists():
        pytest.skip("transfer measurement not run")
    low, high = json.loads(ARTIFACT.read_text(encoding="utf-8"))["auc"]["relevance_here_ci"]
    assert low <= 0.5 <= high, "the interval no longer spans chance; update the write-up"
