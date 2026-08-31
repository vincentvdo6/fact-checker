"""
The pre-registered comparison: fair on both sides, and fitting nothing.

The result is a difference between two models, so every guard here is about the comparison being
like-for-like. The dangerous failures are quiet ones: reading one model through a calibrator and
the other raw, measuring groundedness from the model under test rather than from retrieval, or
quoting a baseline that was never what this script computes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.eval_regrounded import CONTROL, GOLD_ORACLE_NEI_F1, f1_per_class
from src.verdict.encode import LABELS

SCRIPT = Path("scripts/eval_regrounded.py")


def test_f1_is_computed_per_class_by_hand():
    """
    Five claims, lopsided on purpose: a balanced fixture makes a per-class F1 and a micro average
    coincide, which is how a vacuous metric test gets written.
    """
    labels = np.array([0, 0, 0, 1, 2])
    predicted = np.array([0, 0, 1, 1, 0])

    f1 = f1_per_class(labels, predicted)
    # supported: tp 2, fp 1, fn 1 -> p 2/3, r 2/3
    assert f1[LABELS[0]] == pytest.approx(2 / 3)
    # contradicted: tp 1, fp 1, fn 0 -> p 1/2, r 1
    assert f1[LABELS[1]] == pytest.approx(2 / 3)
    # not_enough_evidence: never predicted
    assert f1[LABELS[2]] == 0.0


def test_a_class_that_is_absent_and_unpredicted_scores_zero_not_nan():
    labels = np.array([0, 0])
    f1 = f1_per_class(labels, np.array([0, 0]))
    assert all(np.isfinite(v) for v in f1.values())
    assert f1[LABELS[1]] == 0.0


def test_the_quoted_control_is_the_calibrated_one():
    """
    Both models are read through their calibrator, so quoting Phase 02's raw 0.7015 and 0.5728
    beside calibrated numbers would credit the intervention with the calibrator's +2.7 accuracy
    and +7.4 NEI F1 -- larger than the effect being measured.
    """
    assert CONTROL["accuracy"] == 0.7285
    assert CONTROL["nei_f1"] == 0.6463
    assert GOLD_ORACLE_NEI_F1 == 0.8915


def test_the_comparison_refuses_a_calibration_mismatch():
    """
    The unfairness that would not announce itself. Reading one model through a calibrator and the
    other raw makes the difference mostly Phase 03's work, in whichever direction.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "calibration mismatch" in source
    assert "if len(set(calibrated.values())) > 1:" in source


def test_groundedness_is_measured_from_retrieval_not_from_the_model_under_test():
    """
    Whether gold was retrieved is a property of the evidence both models saw. Measuring it from
    each model's own budgets would give the two different subsets and make the columns describe
    different claims.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "control_budgets = models[args.control][\"budgets\"]" in source
    assert "zip(rows, control_budgets, strict=True)" in source


def test_the_comparison_fits_nothing():
    """The calibrator and the sufficiency model were frozen on splits this script never re-reads."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "fit_bands" not in source


def test_overall_accuracy_is_labelled_as_not_the_criterion():
    """
    It is expected to fall. Presenting it as the headline would reward the model that guesses from
    claim artifacts, which is the behaviour the whole phase exists to remove.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "reported, not judged" in source
