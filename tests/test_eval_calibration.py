"""
The reporting script's arithmetic, and the guard that it stays a reporting script.

eval_calibration is allowed to read test -- that is its job. What it must never do is fit
anything, because a threshold nudged until its promise holds on test is a promise about nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.eval_calibration import NEI_INDEX, macro_f1, per_claim_brier
from src.verdict.encode import LABELS
from src.verdict.labels import Verdict

SCRIPT = Path("scripts/eval_calibration.py")


def test_macro_f1_is_the_unweighted_mean_over_classes():
    """
    Deliberately lopsided support: a weighted mean and an unweighted one coincide on balanced
    fixtures, which is how a vacuous macro-F1 test slipped into this project once already.
    """
    labels = np.array([0] * 90 + [1] * 5 + [2] * 5)
    predictions = np.array([0] * 90 + [1] * 5 + [0] * 5)   # class 2 never predicted

    macro, per_class = macro_f1(labels, predictions)
    assert per_class[LABELS[2]] == 0.0
    assert per_class[LABELS[1]] == pytest.approx(1.0)
    # supported: precision 90/95, recall 1.0
    expected_supported = 2 * (90 / 95) * 1.0 / ((90 / 95) + 1.0)
    assert per_class[LABELS[0]] == pytest.approx(expected_supported)
    assert macro == pytest.approx((expected_supported + 1.0 + 0.0) / 3)


def test_macro_f1_is_perfect_only_when_every_class_is():
    labels = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(labels, labels)[0] == pytest.approx(1.0)


def test_a_class_with_no_support_and_no_prediction_scores_zero_not_nan():
    """An unpredicted, unobserved class must not turn the macro average into NaN."""
    labels = np.array([0, 0, 0])
    macro, per_class = macro_f1(labels, np.array([0, 0, 0]))
    assert np.isfinite(macro)
    assert per_class[LABELS[1]] == 0.0


def test_per_claim_brier_matches_the_hand_computed_value():
    probabilities = np.array([[0.7, 0.2, 0.1]])
    # (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09 + 0.04 + 0.01
    assert per_claim_brier(probabilities, np.array([0]))[0] == pytest.approx(0.14)


def test_per_claim_brier_is_zero_when_confident_and_right():
    probabilities = np.array([[1.0, 0.0, 0.0]])
    assert per_claim_brier(probabilities, np.array([0]))[0] == pytest.approx(0.0)


def test_per_claim_brier_returns_one_value_per_claim():
    """It feeds paired_bootstrap, which needs per-claim values rather than an aggregate."""
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet(np.ones(3), size=50)
    assert per_claim_brier(probabilities, rng.integers(0, 3, size=50)).shape == (50,)


def test_the_nei_index_tracks_the_label_space():
    """
    Hard-coding 2 would survive a reorder of LABELS and silently cross-tabulate the wrong class
    against abstention, which is the one distinction this script exists to keep straight.
    """
    assert LABELS[NEI_INDEX] == Verdict.NOT_ENOUGH_EVIDENCE.value


def test_the_reporting_script_never_writes_a_calibration_artifact():
    """
    The leak that would make every band promise circular: refitting anything against test. Fitting
    lives in scripts/calibrate.py, and this script may only read what that froze.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "fit_bands" not in source
    assert ".fit(" not in source
    # It reads the frozen artifact; it must not write one back.
    assert 'calibration.json").write_text' not in source
    assert "calibration.json" in source
