"""
The held-out evaluation, and the interval that keeps 101 sentences from being over-read.

The SOTU comparison carries one bound it cannot shed: those labels and the hand-written rules share
an author. A second transcript labelled by the same person would add a sample and leave the bound
untouched. ClaimBuster's expert subset is what actually loosens it -- a different annotator pool
from the training labels and from the rubric, on sentences the crowd split does not contain.

It is 101 sentences, which is small enough that a bare point estimate invites over-reading, so the
accuracy carries a Wilson interval and the script says the size out loud.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.eval_detector import prf
from src.eval.intervals import wilson

SCRIPT = Path("scripts/eval_detector.py")


def test_precision_and_recall_are_not_interchangeable():
    predicted = np.array([True] * 3 + [True] * 5 + [False] * 2)
    actual = np.array([True] * 3 + [False] * 5 + [True] * 2)
    got = prf(predicted, actual)
    assert got["precision"] == pytest.approx(0.375)
    assert got["recall"] == pytest.approx(0.6)


def test_predicting_nothing_scores_zero_rather_than_perfect_precision():
    got = prf(np.zeros(4, dtype=bool), np.array([True, True, False, False]))
    assert got["precision"] == 0.0
    assert got["f1"] == pytest.approx(0.0)


def test_the_wilson_interval_brackets_the_estimate():
    low, high = wilson(80, 101)
    assert low < 80 / 101 < high


def test_the_interval_stays_inside_zero_and_one_at_the_edges():
    """
    The reason for Wilson over a normal interval: a small sample near 100% would otherwise report
    an upper bound above 1, which reads as a broken number in a report meant to be careful.
    """
    low, high = wilson(101, 101)
    assert 0.0 <= low <= 1.0
    assert high <= 1.0


def test_a_smaller_sample_gives_a_wider_interval():
    """The whole point of showing it: 101 sentences buys less certainty than 3,509."""
    narrow = wilson(2800, 3509)
    wide = wilson(80, 101)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_an_empty_sample_does_not_divide_by_zero():
    low, high = wilson(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_both_label_sources_are_scored():
    """
    Reporting only the crowd labels would leave the annotator question open, which is the entire
    reason this script exists alongside the SOTU comparison.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'for source in ("test", "groundtruth"):' in source
    assert "expert annotators" in source


def test_the_rules_are_scored_beside_the_detector():
    """
    If the detector's advantage were an artifact of the SOTU labels agreeing with ClaimBuster's
    notion of check-worthy, it would shrink here. Scoring only the detector could not show that.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "check_worthy(t).worthy for t in texts" in source
    assert '"hand-written rules"' in source


def test_the_sample_size_is_stated_rather_than_rounded_off():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "101 sentences" in source
    assert "not a larger sample" in source


def test_the_threshold_comes_from_the_frozen_filter():
    """Evaluating at a threshold other than the pipeline's would measure a different system."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "DetectorFilter(args.binarization)" in source
    assert "learned.threshold" in source
