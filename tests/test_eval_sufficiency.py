"""
The test-split report: it may read test, and it may not fit anything on it.

The criterion this script checks was fixed in the plan before the split was opened. What the tests
guard is that the script cannot quietly become the thing that decides the criterion -- by fitting a
threshold, by moving the baseline it compares against, or by counting accuracy over claims it
declined to answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.eval_sufficiency import PHASE_03

SCRIPT = Path("scripts/eval_sufficiency.py")


def test_the_reporting_script_fits_nothing():
    """
    Every artifact it applies was frozen on a split it does not re-read. A threshold adjusted until
    its promise held on test would be a promise about nothing, and nothing would raise.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "fit_bands" not in source
    assert "out_of_fold" not in source


def test_the_phase_03_baseline_is_the_recorded_one():
    """
    The comparison point is what Phase 03 actually measured. Editing it to make a later number
    look better is the cheapest possible way to fake this result.
    """
    assert PHASE_03["gold_missed_share"] == 0.186
    assert PHASE_03["base_rate"] == 0.226


def test_the_criterion_is_a_strict_comparison_against_the_base_rate():
    """
    Passing requires abstentions to *over*-represent gold-missed claims. Matching the base rate is
    what a random declines-at-random policy achieves, so it is not evidence of anything.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "passed = share_now > base_rate" in source


def test_accuracy_is_measured_over_answered_claims_only():
    """
    Dividing by the whole split would make abstention look free -- decline everything and the
    numerator falls while the denominator does not, so 'accuracy' rises toward zero errors.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "correct[answered].mean()" in source
    assert "correct.mean()" not in source


def test_the_gate_is_still_a_conjunction_at_report_time():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "banded & grounded" in source
    assert "banded | grounded" not in source


def test_the_share_calculation_counts_declines_not_answers():
    """
    The quantity is 'of the claims we declined, how many were ungrounded'. Inverting the mask
    computes something that also looks like a percentage and means the opposite.
    """
    gold_missed = np.array([True, True, True, False, False])
    declined = np.array([True, True, False, False, False])

    share = (gold_missed & declined).sum() / declined.sum()
    inverted = (gold_missed & ~declined).sum() / (~declined).sum()
    assert share == 1.0                      # both declined claims were ungrounded
    assert inverted == pytest.approx(1 / 3)  # of the three answered, one was
    assert share != inverted, "the fixture must distinguish the two directions"

    source = SCRIPT.read_text(encoding="utf-8")
    assert "(gold_missed & declined).sum() / max(declined.sum(), 1)" in source


def test_no_verdict_sources_are_cross_tabulated_not_summed():
    """
    Predicting NEI and declining to answer both reach a reader as 'no verdict' and are different
    claims. A single merged rate reads identically for a system that always abstains and one that
    always predicts NEI.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Counter(zip(" in source
    assert "predicted a verdict" in source
