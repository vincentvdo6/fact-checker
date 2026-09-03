"""
The rules-against-detector comparison, and the discipline that keeps it from being a tuned number.

This script reports a +0.45 F1 swing, which is exactly the size of result that invites a mistake:
sweep the threshold, quote the best cell, call it the headline. That is selection on the test set
and it would void the comparison entirely -- the 120 SOTU labels are the only held-out data this
task has.

So the headline is pinned to 0.5, the model's own decision boundary, fixed before the sweep was
looked at. The sweep is printed as a robustness check and the script must say so; if the result
only held at one threshold, that would be the finding instead of a footnote.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.compare_checkworthy import prf

SCRIPT = Path("scripts/compare_checkworthy.py")


# --- the arithmetic ------------------------------------------------------------------------------

def test_precision_and_recall_are_not_interchangeable():
    """
    3 true positives, 5 false positives, 2 false negatives: precision 3/8, recall 3/5. Different
    values, so a swap changes both rather than cancelling.
    """
    predicted = np.array([True] * 3 + [True] * 5 + [False] * 2)
    actual = np.array([True] * 3 + [False] * 5 + [True] * 2)
    got = prf(predicted, actual)
    assert got["precision"] == pytest.approx(0.375)
    assert got["recall"] == pytest.approx(0.6)
    assert got["f1"] == pytest.approx(2 * 0.375 * 0.6 / (0.375 + 0.6))


def test_predicting_everything_gives_full_recall_and_base_rate_precision():
    actual = np.array([True, True, False, False, False, False])
    got = prf(np.ones(6, dtype=bool), actual)
    assert got["recall"] == pytest.approx(1.0)
    assert got["precision"] == pytest.approx(1 / 3)


def test_predicting_nothing_scores_zero_rather_than_a_perfect_precision():
    """The 0/0 guard must not resolve in the flattering direction."""
    got = prf(np.zeros(4, dtype=bool), np.array([True, True, False, False]))
    assert got["precision"] == 0.0
    assert got["recall"] == 0.0
    assert got["f1"] == pytest.approx(0.0)


def test_accuracy_counts_both_kinds_of_agreement():
    """Two right and two wrong, arranged so a precision-only reading would disagree."""
    got = prf(np.array([True, False, True, False]), np.array([True, False, False, True]))
    assert got["accuracy"] == pytest.approx(0.5)


# --- the operating point is not chosen on the test set ------------------------------------------

def test_the_headline_is_the_pre_specified_boundary_not_the_sweep_maximum():
    """
    An earlier version of this script printed `best: max(results)`, which picks a setting by its
    score on the only held-out labels there are. The headline must name 0.5 explicitly.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "pre-specified 0.5 boundary" in source
    assert 'results["detector, check_worthy >= 0.5"]' in source
    assert "max(results, key=" not in source, "no argmax over settings scored on the test set"


def test_the_sweep_is_labelled_as_robustness_rather_than_selection():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "robustness, not as a choice" in source
    assert "NOT a selection" in source


def test_the_sweep_reports_whether_each_threshold_beats_the_rules():
    """
    The claim being supported is "every threshold beats the heuristic", so each row has to say so
    or not. A sweep that only printed F1 would leave the reader to eyeball it.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "beats the rules?" in source
    assert "'yes' if min(f_one, w_one) > baseline else 'NO'" in source


def test_labels_out_of_step_with_the_segmentation_are_a_hard_error():
    """Same guard as Checkpoint 3: index 51 must still be the sentence that was labelled."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "raise SystemExit(" in source
    assert "set(labelled) - set(sentences)" in source


def test_both_systems_are_scored_on_the_same_sentences():
    """
    The heuristic and the detector must see one list, in one order. Scoring them on separately
    built lists is how an off-by-one becomes a +0.45 result.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "order = sorted(labelled)" in source
    assert "texts = [sentences[i].text for i in order]" in source
    assert "heuristic = np.array([check_worthy(t).worthy for t in texts]" in source
    assert "detector.score_batch(texts)" in source


def test_the_labels_limitation_is_carried_into_the_output():
    """
    The shared-author caveat bounds the heuristic's number. It does not bound the detector's, but
    the reader still needs it to interpret the baseline being beaten.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "limitation carried from the labels" in source
