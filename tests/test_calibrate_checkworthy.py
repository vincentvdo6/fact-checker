"""
Calibrating the detector, and the rule about where its threshold is allowed to come from.

The 120 SOTU labels are the only held-out data this task has. Sweeping a threshold against them
and keeping the best would spend that independence on a hyperparameter and quietly turn the
comparison into a fitted number -- so the sweep happens on ClaimBuster's calibration debates and
the result is frozen into an artifact before the demo ever runs.

Selection is by out-of-fold NLL, not in-fold ECE. Phase 03 recorded why and this run reproduced
it: isotonic had the best in-fold ECE (0.0097 against vector scaling's 0.0120) and lost on
out-of-fold NLL, because ECE depends on a binning choice and selecting on it rewards whichever
calibrator happens to suit the bin edges.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_checkworthy import CANDIDATES, GRID, best_threshold, read_predictions

SCRIPT = Path("scripts/calibrate_checkworthy.py")
ARTIFACT = Path("models/checkworthy/v1/calibration.json")


# --- the threshold search --------------------------------------------------------------------

def test_the_best_threshold_is_the_one_with_the_best_f1():
    """
    Scores are separable at 0.5: everything at or above is positive, everything below negative.
    The grid contains 0.5, so a working search finds it and scores a perfect F1.
    """
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    actual = np.array([True, True, True, False, False, False])
    cut, f1 = best_threshold(scores, actual)
    assert f1 == pytest.approx(1.0)
    assert 0.35 <= cut <= 0.7, f"a separating cut, got {cut}"


def test_a_threshold_above_every_score_is_not_chosen():
    """Predicting nothing gives F1 0, and the 0/0 guards must not make that look perfect."""
    scores = np.array([0.1, 0.2, 0.3])
    actual = np.array([True, True, False])
    cut, f1 = best_threshold(scores, actual)
    assert cut <= 0.3
    assert f1 > 0.0


def test_the_grid_spans_the_whole_probability_range():
    """
    A grid that stopped at 0.5 would hide an optimum above it and silently report the edge as the
    choice. Both binarizations landed at 0.35, which only reads as a decision if lower and higher
    cuts were actually available.
    """
    assert min(GRID) <= 0.05
    assert max(GRID) >= 0.95
    assert 0.35 in GRID and 0.5 in GRID


def test_every_grid_point_is_a_probability():
    assert all(0.0 < cut < 1.0 for cut in GRID)


# --- reading the predictions ---------------------------------------------------------------------

def test_labels_are_read_into_the_contract_s_class_order(tmp_path):
    """
    Class index order is the on-disk contract. Reading "check_worthy" as index 0 would invert the
    calibrator's per-class biases while every number still computed.
    """
    path = tmp_path / "p.jsonl"
    rows = [
        {"id": "a", "label": "non_factual", "logits": [3.0, 0.0, 0.0]},
        {"id": "b", "label": "unimportant_factual", "logits": [0.0, 3.0, 0.0]},
        {"id": "c", "label": "check_worthy", "logits": [0.0, 0.0, 3.0]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    logits, labels = read_predictions(path)
    assert labels.tolist() == [0, 1, 2]
    assert logits.shape == (3, 3)


def test_a_trailing_blank_line_does_not_become_a_row(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"id": "a", "label": "non_factual", "logits": [1.0, 0.0, 0.0]})
                    + "\n\n", encoding="utf-8")
    _, labels = read_predictions(path)
    assert len(labels) == 1


# --- the discipline ------------------------------------------------------------------------------

def test_the_threshold_is_swept_on_calibration_not_on_the_demo_labels():
    """
    The one rule this script exists to enforce. If it ever read the SOTU labels, the comparison
    against the hand-written filter would stop being independent.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "predictions_calibration.jsonl" in source

    # What is forbidden is *reading* that data, not naming it. The script is required to say in
    # its docstring and in the artifact where the threshold came from, so banning the word would
    # forbid the disclosure. Paths and the segmentation entry points are the real guard, and
    # docstrings are stripped so prose cannot trip them either.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            node.value.value = ""
    code = ast.unparse(tree).lower()
    for forbidden in ("labels/", "data/transcripts", "segment(", "check_worthy(t"):
        assert forbidden not in code, f"the calibration script reaches for {forbidden!r}"
    assert "labels/checkworthy-" not in source, "the hand labels are never opened, even in prose"


def test_selection_is_by_out_of_fold_nll_rather_than_in_fold_ece():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from src.calibration.crossfit import select" in source
    assert "select(CANDIDATES, logits, labels" in source


def test_an_uncalibrated_baseline_is_always_a_candidate():
    """
    Without it there is no way to see that calibration did anything, and the ECE improvement is
    the entire justification for the extra artifact.
    """
    assert "uncalibrated" in CANDIDATES


def test_the_frozen_artifact_records_what_justified_it():
    if not ARTIFACT.exists():
        pytest.skip("detector not calibrated; run scripts.calibrate_checkworthy")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for field in ("selected", "calibrator", "fitted_on", "thresholds", "ece", "threshold_note"):
        assert field in payload, f"{field} missing from the artifact"
    assert set(payload["thresholds"]) == {"factual", "check_worthy"}
    assert payload["ece"]["calibrated"] < payload["ece"]["raw"], "calibration must improve ECE"
    assert "SOTU" in payload["threshold_note"], "the artifact states where the cut came from"


def test_the_recorded_thresholds_are_on_the_grid():
    if not ARTIFACT.exists():
        pytest.skip("detector not calibrated")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for name, cut in payload["thresholds"].items():
        assert cut in GRID, f"{name} threshold {cut} is not a grid point"


def test_the_script_writes_the_provenance_note_not_just_the_number():
    """
    The artifact test above validates the file on disk, so it cannot notice the script dropping
    the note until someone re-runs it. This closes that gap: a threshold with no record of where
    it came from is how a tuned number gets mistaken for a measured one.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"threshold_note"' in source
    assert "not spent on choosing a hyperparameter" in source


def test_the_script_always_offers_an_uncalibrated_comparison():
    """
    Source-level, because CANDIDATES is a dict the test above reads at import: dropping the entry
    from the literal is the edit a refactor would actually make.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"uncalibrated": Uncalibrated,' in source
    assert "Uncalibrated().fit(logits, labels)" in source, "and it is scored as the raw baseline"
