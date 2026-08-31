"""
The gate: two conditions, one of them untuned, and neither fitted where it would leak.

Nothing here checks a score. It checks that the gate is the thing the plan committed to before
the numbers were known -- a conjunction, with a threshold nobody chose against an outcome.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from scripts import fit_sufficiency
from scripts.fit_sufficiency import SUFFICIENCY_THRESHOLD, features_for_rows
from src.retrieval.features import RETRIEVAL_NAMES

SCRIPT = Path("scripts/fit_sufficiency.py")


def test_the_threshold_is_the_models_own_decision_boundary():
    """
    0.5 means "more likely than not that retrieval missed". Any other value would be a parameter
    tuned against an outcome, and the phase would be reporting a number it had optimised for.
    """
    assert SUFFICIENCY_THRESHOLD == 0.5


def test_the_gate_requires_both_conditions():
    """
    Answering needs a confidence band AND adequate evidence. An OR would let a confident model
    answer an ungrounded claim, which is the exact behaviour Phase 03 measured and Phase 04 exists
    to stop.
    """
    banded = np.array([True, True, False, False])
    grounded = np.array([True, False, True, False])
    answered = banded & grounded
    assert answered.tolist() == [True, False, False, False]

    source = inspect.getsource(fit_sufficiency.main)
    assert "banded & grounded" in source, "the gate must be a conjunction"
    assert "banded | grounded" not in source


def test_sufficiency_is_compared_with_a_greater_or_equal():
    """
    A claim scoring exactly at the boundary is answered, not declined. Flipping this to a strict
    comparison silently declines every claim the model rates exactly 0.5, which for a
    piecewise-flat model is not a measure-zero event.
    """
    source = inspect.getsource(fit_sufficiency.main)
    assert "sufficiency >= SUFFICIENCY_THRESHOLD" in source


def test_features_are_built_for_every_claim_including_nei():
    """
    The gate applies to NEI predictions too -- that was the policy decision. Building features
    only for verifiable claims would leave the gate unable to decline an ungrounded NEI.
    """
    rows = [
        {"id": 1, "label": "supported", "evidence": [["A", 0, "t"], ["A", 1, "t"]]},
        {"id": 2, "label": "not_enough_evidence", "evidence": [["B", 0, "t"]]},
    ]
    features = features_for_rows(rows, {1: [9.0, 5.0], 2: [4.0]}, [2, 1])
    assert len(features) == 2
    assert set(features[0]) == set(RETRIEVAL_NAMES)


def test_the_sufficiency_model_is_fitted_on_trainval_only():
    """Calibration already carries the calibrator and the bands; a third fit there compounds."""
    source = inspect.getsource(fit_sufficiency.main)
    statement = source[source.index("model = SufficiencyModel"):source.index("# --- place the gate")]
    assert "train_features" in statement
    assert "trainval" in source

    read_calls = [line for line in source.splitlines() if "read_dataset(" in line]
    assert any('"trainval"' in line for line in read_calls)


def test_the_confidence_bands_are_phase_03s_unchanged():
    """
    The cost of the second condition has to show up as a coverage difference. Re-fitting the bands
    on a blended score would absorb it into moved thresholds and hide what the gate actually cost.
    """
    source = inspect.getsource(fit_sufficiency.main)
    assert "fit_bands(confidence, correct" in source
    assert "fit_bands(confidence * sufficiency" not in source


def test_the_fitting_script_never_opens_the_test_split():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "predictions_test" not in source
    assert "bm25-test" not in source
