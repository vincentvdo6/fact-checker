"""
The Checkpoint 1 measurement: what it counts, what it excludes, and where it is allowed to fit.

The arithmetic here is small; the ways it can be quietly wrong are not. Counting NEI as a
retrieval failure would invent one on a third of the split. Fitting the retrieval model anywhere
other than trainval would reuse a split that is already carrying the calibrator and the bands.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from scripts import measure_sufficiency
from scripts.measure_sufficiency import GATE_STRONG, GATE_WEAK, assemble, out_of_fold_auc
from src.retrieval.features import RETRIEVAL_NAMES

SCRIPT = Path("scripts/measure_sufficiency.py")


def row(claim_id: int, label: str, refs, gold=()):
    return {
        "id": claim_id,
        "label": label,
        "claim": "a claim",
        "evidence": [[title, index, "text"] for title, index in refs],
        "gold": [[title, index, "text"] for title, index in gold],
    }


def test_nei_claims_are_excluded_rather_than_counted_as_failures():
    """
    NOT ENOUGH INFO has no gold by construction. Scoring it would mark every such claim as a
    retrieval failure and invent a third of a split's worth of negatives.
    """
    rows = [
        row(1, "supported", [("A", 0)], gold=[("A", 0)]),
        row(2, "not_enough_evidence", [("B", 0)]),
        row(3, "contradicted", [("C", 0)], gold=[("C", 0)]),
    ]
    scores = {1: [9.0], 2: [9.0], 3: [9.0]}
    groups = {1: ((("A", 0),),), 3: ((("C", 0),),)}
    features, labels, ids = assemble(rows, scores, [1, 1, 1], groups, None)
    assert ids == [1, 3]
    assert len(features) == 2 and len(labels) == 2


def test_the_label_is_whether_gold_landed_inside_the_read_prefix():
    """
    Not whether gold is anywhere in the stored 25. A claim whose gold sits at rank 20 but whose
    budget stopped at 2 was not given its evidence, and calling that a success would erase the
    entire effect Phase 04 exists to measure.
    """
    rows = [row(1, "supported", [("A", 0), ("A", 1), ("A", 2)], gold=[("A", 2)])]
    scores = {1: [9.0, 8.0, 7.0]}
    groups = {1: ((("A", 2),),)}

    _, read_all, _ = assemble(rows, scores, [3], groups, None)
    _, read_two, _ = assemble(rows, scores, [2], groups, None)
    assert bool(read_all[0]) is True
    assert bool(read_two[0]) is False


def test_a_claim_without_retrieval_scores_is_skipped_not_defaulted():
    rows = [row(1, "supported", [("A", 0)], gold=[("A", 0)])]
    features, labels, ids = assemble(rows, {}, [1], {1: ((("A", 0),),)}, None)
    assert ids == [] and len(features) == 0 and len(labels) == 0


def test_verdict_features_are_added_only_when_logits_are_supplied():
    """trainval has no exported logits, so the retrieval model must fit without them."""
    rows = [row(1, "supported", [("A", 0)], gold=[("A", 0)])]
    groups = {1: ((("A", 0),),)}
    without, _, _ = assemble(rows, {1: [9.0]}, [1], groups, None)
    with_logits, _, _ = assemble(rows, {1: [9.0]}, [1], groups, {1: [2.0, 1.0, 0.0]})
    assert set(without[0]) == set(RETRIEVAL_NAMES)
    assert "max_prob" in with_logits[0]


def test_out_of_fold_auc_never_scores_a_row_with_a_model_that_saw_it():
    """
    An in-fold AUC on 14 features and a thousand rows is optimistic by several points, which is
    the difference between clearing the gate and only appearing to.
    """
    rng = np.random.default_rng(0)
    labels = rng.random(400) < 0.5
    # Pure noise: an honest out-of-fold estimate must sit near chance.
    features = [dict.fromkeys(RETRIEVAL_NAMES, 0.0) | {"top_score": float(v)}
                for v in rng.normal(size=400)]
    assert out_of_fold_auc(features, labels, RETRIEVAL_NAMES, k=5, seed=0) < 0.60


def test_the_gate_thresholds_are_the_ones_the_plan_fixed():
    """Written down before the number was seen; a moved goalpost is not a result."""
    assert (GATE_WEAK, GATE_STRONG) == (0.60, 0.70)


def test_the_retrieval_model_is_fitted_on_trainval_only():
    """
    Calibration already carries the calibrator and the band thresholds. A third fit there would
    make all three optimistic at once, and nothing would raise.
    """
    source = inspect.getsource(measure_sufficiency.main)
    statement = source[source.index("retrieval_model = "):source.index("results = {")]
    assert "train_features" in statement, f"not fitted on trainval:\n{statement}"
    assert "eval_features" not in statement, f"fitted on the evaluation split:\n{statement}"


@pytest.mark.parametrize("split", ["test"])
def test_the_measurement_never_opens_the_test_split(split):
    """Checkpoint 1 is a decision about whether to continue; test stays sealed until the report."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert f"predictions_{split}" not in source
    assert f'bm25-{split}' not in source
