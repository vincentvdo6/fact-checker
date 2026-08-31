"""
The sufficiency model and the AUC it is judged by, against separable and unlearnable fixtures.

Two failures matter. A model that cannot learn a signal that is plainly there would send Phase 04
into a false negative result. An AUC that mishandles ties would flatter it, because a logistic
regression over a dozen features produces many identical scores.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.calibration.sufficiency import SufficiencyModel, rows_from
from src.eval.selective import roc_auc
from src.retrieval.features import RETRIEVAL_NAMES


def separable(n: int = 600, seed: int = 0, noise: float = 0.5):
    """One informative column plus noise; the label is a shifted version of it."""
    rng = np.random.default_rng(seed)
    positive = rng.random(n) < 0.5
    signal = np.where(positive, 2.0, 0.0) + rng.normal(0, noise, size=n)
    rows = np.column_stack([signal, rng.normal(size=n), rng.normal(size=n)])
    return rows, positive


# --- AUC ---------------------------------------------------------------------------------------

def test_auc_is_one_for_a_perfect_ranking():
    assert roc_auc(np.array([0.1, 0.2, 0.9, 0.95]), np.array([False, False, True, True])) == 1.0


def test_auc_is_zero_for_a_perfectly_inverted_ranking():
    assert roc_auc(np.array([0.9, 0.95, 0.1, 0.2]), np.array([False, False, True, True])) == 0.0


def test_auc_is_a_half_when_every_score_ties():
    """
    The case a threshold sweep gets wrong. A model over a few features emits many identical
    scores, and crediting them as ordered would manufacture a signal that is not there.
    """
    assert roc_auc(np.full(8, 0.7), np.array([True] * 4 + [False] * 4)) == pytest.approx(0.5)


def test_auc_counts_a_tie_as_exactly_half():
    """One positive and one negative share a score; the pair is worth 0.5, so AUC is 0.75."""
    score = np.array([0.9, 0.5, 0.5, 0.1])
    positive = np.array([True, True, False, False])
    # pairs: (0.9,0.5)=1, (0.9,0.1)=1, (0.5,0.5)=0.5, (0.5,0.1)=1  -> 3.5/4
    assert roc_auc(score, positive) == pytest.approx(0.875)


def test_auc_is_invariant_to_a_monotone_rescaling():
    rng = np.random.default_rng(3)
    score, positive = rng.random(200), rng.random(200) < 0.4
    assert roc_auc(score, positive) == pytest.approx(roc_auc(score * 7.5 + 3.0, positive))


def test_auc_refuses_a_single_class_rather_than_returning_a_half():
    """0.5 would read as 'no signal' where the truth is 'no measurement'."""
    with pytest.raises(ValueError, match="both classes"):
        roc_auc(np.array([0.1, 0.9]), np.array([True, True]))


# --- the model ---------------------------------------------------------------------------------

def test_the_model_learns_a_signal_that_is_plainly_there():
    rows, positive = separable()
    model = SufficiencyModel(names=("signal", "noise_a", "noise_b")).fit(rows, positive)
    assert roc_auc(model.predict(rows), positive) > 0.95


def test_the_model_reports_chance_on_pure_noise():
    """A false positive here would send the phase chasing a signal that does not exist."""
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(600, 3))
    positive = rng.random(600) < 0.5
    model = SufficiencyModel(names=("a", "b", "c")).fit(rows, positive)
    assert roc_auc(model.predict(rows), positive) < 0.62


def test_the_informative_feature_carries_the_largest_weight():
    """Coefficients are read as a finding, so they have to mean what they appear to mean."""
    rows, positive = separable()
    model = SufficiencyModel(names=("signal", "noise_a", "noise_b")).fit(rows, positive)
    coefficients = model.coefficients()
    assert abs(coefficients["signal"]) > 5 * max(
        abs(coefficients["noise_a"]), abs(coefficients["noise_b"])
    )


def test_predictions_are_probabilities():
    rows, positive = separable()
    predicted = SufficiencyModel(names=("a", "b", "c")).fit(rows, positive).predict(rows)
    assert np.all((predicted >= 0) & (predicted <= 1))


def test_standardisation_comes_from_the_fitting_split_only():
    """
    Applying the model to a new split must not renormalise against that split's own distribution,
    which would leak it into its own scoring and shift every prediction.
    """
    rows, positive = separable()
    model = SufficiencyModel(names=("a", "b", "c")).fit(rows, positive)
    stored = list(model.mean)
    model.predict(rows * 100.0 + 50.0)
    assert model.mean == stored


def test_a_feature_scaled_up_a_thousandfold_changes_nothing():
    """Standardisation, checked behaviourally: BM25 scores and probabilities are incommensurable."""
    rows, positive = separable()
    plain = SufficiencyModel(names=("a", "b", "c")).fit(rows, positive).predict(rows)
    blown = rows.copy()
    blown[:, 0] *= 1000.0
    scaled = SufficiencyModel(names=("a", "b", "c")).fit(blown, positive).predict(blown)
    assert scaled == pytest.approx(plain, abs=1e-4)


def test_a_constant_column_does_not_divide_by_zero():
    rows, positive = separable()
    rows[:, 1] = 4.0
    predicted = SufficiencyModel(names=("a", "b", "c")).fit(rows, positive).predict(rows)
    assert np.all(np.isfinite(predicted))


def test_the_model_round_trips_through_disk(tmp_path):
    rows, positive = separable()
    model = SufficiencyModel(names=("a", "b", "c")).fit(rows, positive)
    path = tmp_path / "sufficiency.json"
    model.save(path)
    json.loads(path.read_text(encoding="utf-8"))       # plain JSON, not numpy scalars
    assert SufficiencyModel.load(path).predict(rows) == pytest.approx(model.predict(rows))


def test_fitting_on_one_class_is_refused():
    rows, _ = separable()
    with pytest.raises(ValueError, match="single class"):
        SufficiencyModel(names=("a", "b", "c")).fit(rows, np.ones(len(rows), dtype=bool))


def test_a_column_count_mismatch_is_refused():
    rows, positive = separable()
    with pytest.raises(ValueError, match="3 columns for 2 feature names"):
        SufficiencyModel(names=("a", "b")).fit(rows, positive)


def test_predicting_with_the_wrong_width_is_refused():
    rows, positive = separable()
    model = SufficiencyModel(names=("a", "b", "c")).fit(rows, positive)
    with pytest.raises(ValueError, match="width 3"):
        model.predict(np.zeros((5, 2)))


def test_rows_from_orders_by_the_name_tuple():
    features = [dict.fromkeys(RETRIEVAL_NAMES, 0.0) | {"top_score": 9.0} for _ in range(3)]
    rows = rows_from(features, RETRIEVAL_NAMES)
    assert rows.shape == (3, len(RETRIEVAL_NAMES))
    assert rows[0][RETRIEVAL_NAMES.index("top_score")] == 9.0
