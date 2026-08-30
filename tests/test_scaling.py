"""
Calibrators: what each one can and cannot change, checked against known answers.

The two properties worth pinning are the ones that decide which calibrator this project needs.
Temperature cannot move a prediction, so it can never fix a prior shift. Vector scaling can, and
must, or the per-class bias is not earning its place.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.calibration.crossfit import folds, out_of_fold, select
from src.calibration.scaling import (
    CALIBRATORS,
    Isotonic,
    Temperature,
    Uncalibrated,
    VectorScaling,
    load,
    nll,
    pav,
    save,
    softmax,
)


def overconfident(n: int = 3000, true_temperature: float = 2.5, seed: int = 0):
    """
    Logits whose calibrated temperature is known: draw honest ones, then multiply them up.

    Fitting must recover `true_temperature`, because dividing by it restores the distribution the
    labels were actually sampled from.
    """
    rng = np.random.default_rng(seed)
    honest = rng.normal(size=(n, 3)) * 1.5
    labels = np.array([rng.choice(3, p=p) for p in softmax(honest)])
    return honest * true_temperature, labels


def shifted(n: int = 3000, seed: int = 1):
    """
    A model trained on one prior, evaluated on another -- FEVER's 55/20/25 against 33/33/33.

    Built so the fix is a per-class bias and nothing else: the logits carry a constant offset
    toward class 0, and the labels are drawn uniformly.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, size=n)
    logits = rng.normal(size=(n, 3))
    logits[np.arange(n), labels] += 2.0        # a real signal, so the task is learnable
    logits[:, 0] += 1.5                        # and a thumb on the scale for class 0
    return logits, labels


def test_temperature_recovers_a_known_scaling():
    logits, labels = overconfident(true_temperature=2.5)
    fitted = Temperature().fit(logits, labels)
    assert fitted.temperature == pytest.approx(2.5, rel=0.12)


def test_temperature_cannot_change_a_single_prediction():
    """
    Dividing by a positive scalar is monotone within a row. This is why temperature alone cannot
    answer a prior shift, and why the project needs the bias.
    """
    logits, labels = overconfident()
    fitted = Temperature().fit(logits, labels)
    before = softmax(logits).argmax(axis=1)
    after = fitted.transform(logits).argmax(axis=1)
    assert np.array_equal(before, after)


def test_temperature_improves_the_likelihood_it_was_fitted_on():
    logits, labels = overconfident()
    fitted = Temperature().fit(logits, labels)
    assert nll(logits / fitted.temperature, labels) < nll(logits, labels)


def test_vector_scaling_corrects_a_prior_shift_and_temperature_does_not():
    """
    The measurement this project runs on. `retrieved` over-predicts supported 1,241 times against
    a truth of 682, which is train's 55/20/25 prior meeting a 33/33/33 split.
    """
    logits, labels = shifted()
    base = (softmax(logits).argmax(axis=1) == labels).mean()

    temperature = Temperature().fit(logits, labels)
    vector = VectorScaling().fit(logits, labels)

    assert (temperature.transform(logits).argmax(axis=1) == labels).mean() == pytest.approx(base)
    assert (vector.transform(logits).argmax(axis=1) == labels).mean() > base + 0.02


def test_vector_scaling_flattens_the_predicted_class_mix():
    logits, labels = shifted()
    before = np.bincount(softmax(logits).argmax(axis=1), minlength=3)
    after = np.bincount(VectorScaling().fit(logits, labels).transform(logits).argmax(axis=1), minlength=3)
    # The truth is uniform, so a smaller spread across classes is the correction landing.
    assert after.max() - after.min() < before.max() - before.min()


def test_a_constant_shift_in_bias_changes_no_prediction():
    """
    Why the bias vector is only identified up to a constant, stated as a test rather than a claim
    in a docstring: the softmax is invariant to a shared shift, so the optimum is a ray.
    """
    logits, labels = shifted(n=500)
    fitted = VectorScaling().fit(logits, labels)
    shifted_copy = VectorScaling(temperature=fitted.temperature, bias=[b + 3.7 for b in fitted.bias])
    assert shifted_copy.transform(logits) == pytest.approx(fitted.transform(logits))


def test_the_fitted_bias_comes_back_centred():
    """
    Not a guard on the centring line -- that is a no-op here, because the per-class gradients
    cancel across classes and a zero-initialised fit never leaves the sum-zero hyperplane. This
    records the invariant so a change of optimiser or initialisation has to face it.
    """
    logits, labels = shifted()
    assert sum(VectorScaling().fit(logits, labels).bias) == pytest.approx(0.0, abs=1e-9)


def test_every_calibrator_returns_a_distribution():
    logits, labels = shifted()
    for name, cls in CALIBRATORS.items():
        probabilities = cls().fit(logits, labels).transform(logits)
        assert probabilities.shape == logits.shape, name
        assert np.all(probabilities >= 0), name
        assert probabilities.sum(axis=1) == pytest.approx(np.ones(len(logits))), name


def test_pav_is_the_identity_on_already_monotone_input():
    y = np.array([0.0, 0.1, 0.4, 0.9])
    assert pav(y, np.ones(4)) == pytest.approx(y)


def test_pav_pools_a_violation_into_its_mean():
    # 0.8 then 0.2 violates; the pair pools to 0.5 and the result is non-decreasing.
    assert pav(np.array([0.0, 0.8, 0.2, 1.0]), np.ones(4)) == pytest.approx([0.0, 0.5, 0.5, 1.0])


def test_pav_output_is_never_decreasing():
    rng = np.random.default_rng(0)
    for _ in range(20):
        fitted = pav(rng.random(50), np.ones(50))
        assert np.all(np.diff(fitted) >= -1e-12)


def test_isotonic_keeps_a_degenerate_row_finite():
    """All-zero rows would divide to NaN and write invalid JSON into the file Phase 04 reads."""
    logits, labels = shifted(n=200)
    fitted = Isotonic().fit(logits, labels)
    probabilities = fitted.transform(np.full((5, 3), -50.0))
    assert np.all(np.isfinite(probabilities))
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(5))


@pytest.mark.parametrize("cls", [Uncalibrated, Temperature, VectorScaling, Isotonic])
def test_a_calibrator_round_trips_through_disk(cls, tmp_path):
    logits, labels = shifted(n=400)
    fitted = cls().fit(logits, labels)
    path = tmp_path / "calibration.json"
    save(fitted, path)
    json.loads(path.read_text(encoding="utf-8"))          # must be plain JSON, not numpy scalars
    assert load(path).transform(logits) == pytest.approx(fitted.transform(logits))


def test_fitting_on_an_empty_split_is_refused():
    with pytest.raises(ValueError, match="empty"):
        Temperature().fit(np.zeros((0, 3)), np.zeros(0, dtype=int))


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="3 logits and 2 labels"):
        Temperature().fit(np.zeros((3, 3)), np.zeros(2, dtype=int))


def test_a_label_outside_the_class_range_is_refused():
    with pytest.raises(ValueError, match="labels outside"):
        Temperature().fit(np.zeros((4, 3)), np.array([0, 1, 2, 3]))


def test_folds_partition_every_row_exactly_once():
    parts = folds(97, 5, seed=0)
    assert len(parts) == 5
    assert sorted(np.concatenate(parts).tolist()) == list(range(97))


def test_out_of_fold_scores_every_row_by_a_fit_that_excluded_it():
    """
    The guarantee the bands depend on. Verified by construction rather than by trusting the loop:
    refitting on the complement of each fold must reproduce the same rows.
    """
    logits, labels = shifted(n=500)
    got = out_of_fold(VectorScaling, logits, labels, k=5, seed=0)
    for held in folds(len(logits), 5, seed=0):
        keep = np.setdiff1d(np.arange(len(logits)), held)
        expected = VectorScaling().fit(logits[keep], labels[keep]).transform(logits[held])
        assert got[held] == pytest.approx(expected)


def test_out_of_fold_differs_from_fitting_on_everything():
    """If these coincided, the cross-fit would be decorative and the band promise still optimistic."""
    logits, labels = shifted(n=500)
    in_fold = VectorScaling().fit(logits, labels).transform(logits)
    assert not np.allclose(out_of_fold(VectorScaling, logits, labels, k=5, seed=0), in_fold)


def test_selection_prefers_the_calibrator_that_fixes_the_shift():
    logits, labels = shifted(n=1500)
    best, scores = select(
        {"uncalibrated": Uncalibrated, "temperature": Temperature, "vector_scaling": VectorScaling},
        logits, labels, k=5, seed=0,
    )
    assert best == "vector_scaling"
    assert scores["vector_scaling"] < scores["temperature"] < scores["uncalibrated"]
