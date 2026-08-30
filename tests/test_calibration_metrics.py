"""
Calibration metrics, checked against samples whose true calibration error is known by construction.

The hard part of testing ECE is that it is easy to write something plausible that is wrong by a
binning convention and still looks sane on real data. So the cases here are built so the answer is
derivable by hand: a perfectly calibrated sample must score ~0, and a sample that claims p and
delivers q must score |p - q| exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.calibration import (
    adaptive_ece,
    brier,
    ece,
    evaluate_calibration,
    mce,
    nll,
    reliability_bins,
)


def two_class(confidence: np.ndarray) -> np.ndarray:
    """Probability rows whose max equals the requested confidence."""
    return np.column_stack([confidence, 1.0 - confidence])


def calibrated_sample(n: int = 20000, seed: int = 0):
    """
    Confidence drawn over [0.5, 1], outcomes drawn *at* that confidence.

    By construction the model claims p and is right with probability p, so the true calibration
    error is zero and any measured ECE is sampling noise.
    """
    rng = np.random.default_rng(seed)
    confidence = rng.uniform(0.5, 1.0, size=n)
    correct = rng.random(n) < confidence
    # class 0 when correct, class 1 otherwise, so argmax matches the label exactly when correct
    return two_class(confidence), np.where(correct, 0, 1)


def test_a_perfectly_calibrated_sample_scores_near_zero():
    probabilities, labels = calibrated_sample()
    assert ece(probabilities, labels) < 0.01
    assert adaptive_ece(probabilities, labels) < 0.01


def test_a_known_gap_is_measured_exactly():
    """
    Every claim asserts 0.90 and exactly 70% are right, so ECE must be |0.90 - 0.70| = 0.20 --
    a single occupied bin, no weighting subtleties, nowhere for a convention error to hide.
    """
    n = 1000
    probabilities = two_class(np.full(n, 0.9))
    labels = np.array([0] * 700 + [1] * 300)
    assert ece(probabilities, labels) == pytest.approx(0.20, abs=1e-9)
    assert mce(probabilities, labels) == pytest.approx(0.20, abs=1e-9)


def test_ece_weights_bins_by_occupancy():
    """
    900 claims perfectly calibrated, 100 claims maximally wrong. The answer is 0.1 * 1.0, not the
    unweighted mean over the two occupied bins, which would be 0.5.
    """
    probabilities = two_class(np.concatenate([np.full(900, 0.5), np.full(100, 1.0)]))
    labels = np.concatenate([[0] * 450, [1] * 450, [1] * 100])
    assert ece(probabilities, labels) == pytest.approx(0.10, abs=1e-9)


def test_confidence_of_exactly_one_lands_in_the_last_bin():
    """An off-by-one at the top edge would drop the most confident claims out of the sum."""
    probabilities = two_class(np.full(50, 1.0))
    labels = np.zeros(50, dtype=int)
    binned = reliability_bins(probabilities, labels, bins=10)
    assert binned[-1].count == 50
    assert sum(b.count for b in binned) == 50


def test_a_confidence_on_an_interior_edge_falls_in_the_lower_bin():
    """
    Bins are right-closed, (low, high]. Not a free convention: `Isotonic` is piecewise-constant,
    so many claims share an identical probability and exact ties on a bin edge are ordinary rather
    than exotic. Whichever way this goes it must be pinned, or a calibrator swap silently moves
    claims between bins and moves the reported ECE with them.
    """
    probabilities = two_class(np.full(40, 0.9))       # 0.9 is exactly an edge for bins=10
    labels = np.zeros(40, dtype=int)
    binned = reliability_bins(probabilities, labels, bins=10)
    assert binned[8].count == 40, "0.9 belongs in (0.8, 0.9]"
    assert binned[9].count == 0


def test_every_claim_lands_in_exactly_one_bin():
    probabilities, labels = calibrated_sample(n=2000)
    for adaptive in (False, True):
        binned = reliability_bins(probabilities, labels, bins=15, adaptive=adaptive)
        assert sum(b.count for b in binned) == 2000


def test_adaptive_bins_carry_equal_mass():
    probabilities, labels = calibrated_sample(n=1500)
    counts = [b.count for b in reliability_bins(probabilities, labels, bins=15, adaptive=True)]
    assert max(counts) - min(counts) <= 1


def test_adaptive_ece_sees_error_that_cancels_inside_one_wide_bin():
    """
    The reason both are reported. Two groups sit inside the same equal-width bin (0.933-1.0) and
    err in opposite directions: one under-confident by 0.05, one over-confident by 0.19. Averaged
    together inside that bin they partly cancel and equal-width reports 0.07. Equal-mass bins
    separate them and report the 0.12 that is really there.
    """
    under = np.full(1000, 0.95)     # claims 0.95, always right      -> gap -0.05
    over = np.full(1000, 0.99)      # claims 0.99, right 80% of time  -> gap +0.19
    probabilities = two_class(np.concatenate([under, over]))
    labels = np.concatenate([np.zeros(1000, dtype=int), [0] * 800 + [1] * 200])

    # Two bins each, so the split is exact and both answers are checkable by hand: equal-width
    # puts both groups above 0.5 and averages to |0.97 - 0.90|; equal-mass cuts at the median,
    # which lands exactly between the groups, and reports (0.05 + 0.19) / 2.
    assert ece(probabilities, labels, bins=2) == pytest.approx(0.07, abs=1e-9)
    assert adaptive_ece(probabilities, labels, bins=2) == pytest.approx(0.12, abs=1e-9)


def test_mce_reports_the_worst_bin_not_the_average():
    probabilities = two_class(np.concatenate([np.full(990, 0.5), np.full(10, 1.0)]))
    labels = np.concatenate([[0] * 495, [1] * 495, [1] * 10])
    assert mce(probabilities, labels) == pytest.approx(1.0, abs=1e-9)
    assert ece(probabilities, labels) < 0.02


def test_brier_is_zero_for_a_confident_and_correct_predictor():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert brier(probabilities, np.array([0, 1])) == pytest.approx(0.0)


def test_brier_is_two_for_a_confident_and_wrong_predictor():
    """Squared error over both the claimed class and the true one, so the worst case is 2."""
    probabilities = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert brier(probabilities, np.array([1, 1])) == pytest.approx(2.0)


def test_nll_stays_finite_at_zero_probability():
    probabilities = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert np.isfinite(nll(probabilities, np.array([1, 1])))


def test_overconfidence_is_signed_toward_the_reader_being_misled():
    probabilities = two_class(np.full(100, 0.9))
    labels = np.array([0] * 60 + [1] * 40)
    report = evaluate_calibration(probabilities, labels, label="test")
    assert report.mean_confidence == pytest.approx(0.9)
    assert report.accuracy == pytest.approx(0.6)
    assert report.overconfidence == pytest.approx(0.3)


def test_the_bootstrap_interval_is_well_formed_and_positive():
    """
    Deliberately not asserting that the interval brackets the point estimate. ECE is an L1
    aggregate, so resampling noise can only push it up: near zero the whole percentile interval
    sits above the point, which is a property of the statistic rather than a bug. The interval is
    a spread, not a confidence statement about the true ECE -- see the note in evaluate_calibration.
    """
    probabilities, labels = calibrated_sample(n=800)
    report = evaluate_calibration(probabilities, labels, label="test", resamples=200, seed=0)
    assert report.ece_interval is not None
    assert 0.0 <= report.ece_interval.low <= report.ece_interval.high


def test_the_bootstrap_is_reproducible_for_a_seed():
    probabilities, labels = calibrated_sample(n=500)
    first = evaluate_calibration(probabilities, labels, label="a", resamples=100, seed=7)
    second = evaluate_calibration(probabilities, labels, label="a", resamples=100, seed=7)
    assert first.ece_interval.to_dict() == second.ece_interval.to_dict()


def test_the_report_serialises_to_plain_json():
    import json

    probabilities, labels = calibrated_sample(n=400)
    payload = evaluate_calibration(probabilities, labels, label="retrieved", resamples=50).to_dict()
    assert json.loads(json.dumps(payload))["label"] == "retrieved"


def test_an_empty_sample_is_refused():
    with pytest.raises(ValueError, match="empty"):
        ece(np.zeros((0, 3)), np.zeros(0, dtype=int))


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="3 probabilities and 2 labels"):
        ece(np.zeros((3, 3)), np.zeros(2, dtype=int))
