"""Bootstrap intervals and the exact McNemar test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.stats import Interval, bootstrap_ci, mcnemar, paired_bootstrap


def test_mcnemar_matches_a_hand_computed_exact_p():
    """
    8 discordant pairs split 7/1. Under the null they split like a fair coin, so the two-sided
    p is 2 * P(X <= 1) = 2 * (C(8,0) + C(8,1)) / 2^8 = 2 * 9 / 256.
    """
    a = [True] * 7 + [False] * 1 + [True] * 5
    b = [False] * 7 + [True] * 1 + [True] * 5
    result = mcnemar(a, b)
    assert (result.a_only, result.b_only) == (7, 1)
    assert result.discordant == 8
    assert result.p_value == pytest.approx(2 * (math.comb(8, 0) + math.comb(8, 1)) / 2**8)


def test_mcnemar_ignores_agreements():
    """Claims both variants got right carry no information about which is better."""
    lean = mcnemar([True, False], [False, True])
    padded = mcnemar([True, False] + [True] * 50, [False, True] + [True] * 50)
    assert lean.p_value == padded.p_value


def test_mcnemar_with_no_disagreement_is_uninformative():
    result = mcnemar([True, False, True], [True, False, True])
    assert result.discordant == 0
    assert result.p_value == 1.0


def test_mcnemar_is_symmetric_in_its_arguments():
    a = [True, True, False, False, True]
    b = [False, True, True, False, False]
    assert mcnemar(a, b).p_value == mcnemar(b, a).p_value


def test_mcnemar_p_never_exceeds_one():
    """An even split doubles a tail just over a half; clamping keeps it a probability."""
    assert mcnemar([True, False], [False, True]).p_value == 1.0


def test_mcnemar_rejects_misaligned_samples():
    with pytest.raises(ValueError, match="must align"):
        mcnemar([True, False], [True])


def test_bootstrap_covers_a_known_mean():
    values = list(np.random.default_rng(0).normal(0.7, 0.1, 500))
    interval = bootstrap_ci(values, seed=1)
    assert interval.low < 0.7 < interval.high
    assert interval.point == pytest.approx(float(np.mean(values)))


def test_bootstrap_is_deterministic_under_a_seed():
    # Continuous, because a discrete sample lands on the same percentiles under many seeds and
    # would make the second assertion pass or fail on the sample rather than on the seeding.
    values = list(np.random.default_rng(9).normal(0.5, 0.2, 200))
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)
    assert bootstrap_ci(values, seed=3) != bootstrap_ci(values, seed=4)


def test_a_wider_alpha_gives_a_narrower_interval():
    values = list(np.random.default_rng(2).normal(0.5, 0.2, 300))
    wide = bootstrap_ci(values, alpha=0.05, seed=0)
    narrow = bootstrap_ci(values, alpha=0.5, seed=0)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_bootstrap_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_rejects_an_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci([1.0, 2.0], alpha=alpha)


def test_a_variant_against_itself_has_zero_difference():
    values = [1.0, 0.0, 1.0, 1.0, 0.0] * 20
    interval = paired_bootstrap(values, values, seed=0)
    assert interval.point == 0.0
    assert interval.low == interval.high == 0.0
    assert not interval.excludes_zero


def test_pairing_is_narrower_than_ignoring_it():
    """
    Both variants see the same claims, so shared difficulty cancels in the difference. Resampling
    each independently throws that away and reports an interval wider than the evidence supports.
    """
    rng = np.random.default_rng(5)
    difficulty = rng.normal(0.0, 1.0, 400)
    a = difficulty + 0.20
    b = difficulty

    paired = paired_bootstrap(a, b, seed=0)
    unpaired_width = (
        bootstrap_ci(a, seed=0).high - bootstrap_ci(a, seed=0).low
    ) + (bootstrap_ci(b, seed=1).high - bootstrap_ci(b, seed=1).low)
    assert (paired.high - paired.low) < unpaired_width
    assert paired.excludes_zero


def test_paired_bootstrap_rejects_misaligned_samples():
    with pytest.raises(ValueError, match="must align"):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_interval_reports_whether_a_difference_has_a_sign():
    assert Interval(0.1, 0.05, 0.15).excludes_zero
    assert Interval(-0.1, -0.15, -0.05).excludes_zero
    assert not Interval(0.01, -0.02, 0.04).excludes_zero


def test_intervals_are_json_serializable():
    import json

    json.dumps(bootstrap_ci([1.0, 2.0, 3.0], seed=0).to_dict())
    json.dumps(mcnemar([True, False], [False, False]).to_dict())
