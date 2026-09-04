"""
The interval that keeps small samples from being over-read.

Several rates in this project sit on samples where a bare point estimate is misleading: 101
expert-labelled sentences, a speaker subgroup of 373, a band with 346 members. The interval is
what separates "this group is worse" from "this group is smaller", and it is used by both the
held-out evaluation and the subgroup audit -- which is why it has one home rather than two copies
that could drift apart.
"""

from __future__ import annotations

import math

import pytest

from src.eval.intervals import wilson


def test_the_interval_brackets_the_observed_rate():
    low, high = wilson(80, 101)
    assert low < 80 / 101 < high


def test_zero_successes_does_not_produce_a_negative_bound():
    """
    The algebra lands a few parts in 1e18 below zero at the boundary, and a printed "-0.0000" in a
    table of rates is exactly the detail that makes a reader wonder what else went unchecked.
    """
    low, high = wilson(0, 40)
    assert low == 0.0
    assert 0.0 < high < 1.0


def test_every_success_does_not_produce_a_bound_above_one():
    low, high = wilson(40, 40)
    assert high == 1.0
    assert 0.0 < low < 1.0


def test_it_does_not_claim_certainty_from_the_least_informative_observation():
    """
    The normal approximation gives [0, 0] for zero successes -- perfect confidence from the one
    result that provides least. Wilson is used precisely because it does not.
    """
    low, high = wilson(0, 20)
    assert high > 0.05, "an interval, not a point"


def test_a_smaller_sample_gives_a_wider_interval():
    wide = wilson(8, 10)
    narrow = wilson(800, 1000)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_an_empty_sample_is_nan_rather_than_a_division_error():
    low, high = wilson(0, 0)
    assert math.isnan(low) and math.isnan(high)


@pytest.mark.parametrize("successes", [0, 1, 25, 49, 50])
def test_the_interval_always_stays_inside_the_unit_range(successes):
    low, high = wilson(successes, 50)
    assert 0.0 <= low <= high <= 1.0


def test_a_wider_z_gives_a_wider_interval():
    ninety_five = wilson(30, 100)
    ninety_nine = wilson(30, 100, z=2.576)
    assert (ninety_nine[1] - ninety_nine[0]) > (ninety_five[1] - ninety_five[0])
