"""Band fitting: thresholds, the two accuracies, ties, and abstention."""

from __future__ import annotations

import json
import random
from collections import defaultdict

import pytest

from src.calibration.bands import BAND_ORDER, DEFAULT_TARGETS, Band, BandPolicy, fit_bands


def ramp(n: int = 1000) -> tuple[list[float], list[bool]]:
    """Distinct probabilities from 1.0 down to 0.0, correctness tracking the probability."""
    probabilities = [1.0 - i / n for i in range(n)]
    correct = [(i % 100) / 100.0 < p for i, p in enumerate(probabilities)]
    return probabilities, correct


def tied(n: int = 2000) -> tuple[list[float], list[bool]]:
    """Coarse probabilities so many rows share a value, the way isotonic output does."""
    rng = random.Random(11)
    probabilities = [round(0.30 + 0.70 * rng.random(), 1) for _ in range(n)]
    correct = [rng.random() < p for p in probabilities]
    return probabilities, correct


def test_thresholds_decrease_from_strict_to_loose():
    policy = fit_bands(*ramp())
    values = [policy.thresholds[b] for b in BAND_ORDER if b in policy.thresholds]
    assert values == sorted(values, reverse=True)


def test_each_band_meets_its_within_target():
    """The promise shown to a reader is the within-band number, so that is what must hold."""
    policy = fit_bands(*ramp())
    for band, achieved in policy.within.items():
        assert achieved >= policy.targets[band]


def test_within_accuracy_never_exceeds_cumulative_below_the_top():
    policy = fit_bands(*ramp())
    for band in list(policy.thresholds)[1:]:
        assert policy.within[band] <= policy.cumulative[band] + 1e-9


@pytest.mark.parametrize("fixture", [ramp, tied], ids=["distinct", "tied"])
def test_assign_reproduces_the_recorded_promise(fixture):
    """
    The invariant the module rests on: replaying assign() over the fitting split must
    reproduce each band's recorded support and within-accuracy exactly.

    This is the test that catches a threshold landing mid-tie. A threshold admits every
    row sharing its probability, so a band cut between two equal probabilities reports
    an accuracy over rows that assign() then widens past.
    """
    probabilities, correct = fixture()
    policy = fit_bands(probabilities, correct)

    realised: dict[Band, list[bool]] = defaultdict(list)
    for probability, is_correct in zip(probabilities, correct, strict=True):
        band = policy.assign(probability)
        if band is not None:
            realised[band].append(is_correct)

    assert set(realised) == set(policy.thresholds)
    for band, rows in realised.items():
        assert len(rows) == policy.support[band]
        assert sum(rows) / len(rows) == pytest.approx(policy.within[band])


def test_ties_do_not_straddle_a_boundary():
    probabilities = [0.9] * 40 + [0.5] * 40
    correct = [True] * 38 + [False] * 2 + [i % 2 == 0 for i in range(40)]
    policy = fit_bands(probabilities, correct)
    # Every row sharing a probability must land in the same band.
    for probability in (0.9, 0.5):
        assigned = {policy.assign(p) for p in probabilities if p == probability}
        assert len(assigned) == 1


def test_just_below_a_threshold_falls_through():
    policy = fit_bands(*ramp())
    strong = policy.thresholds[Band.STRONG]
    # ramp() fits no moderate band, so "the next one down" is whichever is present.
    looser = [b for b in BAND_ORDER if b in policy.thresholds and b is not Band.STRONG]
    assert policy.assign(strong) is Band.STRONG
    assert policy.assign(strong - 1e-9) is looser[0]


def test_low_probability_abstains():
    assert fit_bands(*ramp()).assign(0.0) is None


def test_unreachable_target_drops_the_band():
    policy = fit_bands([0.9, 0.8, 0.7], [False] * 3, min_support=1)
    assert policy.thresholds == {}
    assert policy.assign(1.0) is None
    assert policy.coverage == 0.0


def test_perfect_split_puts_everything_in_strong():
    policy = fit_bands([0.9, 0.8, 0.7], [True] * 3, min_support=1)
    assert policy.thresholds[Band.STRONG] == 0.7
    assert policy.support[Band.STRONG] == 3
    assert policy.coverage == 1.0


def test_thin_band_is_refused_unless_min_support_allows_it():
    # Only the top three rows can reach 90%; the default floor refuses to certify on that.
    probabilities = [0.99, 0.98, 0.97] + [0.50] * 100
    correct = [True] * 3 + [i % 2 == 0 for i in range(100)]
    assert Band.STRONG not in fit_bands(probabilities, correct).thresholds
    assert Band.STRONG in fit_bands(probabilities, correct, min_support=3).thresholds


def test_every_band_meets_min_support():
    policy = fit_bands(*ramp())
    assert all(count >= policy.min_support for count in policy.support.values())


class Scalar:
    """
    Stand-in for numpy.float64: sortable, summable, truth-testable and float-able, but
    not JSON-serializable. It has to support all of those, or a policy that failed to
    coerce would die in sorted() rather than reaching the json.dumps this test is for.
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __float__(self) -> float:
        return self.value

    def __bool__(self) -> bool:
        return bool(self.value)

    def __lt__(self, other: object) -> bool:
        return self.value < float(other)  # type: ignore[arg-type]

    def __radd__(self, other: float) -> Scalar:
        # Returning Scalar rather than float is the point: np.bool_ does not launder into
        # a builtin on the first addition, so an uncoerced value has to survive the whole
        # accuracy computation and reach json.dumps.
        return Scalar(other + self.value)

    def __sub__(self, other: object) -> Scalar:
        return Scalar(self.value - float(other))  # type: ignore[arg-type]

    def __truediv__(self, other: object) -> Scalar:
        return Scalar(self.value / float(other))  # type: ignore[arg-type]

    def __ge__(self, other: object) -> bool:
        return self.value >= float(other)  # type: ignore[arg-type]


def test_foreign_scalars_are_coerced_at_fit_time():
    probabilities, correct = ramp()
    policy = fit_bands([Scalar(p) for p in probabilities], [Scalar(c) for c in correct])
    assert all(type(v) is float for v in policy.thresholds.values())
    json.dumps(policy.to_dict())  # raises if a foreign scalar survived into the policy


def test_empty_targets_rejected():
    with pytest.raises(ValueError, match="no target given for strong, moderate, weak"):
        fit_bands(*ramp(), targets={})


def test_unknown_target_key_rejected():
    with pytest.raises(ValueError, match="unknown target key"):
        fit_bands(*ramp(), targets=dict(DEFAULT_TARGETS) | {"elite": 0.99})


def test_partial_targets_rejected():
    with pytest.raises(ValueError, match="no target given for moderate, weak"):
        fit_bands(*ramp(), targets={Band.STRONG: 0.9})


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="probabilities"):
        fit_bands([0.5, 0.4], [True])


def test_empty_split_rejected():
    with pytest.raises(ValueError, match="empty split"):
        fit_bands([], [])


def test_policy_round_trips_through_json(tmp_path):
    policy = fit_bands(*ramp())
    path = tmp_path / "bands.json"
    policy.save(path)
    restored = BandPolicy.load(path)
    assert restored == policy
    assert restored.assign(0.95) == policy.assign(0.95)
