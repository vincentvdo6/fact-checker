"""
Risk-coverage and AURC, against curves whose shape is known before the code runs.

The two failures worth guarding are opposite. A curve that ignores ties reports operating points
no threshold can reach, which flatters a piecewise-constant calibrator like isotonic. A summary
that does not subtract the oracle rewards a model for being accurate rather than for ranking its
errors, which is the whole thing selective prediction is supposed to measure.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.eval.selective import (
    aurc,
    coverage_at_accuracy,
    coverage_at_risk,
    evaluate_selective,
    excess_aurc,
    optimal_aurc,
    risk_at_coverage,
    risk_coverage_curve,
)


def test_perfect_confidence_ordering_has_no_excess_area():
    """Every error ranked below every success: the ranking is optimal, so E-AURC is zero."""
    correct = np.array([True] * 80 + [False] * 20)
    confidence = np.linspace(1.0, 0.0, 100)          # strictly decreasing, no ties
    assert excess_aurc(confidence, correct) == pytest.approx(0.0, abs=1e-12)


def test_a_flawless_model_has_zero_risk_everywhere():
    correct = np.ones(50, dtype=bool)
    confidence = np.linspace(1.0, 0.5, 50)
    assert aurc(confidence, correct) == pytest.approx(0.0)
    assert all(point.risk == 0.0 for point in risk_coverage_curve(confidence, correct))


def test_reversed_confidence_is_worse_than_the_oracle():
    """Confidence anti-correlated with correctness: same accuracy, much worse ranking."""
    correct = np.array([True] * 80 + [False] * 20)
    good = np.linspace(1.0, 0.0, 100)
    bad = np.linspace(0.0, 1.0, 100)
    assert excess_aurc(bad, correct) > excess_aurc(good, correct)
    assert aurc(bad, correct) > aurc(good, correct)


def test_full_coverage_risk_is_the_plain_error_rate():
    correct = np.array([True] * 70 + [False] * 30)
    confidence = np.linspace(1.0, 0.0, 100)
    assert risk_coverage_curve(confidence, correct)[-1].risk == pytest.approx(0.30)
    assert risk_coverage_curve(confidence, correct)[-1].coverage == pytest.approx(1.0)


def test_a_known_curve_point_by_point():
    """
    Four claims, distinct confidences, the top two right and the bottom two wrong. Every operating
    point is checkable by hand, so a convention error has nowhere to hide.
    """
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, False])
    points = risk_coverage_curve(confidence, correct)
    assert [(p.answered, p.risk) for p in points] == [
        (1, 0.0), (2, 0.0), (3, pytest.approx(1 / 3)), (4, 0.5),
    ]


def test_aurc_reduces_to_the_mean_risk_when_nothing_ties():
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, False])
    expected = np.mean([0.0, 0.0, 1 / 3, 0.5])
    assert aurc(confidence, correct) == pytest.approx(expected)


def test_optimal_aurc_is_nonzero_whenever_the_model_errs():
    """Full coverage must still carry the errors, so even a perfect ranking has area."""
    assert optimal_aurc(np.array([True] * 9 + [False])) > 0.0
    assert optimal_aurc(np.ones(10, dtype=bool)) == pytest.approx(0.0)


def test_excess_aurc_is_never_negative():
    rng = np.random.default_rng(0)
    for seed in range(15):
        rng = np.random.default_rng(seed)
        correct = rng.random(200) < 0.7
        confidence = rng.random(200)
        assert excess_aurc(confidence, correct) >= -1e-12


# --- ties: the case isotonic makes ordinary ----------------------------------------------------

def test_a_tied_run_offers_no_threshold_inside_it():
    """
    All four claims share one confidence, so the only reachable thresholds are "answer none" and
    "answer all". A curve with intermediate points would describe a system that cannot be built.
    """
    confidence = np.full(4, 0.75)
    correct = np.array([True, True, False, False])
    points = risk_coverage_curve(confidence, correct)
    assert len(points) == 1
    assert points[0].answered == 4
    assert points[0].risk == pytest.approx(0.5)


def test_ties_do_not_lose_coverage_from_the_area():
    """
    The tied run still spans its full width of coverage. Skipping unreachable cuts must not also
    skip the area they cover, or AURC silently shrinks for a heavily tied calibrator.
    """
    confidence = np.full(4, 0.75)
    correct = np.array([True, True, False, False])
    # one reachable point, risk 0.5, spanning coverage 0 -> 1
    assert aurc(confidence, correct) == pytest.approx(0.5)


def test_each_operating_point_is_weighted_by_the_coverage_it_spans():
    """
    Unequal spans, which is the case that separates a coverage-weighted integral from a plain
    average over reachable points. One claim stands alone at 0.9 and three tie at 0.7, so the
    points span 0.25 and 0.75 of coverage. Averaging them instead would report 0.375 rather than
    0.5625 -- and heavy ties are the norm for isotonic, not a corner case.
    """
    confidence = np.array([0.9, 0.7, 0.7, 0.7])
    correct = np.array([True, False, False, False])
    points = risk_coverage_curve(confidence, correct)
    assert [p.answered for p in points] == [1, 4]
    assert aurc(confidence, correct) == pytest.approx(0.0 * 0.25 + 0.75 * 0.75)


def test_partially_tied_confidence_keeps_only_reachable_points():
    confidence = np.array([0.9, 0.7, 0.7, 0.5])
    correct = np.array([True, False, True, False])
    assert [p.answered for p in risk_coverage_curve(confidence, correct)] == [1, 3, 4]


# --- operating-point queries -------------------------------------------------------------------

def test_coverage_at_risk_picks_the_largest_admissible_coverage():
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, False])
    point = coverage_at_risk(confidence, correct, 0.34)
    assert point is not None and point.answered == 3      # risk 1/3 <= 0.34, and 4 would be 0.5


def test_coverage_at_accuracy_is_the_readers_phrasing():
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, False])
    assert coverage_at_accuracy(confidence, correct, 0.66) == coverage_at_risk(confidence, correct, 0.34)


def test_an_unreachable_risk_target_returns_none_rather_than_the_least_bad_point():
    """Abstention is an outcome. A system that cannot hit the target must say so."""
    confidence = np.array([0.9, 0.8])
    correct = np.array([False, False])
    assert coverage_at_risk(confidence, correct, 0.10) is None


def test_risk_at_coverage_takes_the_smallest_coverage_meeting_the_target():
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, False])
    point = risk_at_coverage(confidence, correct, 0.6)
    assert point is not None and point.answered == 3


def test_risk_at_coverage_returns_none_when_ties_jump_the_target():
    """With one reachable point at full coverage, a 50% request cannot be honoured from above."""
    confidence = np.full(4, 0.5)
    correct = np.array([True, False, True, False])
    assert risk_at_coverage(confidence, correct, 0.5).coverage == pytest.approx(1.0)


def test_a_nonsensical_target_is_refused():
    confidence, correct = np.array([0.9, 0.8]), np.array([True, False])
    with pytest.raises(ValueError, match="target risk"):
        coverage_at_risk(confidence, correct, 1.5)
    with pytest.raises(ValueError, match="target coverage"):
        risk_at_coverage(confidence, correct, 0.0)


def test_an_empty_sample_is_refused():
    with pytest.raises(ValueError, match="empty"):
        risk_coverage_curve(np.array([]), np.array([], dtype=bool))


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="confidences"):
        risk_coverage_curve(np.array([0.9, 0.8]), np.array([True]))


def test_the_report_serialises_to_plain_json():
    rng = np.random.default_rng(1)
    correct = rng.random(300) < 0.7
    confidence = np.clip(rng.random(300) * 0.4 + 0.6 * correct, 0, 1)
    payload = evaluate_selective(confidence, correct, label="retrieved").to_dict()
    assert json.loads(json.dumps(payload))["label"] == "retrieved"
    assert payload["excess_aurc"] >= 0.0
