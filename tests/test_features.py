"""
Sufficiency features, against rankings whose every value is computable by eye.

A feature that is plausible but wrong is the failure to guard against here: it produces a number,
the model fits on it, the AUC lands somewhere believable, and nothing raises. So the fixtures are
built so each expected value can be checked by hand rather than by re-running the implementation.
"""

from __future__ import annotations

import math

import pytest

from src.retrieval.features import (
    FEATURE_NAMES,
    RETRIEVAL_NAMES,
    VERDICT_NAMES,
    as_row,
    features_for,
    retrieval_features,
    verdict_features,
)


def ranking(*pairs):
    """(title, score) pairs -> the (evidence, scores) shape the retriever writes."""
    evidence = [(title, i) for i, (title, _) in enumerate(pairs)]
    return evidence, [score for _, score in pairs]


def test_a_hand_computable_ranking():
    """One page, scores 10, 8, 6, 4. Every feature is arithmetic a reader can verify."""
    evidence, scores = ranking(("A", 10.0), ("A", 8.0), ("A", 6.0), ("A", 4.0))
    f = retrieval_features(evidence, scores, budget=4)

    assert f["top_score"] == 10.0
    assert f["mean_score"] == 7.0                      # (10+8+6+4)/4
    assert f["min_score"] == 4.0
    assert f["score_decay"] == 6.0                     # 10 - 4
    assert f["score_margin"] == 2.0                    # 10 - 8
    assert f["score_ratio"] == pytest.approx(10 / 7)
    assert f["score_std"] == pytest.approx(math.sqrt(5.0))   # var = (9+1+1+9)/4
    assert f["distinct_pages"] == 1.0
    assert f["top_page_share"] == 1.0
    assert f["first_page_change"] == 4.0               # never changes; equals the length
    assert f["n_read"] == 4.0


def test_features_describe_only_the_read_prefix():
    """
    The target is recall over n_evidence_used, so a feature covering unread sentences would
    describe evidence the model never saw and could not have reasoned from.
    """
    evidence, scores = ranking(("A", 10.0), ("A", 8.0), ("B", 1.0), ("C", 0.5))
    read_two = retrieval_features(evidence, scores, budget=2)
    assert read_two["min_score"] == 8.0
    assert read_two["distinct_pages"] == 1.0
    assert read_two["n_read"] == 2.0
    # The unread tail would have changed all three.
    assert retrieval_features(evidence, scores, budget=4)["distinct_pages"] == 3.0


def test_page_structure_is_measured_from_the_top_page():
    evidence, scores = ranking(("A", 9.0), ("A", 8.0), ("B", 7.0), ("A", 6.0))
    f = retrieval_features(evidence, scores, budget=4)
    assert f["distinct_pages"] == 2.0
    assert f["top_page_share"] == 0.75          # three of four rows are page A
    assert f["first_page_change"] == 2.0        # rank 2 is the first row that is not A


def test_a_flat_ranking_is_distinguishable_from_a_peaked_one():
    """The signal the whole phase rests on: a confident retrieval looks different from a lost one."""
    peaked, peaked_scores = ranking(("A", 30.0), ("B", 5.0), ("C", 4.0), ("D", 3.0))
    flat, flat_scores = ranking(("A", 9.0), ("B", 9.0), ("C", 9.0), ("D", 9.0))
    sharp = retrieval_features(peaked, peaked_scores, budget=4)
    dull = retrieval_features(flat, flat_scores, budget=4)

    assert sharp["score_margin"] > dull["score_margin"]
    assert sharp["score_decay"] > dull["score_decay"]
    assert sharp["score_ratio"] > dull["score_ratio"]
    assert dull["score_margin"] == 0.0
    assert dull["score_std"] == 0.0


def test_a_single_sentence_has_no_margin_rather_than_crashing():
    evidence, scores = ranking(("A", 12.0))
    f = retrieval_features(evidence, scores, budget=1)
    assert f["score_margin"] == 0.0
    assert f["score_decay"] == 0.0
    assert f["n_read"] == 1.0


def test_reading_nothing_yields_zeros_not_an_error():
    """claim_only reads no evidence, and a claim can pack to zero sentences under a tight budget."""
    evidence, scores = ranking(("A", 5.0), ("B", 4.0))
    f = retrieval_features(evidence, scores, budget=0)
    assert set(f) == set(RETRIEVAL_NAMES)
    assert all(value == 0.0 for value in f.values())


def test_a_budget_beyond_the_stored_evidence_reads_what_exists():
    evidence, scores = ranking(("A", 5.0), ("B", 4.0))
    assert retrieval_features(evidence, scores, budget=99)["n_read"] == 2.0


def test_mismatched_refs_and_scores_are_refused():
    with pytest.raises(ValueError, match="2 refs and 3 scores"):
        retrieval_features([("A", 0), ("B", 1)], [1.0, 2.0, 3.0], budget=2)


def test_a_negative_budget_is_refused():
    with pytest.raises(ValueError, match="negative"):
        retrieval_features([("A", 0)], [1.0], budget=-1)


# --- verdict side ------------------------------------------------------------------------------

def test_verdict_features_on_a_confident_row():
    f = verdict_features([10.0, 0.0, 0.0])
    assert f["max_prob"] > 0.99
    assert f["prob_margin"] > 0.99
    assert f["entropy"] < 0.05


def test_verdict_features_on_a_uniform_row():
    f = verdict_features([1.0, 1.0, 1.0])
    assert f["max_prob"] == pytest.approx(1 / 3)
    assert f["prob_margin"] == pytest.approx(0.0)
    assert f["entropy"] == pytest.approx(math.log(3))


def test_entropy_separates_rows_that_share_a_maximum():
    """
    Why entropy is carried alongside max_prob. Both rows peak at 0.5; one is a two-way split and
    the other spreads its remainder, and they are different situations.
    """
    two_way = verdict_features([math.log(0.5), math.log(0.5), math.log(1e-9)])
    three_way = verdict_features([math.log(0.5), math.log(0.25), math.log(0.25)])
    assert two_way["max_prob"] == pytest.approx(three_way["max_prob"], abs=1e-6)
    assert three_way["entropy"] > two_way["entropy"]


def test_a_single_logit_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        verdict_features([1.0])


# --- the row contract --------------------------------------------------------------------------

def test_feature_names_are_the_two_families_in_order():
    assert FEATURE_NAMES == RETRIEVAL_NAMES + VERDICT_NAMES
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_features_for_produces_exactly_the_declared_names():
    evidence, scores = ranking(("A", 9.0), ("B", 3.0))
    assert set(features_for(evidence, scores, [1.0, 0.0, 0.0], budget=2)) == set(FEATURE_NAMES)


def test_as_row_orders_by_the_name_tuple_not_by_insertion():
    """
    A model fitted on one order and applied in another still returns a number, and every metric
    downstream still computes. Ordering has to come from the contract, not from dict order.
    """
    evidence, scores = ranking(("A", 9.0), ("B", 3.0))
    features = features_for(evidence, scores, [2.0, 1.0, 0.0], budget=2)
    scrambled = dict(reversed(list(features.items())))
    assert as_row(scrambled) == as_row(features)
    assert as_row(features)[0] == features["top_score"]


def test_as_row_can_select_one_family_for_the_ablation():
    evidence, scores = ranking(("A", 9.0), ("B", 3.0))
    features = features_for(evidence, scores, [2.0, 1.0, 0.0], budget=2)
    assert len(as_row(features, RETRIEVAL_NAMES)) == len(RETRIEVAL_NAMES)
    assert len(as_row(features, VERDICT_NAMES)) == len(VERDICT_NAMES)


def test_a_missing_feature_is_refused_rather_than_defaulted():
    with pytest.raises(KeyError, match="missing features"):
        as_row({"top_score": 1.0})
