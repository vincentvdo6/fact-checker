"""Verdict scoring, and the split that separates reasoning failure from retrieval failure."""

from __future__ import annotations

import json

import pytest

from src.eval.retrieval import Ceiling
from src.eval.verdict import evaluate_verdicts
from src.verdict.encode import LABELS

S, C, N = LABELS   # supported, contradicted, not_enough_evidence
CEILING = Ceiling(verifiable=0.774, overall=0.85, nei_share=0.33)


def report(labels, predictions, **kw):
    return evaluate_verdicts(labels, predictions, variant="retrieved", ceiling=CEILING, **kw)


def test_accuracy_and_confusion_against_hand_counts():
    labels = [S, S, C, C, N, N]
    predictions = [S, C, C, C, N, S]
    r = report(labels, predictions)
    assert r.accuracy == pytest.approx(4 / 6)
    # rows gold, columns predicted, in LABELS order
    assert r.confusion == ((1, 1, 0), (0, 2, 0), (1, 0, 1))


def test_per_class_metrics_against_hand_computation():
    labels = [S, S, C, C, N, N]
    predictions = [S, C, C, C, N, S]
    by_label = {c.label: c for c in report(labels, predictions).per_class}

    # supported: predicted twice (one gold S, one gold N) -> precision 1/2; recalled 1 of 2.
    assert by_label[S].precision == pytest.approx(0.5)
    assert by_label[S].recall == pytest.approx(0.5)
    assert by_label[S].f1 == pytest.approx(0.5)
    # contradicted: predicted three times, two correct; both golds found.
    assert by_label[C].precision == pytest.approx(2 / 3)
    assert by_label[C].recall == pytest.approx(1.0)
    assert by_label[C].f1 == pytest.approx(2 * (2 / 3) / (2 / 3 + 1))
    assert by_label[N].support == 2


def test_macro_f1_is_unweighted_so_a_rare_class_still_counts():
    """
    Support is deliberately lopsided: 4 supported against 1 each of the others, and the rare
    contradicted class is missed entirely. Weighting by support would hide that behind the
    common class -- 0.759 against the true macro 0.630 -- which is the whole reason to prefer
    macro here, since the dev splits are balanced but train is not.
    """
    labels = [S, S, S, S, C, N]
    predictions = [S, S, S, S, S, N]
    r = report(labels, predictions)
    by_label = {c.label: c for c in r.per_class}

    assert by_label[S].f1 == pytest.approx(2 * (4 / 5) * 1.0 / ((4 / 5) + 1.0))
    assert by_label[C].f1 == 0.0
    assert by_label[N].f1 == pytest.approx(1.0)

    unweighted = sum(c.f1 for c in r.per_class) / 3
    weighted = sum(c.f1 * c.support for c in r.per_class) / sum(c.support for c in r.per_class)
    assert unweighted != pytest.approx(weighted)
    assert r.macro_f1 == pytest.approx(unweighted)


def test_a_class_never_predicted_scores_zero_rather_than_dividing_by_zero():
    r = report([S, C], [S, S])
    by_label = {c.label: c for c in r.per_class}
    assert by_label[N].precision == 0.0 and by_label[N].recall == 0.0 and by_label[N].f1 == 0.0


def test_the_ceiling_travels_inside_the_report():
    """Accuracy without its bound is unreadable, so it must not be serializable alone."""
    payload = report([S], [S]).to_dict()
    assert payload["ceiling"]["verifiable"] == pytest.approx(0.774)
    assert "nei_share" in payload["ceiling"]


def test_by_retrieval_splits_on_what_the_model_read():
    labels = [S, S, C, C]
    predictions = [S, C, C, S]
    # first two had gold in the packed evidence, last two did not
    r = report(labels, predictions, gold_read=[True, True, False, False])
    assert r.by_retrieval.with_gold == pytest.approx(0.5)
    assert r.by_retrieval.without_gold == pytest.approx(0.5)
    assert r.by_retrieval.n_with_gold == 2 and r.by_retrieval.n_without_gold == 2


def test_nei_is_excluded_from_the_retrieval_split_rather_than_counted_a_miss():
    """
    NOT ENOUGH INFO has no gold to read, so counting it as a retrieval miss would blame retrieval
    for a claim that had nothing to retrieve.
    """
    labels = [S, C, N, N]
    predictions = [S, C, N, S]
    r = report(labels, predictions, gold_read=[True, True, None, None])
    assert r.by_retrieval.n_with_gold == 2
    assert r.by_retrieval.n_without_gold == 0
    assert r.by_retrieval.with_gold == pytest.approx(1.0)


def test_the_two_halves_recombine_to_verifiable_accuracy():
    labels = [S, S, C, C, C]
    predictions = [S, C, C, C, S]
    reads = [True, True, False, False, False]
    r = report(labels, predictions, gold_read=reads)
    b = r.by_retrieval
    combined = (b.with_gold * b.n_with_gold + b.without_gold * b.n_without_gold) / len(labels)
    assert combined == pytest.approx(r.accuracy)


def test_by_retrieval_is_absent_when_not_supplied():
    assert report([S, C], [S, C]).by_retrieval is None
    assert report([S, C], [S, C]).to_dict()["by_retrieval"] is None


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="labels and"):
        report([S, C], [S])
    with pytest.raises(ValueError, match="gold_read flags"):
        report([S, C], [S, C], gold_read=[True])


def test_an_empty_split_is_rejected():
    with pytest.raises(ValueError, match="empty split"):
        report([], [])


def test_a_label_outside_the_verdict_space_is_rejected():
    """A stray label would land in no confusion row and silently vanish from the totals."""
    with pytest.raises(ValueError, match="outside the verdict space"):
        report([S, "mostly_true"], [S, S])


def test_the_report_is_json_serializable():
    json.dumps(report([S, C, N], [S, C, N], gold_read=[True, False, None]).to_dict())


def test_an_empty_retrieval_bucket_is_undefined_not_zero():
    """
    claim_only reads no evidence, so nothing lands in with_gold. Reporting 0.0 there would read
    as the model getting every gold-read claim wrong, and that zero travels into metrics.json.
    """
    labels = ["supported", "contradicted", "not_enough_evidence"]
    report = evaluate_verdicts(
        labels, list(labels), variant="claim_only",
        ceiling=Ceiling(verifiable=0.0, overall=0.0, nei_share=1 / 3),
        gold_read=[False, False, None],
    )
    assert report.by_retrieval.n_with_gold == 0
    assert report.by_retrieval.with_gold is None
    assert report.by_retrieval.to_dict()["with_gold"] is None
    # The populated half still reports a number.
    assert report.by_retrieval.without_gold == 1.0
