"""
Scoring a verdict model against what its evidence made possible.

Accuracy alone is unreadable here. A model that gets 70% might be reasoning well over evidence
that only contained the answer 77% of the time, or reasoning badly over evidence that almost
always did. Those call for opposite work, and the difference is invisible in a single number.

So every report carries its Ceiling, in the same structure rather than beside it, for the reason
Ceiling itself carries nei_share: a bound that can be dropped on the way to a slide deck will be.

`by_retrieval` is the measurement that actually separates the two failures. It splits verifiable
accuracy by whether the gold evidence was inside the packed prefix the model read -- not whether
retrieval returned it somewhere, but whether the model saw it. Accuracy on claims whose evidence
was present is reasoning quality; accuracy on the rest is what the model guesses when it has
nothing, and a model scoring well there is reading dataset artifacts rather than evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.eval.retrieval import Ceiling
from src.verdict.encode import LABELS


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int         # gold count, so the class shares are readable off the report

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class ByRetrieval:
    """
    Verifiable accuracy split by whether the model actually read the gold evidence.

    `without_gold` is the more informative half. The model has nothing to reason from there, so
    anything much above the class prior is the model recognising the claim rather than checking
    it -- the same artifact the evidence-free baseline exists to measure, seen from inside the
    real model.
    """

    # None, not 0.0, when the bucket is empty: an accuracy over no claims is undefined, and
    # a zero there reads as the model getting every one of them wrong. claim_only reads no
    # evidence at all, so its with_gold bucket is always empty.
    with_gold: float | None
    without_gold: float | None
    n_with_gold: int
    n_without_gold: int

    def to_dict(self) -> dict[str, object]:
        return {
            "with_gold": self.with_gold,
            "without_gold": self.without_gold,
            "n_with_gold": self.n_with_gold,
            "n_without_gold": self.n_without_gold,
        }


@dataclass(frozen=True, slots=True)
class VerdictReport:
    variant: str
    claims: int
    accuracy: float
    macro_f1: float
    per_class: tuple[ClassMetrics, ...]
    confusion: tuple[tuple[int, ...], ...]   # rows gold, columns predicted, in LABELS order
    ceiling: Ceiling
    by_retrieval: ByRetrieval | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "claims": self.claims,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": [c.to_dict() for c in self.per_class],
            "confusion": [list(row) for row in self.confusion],
            "labels": list(LABELS),
            "ceiling": self.ceiling.to_dict(),
            "by_retrieval": self.by_retrieval.to_dict() if self.by_retrieval else None,
        }


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)


def evaluate_verdicts(
    labels: Sequence[str],
    predictions: Sequence[str],
    *,
    variant: str,
    ceiling: Ceiling,
    gold_read: Sequence[bool | None] | None = None,
) -> VerdictReport:
    """
    Score predictions against gold labels.

    `gold_read` marks, per claim, whether the gold evidence was inside what the model read. None
    means the question does not apply -- NOT ENOUGH INFO has no gold to read -- and those claims
    are excluded from `by_retrieval` rather than counted as a miss, which would blame retrieval
    for a claim that had nothing to retrieve.
    """
    if len(labels) != len(predictions):
        raise ValueError(f"got {len(labels)} labels and {len(predictions)} predictions")
    if not labels:
        raise ValueError("cannot score an empty split")

    unknown = ({*labels} | {*predictions}) - set(LABELS)
    if unknown:
        raise ValueError(f"labels outside the verdict space: {sorted(unknown)}")

    index = {label: i for i, label in enumerate(LABELS)}
    confusion = [[0] * len(LABELS) for _ in LABELS]
    for gold, predicted in zip(labels, predictions, strict=True):
        confusion[index[gold]][index[predicted]] += 1

    correct = sum(confusion[i][i] for i in range(len(LABELS)))
    per_class: list[ClassMetrics] = []
    for i, label in enumerate(LABELS):
        true_positive = confusion[i][i]
        predicted_positive = sum(confusion[r][i] for r in range(len(LABELS)))
        actual_positive = sum(confusion[i])
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        per_class.append(ClassMetrics(label, precision, recall, _f1(precision, recall), actual_positive))

    by_retrieval = None
    if gold_read is not None:
        if len(gold_read) != len(labels):
            raise ValueError(f"got {len(gold_read)} gold_read flags and {len(labels)} labels")
        hits = [
            (gold == predicted, read)
            for gold, predicted, read in zip(labels, predictions, gold_read, strict=True)
            if read is not None
        ]
        with_gold = [ok for ok, read in hits if read]
        without_gold = [ok for ok, read in hits if not read]
        by_retrieval = ByRetrieval(
            with_gold=sum(with_gold) / len(with_gold) if with_gold else None,
            without_gold=sum(without_gold) / len(without_gold) if without_gold else None,
            n_with_gold=len(with_gold),
            n_without_gold=len(without_gold),
        )

    return VerdictReport(
        variant=variant,
        claims=len(labels),
        accuracy=correct / len(labels),
        macro_f1=sum(c.f1 for c in per_class) / len(per_class),
        per_class=tuple(per_class),
        confusion=tuple(tuple(row) for row in confusion),
        ceiling=ceiling,
        by_retrieval=by_retrieval,
    )
