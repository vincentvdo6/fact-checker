"""
Features describing whether retrieval found enough to settle a claim.

Phase 03 measured the gap this exists to close. The verdict model's confidence barely moves when
retrieval missed the gold evidence -- it falls 0.0365 -- so abstention driven by confidence alone
declines the wrong claims: 18.6% of its abstentions were gold-missed against a 22.6% base rate.
Confidence is a statement about the softmax, and the softmax never saw how the evidence was found.

These features do. They come from the retriever rather than the verdict head, so they can say
"nothing here matched the claim well" on a claim the model is nonetheless happy to answer.

Everything is computed over the **read prefix** -- the first `budget` sentences, which is what
`n_evidence_used` records and what the model actually saw. The target these predict is
`recall_at_k(refs, groups, n_evidence_used)`, defined over exactly that prefix, so a feature
describing all 25 stored sentences would describe evidence the model was never shown.

The two families stay separate because the ablation depends on it: if retrieval features add
nothing over verdict features, the phase has learned that confidence was already all there was.

The names are the contract. They are ordered, exported, and asserted by test, because a model
fitted on one order and applied in another produces plausible numbers from scrambled inputs and
nothing raises.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Ref = tuple[str, int]

RETRIEVAL_NAMES: tuple[str, ...] = (
    "top_score",
    "mean_score",
    "min_score",
    "score_std",
    "score_decay",
    "score_margin",
    "score_ratio",
    "distinct_pages",
    "top_page_share",
    "first_page_change",
    "n_read",
)

VERDICT_NAMES: tuple[str, ...] = (
    "max_prob",
    "prob_margin",
    "entropy",
)

FEATURE_NAMES: tuple[str, ...] = RETRIEVAL_NAMES + VERDICT_NAMES


def _softmax(logits: Sequence[float]) -> list[float]:
    largest = max(logits)
    exponentiated = [math.exp(value - largest) for value in logits]
    total = sum(exponentiated)
    return [value / total for value in exponentiated]


def retrieval_features(
    evidence: Sequence[Ref], scores: Sequence[float], *, budget: int
) -> dict[str, float]:
    """
    Shape of the retrieval result over the sentences the model read.

    A claim whose top hit scores far above the rest, drawn from one page, looks like a claim whose
    entity was found. A flat spread across many pages looks like a claim retrieval never located,
    which is the case where the verdict head is reasoning over text about something else.
    """
    if len(evidence) != len(scores):
        raise ValueError(f"got {len(evidence)} refs and {len(scores)} scores")
    if budget < 0:
        raise ValueError(f"budget must not be negative, got {budget}")

    read = [float(s) for s in scores[:budget]]
    refs = [tuple(r) for r in evidence[:budget]]
    if not read:
        # claim_only reads nothing, and a claim can pack to zero sentences. Zeros throughout is
        # the honest encoding: there is no retrieval evidence to describe, and the head sees a
        # row that cannot look like a well-retrieved one.
        return dict.fromkeys(RETRIEVAL_NAMES, 0.0)

    top = read[0]
    mean = sum(read) / len(read)
    variance = sum((s - mean) ** 2 for s in read) / len(read)
    titles = [title for title, _ in refs]
    first_title = titles[0]

    return {
        "top_score": top,
        "mean_score": mean,
        "min_score": read[-1],
        "score_std": math.sqrt(variance),
        # How fast the ranking falls off. A steep fall means one clear match; a flat one means
        # the retriever found nothing it preferred.
        "score_decay": top - read[-1],
        "score_margin": top - read[1] if len(read) > 1 else 0.0,
        # Scale-free companion to decay: BM25 scores drift with claim length, and a ratio does
        # not, so the two disagree in exactly the cases where length is doing the work.
        "score_ratio": top / mean if mean > 0 else 0.0,
        "distinct_pages": float(len(set(titles))),
        "top_page_share": titles.count(first_title) / len(titles),
        # Rank where the ranking leaves the best page. Late means one page dominated.
        "first_page_change": float(
            next((i for i, t in enumerate(titles) if t != first_title), len(titles))
        ),
        "n_read": float(len(read)),
    }


def verdict_features(logits: Sequence[float]) -> dict[str, float]:
    """
    What the verdict head already knew. The ablation baseline, not a contribution.

    Entropy rather than max probability alone: a row split evenly between two classes and a row
    split evenly across three are both "unconfident" by the maximum, and they are different
    situations.
    """
    if len(logits) < 2:
        raise ValueError(f"need at least two logits, got {len(logits)}")
    probabilities = sorted(_softmax(logits), reverse=True)
    return {
        "max_prob": probabilities[0],
        "prob_margin": probabilities[0] - probabilities[1],
        "entropy": -sum(p * math.log(p) for p in probabilities if p > 0),
    }


def features_for(
    evidence: Sequence[Ref], scores: Sequence[float], logits: Sequence[float], *, budget: int
) -> dict[str, float]:
    """Both families for one claim, keyed by FEATURE_NAMES."""
    return retrieval_features(evidence, scores, budget=budget) | verdict_features(logits)


def as_row(features: dict[str, float], names: Sequence[str] = FEATURE_NAMES) -> list[float]:
    """
    Order a feature dict into a model row.

    Ordering by an explicit name tuple rather than by dict insertion: a model fitted on one order
    and applied in another still produces a number, and every metric downstream still computes.
    """
    missing = [name for name in names if name not in features]
    if missing:
        raise KeyError(f"missing features: {missing}")
    return [float(features[name]) for name in names]
