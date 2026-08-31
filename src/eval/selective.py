"""
Selective prediction: what the system buys by declining to answer.

A fact checker that answers everything is judged on one number. A fact checker that may abstain is
judged on a curve, because it can trade coverage for accuracy, and the shape of that trade is the
product. Reporting accuracy alone for a system with an abstention option hides the only interesting
degree of freedom it has.

  coverage  the share of claims answered
  risk      the error rate among the claims it answered -- not among all claims, which is the
            mistake that makes abstention look free

The curve runs from answering everything (coverage 1, risk = the plain error rate) down toward
answering only the most confident. A system whose confidence is informative loses risk quickly as
coverage falls. A system whose confidence is noise traces a flat line, and abstention buys nothing
no matter how the threshold is tuned.

AURC summarises the curve, but it is contaminated by accuracy: a stronger model has a lower AURC
even with identically useless confidence. **E-AURC** subtracts the AURC an oracle ordering would
achieve at the same accuracy, so what is left measures the ranking alone. That is the number to
compare two calibrators with.

**Ties are operating points, not rows.** A threshold admits every claim sharing its confidence, so
coverage can only change where confidence strictly drops -- the same rule `fit_bands` follows for
the same reason. It matters here rather than being a technicality: `Isotonic` is piecewise
constant, so hundreds of claims can share one probability, and a curve evaluated at every k would
report operating points no threshold can actually reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Operating:
    """One reachable threshold: answer everything at or above `confidence`."""

    confidence: float
    coverage: float
    risk: float
    answered: int

    @property
    def accuracy(self) -> float:
        return 1.0 - self.risk

    def to_dict(self) -> dict[str, float | int]:
        return {
            "confidence": self.confidence, "coverage": self.coverage,
            "risk": self.risk, "accuracy": self.accuracy, "answered": self.answered,
        }


def _prepare(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if confidence.shape != correct.shape:
        raise ValueError(f"got {confidence.shape} confidences and {correct.shape} outcomes")
    if confidence.ndim != 1:
        raise ValueError(f"confidence must be 1-D, got shape {confidence.shape}")
    if confidence.size == 0:
        raise ValueError("cannot build a risk-coverage curve from an empty sample")
    return confidence, correct


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> list[Operating]:
    """
    Every reachable operating point, most selective first.

    Only thresholds where confidence strictly drops appear: a cut inside a run of equal
    confidences would describe a system that answers some of them and declines the rest, which no
    threshold can express.
    """
    confidence, correct = _prepare(confidence, correct)
    order = np.argsort(-confidence, kind="stable")
    ranked_confidence = confidence[order]
    errors = np.cumsum(~correct[order])
    n = len(confidence)

    points: list[Operating] = []
    for i in range(n):
        if i + 1 < n and ranked_confidence[i] == ranked_confidence[i + 1]:
            continue                       # mid-tie: no threshold stops here
        answered = i + 1
        points.append(Operating(
            confidence=float(ranked_confidence[i]),
            coverage=answered / n,
            risk=float(errors[i]) / answered,
            answered=answered,
        ))
    return points


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """
    Area under the risk-coverage curve, integrating risk over coverage.

    Each reachable point carries the coverage span it is responsible for, so a run of tied
    confidences contributes its whole width once rather than being silently skipped. With no ties
    this reduces exactly to the usual mean of risk over k = 1..n.
    """
    points = risk_coverage_curve(confidence, correct)
    total = 0.0
    previous = 0.0
    for point in points:
        total += point.risk * (point.coverage - previous)
        previous = point.coverage
    return float(total)


def optimal_aurc(correct: np.ndarray) -> float:
    """
    The AURC an oracle ranking reaches at this accuracy: every correct claim ordered above every
    error. Nonzero whenever the model is wrong at all, since full coverage must still carry them.
    """
    correct = np.asarray(correct, dtype=bool)
    n = len(correct)
    if n == 0:
        raise ValueError("cannot score an empty sample")
    hits = int(correct.sum())
    k = np.arange(1, n + 1)
    risk = np.maximum(0, k - hits) / k
    return float(risk.mean())


def excess_aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """
    AURC minus what an oracle ordering would achieve at the same accuracy.

    The comparable number. Raw AURC rewards a model for being accurate, which is already reported
    elsewhere; this isolates whether the confidence *ranks* claims usefully. Zero means the
    confidence orders errors perfectly; larger is worse.
    """
    return aurc(confidence, correct) - optimal_aurc(correct)


def coverage_at_risk(confidence: np.ndarray, correct: np.ndarray, target: float) -> Operating | None:
    """
    The most claims answerable while holding measured risk at or below `target`.

    None when no reachable threshold achieves it -- an outcome, not an error. A system that cannot
    reach 5% risk at any coverage should say so rather than return its least-bad point.
    """
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"target risk must lie in [0, 1], got {target}")
    admissible = [p for p in risk_coverage_curve(confidence, correct) if p.risk <= target]
    return max(admissible, key=lambda p: p.coverage) if admissible else None


def coverage_at_accuracy(confidence: np.ndarray, correct: np.ndarray, target: float) -> Operating | None:
    """Coverage reachable while holding accuracy at or above `target`. The reader-facing phrasing."""
    return coverage_at_risk(confidence, correct, 1.0 - target)


def risk_at_coverage(confidence: np.ndarray, correct: np.ndarray, target: float) -> Operating | None:
    """
    Risk at the least-selective threshold that still answers at least `target` of the claims.

    None when ties make that coverage unreachable from above -- with heavy ties the curve can jump
    across a requested level, and inventing a point inside the jump would describe a threshold
    that does not exist.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target coverage must lie in (0, 1], got {target}")
    admissible = [p for p in risk_coverage_curve(confidence, correct) if p.coverage >= target]
    return min(admissible, key=lambda p: p.coverage) if admissible else None


@dataclass(frozen=True, slots=True)
class SelectiveReport:
    label: str
    claims: int
    full_coverage_risk: float
    aurc: float
    optimal_aurc: float
    excess_aurc: float
    curve: list[Operating]
    at_risk: dict[str, Operating | None]
    at_coverage: dict[str, Operating | None]

    def to_dict(self) -> dict[str, object]:
        def maybe(point: Operating | None):
            return point.to_dict() if point is not None else None

        return {
            "label": self.label,
            "claims": self.claims,
            "full_coverage_risk": self.full_coverage_risk,
            "aurc": self.aurc,
            "optimal_aurc": self.optimal_aurc,
            "excess_aurc": self.excess_aurc,
            "at_risk": {k: maybe(v) for k, v in self.at_risk.items()},
            "at_coverage": {k: maybe(v) for k, v in self.at_coverage.items()},
            "curve": [p.to_dict() for p in self.curve],
        }


DEFAULT_RISKS = (0.05, 0.10, 0.15, 0.20)
DEFAULT_COVERAGES = (0.5, 0.7, 0.9)


def evaluate_selective(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    label: str,
    risks: tuple[float, ...] = DEFAULT_RISKS,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> SelectiveReport:
    confidence, correct = _prepare(confidence, correct)
    curve = risk_coverage_curve(confidence, correct)
    return SelectiveReport(
        label=label,
        claims=len(correct),
        full_coverage_risk=float((~correct).mean()),
        aurc=aurc(confidence, correct),
        optimal_aurc=optimal_aurc(correct),
        excess_aurc=excess_aurc(confidence, correct),
        curve=curve,
        at_risk={f"{r:.2f}": coverage_at_risk(confidence, correct, r) for r in risks},
        at_coverage={f"{c:.2f}": risk_at_coverage(confidence, correct, c) for c in coverages},
    )


def roc_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """
    Probability a random positive outscores a random negative, ties counting a half.

    Computed from rank sums rather than by sweeping thresholds, which makes tie handling exact
    rather than a function of how finely the sweep happened to step. That matters here because a
    sufficiency model over a handful of features produces many identical scores, and a sweep would
    quietly credit half of them.

    Undefined when one class is absent -- there is no pair to compare -- so it raises rather than
    returning 0.5, which would read as "no signal" instead of "no measurement".
    """
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    if score.shape != positive.shape:
        raise ValueError(f"got {score.shape} scores and {positive.shape} labels")
    hits, misses = int(positive.sum()), int((~positive).sum())
    if hits == 0 or misses == 0:
        raise ValueError(f"AUC needs both classes; got {hits} positive and {misses} negative")

    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # Average the ranks inside each tied run, which is what makes a tie worth exactly a half.
    sorted_scores = score[order]
    start = 0
    for stop in range(1, len(score) + 1):
        if stop == len(score) or sorted_scores[stop] != sorted_scores[start]:
            ranks[order[start:stop]] = (start + stop + 1) / 2
            start = stop
    return float((ranks[positive].sum() - hits * (hits + 1) / 2) / (hits * misses))
