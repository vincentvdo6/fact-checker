"""
Is the probability honest? Accuracy cannot answer, because none of this moves the argmax.

A model can rank perfectly and still be badly calibrated: if every claim it gets right is labelled
0.99 and every claim it gets wrong is labelled 0.97, the ordering is flawless and the numbers are
fiction. Selective prediction runs on the numbers, not the ordering -- an abstention threshold is
a statement about what 0.80 means -- so calibration error is the metric that decides whether the
rest of Phase 03 is standing on anything.

Three views, because each hides a different failure:

  ECE           the headline. Equal-width bins over confidence, weighted by occupancy. Its flaw
                is the binning: a model whose confidence piles into the top bin gets averaged
                against a handful of sparse low bins and looks better than it is.
  adaptive ECE  equal-mass bins instead. Every bin carries the same number of claims, so a
                confidence pile-up cannot hide inside one wide bin. This is the more honest
                number for a model like ours, whose confidence clusters high.
  MCE           the worst bin rather than the average. A system that is well calibrated overall
                and badly wrong in one region is exactly the system that surprises a reader,
                because the region it is wrong about is usually the confident one.

Brier is reported alongside as a proper scoring rule: unlike ECE it cannot be gamed by a constant
predictor, and unlike NLL it stays finite when a probability hits zero.

Everything here takes probabilities, not logits, so a calibrator sits upstream and this module
never needs to know which one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eval.stats import Interval, bootstrap_ci

DEFAULT_BINS = 15


@dataclass(frozen=True, slots=True)
class Bin:
    """One reliability bin. `confidence` is what the model claimed, `accuracy` what it delivered."""

    low: float
    high: float
    confidence: float
    accuracy: float
    count: int

    @property
    def gap(self) -> float:
        """Signed: positive means overconfident, which is the direction a reader is hurt by."""
        return self.confidence - self.accuracy

    def to_dict(self) -> dict[str, float | int]:
        return {
            "low": self.low, "high": self.high, "confidence": self.confidence,
            "accuracy": self.accuracy, "count": self.count, "gap": self.gap,
        }


def _prepare(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {probabilities.shape}")
    if len(probabilities) != len(labels):
        raise ValueError(f"got {len(probabilities)} probabilities and {len(labels)} labels")
    if len(probabilities) == 0:
        raise ValueError("cannot measure calibration on an empty sample")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    return confidence, correct


def _collect(
    confidence: np.ndarray, correct: np.ndarray, index: np.ndarray, bins: int, edges: np.ndarray
) -> list[Bin]:
    out: list[Bin] = []
    for slot in range(bins):
        members = index == slot
        count = int(members.sum())
        if count == 0:
            # Kept, with zeros, so a plot shows the gap rather than silently closing it.
            out.append(Bin(float(edges[slot]), float(edges[slot + 1]), 0.0, 0.0, 0))
            continue
        out.append(Bin(
            low=float(edges[slot]),
            high=float(edges[slot + 1]),
            confidence=float(confidence[members].mean()),
            accuracy=float(correct[members].mean()),
            count=count,
        ))
    return out


def _bins_equal_width(confidence: np.ndarray, correct: np.ndarray, bins: int) -> list[Bin]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    # right=True so a confidence of exactly 1.0 lands in the last bin instead of falling past it.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    return _collect(confidence, correct, index, bins, edges)


def _bins_equal_mass(confidence: np.ndarray, correct: np.ndarray, bins: int) -> list[Bin]:
    order = np.argsort(confidence, kind="stable")
    parts = np.array_split(order, bins)
    index = np.empty(len(confidence), dtype=np.int64)
    for slot, part in enumerate(parts):
        index[part] = slot
    edges = np.array([0.0] + [float(confidence[part].max()) if len(part) else 0.0 for part in parts])
    return _collect(confidence, correct, index, bins, edges)


def reliability_bins(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = DEFAULT_BINS, adaptive: bool = False
) -> list[Bin]:
    confidence, correct = _prepare(probabilities, labels)
    return (_bins_equal_mass if adaptive else _bins_equal_width)(confidence, correct, bins)


def _weighted_gap(binned: list[Bin], total: int) -> float:
    return float(sum(b.count * abs(b.gap) for b in binned) / total)


def ece(probabilities: np.ndarray, labels: np.ndarray, *, bins: int = DEFAULT_BINS) -> float:
    """Expected calibration error: occupancy-weighted mean gap over equal-width bins."""
    return _weighted_gap(reliability_bins(probabilities, labels, bins=bins), len(labels))


def adaptive_ece(probabilities: np.ndarray, labels: np.ndarray, *, bins: int = DEFAULT_BINS) -> float:
    """ECE over equal-mass bins, so a pile-up of confidence cannot hide inside one wide bin."""
    return _weighted_gap(reliability_bins(probabilities, labels, bins=bins, adaptive=True), len(labels))


def mce(probabilities: np.ndarray, labels: np.ndarray, *, bins: int = DEFAULT_BINS) -> float:
    """Maximum calibration error: the worst occupied bin, not the average."""
    occupied = [abs(b.gap) for b in reliability_bins(probabilities, labels, bins=bins) if b.count > 0]
    return float(max(occupied)) if occupied else 0.0


def brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot truth. Lower is better."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(probabilities) != len(labels):
        raise ValueError(f"got {len(probabilities)} probabilities and {len(labels)} labels")
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probabilities - onehot) ** 2).sum(axis=1).mean())


def nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log likelihood, clipped so a zero probability does not return infinity."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    picked = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    label: str
    claims: int
    accuracy: float
    ece: float
    adaptive_ece: float
    mce: float
    brier: float
    nll: float
    mean_confidence: float
    ece_interval: Interval | None
    bins: list[Bin]

    @property
    def overconfidence(self) -> float:
        """Mean confidence minus accuracy. Positive is the direction that misleads a reader."""
        return self.mean_confidence - self.accuracy

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "claims": self.claims,
            "accuracy": self.accuracy,
            "ece": self.ece,
            "adaptive_ece": self.adaptive_ece,
            "mce": self.mce,
            "brier": self.brier,
            "nll": self.nll,
            "mean_confidence": self.mean_confidence,
            "overconfidence": self.overconfidence,
            "ece_interval": self.ece_interval.to_dict() if self.ece_interval else None,
            "bins": [b.to_dict() for b in self.bins],
        }


def evaluate_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    label: str,
    bins: int = DEFAULT_BINS,
    resamples: int = 0,
    seed: int = 0,
) -> CalibrationReport:
    """
    Every calibration number for one set of probabilities.

    `resamples` above zero adds a bootstrap interval on ECE. It resamples claim *indices* through
    the existing `bootstrap_ci` rather than reimplementing the bootstrap: ECE is a binned
    aggregate, not a mean of per-claim values, so there is nothing to average directly.

    Read that interval as a spread, not as a confidence statement about the true ECE. ECE is an L1
    aggregate of bin gaps, so resampling noise can only inflate it -- for a well calibrated model
    the entire percentile interval sits *above* the point estimate. To decide whether one
    calibrator genuinely beats another, use `paired_bootstrap` on a per-claim proper score
    (Brier), which is unbiased and paired across the same claims; do not difference two ECEs.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidence, correct = _prepare(probabilities, labels)

    interval = None
    if resamples > 0:
        def resampled_ece(draw: np.ndarray) -> float:
            picked = draw.astype(np.int64)
            return ece(probabilities[picked], labels[picked], bins=bins)

        interval = bootstrap_ci(
            np.arange(len(labels), dtype=np.float64),
            statistic=resampled_ece,
            resamples=resamples,
            seed=seed,
        )

    return CalibrationReport(
        label=label,
        claims=len(labels),
        accuracy=float(correct.mean()),
        ece=ece(probabilities, labels, bins=bins),
        adaptive_ece=adaptive_ece(probabilities, labels, bins=bins),
        mce=mce(probabilities, labels, bins=bins),
        brier=brier(probabilities, labels),
        nll=nll(probabilities, labels),
        mean_confidence=float(confidence.mean()),
        ece_interval=interval,
        bins=reliability_bins(probabilities, labels, bins=bins),
    )
