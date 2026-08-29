"""
Confidence bands.

The model produces a calibrated probability and a reader never sees it. What they see
is a band, and each band carries a promise that was measured rather than chosen:
"strong" means claims shown as strong were right about nine times in ten on held-out
data. Thresholds are fitted on the calibration split and then frozen -- fitting them
on anything that overlaps test makes the promise circular.

Two accuracies are recorded per band and they are not the same number:

  within      accuracy of the band's own interval. This is what the scan fits against
              and the only one honest to show a reader, because someone looking at a
              "moderate" chip is looking at a moderate claim, not a moderate-or-better
              one.
  cumulative  accuracy of everything at or above the threshold. Reported as a
              diagnostic only. Fitting against it looks fine and quietly lies: a band
              can clear a 75% cumulative target while its own interval sits at 67%,
              because the stricter band above is carrying it.

A band that cannot reach its target, or can only reach it on fewer than `min_support`
examples, is left out entirely rather than given a lenient threshold. Falling back to
"strong is whatever the top decile happens to be" would hand the reader a promise the
split never supported, and an accuracy measured on a handful of rows is noise wearing
a promise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Band(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


# Strictest first. Both fitting and assignment depend on this order.
BAND_ORDER: tuple[Band, ...] = (Band.STRONG, Band.MODERATE, Band.WEAK)

DEFAULT_TARGETS: dict[Band, float] = {
    Band.STRONG: 0.90,
    Band.MODERATE: 0.75,
    Band.WEAK: 0.60,
}

# Below roughly this many examples a band's measured accuracy is noise, and the
# promise on the chip stops meaning anything.
DEFAULT_MIN_SUPPORT = 30


@dataclass(frozen=True, slots=True)
class BandPolicy:
    thresholds: dict[Band, float]
    within: dict[Band, float]
    cumulative: dict[Band, float]
    support: dict[Band, int]
    targets: dict[Band, float]
    min_support: int
    fitted_on: int

    def assign(self, probability: float) -> Band | None:
        """Band for a probability, or None to abstain."""
        for band in BAND_ORDER:
            threshold = self.thresholds.get(band)
            if threshold is not None and probability >= threshold:
                return band
        return None

    @property
    def coverage(self) -> float:
        """Share of the fitting split that received any band at all."""
        return sum(self.support.values()) / self.fitted_on

    def to_dict(self) -> dict[str, object]:
        return {
            "thresholds": {b.value: v for b, v in self.thresholds.items()},
            "within": {b.value: v for b, v in self.within.items()},
            "cumulative": {b.value: v for b, v in self.cumulative.items()},
            "support": {b.value: v for b, v in self.support.items()},
            "targets": {b.value: v for b, v in self.targets.items()},
            "min_support": self.min_support,
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BandPolicy:
        def floats(key: str) -> dict[Band, float]:
            return {Band(name): float(value) for name, value in payload[key].items()}

        return cls(
            thresholds=floats("thresholds"),
            within=floats("within"),
            cumulative=floats("cumulative"),
            support={Band(n): int(v) for n, v in payload["support"].items()},
            targets=floats("targets"),
            min_support=int(payload["min_support"]),
            fitted_on=int(payload["fitted_on"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BandPolicy:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_bands(
    probabilities: Sequence[float],
    correct: Sequence[bool],
    *,
    targets: dict[Band, float] | None = None,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> BandPolicy:
    """Fit band thresholds on a held-out split. See module docstring for the two accuracies."""
    targets = dict(DEFAULT_TARGETS if targets is None else targets)
    unknown = [str(key) for key in targets if not isinstance(key, Band)]
    if unknown:
        raise ValueError(f"unknown target key(s): {', '.join(unknown)}")
    missing = [band.value for band in BAND_ORDER if band not in targets]
    if missing:
        raise ValueError(f"no target given for {', '.join(missing)}")
    if len(probabilities) != len(correct):
        raise ValueError(f"got {len(probabilities)} probabilities and {len(correct)} labels")
    if len(probabilities) == 0:
        raise ValueError("cannot fit bands on an empty split")

    # float()/bool() rather than the values as given: numpy scalars survive the sort
    # but are not JSON-serializable, which would only surface when a policy is saved.
    ranked = sorted(
        ((float(p), bool(c)) for p, c in zip(probabilities, correct, strict=True)),
        key=lambda row: row[0],
        reverse=True,
    )

    # hits[i] is the number correct among the i highest-scoring predictions.
    hits: list[int] = [0]
    for _, is_correct in ranked:
        hits.append(hits[-1] + is_correct)

    def accuracy(first: int, last: int) -> float:
        """Accuracy over the inclusive index range [first, last]."""
        return (hits[last + 1] - hits[first]) / (last - first + 1)

    # A threshold admits every row sharing its probability, so a band may only end where
    # the next probability is strictly lower. Cutting mid-tie records an accuracy that
    # assign() cannot reproduce -- and isotonic calibration is piecewise-constant, so
    # ties are the normal case rather than an edge case.
    boundaries = [i for i in range(len(ranked)) if i + 1 == len(ranked) or ranked[i][0] > ranked[i + 1][0]]

    thresholds: dict[Band, float] = {}
    within: dict[Band, float] = {}
    cumulative: dict[Band, float] = {}
    support: dict[Band, int] = {}

    start = 0  # first index not yet claimed by a stricter band
    for band in BAND_ORDER:
        target = targets[band]
        end = next(
            (
                i
                for i in reversed(boundaries)
                if i >= start and i - start + 1 >= min_support and accuracy(start, i) >= target
            ),
            None,
        )
        if end is None:
            continue

        thresholds[band] = ranked[end][0]
        within[band] = accuracy(start, end)
        cumulative[band] = accuracy(0, end)
        support[band] = end - start + 1
        start = end + 1

    return BandPolicy(
        thresholds=thresholds,
        within=within,
        cumulative=cumulative,
        support=support,
        targets=targets,
        min_support=min_support,
        fitted_on=len(ranked),
    )
