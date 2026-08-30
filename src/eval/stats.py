"""
Confidence intervals and paired significance tests for comparing model variants.

Three variants get trained on the same claims -- retrieved evidence, claim only, gold evidence --
and the questions asked of them are all differences: is the model better than the evidence-free
baseline, how far below the oracle does it sit. A difference between two systems scored on the
same claims is a paired quantity, and treating it as unpaired throws away the pairing and reports
an interval wider than the evidence supports.

Resampling is over claims, never over predictions. A claim is the unit that was sampled from the
world; a prediction is a deterministic function of one. Bootstrapping predictions would treat the
model's own arithmetic as a source of randomness.

No scipy. McNemar is a binomial tail that math.comb computes exactly, and the bootstrap is a
numpy resample -- neither justifies a dependency whose install would then have to be pinned
across the Kaggle boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval supports a claim that the difference has a sign."""
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> dict[str, float]:
        return {"point": self.point, "low": self.low, "high": self.high}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for a statistic of per-claim values."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    rng = _rng(seed)
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    stats = np.array([statistic(array[row]) for row in draws])
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(statistic(array)), float(low), float(high))


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """
    Interval on the mean difference a - b, resampling claims once and scoring both variants on
    the same draw.

    Drawing separately for each variant would discard the pairing that makes the comparison
    sharp: both systems saw the same claims, so a claim that is hard for one is usually hard for
    the other, and that shared difficulty cancels in the difference.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"paired samples must align; got {first.shape} and {second.shape}")
    return bootstrap_ci(first - second, resamples=resamples, alpha=alpha, seed=seed)


@dataclass(frozen=True, slots=True)
class McNemar:
    a_only: int      # claims a got right and b got wrong
    b_only: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.a_only + self.b_only

    def to_dict(self) -> dict[str, float | int]:
        return {"a_only": self.a_only, "b_only": self.b_only, "p_value": self.p_value}


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemar:
    """
    Exact two-sided McNemar test on the claims where the two variants disagree.

    Only discordant pairs carry information: a claim both got right, or both got wrong, says
    nothing about which is better. Under the null the discordant pairs split like a fair coin,
    so the p-value is an exact binomial tail rather than a chi-square approximation with a
    continuity correction -- the counts here run in the hundreds, where the two agree, but exact
    needs no caveat about when it stops being valid.
    """
    first = np.asarray(a_correct, dtype=bool)
    second = np.asarray(b_correct, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"paired samples must align; got {first.shape} and {second.shape}")

    a_only = int(np.sum(first & ~second))
    b_only = int(np.sum(~first & second))
    n = a_only + b_only
    if n == 0:
        # The variants never disagree, so there is no evidence either way.
        return McNemar(0, 0, 1.0)

    smaller = min(a_only, b_only)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return McNemar(a_only, b_only, min(1.0, 2 * tail))
