"""
A model of whether retrieval found enough to settle a claim, independent of the verdict.

Phase 03 ended on a specific failure: abstention driven by the verdict head's confidence declines
the wrong claims. Confidence falls only 0.0365 when the gold evidence was missed, and 18.6% of
abstentions were gold-missed against a 22.6% base rate -- the system was slightly *less* likely to
decline a claim it could not answer. The verdict head cannot see this, because it is handed text
and never told how that text was found.

This model is told. It reads the shape of the retrieval result -- how far the top score sits above
the rest, how fast the ranking decays, whether one page dominates -- and predicts whether a
complete gold group was inside what the model read. That prediction is a second gate on answering,
orthogonal to confidence.

Deliberately a logistic regression over about a dozen named features, not something larger. The
target has a few thousand training rows, the features are hand-built and interpretable, and a
model whose coefficients can be read is one whose failure can be explained. It also keeps the
whole phase on CPU.

Features are standardised because they are not commensurable: BM25 scores run to the tens while
probabilities sit in the unit interval, and an unstandardised L2 penalty would fall almost
entirely on the probabilities. Mean and scale come from the fitting split alone and travel with
the model, so applying it to a new split cannot leak that split's distribution into its own
normalisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.retrieval.features import FEATURE_NAMES, as_row

# Enough to keep a separating direction from running away on a feature that happens to be nearly
# separable in a few thousand rows, and small enough not to flatten a real signal.
DEFAULT_L2 = 1.0


@dataclass
class SufficiencyModel:
    """Logistic regression over named features, carrying its own standardisation."""

    names: tuple[str, ...] = FEATURE_NAMES
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    intercept: float = 0.0
    l2: float = DEFAULT_L2

    def _standardise(self, rows: np.ndarray) -> np.ndarray:
        return (rows - np.asarray(self.mean)) / np.asarray(self.scale)

    def fit(self, rows: np.ndarray, positive: np.ndarray) -> SufficiencyModel:
        rows = np.asarray(rows, dtype=np.float64)
        positive = np.asarray(positive, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError(f"rows must be 2-D, got shape {rows.shape}")
        if len(rows) != len(positive):
            raise ValueError(f"got {len(rows)} rows and {len(positive)} labels")
        if rows.shape[1] != len(self.names):
            raise ValueError(f"got {rows.shape[1]} columns for {len(self.names)} feature names")
        if len(set(positive.tolist())) < 2:
            raise ValueError("cannot fit on a single class")

        self.mean = rows.mean(axis=0).tolist()
        # A constant column has zero spread; dividing by 1 leaves it at zero, which the penalty
        # then holds at zero weight rather than producing a division by zero.
        spread = rows.std(axis=0)
        self.scale = np.where(spread > 0, spread, 1.0).tolist()
        standardised = self._standardise(rows)

        def objective(params: np.ndarray):
            weights, intercept = params[:-1], params[-1]
            margin = standardised @ weights + intercept
            # log(1 + exp(-y*m)) via logaddexp, which does not overflow for large |margin|.
            signed = np.where(positive > 0, margin, -margin)
            loss = float(np.logaddexp(0.0, -signed).mean() + self.l2 * (weights @ weights) / 2 / len(rows))
            residual = (1.0 / (1.0 + np.exp(-margin))) - positive
            grad_weights = standardised.T @ residual / len(rows) + self.l2 * weights / len(rows)
            return loss, np.concatenate([grad_weights, [float(residual.mean())]])

        result = minimize(
            objective, x0=np.zeros(len(self.names) + 1), jac=True, method="L-BFGS-B"
        )
        self.weights = result.x[:-1].tolist()
        self.intercept = float(result.x[-1])
        return self

    def predict(self, rows: np.ndarray) -> np.ndarray:
        """Probability that retrieval was sufficient, one per row."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self.names):
            raise ValueError(f"expected rows of width {len(self.names)}, got shape {rows.shape}")
        margin = self._standardise(rows) @ np.asarray(self.weights) + self.intercept
        return 1.0 / (1.0 + np.exp(-margin))

    def coefficients(self) -> dict[str, float]:
        """
        Weights in standardised units, so they are comparable across features.

        The point of a small model: these are readable, and a sufficiency signal that turns out to
        rest entirely on n_read is a different finding from one that rests on score_margin.
        """
        return dict(zip(self.names, self.weights, strict=True))

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "mean": self.mean,
            "scale": self.scale,
            "weights": self.weights,
            "intercept": self.intercept,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SufficiencyModel:
        return cls(
            names=tuple(payload["names"]),
            mean=[float(v) for v in payload["mean"]],
            scale=[float(v) for v in payload["scale"]],
            weights=[float(v) for v in payload["weights"]],
            intercept=float(payload["intercept"]),
            l2=float(payload["l2"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SufficiencyModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def rows_from(features: list[dict[str, float]], names: tuple[str, ...]) -> np.ndarray:
    """Feature dicts to a matrix, ordered by `names` rather than by dict insertion."""
    return np.asarray([as_row(f, names) for f in features], dtype=np.float64)
