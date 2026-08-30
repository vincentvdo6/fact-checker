"""
Out-of-fold calibrated probabilities, so the bands are fitted on numbers nobody has seen.

The calibrator and the confidence bands both need held-out data, and the project has exactly one
held-out split to give them. Fitting the calibrator on all 2,000 calibration rows and then fitting
band thresholds on those same rows means the thresholds are chosen against probabilities the
calibrator already optimised -- the bands inherit its overfitting and promise slightly more than
they can keep.

The bias is small with four parameters. It is not small with isotonic, which is exactly the
calibrator most likely to look best on the split it was fitted on. Since the band promise is the
deliverable of this whole project -- "claims shown as strong were right nine times in ten" -- the
one place not to accept a small known bias is here.

So: K folds, each row scored by a calibrator that never saw it, bands fitted on the assembled
out-of-fold probabilities, and the deployed calibrator refitted on everything afterwards. Costs
no data and one extra pass. The same machinery answers "which calibrator generalises best" without
consulting test, which is the other thing that would quietly invalidate the report.
"""

from __future__ import annotations

import numpy as np

from src.calibration.scaling import nll


def folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    """K index arrays, shuffled once, as equal in size as n allows."""
    if k < 2:
        raise ValueError(f"need at least 2 folds, got {k}")
    if n < k:
        raise ValueError(f"cannot split {n} rows into {k} folds")
    order = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(order, k)]


def out_of_fold(
    calibrator_cls,
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """
    Calibrated probabilities where every row was scored by a fit that excluded it.

    The returned array is in the original row order, so it lines up with `labels` and with the
    prediction file it came from.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    out = np.empty_like(logits)
    for held in folds(len(logits), k, seed):
        keep = np.setdiff1d(np.arange(len(logits)), held, assume_unique=True)
        fitted = calibrator_cls().fit(logits[keep], labels[keep])
        out[held] = fitted.transform(logits[held])
    return out


def select(
    candidates: dict[str, type],
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 5,
    seed: int = 0,
) -> tuple[str, dict[str, float]]:
    """
    Pick the calibrator with the best out-of-fold NLL, and return every score for the record.

    NLL rather than ECE: ECE depends on a binning choice, and selecting on it rewards whichever
    calibrator happens to suit the bin edges. NLL is a proper scoring rule with nothing to tune.
    """
    scores: dict[str, float] = {}
    for name, cls in candidates.items():
        probabilities = out_of_fold(cls, logits, labels, k=k, seed=seed)
        # Back to log space for a proper score; clip keeps a zero from becoming -inf.
        scores[name] = nll(np.log(np.clip(probabilities, 1e-12, None)), labels)
    best = min(scores, key=lambda name: scores[name])
    return best, scores
