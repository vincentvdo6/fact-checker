"""
Post-hoc calibrators: turn a trained model's logits into probabilities worth believing.

A cross-encoder trained with cross-entropy is systematically overconfident -- it is rewarded for
driving the correct logit up long after the ranking has stopped changing, so the softmax ends up
sharper than the accuracy justifies. None of this changes the argmax, so accuracy is blind to it
and only a calibration metric can see it.

Two distinct defects, and they need different instruments:

  sharpness   the whole distribution is too peaked. A single temperature fixes this, and cannot
              do anything else -- dividing every logit by one scalar is monotone per row, so the
              prediction never moves.
  prior shift the classes arrive in different proportions than in training. FEVER's train split
              is 55/20/25 and both dev halves are 33/33/33, so the model over-predicts supported
              (1,241 times on test where the truth is 682). A scalar temperature *structurally*
              cannot correct this: it has no per-class degree of freedom. A per-class bias does,
              and it moves the argmax, so it is the only calibrator here that can change accuracy.

`VectorScaling` is therefore the one expected to matter for this project. `Temperature` is kept
because it is the standard baseline and separates the two effects: if temperature alone fixes ECE
but not accuracy, the miscalibration was sharpness; if the bias is what helps, it was prior shift.

Everything here fits on the calibration split and nowhere else. Fitting on test roughly halves
the reported ECE and makes every promise downstream circular.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

# Temperature is searched in log space: the NLL is far more sensitive near 0 than near 10, and a
# bounded scalar search in log space spends its iterations where the curvature is.
LOG_T_BOUNDS = (np.log(0.05), np.log(20.0))


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(logits))


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log likelihood. The objective every calibrator here minimises."""
    return float(-log_softmax(logits)[np.arange(len(labels)), labels].mean())


def _check(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {logits.shape}")
    if len(logits) != len(labels):
        raise ValueError(f"got {len(logits)} logits and {len(labels)} labels")
    if len(logits) == 0:
        raise ValueError("cannot fit a calibrator on an empty split")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError(f"labels outside [0, {logits.shape[1]})")
    return logits, labels


@dataclass
class Uncalibrated:
    """Identity. Exists so "before" appears in the same table on the same footing as "after"."""

    name: str = "uncalibrated"

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> Uncalibrated:
        _check(logits, labels)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=np.float64))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Uncalibrated:
        return cls()


@dataclass
class Temperature:
    """
    One scalar, `logits / T`. The standard baseline (Guo et al. 2017).

    Monotone within a row, so it cannot change a single prediction -- accuracy before and after is
    identical by construction, and a test asserts exactly that. What it can do is fix sharpness.
    """

    temperature: float = 1.0
    name: str = "temperature"

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> Temperature:
        logits, labels = _check(logits, labels)
        result = minimize_scalar(
            lambda log_t: nll(logits / np.exp(log_t), labels),
            bounds=LOG_T_BOUNDS,
            method="bounded",
        )
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=np.float64) / self.temperature)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "temperature": self.temperature}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Temperature:
        return cls(temperature=float(payload["temperature"]))


@dataclass
class VectorScaling:
    """
    `logits / T + b`, with one bias per class. The calibrator that can move the argmax.

    The bias vector is the point. A scalar temperature has no per-class degree of freedom, so it
    cannot answer a prior shift; three biases can, by shifting the decision boundary toward the
    classes the evaluation split actually contains.

    Adding a constant to all three biases leaves the softmax exactly unchanged, so the optimum is
    a ray and not a point. In practice the fit lands centred on its own: the per-class gradients
    are `(softmax - onehot)` summed over rows, which cancels to zero across classes, so a
    zero-initialised descent never leaves the sum-zero hyperplane. The explicit centring below is
    there to keep that true if the initialisation or the optimiser ever changes -- it is a
    guardrail, not the mechanism, and `test_a_constant_shift_in_bias_changes_no_prediction` pins
    the property that makes it safe.
    """

    temperature: float = 1.0
    bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    name: str = "vector_scaling"

    def _objective(self, params: np.ndarray, logits: np.ndarray, labels: np.ndarray):
        """NLL and its gradient in (log T, b). Analytic, so L-BFGS-B converges in a few dozen."""
        log_t, bias = params[0], params[1:]
        inv_t = np.exp(-log_t)
        scaled = logits * inv_t + bias
        log_p = log_softmax(scaled)
        rows = np.arange(len(labels))
        loss = float(-log_p[rows, labels].mean())

        # d(loss)/d(scaled) = (softmax - onehot) / n
        residual = np.exp(log_p)
        residual[rows, labels] -= 1.0
        residual /= len(labels)

        grad_bias = residual.sum(axis=0)
        # scaled = logits * exp(-log_t) + b, so d(scaled)/d(log_t) = -logits * exp(-log_t)
        grad_log_t = float((residual * (-logits * inv_t)).sum())
        return loss, np.concatenate(([grad_log_t], grad_bias))

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> VectorScaling:
        logits, labels = _check(logits, labels)
        classes = logits.shape[1]
        result = minimize(
            self._objective,
            x0=np.zeros(1 + classes),
            args=(logits, labels),
            jac=True,
            method="L-BFGS-B",
            bounds=[LOG_T_BOUNDS] + [(-20.0, 20.0)] * classes,
        )
        bias = result.x[1:]
        self.temperature = float(np.exp(result.x[0]))
        self.bias = (bias - bias.mean()).tolist()
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float64) / self.temperature + np.asarray(self.bias)
        return softmax(scaled)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "temperature": self.temperature, "bias": list(self.bias)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> VectorScaling:
        return cls(temperature=float(payload["temperature"]), bias=[float(b) for b in payload["bias"]])


def pav(y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Pool adjacent violators: the isotonic (non-decreasing) least-squares fit of `y`.

    Exact and O(n) -- blocks are merged onto a stack while the previous block's mean exceeds the
    current one, which is the whole algorithm.
    """
    values: list[float] = []
    weights: list[float] = []
    for value, w in zip(y, weight, strict=True):
        values.append(float(value))
        weights.append(float(w))
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2 = values.pop(), weights.pop()
            v1, w1 = values.pop(), weights.pop()
            merged = w1 + w2
            values.append((v1 * w1 + v2 * w2) / merged)
            weights.append(merged)
    out = np.empty(len(y), dtype=np.float64)
    at = 0
    for value, w in zip(values, weights, strict=True):
        count = int(round(w))
        out[at:at + count] = value
        at += count
    return out


@dataclass
class Isotonic:
    """
    One-vs-rest isotonic regression per class, renormalised to a distribution.

    Non-parametric, so it can fit shapes the four-parameter calibrators cannot -- and can equally
    overfit 2,000 rows where they cannot. It is reported, never assumed to win; the selection in
    `scripts/calibrate.py` is made out-of-fold for exactly this reason.

    Renormalising breaks the per-class monotonicity slightly, which is the accepted cost of
    getting a distribution back out of three independent fits.
    """

    knots_x: list[list[float]] = field(default_factory=list)
    knots_y: list[list[float]] = field(default_factory=list)
    name: str = "isotonic"

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> Isotonic:
        logits, labels = _check(logits, labels)
        probabilities = softmax(logits)
        self.knots_x, self.knots_y = [], []
        for k in range(logits.shape[1]):
            column = probabilities[:, k]
            target = (labels == k).astype(np.float64)
            order = np.argsort(column, kind="stable")
            fitted = pav(target[order], np.ones(len(order)))
            self.knots_x.append(column[order].tolist())
            self.knots_y.append(fitted.tolist())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        probabilities = softmax(np.asarray(logits, dtype=np.float64))
        out = np.empty_like(probabilities)
        for k in range(probabilities.shape[1]):
            out[:, k] = np.interp(probabilities[:, k], self.knots_x[k], self.knots_y[k])
        # A row can be all-zero when every class is mapped to zero; fall back to the raw softmax
        # rather than dividing by zero and emitting NaN into a file Phase 04 reads.
        total = out.sum(axis=1, keepdims=True)
        degenerate = total[:, 0] <= 0
        out[degenerate] = probabilities[degenerate]
        total = out.sum(axis=1, keepdims=True)
        return out / total

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "knots_x": self.knots_x, "knots_y": self.knots_y}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Isotonic:
        return cls(
            knots_x=[[float(v) for v in col] for col in payload["knots_x"]],
            knots_y=[[float(v) for v in col] for col in payload["knots_y"]],
        )


CALIBRATORS = {
    "uncalibrated": Uncalibrated,
    "temperature": Temperature,
    "vector_scaling": VectorScaling,
    "isotonic": Isotonic,
}


def from_dict(payload: dict[str, object]):
    name = payload["name"]
    if name not in CALIBRATORS:
        raise ValueError(f"unknown calibrator {name!r}; known: {sorted(CALIBRATORS)}")
    return CALIBRATORS[name].from_dict(payload)


def save(calibrator, path: str | Path) -> None:
    Path(path).write_text(json.dumps(calibrator.to_dict(), indent=2), encoding="utf-8")


def load(path: str | Path):
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
