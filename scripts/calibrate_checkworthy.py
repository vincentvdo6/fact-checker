"""
Calibrate the check-worthiness detector, and choose its operating point on held-out debates.

The detector shipped uncalibrated -- raw softmax over three classes -- and this repository's whole
argument is that a raw softmax is not a probability. Phase 03 measured ECE 0.1287 on the verdict
head's raw output against 0.0314 after vector scaling, and bought 2.7 accuracy points for nothing
in the process. The same machinery applies here unchanged, which is the point of having built it
as `src/calibration/` rather than as part of the verdict pipeline.

**The threshold is chosen here and not in the comparison.** `scripts/compare_checkworthy.py` scores
against the 120 SOTU labels, which are the only held-out data this task has; picking an operating
point by sweeping there and keeping the best would spend that independence on a hyperparameter.
So the threshold is fitted on ClaimBuster's calibration debates -- 3,480 sentences the model never
trained on and which have nothing to do with the demo transcript -- and then frozen.

Selection is by out-of-fold NLL rather than in-fold ECE, for the reason Phase 03 recorded: isotonic
won the in-fold ECE every single time and lost out-of-fold every single time, because ECE depends
on a binning choice and selecting on it rewards whichever calibrator suits the bin edges.

What gets written is a frozen artifact beside the weights: the calibrator, the chosen threshold for
each binarization, and the numbers that justified them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.build_checkworthy_dataset import LABELS
from src.calibration.crossfit import select
from src.calibration.scaling import Isotonic, Temperature, Uncalibrated, VectorScaling
from src.eval.calibration import evaluate_calibration

MODEL = Path("models/checkworthy/v1")
CALIBRATION_FILE = "calibration.json"

CANDIDATES = {
    "uncalibrated": Uncalibrated,
    "temperature": Temperature,
    "vector_scaling": VectorScaling,
    "isotonic": Isotonic,
}

# Swept on calibration only. Wider than the comparison's sweep because this is where the choice is
# actually made, and a grid that stops short of the optimum hides it.
GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    index = {name: i for i, name in enumerate(LABELS)}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    logits = np.asarray([r["logits"] for r in rows], dtype=np.float64)
    labels = np.asarray([index[r["label"]] for r in rows], dtype=np.int64)
    return logits, labels


def best_threshold(scores: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """The grid point with the best F1, and that F1. Chosen on calibration, never on the demo."""
    best, best_f1 = 0.5, -1.0
    for cut in GRID:
        predicted = scores >= cut
        tp = int((predicted & actual).sum())
        precision = tp / max(int(predicted.sum()), 1)
        recall = tp / max(int(actual.sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best, best_f1 = cut, f1
    return best, best_f1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.model)
    logits, labels = read_predictions(root / "predictions_calibration.jsonl")
    print(f"{len(labels):,} calibration sentences from held-out debates")

    chosen, scores = select(CANDIDATES, logits, labels, k=args.folds, seed=args.seed)
    print(f"\n{'calibrator':<16} {'oof NLL':>9} {'ECE':>8} {'adaECE':>8} {'Brier':>8} {'acc':>8}")
    for name, cls in CANDIDATES.items():
        fitted = cls().fit(logits, labels)
        report = evaluate_calibration(fitted.transform(logits), labels, label=name)
        marker = "  <-- selected" if name == chosen else ""
        print(f"{name:<16} {scores[name]:>9.4f} {report.ece:>8.4f} {report.adaptive_ece:>8.4f} "
              f"{report.brier:>8.4f} {report.accuracy:>8.4f}{marker}")

    calibrator = CANDIDATES[chosen]().fit(logits, labels)
    raw = evaluate_calibration(
        Uncalibrated().fit(logits, labels).transform(logits), labels, label="uncalibrated"
    )
    fitted = evaluate_calibration(calibrator.transform(logits), labels, label=chosen)
    print(f"\nECE {raw.ece:.4f} -> {fitted.ece:.4f}   "
          f"accuracy {raw.accuracy:.4f} -> {fitted.accuracy:.4f}")

    # --- the operating point, on these debates and not on the demo -----------------------------
    probabilities = calibrator.transform(logits)
    factual_score = probabilities[:, 1] + probabilities[:, 2]
    worthy_score = probabilities[:, 2]
    thresholds = {}
    print(f"\n{'binarization':<16} {'threshold':>10} {'F1 here':>9} {'base rate':>10}")
    for name, score, actual in (
        ("factual", factual_score, labels >= 1),
        ("check_worthy", worthy_score, labels == 2),
    ):
        cut, f1 = best_threshold(score, actual)
        thresholds[name] = cut
        print(f"{name:<16} {cut:>10.2f} {f1:>9.4f} {actual.mean():>10.4f}")

    payload = {
        "selected": chosen,
        "calibrator": calibrator.to_dict(),
        "fitted_on": "ClaimBuster calibration split, 5 held-out debates",
        "claims": int(len(labels)),
        "out_of_fold_nll": scores,
        "ece": {"raw": raw.ece, "calibrated": fitted.ece},
        "accuracy": {"raw": raw.accuracy, "calibrated": fitted.accuracy},
        "thresholds": thresholds,
        "threshold_note": (
            "Swept on the calibration split only. The 120 SOTU labels are the sole held-out data "
            "for this task and are not spent on choosing a hyperparameter."
        ),
    }
    (root / CALIBRATION_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {root / CALIBRATION_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
