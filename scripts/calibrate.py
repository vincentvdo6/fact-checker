"""
Fit the calibrator and the confidence bands on the calibration split, then freeze them.

This script is the only place in the project allowed to look at labels and choose something. Every
number Phase 03 reports comes from applying what is frozen here to a split this script never
opens. If the selection leaks -- if a calibrator is picked because it scored well on test, or a
band threshold is nudged until its promise holds -- then the promise is circular and the whole
contribution evaporates. There is no warning when that happens; the numbers simply get better.

Two nested fits share one held-out split, which is the subtlety worth stating plainly. The
calibrator is chosen and fitted on calibration, and the bands need calibrated probabilities to
find their thresholds. Fitting the bands on probabilities the calibrator already optimised makes
them inherit its overfitting and promise slightly more than they can keep. So the bands are fitted
on out-of-fold probabilities -- every row scored by a calibrator that never saw it -- and only then
is the winning calibrator refitted on all 2,000 rows for deployment.

Selection is by out-of-fold NLL rather than ECE. ECE depends on a binning choice and selecting on
it would reward whichever calibrator happens to suit the bin edges; NLL is a proper scoring rule
with nothing to tune.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.calibration.bands import BAND_ORDER, fit_bands
from src.calibration.crossfit import out_of_fold, select
from src.calibration.scaling import Isotonic, Temperature, Uncalibrated, VectorScaling
from src.eval.calibration import evaluate_calibration
from src.verdict.contract import installed_variants
from src.verdict.encode import LABELS

MODELS = Path("models/verdict")
CALIBRATION_FILE = "calibration.json"


# Uncalibrated is a candidate on purpose: if it wins out-of-fold, the honest thing is to ship no
# calibration rather than a transform that made the split it was fitted on look better.
CANDIDATES = {
    "uncalibrated": Uncalibrated,
    "temperature": Temperature,
    "vector_scaling": VectorScaling,
    "isotonic": Isotonic,
}


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    logits, labels, ids = [], [], []
    label_index = {name: i for i, name in enumerate(LABELS)}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            logits.append(row["logits"])
            labels.append(label_index[row["label"]])
            ids.append(row["id"])
    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64), ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=str(MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-support", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.models)
    available = [
        v for v in installed_variants(root)
        if (root / v / "predictions_calibration.jsonl").exists()
    ]
    if not available:
        raise SystemExit(
            f"no calibration predictions under {root}; install an artifact first:\n"
            "  python -m scripts.install_artifacts artifacts_<variant>_v1.zip"
        )

    for variant in available:
        logits, labels, _ = read_predictions(root / variant / "predictions_calibration.jsonl")
        print(f"\n{'=' * 68}\n{variant}  --  {len(labels):,} calibration claims")

        best, scores = select(CANDIDATES, logits, labels, k=args.folds, seed=args.seed)
        print(f"\n  {'calibrator':<16} {'oof NLL':>9} {'ECE':>8} {'adaECE':>8} {'Brier':>8} {'acc':>8}")
        for name, cls in CANDIDATES.items():
            report = evaluate_calibration(
                cls().fit(logits, labels).transform(logits), labels, label=name
            )
            mark = "  <-- selected" if name == best else ""
            print(f"  {name:<16} {scores[name]:>9.4f} {report.ece:>8.4f} "
                  f"{report.adaptive_ece:>8.4f} {report.brier:>8.4f} {report.accuracy:>8.4f}{mark}")
        print("  (ECE and accuracy above are in-fold, for orientation only; selection is the oof NLL)")

        # Bands see probabilities from calibrators that never saw the row they are scoring.
        oof = out_of_fold(CANDIDATES[best], logits, labels, k=args.folds, seed=args.seed)
        policy = fit_bands(
            oof.max(axis=1),
            oof.argmax(axis=1) == labels,
            min_support=args.min_support,
        )

        # Deployment fit: the same class, refit on everything now that selection is settled.
        calibrator = CANDIDATES[best]().fit(logits, labels)

        print(f"\n  bands, fitted on out-of-fold probabilities ({policy.fitted_on:,} claims)")
        print(f"  {'band':<10} {'threshold':>10} {'promised':>9} {'within':>8} {'cumul':>8} {'n':>7}")
        for band in BAND_ORDER:
            if band not in policy.thresholds:
                print(f"  {band.value:<10} {'--':>10}   not reachable at min_support="
                      f"{args.min_support} on this split")
                continue
            print(f"  {band.value:<10} {policy.thresholds[band]:>10.4f} "
                  f"{policy.targets[band]:>9.2f} {policy.within[band]:>8.4f} "
                  f"{policy.cumulative[band]:>8.4f} {policy.support[band]:>7,}")
        print(f"  coverage {policy.coverage:.4f}  "
              f"({policy.fitted_on - sum(policy.support.values()):,} claims abstained)")

        payload = {
            "variant": variant,
            "selected": best,
            "oof_nll": scores,
            "folds": args.folds,
            "seed": args.seed,
            "fitted_on": {"split": "calibration", "claims": int(len(labels))},
            "calibrator": calibrator.to_dict(),
            "bands": policy.to_dict(),
        }
        destination = root / variant / CALIBRATION_FILE
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  wrote {destination}")

    print(f"\n{'=' * 68}\nnext: python -m scripts.eval_calibration --split test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
