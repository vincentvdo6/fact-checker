"""
Fit the pair judge's calibrator and bands on its calibration logits, and report them on test.

The same procedure Phase 03 applied to the verdict model, on the four-relation head: candidate
calibrators compete out of fold on the calibration split, the winner is refitted on all of it,
and band thresholds are cut there. Test is touched once, at the end, and every number it reports
is quoted beside the subgroup it was measured on -- source (FEVER or MNLI), negated hypothesis,
and each band -- because the extension will count a sentence only on a banded `states` or
`states_negation`, and the counting error at each band is the number that decides whether it may.

Two band policies are fitted. The first is Phase 03's, on every pair. The second is the one the
chain uses: fitted only on pairs the judge would *count*, with targets stated here in advance --
strong 0.95, moderate 0.90, weak 0.85 -- because a counted sentence moves a verdict and a
bears_on sentence shown as relevant does not. Overall accuracy near 0.89 lets the first policy's
0.90 band swallow almost everything; the second answers the question that matters.

Nothing here says anything about news sentences. That measurement is `scripts/eval_pair_judge.py`
on the hand-labelled pairs.

    python -m scripts.calibrate_pair_judge --models models/pair_judge/v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.calibration.bands import BAND_ORDER, Band, fit_bands
from src.calibration.crossfit import out_of_fold, select
from src.calibration.scaling import Isotonic, Temperature, Uncalibrated, VectorScaling, nll
from src.eval.calibration import ece
from src.verdict.pair_judgment import RELATIONS

CANDIDATES = {"uncalibrated": Uncalibrated, "temperature": Temperature,
              "vector_scaling": VectorScaling, "isotonic": Isotonic}
COUNTED = {RELATIONS.index("states"), RELATIONS.index("states_negation")}
COUNTING_TARGETS = {Band.STRONG: 0.95, Band.MODERATE: 0.90, Band.WEAK: 0.85}


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    index = {name: i for i, name in enumerate(RELATIONS)}
    logits, labels, rows = [], [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            logits.append(row["logits"])
            labels.append(index[row["relation"]])
            rows.append({"source": row["source"], "negated": bool(row["negated_hypothesis"])})
    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64), rows


def counting_bands(probabilities: np.ndarray, labels: np.ndarray, *, min_support: int):
    """Bands over the counted predictions only: the promise is about sentences that would move a verdict."""
    predicted = probabilities.argmax(axis=1)
    mask = np.isin(predicted, list(COUNTED))
    return fit_bands(probabilities.max(axis=1)[mask], (predicted == labels)[mask],
                     targets=COUNTING_TARGETS, min_support=min_support)


def subgroup(probabilities: np.ndarray, labels: np.ndarray, rows: list[dict], bands, counting=None) -> dict:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    counted = np.isin(predicted, list(COUNTED))
    out = {"n": int(len(labels)), "accuracy": float(correct.mean()),
           "ece": float(ece(probabilities, labels)),
           "counting_error": float((counted & ~correct).sum() / max(counted.sum(), 1)), "counted": int(counted.sum())}
    for name, mask in (("fever", np.asarray([r["source"] == "fever" for r in rows])),
                       ("mnli", np.asarray([r["source"] == "mnli" for r in rows])),
                       ("negated_hypothesis", np.asarray([r["negated"] for r in rows]))):
        if mask.any():
            out[name] = {"n": int(mask.sum()), "accuracy": float(correct[mask].mean()),
                         "counting_error": float((counted & ~correct & mask).sum() / max((counted & mask).sum(), 1))}
    assigned = np.asarray([bands.assign(float(value)) for value in confidence], dtype=object)
    out["bands"] = {}
    for band in BAND_ORDER:
        mask = assigned == band
        if mask.any():
            out["bands"][band.value] = {"n": int(mask.sum()), "accuracy": float(correct[mask].mean()),
                                        "counting_error": float((counted & ~correct & mask).sum() / max((counted & mask).sum(), 1)),
                                        "counted": int((counted & mask).sum())}
    out["abstained"] = int((assigned == None).sum())  # noqa: E711
    if counting is not None:
        out["counting_bands"] = {}
        assigned = np.asarray([counting.assign(float(value)) if flag else None
                               for value, flag in zip(confidence, counted)], dtype=object)
        for band in BAND_ORDER:
            mask = assigned == band
            if mask.any():
                out["counting_bands"][band.value] = {
                    "counted": int(mask.sum()), "share_of_counted": float(mask.sum() / max(counted.sum(), 1)),
                    "counting_error": float((mask & ~correct).sum() / mask.sum()),
                    "promised": 1 - COUNTING_TARGETS[band]}
        out["counting_bands"]["abstained"] = int((counted & (assigned == None)).sum())  # noqa: E711
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="models/pair_judge/v4")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-support", type=int, default=30)
    args = parser.parse_args()
    root = Path(args.models)
    logits, labels, rows = read_predictions(root / "predictions_calibration.jsonl")
    best, scores = select(CANDIDATES, logits, labels, k=args.folds, seed=args.seed)
    print(f"out-of-fold NLL on {len(labels):,} calibration pairs: " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items()))
    print(f"selected {best}")
    calibrator = CANDIDATES[best]().fit(logits, labels)
    oof = out_of_fold(CANDIDATES[best], logits, labels, k=args.folds, seed=args.seed)
    bands = fit_bands(oof.max(axis=1), oof.argmax(axis=1) == labels, min_support=args.min_support)
    counting = counting_bands(oof, labels, min_support=args.min_support)
    payload = {"variant": "pair_judge", "selected": best, "oof_nll": scores, "folds": args.folds, "seed": args.seed,
               "fitted_on": {"split": "calibration", "pairs": int(len(labels))},
               "calibrator": calibrator.to_dict(), "bands": bands.to_dict(), "counting_bands": counting.to_dict(),
               "counting_targets": {band.value: target for band, target in COUNTING_TARGETS.items()},
               "calibration_nll": {"raw": nll(logits, labels), "calibrated": float(-np.log(
                   calibrator.transform(logits)[np.arange(len(labels)), labels]).mean())}}
    (root / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = {"calibration_oof": subgroup(oof, labels, rows, bands, counting)}
    test = root / "predictions_test.jsonl"
    if test.exists():
        t_logits, t_labels, t_rows = read_predictions(test)
        report["test_raw"] = subgroup(Uncalibrated().transform(t_logits), t_labels, t_rows, bands, counting)
        report["test_calibrated"] = subgroup(calibrator.transform(t_logits), t_labels, t_rows, bands, counting)
    (root / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, block in report.items():
        print(f"\n{name}: n={block['n']:,} accuracy {block['accuracy']:.4f} ECE {block['ece']:.4f} "
              f"counting error {block['counting_error']:.4f} over {block['counted']:,} counted, abstained {block['abstained']:,}")
        for key in ("fever", "mnli", "negated_hypothesis"):
            if key in block:
                print(f"  {key:<20} n={block[key]['n']:>6,} accuracy {block[key]['accuracy']:.4f} counting error {block[key]['counting_error']:.4f}")
        for band in Band:
            if band.value in block["bands"]:
                row = block["bands"][band.value]
                print(f"  band {band.value:<9} n={row['n']:>6,} accuracy {row['accuracy']:.4f} counting error {row['counting_error']:.4f} over {row['counted']:,}")
        for band in Band:
            if band.value in block.get("counting_bands", {}):
                row = block["counting_bands"][band.value]
                print(f"  counting {band.value:<9} counted={row['counted']:>6,} ({row['share_of_counted']:.1%}) "
                      f"error {row['counting_error']:.4f}  promised <= {row['promised']:.2f}")
        if "counting_bands" in block:
            print(f"  counting abstained  {block['counting_bands']['abstained']:,} counted predictions earn no band")
    print(f"\nwrote {root / 'calibration.json'} and calibration_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
