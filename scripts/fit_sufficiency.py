"""
Fit the sufficiency model and place it as a second, independent gate on answering.

Phase 03's gate asked one question -- is the softmax confident? -- and Checkpoint 1 showed that
question is nearly uninformative about whether the model was handed the evidence it needed
(AUC 0.5856 against the retriever's 0.7485). This script fits the retrieval-side answer and puts
it beside the confidence bands.

The two signals are kept apart rather than blended, because measuring on the calibration split
showed blending destroys information:

    sufficiency -> groundedness (gold_read)   AUC 0.7485
    sufficiency -> correctness                AUC 0.5311
    confidence  -> correctness                AUC 0.7169

Sufficiency predicts whether the evidence was there. It barely predicts whether the answer is
right, because the model answers 65% of gold-missed claims correctly anyway by reading FEVER's
claim-only artifacts. Multiplying the two scores therefore *degrades* the correctness ranking
(E-AURC 0.1136 -> 0.1492), and a two-parameter model given a free hand learns to nearly ignore
sufficiency -- weights +0.75 confidence against +0.09 sufficiency -- and still does not beat
confidence alone. Blending a groundedness signal into a correctness ranking is not a free lunch;
it is a category error, and the calibration split says so before test is opened.

So answering requires both conditions, independently:

    a confidence band is assigned   AND   predicted sufficiency >= 0.5

The threshold is the model's own decision boundary -- more likely than not that retrieval found
what the claim needed -- rather than a value tuned against an outcome. The confidence bands are
Phase 03's, fitted exactly as before, so the cost of the second condition appears as a difference
between two policies rather than being absorbed into moved thresholds.

Where each piece is fitted:

  sufficiency model   trainval, idle since Phase 02 model selection
  calibrator          calibration, unchanged, frozen by scripts/calibrate.py
  band thresholds     calibration, on out-of-fold probabilities, exactly as before
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from src.calibration.bands import BAND_ORDER, fit_bands
from src.calibration.crossfit import out_of_fold
from src.calibration.scaling import CALIBRATORS
from src.calibration.sufficiency import SufficiencyModel, rows_from
from src.data.fever import NOT_ENOUGH_INFO, load_claims
from src.data.splits import split_dev
from src.eval.retrieval import recall_at_k
from src.eval.selective import roc_auc
from src.retrieval.features import RETRIEVAL_NAMES, retrieval_features
from src.verdict.encode import LABELS
from src.verdict.labels import Verdict

DATA = Path("data/kaggle/fever-verdict-v1")
MODELS = Path("models/verdict")
RUNS = Path("runs")
DEV = "data/fever/shared_task_dev.jsonl"
SUFFICIENCY_FILE = "sufficiency.json"
NOT_ENOUGH_EVIDENCE = Verdict.NOT_ENOUGH_EVIDENCE.value

# The model's own decision boundary, not a tuned value: below this, retrieval is more likely than
# not to have missed what the claim needed.
SUFFICIENCY_THRESHOLD = 0.5


def read_dataset(split: str) -> list[dict]:
    path = DATA / f"verdict_{split}.jsonl.gz"
    opener, target = (gzip.open, path) if path.exists() else (open, DATA / f"verdict_{split}.jsonl")
    with opener(target, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_scores(directory: Path) -> dict[int, list[float]]:
    path = directory / "retrieved.jsonl"
    out: dict[int, list[float]] = {}
    with open(path, encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            if "scores" not in row:
                raise SystemExit(f"{path} has no scores; re-run retrieval for this split")
            out[row["id"]] = row["scores"]
    return out


def features_for_rows(rows: list[dict], scores: dict[int, list[float]], budgets: list[int]):
    """One feature dict per row, in row order, for every claim -- NEI included."""
    return [
        retrieval_features([(t, i) for t, i, _ in row["evidence"]], scores[row["id"]], budget=budget)
        for row, budget in zip(rows, budgets, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="retrieved")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=30)
    args = parser.parse_args()

    root = MODELS / args.variant
    frozen = json.loads((root / "calibration.json").read_text(encoding="utf-8"))

    from scripts.measure_sufficiency import budgets_from_encoding

    print("fitting sufficiency on trainval ...", flush=True)
    trainval_rows = read_dataset("trainval")
    trainval_scores = read_scores(RUNS / "evidence-trainval")
    trainval_budgets = budgets_from_encoding(trainval_rows, args.variant, args.max_length, args.seed)

    train_features, train_labels = [], []
    for row, budget in zip(trainval_rows, trainval_budgets, strict=True):
        if row["label"] == NOT_ENOUGH_EVIDENCE or not row["gold"]:
            continue                      # sufficiency is undefined where there is no gold
        refs = [(t, i) for t, i, _ in row["evidence"]]
        train_features.append(retrieval_features(refs, trainval_scores[row["id"]], budget=budget))
        train_labels.append(recall_at_k(refs, (tuple((t, i) for t, i, _ in row["gold"]),), budget))

    train_labels = np.asarray(train_labels, dtype=bool)
    model = SufficiencyModel(names=RETRIEVAL_NAMES).fit(
        rows_from(train_features, RETRIEVAL_NAMES), train_labels
    )
    print(f"  {len(train_labels):,} verifiable trainval claims, "
          f"gold_read base rate {train_labels.mean():.4f}")

    # --- place the gate on calibration ------------------------------------------------------
    rows = read_dataset("calibration")
    scores = read_scores(RUNS / "bm25-calibration")
    predictions = {
        row["id"]: row
        for row in map(json.loads, open(root / "predictions_calibration.jsonl", encoding="utf-8"))
    }
    budgets = [predictions[r["id"]]["n_evidence_used"] for r in rows]
    sufficiency = model.predict(rows_from(features_for_rows(rows, scores, budgets), RETRIEVAL_NAMES))

    label_index = {name: i for i, name in enumerate(LABELS)}
    logits = np.asarray([predictions[r["id"]]["logits"] for r in rows], dtype=np.float64)
    labels = np.asarray([label_index[r["label"]] for r in rows], dtype=np.int64)

    # Out-of-fold for the reason Phase 03 did it: bands placed on probabilities the calibrator
    # already optimised promise more than they can keep.
    oof = out_of_fold(CALIBRATORS[frozen["selected"]], logits, labels, k=args.folds, seed=args.seed)
    confidence = oof.max(axis=1)
    correct = oof.argmax(axis=1) == labels

    policy = fit_bands(confidence, correct, min_support=args.min_support)
    banded = np.asarray([policy.assign(float(c)) is not None for c in confidence])
    grounded = sufficiency >= SUFFICIENCY_THRESHOLD

    calibration_claims, _ = split_dev(load_claims(DEV))
    groups = {c.id: c.groups for c in calibration_claims if c.label != NOT_ENOUGH_INFO}
    verifiable = np.asarray([r["id"] in groups for r in rows])
    gold_read = np.asarray([
        bool(r["id"] in groups
             and recall_at_k([(t, i) for t, i, _ in r["evidence"]], groups[r["id"]], budget))
        for r, budget in zip(rows, budgets, strict=True)
    ])

    auc = roc_auc(sufficiency[verifiable], gold_read[verifiable])
    print(f"\nsufficiency on calibration: AUC {auc:.4f} against gold_read")
    print(f"  mean when gold read   {sufficiency[verifiable & gold_read].mean():.4f}")
    print(f"  mean when gold missed {sufficiency[verifiable & ~gold_read].mean():.4f}")

    base_rate = float((verifiable & ~gold_read).sum() / max(verifiable.sum(), 1))
    header = f"{'policy':<26} {'coverage':>9} {'risk':>8} {'declined':>9} {'gold-missed share':>19}"
    print("\n" + header)
    summary = {}
    for name, answered in (("confidence only", banded), ("plus sufficiency", banded & grounded)):
        declined = ~answered
        share = float((verifiable & ~gold_read & declined).sum() / max(declined.sum(), 1))
        risk = float((~correct[answered]).mean()) if answered.any() else float("nan")
        print(f"{name:<26} {answered.mean():>9.4f} {risk:>8.4f} {int(declined.sum()):>9,} "
              f"{share:>18.1%}")
        summary[name] = {
            "coverage": float(answered.mean()), "risk": risk,
            "declined": int(declined.sum()), "gold_missed_share": share,
        }
    print(f"{'(base rate among verifiable)':<26} {'':>9} {'':>8} {'':>9} {base_rate:>18.1%}")

    print(f"\n{'band':<10} {'promised':>9} {'threshold':>10} {'within':>8} {'n':>8}")
    for band in BAND_ORDER:
        if band in policy.thresholds:
            print(f"{band.value:<10} {policy.targets[band]:>9.2f} {policy.thresholds[band]:>10.4f} "
                  f"{policy.within[band]:>8.4f} {policy.support[band]:>8,}")
        else:
            print(f"{band.value:<10} {policy.targets[band]:>9.2f} {'--':>10}   not reachable")

    payload = {
        "variant": args.variant,
        "fitted_on": {"sufficiency": "trainval", "bands": "calibration (out-of-fold)"},
        "gate": "band assigned AND sufficiency >= threshold",
        "threshold": SUFFICIENCY_THRESHOLD,
        "trainval_claims": int(len(train_labels)),
        "trainval_base_rate": float(train_labels.mean()),
        "calibration_auc": float(auc),
        "calibration_base_rate": base_rate,
        "calibration_summary": summary,
        "model": model.to_dict(),
        "bands": policy.to_dict(),
    }
    (root / SUFFICIENCY_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {root / SUFFICIENCY_FILE}")
    print("next: python -m scripts.eval_sufficiency --split test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
