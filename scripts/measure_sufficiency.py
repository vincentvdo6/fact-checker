"""
Checkpoint 1: is evidence sufficiency learnable at all, and does retrieval know something
confidence does not?

Phase 03 ended on a measured failure. Abstention driven by the verdict head's confidence declined
the wrong claims -- 18.6% of its abstentions were gold-missed against a 22.6% base rate -- because
confidence describes the softmax and the softmax never saw how the evidence was found. Phase 04
proposes that the retriever knows. This script decides whether that is true before anything is
built on it.

The target is `gold_read`: was a complete gold group inside the sentences the model actually read.
Defined on verifiable claims only, since NOT ENOUGH INFO has no gold by construction and counting
it as a retrieval failure would invent one.

Three fits, answering three different questions:

  verdict only     what Phase 03's confidence already had. The baseline to beat.
  retrieval only   is sufficiency visible in the shape of the retrieval result at all.
  both             does retrieval carry anything the verdict head did not already know.

The retrieval model fits on **trainval**, which has been idle since Phase 02 model selection, and
is scored on calibration -- so its number involves no reuse at all. The verdict and combined
models need logits, which exist only for calibration and test, so they are cross-fitted within
calibration and scored out-of-fold. Both estimates are honest; they are not fitted the same way,
and the printout says so rather than implying a cleaner comparison than exists.

The gate, fixed before the number was seen: AUC >= 0.70 proceeds, 0.60-0.70 is weak and gets
reported as such, below 0.60 is a negative result and the phase pivots rather than pressing on.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from src.calibration.crossfit import folds
from src.calibration.sufficiency import SufficiencyModel, rows_from
from src.data.fever import NOT_ENOUGH_INFO, load_claims
from src.data.splits import split_dev
from src.eval.retrieval import recall_at_k
from src.eval.selective import roc_auc
from src.retrieval.features import (
    FEATURE_NAMES,
    RETRIEVAL_NAMES,
    VERDICT_NAMES,
    retrieval_features,
    verdict_features,
)
from src.verdict.budgets import budgets_for
from src.verdict.labels import Verdict

DATA = Path("data/kaggle/fever-verdict-v1")
MODELS = Path("models/verdict")
RUNS = Path("runs")
DEV = "data/fever/shared_task_dev.jsonl"

GATE_STRONG, GATE_WEAK = 0.70, 0.60
NOT_ENOUGH_EVIDENCE = Verdict.NOT_ENOUGH_EVIDENCE.value


def read_dataset(split: str) -> list[dict]:
    path = DATA / f"verdict_{split}.jsonl.gz"
    opener, target = (gzip.open, path) if path.exists() else (open, DATA / f"verdict_{split}.jsonl")
    with opener(target, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_scores(directory: Path) -> dict[int, list[float]]:
    path = directory / "retrieved.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} does not exist; re-run retrieval to persist scores")
    out: dict[int, list[float]] = {}
    with open(path, encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            if "scores" not in row:
                raise SystemExit(
                    f"{path} predates the score-persisting writer; re-run retrieval for this split"
                )
            out[row["id"]] = row["scores"]
    return out


def budgets_from_encoding(rows: list[dict], variant: str, max_length: int, seed: int) -> list[int]:
    """The encoder's packing budget per row; see src/verdict/budgets.py for why it is recomputed."""
    return budgets_for(
        rows,
        variant=variant,
        max_length=max_length,
        seed=seed,
        tokenizer_source=str(MODELS / variant / "model_v1"),
    )


def assemble(rows: list[dict], scores: dict[int, list[float]], budgets: list[int],
             groups: dict[int, object], logits: dict[int, list[float]] | None):
    """Feature dicts and gold_read labels for the verifiable claims of one split."""
    features, labels, ids = [], [], []
    for row, budget in zip(rows, budgets, strict=True):
        claim_id = row["id"]
        if row["label"] == NOT_ENOUGH_EVIDENCE or claim_id not in groups:
            continue                          # NEI has no gold; sufficiency is undefined for it
        if claim_id not in scores:
            continue
        refs = [(t, i) for t, i, _ in row["evidence"]]
        entry = retrieval_features(refs, scores[claim_id], budget=budget)
        if logits is not None:
            entry |= verdict_features(logits[claim_id])
        features.append(entry)
        labels.append(recall_at_k(refs, groups[claim_id], budget))
        ids.append(claim_id)
    return features, np.asarray(labels, dtype=bool), ids


def out_of_fold_auc(features, labels, names, *, k: int = 5, seed: int = 0) -> float:
    """Fit and score within one split without a row ever scoring itself."""
    rows = rows_from(features, names)
    predicted = np.empty(len(labels), dtype=np.float64)
    for held in folds(len(rows), k, seed):
        keep = np.setdiff1d(np.arange(len(rows)), held, assume_unique=True)
        model = SufficiencyModel(names=names).fit(rows[keep], labels[keep])
        predicted[held] = model.predict(rows[held])
    return roc_auc(predicted, labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="retrieved")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    calibration_claims, _ = split_dev(load_claims(DEV))
    groups = {c.id: c.groups for c in calibration_claims if c.label != NOT_ENOUGH_INFO}

    print("reading trainval ...", flush=True)
    trainval_rows = read_dataset("trainval")
    trainval_scores = read_scores(RUNS / "evidence-trainval")
    # trainval gold travels inside the dataset row rather than coming from a claim file: the split
    # is carved from train and the builder already resolved and stored the smallest gold group,
    # which is the one the encoder reserves. One group, so recall_at_k sees a 1-tuple.
    trainval_groups = {
        row["id"]: (tuple((t, i) for t, i, _ in row["gold"]),)
        for row in trainval_rows if row["gold"]
    }
    trainval_budgets = budgets_from_encoding(trainval_rows, args.variant, args.max_length, args.seed)
    train_features, train_labels, _ = assemble(
        trainval_rows, trainval_scores, trainval_budgets, trainval_groups, None
    )

    print("reading calibration ...", flush=True)
    calibration_rows = read_dataset("calibration")
    calibration_scores = read_scores(RUNS / "bm25-calibration")
    predictions = {}
    with open(MODELS / args.variant / "predictions_calibration.jsonl", encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            predictions[row["id"]] = row
    calibration_budgets = [predictions[r["id"]]["n_evidence_used"] for r in calibration_rows]
    logits = {i: r["logits"] for i, r in predictions.items()}
    eval_features, eval_labels, _ = assemble(
        calibration_rows, calibration_scores, calibration_budgets, groups, logits
    )

    print(f"\ntrainval     {len(train_labels):,} verifiable claims, "
          f"gold_read base rate {train_labels.mean():.4f}")
    print(f"calibration  {len(eval_labels):,} verifiable claims, "
          f"gold_read base rate {eval_labels.mean():.4f}")

    retrieval_model = SufficiencyModel(names=RETRIEVAL_NAMES).fit(
        rows_from(train_features, RETRIEVAL_NAMES), train_labels
    )
    results = {
        "verdict_only": (
            out_of_fold_auc(eval_features, eval_labels, VERDICT_NAMES, k=args.folds),
            "calibration, out-of-fold",
        ),
        "retrieval_only": (
            roc_auc(retrieval_model.predict(rows_from(eval_features, RETRIEVAL_NAMES)), eval_labels),
            "trainval -> calibration",
        ),
        "both": (
            out_of_fold_auc(eval_features, eval_labels, FEATURE_NAMES, k=args.folds),
            "calibration, out-of-fold",
        ),
    }

    print(f"\nAUC for predicting gold_read on calibration ({len(eval_labels):,} verifiable claims)")
    print(f"  {'features':<16} {'AUC':>7}   fitted")
    for name, (auc, how) in results.items():
        print(f"  {name:<16} {auc:>7.4f}   {how}")

    print("\nretrieval model coefficients, standardised units")
    for name, weight in sorted(
        retrieval_model.coefficients().items(), key=lambda kv: -abs(kv[1])
    ):
        print(f"  {name:<20} {weight:>+8.4f}")

    headline = results["retrieval_only"][0]
    lift = headline - results["verdict_only"][0]
    print(f"\nretrieval features beat verdict features by {lift:+.4f} AUC")
    if headline >= GATE_STRONG:
        verdict = "PROCEED -- sufficiency is learnable from retrieval alone"
    elif headline >= GATE_WEAK:
        verdict = "WEAK -- report the number and reconsider before building the gate"
    else:
        verdict = "STOP -- negative result; sufficiency is not recoverable from these features"
    print(f"gate ({GATE_WEAK:.2f} / {GATE_STRONG:.2f}): {verdict}")

    out = RUNS / "sufficiency"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint1.json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "trainval_claims": int(len(train_labels)),
                "calibration_claims": int(len(eval_labels)),
                "base_rate": {
                    "trainval": float(train_labels.mean()),
                    "calibration": float(eval_labels.mean()),
                },
                "auc": {name: auc for name, (auc, _) in results.items()},
                "coefficients": retrieval_model.coefficients(),
                "gate": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out / 'checkpoint1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
