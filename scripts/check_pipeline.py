"""
Checkpoint 2: does the assembled pipeline reproduce the offline report it was assembled from?

`scripts/eval_sufficiency.py` computed the Phase 04 numbers by holding logits, features, calibrator
and gate as loose arrays in one function. `src/pipeline/verify.py` reassembles the same pieces
behind an object the demo can call one claim at a time. Those are two implementations of one
policy, and the failure mode is not a crash: a mis-wired calibrator, a gate reading the wrong
condition, or a band assigned from raw instead of calibrated probability all produce a page that
looks entirely plausible and promises something nobody measured.

So the pipeline is required to land on the offline report exactly -- coverage, accuracy, the
declined count, the gold-missed share, and every band's measured accuracy:

    coverage 0.7580   accuracy 0.7553   declined 484   gold-missed 27.7% against a 22.6% base rate

**Nothing here re-scores the model.** Checkpoint 1 (`scripts/check_onnx_parity.py`) is what
establishes that the local graph reproduces the logits in `predictions_test.jsonl` -- to 1e-3, with
identical argmax and identical band assignment. This drives `judge_scored` from those same stored
logits, so the two checkpoints compose into a claim about the whole path while each stays cheap
enough to run. Feeding stored logits is what makes this one a test of the *wiring* and not of the
model, and it is only meaningful once Checkpoint 1 has passed on the same split.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from src.data.fever import NOT_ENOUGH_INFO, load_claims
from src.data.splits import split_dev
from src.eval.retrieval import recall_at_k
from src.pipeline.verify import Verifier
from src.retrieval.features import retrieval_features
from src.verdict.encode import LABELS

DATA = Path("data/kaggle/fever-verdict-v1")
MODELS = Path("models/verdict")
RUNS = Path("runs")
DEV = "data/fever/shared_task_dev.jsonl"

TOLERANCE = 1e-9        # two arithmetic paths over identical inputs; anything above this is a bug


def read_dataset(split: str) -> list[dict]:
    path = DATA / f"verdict_{split}.jsonl.gz"
    opener, target = (gzip.open, path) if path.exists() else (open, DATA / f"verdict_{split}.jsonl")
    with opener(target, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--variant", default="retrieved")
    args = parser.parse_args()

    root = MODELS / args.variant
    rows = read_dataset(args.split)
    scores = {
        row["id"]: row["scores"]
        for row in map(json.loads,
                       open(RUNS / f"bm25-{args.split}" / "retrieved.jsonl", encoding="utf-8"))
    }
    predictions = {
        row["id"]: row
        for row in map(json.loads,
                       open(root / f"predictions_{args.split}.jsonl", encoding="utf-8"))
    }
    budgets = [predictions[r["id"]]["n_evidence_used"] for r in rows]

    # --- the pipeline's own answer ---------------------------------------------------------
    verifier = Verifier(args.variant)
    features = [
        retrieval_features([(t, i) for t, i, _ in r["evidence"]], scores[r["id"]], budget=b)
        for r, b in zip(rows, budgets, strict=True)
    ]
    judgements = verifier.judge_scored(
        [r["claim"] for r in rows],
        [predictions[r["id"]]["logits"] for r in rows],
        features,
    )

    answered = np.asarray([j.answered for j in judgements])
    predicted = np.asarray([LABELS.index(str(j.predicted)) for j in judgements])
    labels = np.asarray([LABELS.index(r["label"]) for r in rows], dtype=np.int64)
    correct = predicted == labels

    # --- gold-missed, computed as the offline report computed it ----------------------------
    claims = {c.id: c for c in split_dev(load_claims(DEV))[1 if args.split == "test" else 0]}
    groups = {i: c.groups for i, c in claims.items() if c.label != NOT_ENOUGH_INFO}
    verifiable = np.asarray([r["id"] in groups for r in rows])
    gold_read = np.asarray([
        bool(r["id"] in groups
             and recall_at_k([(t, i) for t, i, _ in r["evidence"]], groups[r["id"]], b))
        for r, b in zip(rows, budgets, strict=True)
    ])
    gold_missed = verifiable & ~gold_read

    got = {
        "coverage": float(answered.mean()),
        "accuracy": float(correct[answered].mean()),
        "declined": int((~answered).sum()),
        "gold_missed_share": float((gold_missed & ~answered).sum() / max((~answered).sum(), 1)),
    }

    # --- against the report ------------------------------------------------------------------
    report = json.loads(
        (RUNS / f"sufficiency-{args.split}" / "metrics.json").read_text(encoding="utf-8")
    )
    want = report["policies"]["plus sufficiency"]

    print(f"{args.split}: {len(rows):,} claims through src.pipeline.verify, variant {args.variant}")
    print(f"\n{'quantity':<22} {'offline report':>16} {'pipeline':>16}   verdict")
    failures = 0
    for name in ("coverage", "accuracy", "declined", "gold_missed_share"):
        ours, theirs = got[name], want[name]
        agrees = abs(float(ours) - float(theirs)) <= TOLERANCE
        failures += not agrees
        print(f"{name:<22} {theirs:>16.10f} {ours:>16.10f}   {'ok' if agrees else 'FAIL'}")

    # --- the band promises, which are what the page actually shows ---------------------------
    print(f"\n{'band':<22} {'offline n':>10} {'pipeline n':>11} {'offline acc':>12} "
          f"{'pipeline acc':>13}   verdict")
    for band, promise in report["bands"].items():
        members = np.asarray([str(j.band) == band for j in judgements]) & answered
        n = int(members.sum())
        accuracy = float(correct[members].mean()) if n else float("nan")
        agrees = n == promise["n"] and abs(accuracy - promise["measured"]) <= TOLERANCE
        failures += not agrees
        print(f"{band:<22} {promise['n']:>10,} {n:>11,} {promise['measured']:>12.10f} "
              f"{accuracy:>13.10f}   {'ok' if agrees else 'FAIL'}")

    if failures:
        print(f"\nCheckpoint 2: FAIL -- {failures} quantities disagree with the offline report.")
        print("  The pipeline is wrong, not the report: the report is what was published.")
        return 1
    print("\nCheckpoint 2: PASS -- the assembled pipeline is the policy that was evaluated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
