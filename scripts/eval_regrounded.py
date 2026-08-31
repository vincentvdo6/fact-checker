"""
The pre-registered comparison: does relabelling the training signal do what Phase 02 predicted?

Phase 02 traced a 38-point NEI gap to label noise -- roughly 30% of verifiable training rows carry
a verdict their evidence cannot support, so "no supporting evidence -> NEI" is contradicted often
enough that the model never learns it. Phases 03 and 04 worked around that at inference. This
compares a model trained on the relabelled data against the Phase 02 control that was not.

**Overall accuracy is expected to fall, and that is not the criterion.** Both models are graded
against FEVER's original labels, so when the regrounded model correctly answers "not enough
evidence" on a claim whose gold retrieval simply missed, it is scored wrong. Judging on the
headline number would reward the model that guesses from claim artifacts. The criteria below were
fixed in the plan before the run, and every one of them is reported as it lands.

  accuracy on gold-read claims   should hold. Those rows were never relabelled, so a drop here
                                 means the intervention damaged ordinary reasoning.
  NEI F1                         should rise toward the gold oracle's 0.8915. This is the defect
                                 Phase 02 measured, stated as a number.
  corr(confidence, sufficiency)  should rise from +0.03. This is the one that closes the loop: if
                                 confidence starts tracking whether the evidence was there, the
                                 gate Phase 04 bolted on becomes native.
  accuracy on gold-missed claims should FALL, and a fall is the intended behaviour -- the model
                                 declining to guess where it has nothing to reason from.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from src.calibration.sufficiency import SufficiencyModel, rows_from
from src.data.fever import NOT_ENOUGH_INFO, load_claims
from src.data.splits import split_dev
from src.eval.retrieval import recall_at_k
from src.eval.stats import mcnemar, paired_bootstrap
from src.retrieval.features import RETRIEVAL_NAMES, retrieval_features
from src.verdict.encode import LABELS
from src.verdict.labels import Verdict

DATA = Path("data/kaggle/fever-verdict-v1")     # evaluation rows; identical in v2 by construction
MODELS = Path("models/verdict")
RUNS = Path("runs")
DEV = "data/fever/shared_task_dev.jsonl"
NEI = Verdict.NOT_ENOUGH_EVIDENCE.value

# The Phase 02 control as this script measures it -- **calibrated**, because both models are read
# through their Phase 03 calibrator. The raw numbers (0.7015 accuracy, 0.5728 NEI F1) belong to a
# different table, and quoting them beside calibrated ones would overstate every change by the
# calibrator's own contribution.
CONTROL = {"accuracy": 0.7285, "nei_f1": 0.6463, "gold_read_accuracy": 0.7990}
GOLD_ORACLE_NEI_F1 = 0.8915


def read_dataset(split: str) -> list[dict]:
    path = DATA / f"verdict_{split}.jsonl.gz"
    opener, target = (gzip.open, path) if path.exists() else (open, DATA / f"verdict_{split}.jsonl")
    with opener(target, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def f1_per_class(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for index, name in enumerate(LABELS):
        tp = int(((predicted == index) & (labels == index)).sum())
        fp = int(((predicted == index) & (labels != index)).sum())
        fn = int(((predicted != index) & (labels == index)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return out


def load_variant(name: str, split: str, rows: list[dict]) -> dict:
    """Predictions and calibrated confidence for one installed model, aligned to `rows`."""
    root = MODELS / name
    predictions = {
        row["id"]: row
        for row in map(json.loads, open(root / f"predictions_{split}.jsonl", encoding="utf-8"))
    }
    logits = np.asarray([predictions[r["id"]]["logits"] for r in rows], dtype=np.float64)

    calibration = root / "calibration.json"
    if calibration.exists():
        from src.calibration.scaling import from_dict as calibrator_from_dict

        probabilities = calibrator_from_dict(
            json.loads(calibration.read_text(encoding="utf-8"))["calibrator"]
        ).transform(logits)
    else:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

    return {
        "budgets": [predictions[r["id"]]["n_evidence_used"] for r in rows],
        "predicted": probabilities.argmax(axis=1),
        "confidence": probabilities.max(axis=1),
        "calibrated": calibration.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", default="retrieved")
    parser.add_argument("--grounded", default="retrieved_grounded")
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    args = parser.parse_args()

    for name in (args.control, args.grounded):
        if not (MODELS / name / f"predictions_{args.split}.jsonl").exists():
            raise SystemExit(
                f"{name} is not installed; run:\n"
                f"  python -m scripts.install_artifacts <zip> --as {name}"
            )

    rows = read_dataset(args.split)
    label_index = {name: i for i, name in enumerate(LABELS)}
    labels = np.asarray([label_index[r["label"]] for r in rows], dtype=np.int64)

    models = {name: load_variant(name, args.split, rows) for name in (args.control, args.grounded)}

    # Comparing a calibrated model against an uncalibrated one would credit or blame the
    # intervention for the calibrator's work. Phase 03 moved accuracy +2.7 and NEI F1 +7.4 on the
    # control alone, which is larger than the effect being measured here.
    calibrated = {name: model["calibrated"] for name, model in models.items()}
    if len(set(calibrated.values())) > 1:
        missing = [name for name, done in calibrated.items() if not done]
        raise SystemExit(
            f"calibration mismatch: {calibrated}. Fit the missing one before comparing:\n"
            f"  python -m scripts.calibrate   # discovers {', '.join(missing)}"
        )

    claims = {c.id: c for c in split_dev(load_claims(DEV))[1 if args.split == "test" else 0]}
    groups = {i: c.groups for i, c in claims.items() if c.label != NOT_ENOUGH_INFO}
    verifiable = np.asarray([r["id"] in groups for r in rows])

    # Groundedness is a property of retrieval, not of either model, so it is measured once against
    # the control's budgets -- the same evidence both were shown.
    control_budgets = models[args.control]["budgets"]
    gold_read = np.asarray([
        bool(r["id"] in groups
             and recall_at_k([(t, i) for t, i, _ in r["evidence"]], groups[r["id"]], b))
        for r, b in zip(rows, control_budgets, strict=True)
    ])
    gold_missed = verifiable & ~gold_read

    scores = {
        row["id"]: row["scores"]
        for row in map(json.loads, open(RUNS / f"bm25-{args.split}" / "retrieved.jsonl", encoding="utf-8"))
    }
    sufficiency_model = SufficiencyModel.from_dict(
        json.loads((MODELS / args.control / "sufficiency.json").read_text(encoding="utf-8"))["model"]
    )
    sufficiency = sufficiency_model.predict(rows_from(
        [retrieval_features([(t, i) for t, i, _ in r["evidence"]], scores[r["id"]], budget=b)
         for r, b in zip(rows, control_budgets, strict=True)],
        RETRIEVAL_NAMES,
    ))

    print(f"{args.split}: {len(rows):,} claims   |   control {args.control}   "
          f"grounded {args.grounded}")
    print(f"gold read {int(gold_read.sum()):,}   gold missed {int(gold_missed.sum()):,}   "
          f"NEI {int((~verifiable).sum()):,}")

    report: dict[str, dict] = {}
    for name, model in models.items():
        correct = model["predicted"] == labels
        f1 = f1_per_class(labels, model["predicted"])
        report[name] = {
            "calibrated": model["calibrated"],
            "accuracy": float(correct.mean()),
            "accuracy_gold_read": float(correct[gold_read].mean()),
            "accuracy_gold_missed": float(correct[gold_missed].mean()),
            "accuracy_nei": float(correct[~verifiable].mean()),
            "nei_f1": f1[NEI],
            "per_class_f1": f1,
            "corr_confidence_sufficiency": float(np.corrcoef(model["confidence"], sufficiency)[0, 1]),
            "predicted_nei": int((model["predicted"] == label_index[NEI]).sum()),
        }

    control, grounded = report[args.control], report[args.grounded]
    print(f"\n{'':<34} {'control':>10} {'grounded':>10} {'change':>10}   judged")
    lines = [
        ("accuracy | gold read", "accuracy_gold_read", "hold"),
        ("NEI F1", "nei_f1", "rise"),
        ("corr(confidence, sufficiency)", "corr_confidence_sufficiency", "rise"),
        ("accuracy | gold missed", "accuracy_gold_missed", "fall is intended"),
        ("accuracy | NEI claims", "accuracy_nei", "rise"),
        ("accuracy overall", "accuracy", "reported, not judged"),
    ]
    for label, key, judged in lines:
        a, b = control[key], grounded[key]
        print(f"{label:<34} {a:>10.4f} {b:>10.4f} {b - a:>+10.4f}   {judged}")

    print(f"\n{'predicted NEI':<34} {control['predicted_nei']:>10,} "
          f"{grounded['predicted_nei']:>10,} {grounded['predicted_nei'] - control['predicted_nei']:>+10,}"
          f"   of {int((~verifiable).sum()):,} true")
    print(f"{'NEI F1 gap to the gold oracle':<34} "
          f"{GOLD_ORACLE_NEI_F1 - control['nei_f1']:>10.4f} "
          f"{GOLD_ORACLE_NEI_F1 - grounded['nei_f1']:>10.4f}")

    a = (models[args.grounded]["predicted"] == labels)
    b = (models[args.control]["predicted"] == labels)
    delta = paired_bootstrap(a.astype(float).tolist(), b.astype(float).tolist(), seed=0)
    test_result = mcnemar(a.tolist(), b.tolist())
    print(f"\npaired accuracy change {delta.point:+.4f} [{delta.low:+.4f}, {delta.high:+.4f}]"
          f"   McNemar p={test_result.p_value:.2e}")

    out = RUNS / f"regrounded-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "control": args.control,
                "grounded": args.grounded,
                "phase_02_control": CONTROL,
                "counts": {
                    "gold_read": int(gold_read.sum()),
                    "gold_missed": int(gold_missed.sum()),
                    "nei": int((~verifiable).sum()),
                },
                "models": report,
                "paired_accuracy_delta": delta.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
