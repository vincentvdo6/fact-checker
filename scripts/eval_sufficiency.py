"""
The test-split report: what the sufficiency gate bought, and what it cost.

The criterion was written into the plan before this split was opened. Phase 03's abstentions were
18.6% gold-missed against a 22.6% base rate -- the system was *less* likely to decline a claim it
could not answer than to decline one it could. The gate passes only if that share moves above the
base rate. It is reported whichever way it lands, and the coverage it costs is reported beside it.

Two things are deliberately not done here. Nothing is fitted: the calibrator, the bands and the
sufficiency model were all frozen on splits this script never re-reads, and a threshold adjusted
until its promise held on test would be a promise about nothing. And the two kinds of "no verdict"
stay apart, per src/verdict/labels.py -- predicting NOT ENOUGH EVIDENCE is a claim about the world,
declining to answer is a claim about the system, and a merged rate would read identically for a
system that always abstains and one that always predicts NEI.

Accuracy is reported over *answered* claims. Dividing by the whole split would make abstention look
free, which is the one arithmetic mistake that would flatter every number below.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.calibration.bands import BAND_ORDER, BandPolicy
from src.calibration.scaling import from_dict as calibrator_from_dict
from src.calibration.sufficiency import SufficiencyModel, rows_from
from src.data.fever import NOT_ENOUGH_INFO, load_claims
from src.data.splits import split_dev
from src.eval.retrieval import recall_at_k
from src.eval.selective import evaluate_selective
from src.eval.stats import paired_bootstrap
from src.retrieval.features import RETRIEVAL_NAMES, retrieval_features
from src.verdict.encode import LABELS
from src.verdict.labels import Verdict

DATA = Path("data/kaggle/fever-verdict-v1")
MODELS = Path("models/verdict")
RUNS = Path("runs")
DEV = "data/fever/shared_task_dev.jsonl"
NEI_INDEX = LABELS.index(Verdict.NOT_ENOUGH_EVIDENCE.value)

# Phase 03, quoted beside every number so a change is read against the right baseline.
PHASE_03 = {"gold_missed_share": 0.186, "coverage": 0.9060, "base_rate": 0.226}


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
    frozen = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "sufficiency.json").read_text(encoding="utf-8"))
    calibrator = calibrator_from_dict(frozen["calibrator"])
    model = SufficiencyModel.from_dict(gate["model"])
    policy = BandPolicy.from_dict(gate["bands"])
    threshold = gate["threshold"]

    rows = read_dataset(args.split)
    scores = {
        row["id"]: row["scores"]
        for row in map(json.loads, open(RUNS / f"bm25-{args.split}" / "retrieved.jsonl", encoding="utf-8"))
    }
    predictions = {
        row["id"]: row
        for row in map(json.loads, open(root / f"predictions_{args.split}.jsonl", encoding="utf-8"))
    }
    budgets = [predictions[r["id"]]["n_evidence_used"] for r in rows]

    label_index = {name: i for i, name in enumerate(LABELS)}
    logits = np.asarray([predictions[r["id"]]["logits"] for r in rows], dtype=np.float64)
    labels = np.asarray([label_index[r["label"]] for r in rows], dtype=np.int64)
    probabilities = calibrator.transform(logits)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels

    features = [
        retrieval_features([(t, i) for t, i, _ in r["evidence"]], scores[r["id"]], budget=b)
        for r, b in zip(rows, budgets, strict=True)
    ]
    sufficiency = model.predict(rows_from(features, RETRIEVAL_NAMES))

    banded = np.asarray([policy.assign(float(c)) is not None for c in confidence])
    grounded = sufficiency >= threshold

    claims = {c.id: c for c in (split_dev(load_claims(DEV))[1 if args.split == "test" else 0])}
    groups = {i: c.groups for i, c in claims.items() if c.label != NOT_ENOUGH_INFO}
    verifiable = np.asarray([r["id"] in groups for r in rows])
    gold_read = np.asarray([
        bool(r["id"] in groups
             and recall_at_k([(t, i) for t, i, _ in r["evidence"]], groups[r["id"]], b))
        for r, b in zip(rows, budgets, strict=True)
    ])
    gold_missed = verifiable & ~gold_read
    base_rate = float(gold_missed.sum() / max(verifiable.sum(), 1))

    print(f"{args.split}: {len(rows):,} claims   |   variant {args.variant}   |   "
          f"calibrator {frozen['selected']}, sufficiency fitted on trainval")

    policies = {"confidence only": banded, "plus sufficiency": banded & grounded}
    print(f"\n{'policy':<22} {'coverage':>9} {'accuracy':>9} {'declined':>9} "
          f"{'gold-missed share':>19}")
    summary = {}
    for name, answered in policies.items():
        declined = ~answered
        share = float((gold_missed & declined).sum() / max(declined.sum(), 1))
        accuracy = float(correct[answered].mean()) if answered.any() else float("nan")
        print(f"{name:<22} {answered.mean():>9.4f} {accuracy:>9.4f} {int(declined.sum()):>9,} "
              f"{share:>18.1%}")
        summary[name] = {
            "coverage": float(answered.mean()), "accuracy": accuracy,
            "declined": int(declined.sum()), "gold_missed_share": share,
        }
    print(f"{'(base rate)':<22} {'':>9} {'':>9} {'':>9} {base_rate:>18.1%}")

    # --- the pre-registered criterion --------------------------------------------------------
    share_now = summary["plus sufficiency"]["gold_missed_share"]
    share_before = summary["confidence only"]["gold_missed_share"]
    passed = share_now > base_rate
    print("\npre-registered criterion: abstentions over-represent gold-missed claims")
    print(f"  Phase 03            {PHASE_03['gold_missed_share']:>6.1%}   "
          f"(base rate {PHASE_03['base_rate']:.1%})   under-selected")
    print(f"  confidence only     {share_before:>6.1%}   (base rate {base_rate:.1%})")
    print(f"  plus sufficiency    {share_now:>6.1%}   (base rate {base_rate:.1%})   "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"  coverage paid       {summary['confidence only']['coverage'] - summary['plus sufficiency']['coverage']:>+6.4f}")

    # --- what it cost, on claims the gate now declines --------------------------------------
    newly = banded & ~grounded
    if newly.any():
        print(f"\nclaims the sufficiency condition newly declines: {int(newly.sum()):,}")
        print(f"  the model was right on {correct[newly].mean():.1%} of them")
        print(f"  {(gold_missed & newly).sum() / max(newly.sum(), 1):.1%} were gold-missed "
              f"against a {base_rate:.1%} base rate")
        nei_share = float((predicted[newly] == NEI_INDEX).mean())
        print(f"  {nei_share:.1%} were predictions of NOT ENOUGH EVIDENCE, declined because we "
              f"cannot tell 'nothing exists' from 'we searched badly'")

    # --- accuracy is not the point, but it must be reported ----------------------------------
    delta = paired_bootstrap(
        correct[banded & grounded].astype(float).tolist(),
        correct[banded].astype(float).tolist(),
        seed=0,
    ) if banded.sum() == (banded & grounded).sum() else None
    selective = {
        name: evaluate_selective(confidence[mask], correct[mask], label=name)
        for name, mask in policies.items()
    }
    print(f"\n{'policy':<22} {'risk':>8} {'AURC':>8} {'E-AURC':>8}")
    for name, report in selective.items():
        print(f"{name:<22} {report.full_coverage_risk:>8.4f} {report.aurc:>8.4f} "
              f"{report.excess_aurc:>8.4f}")
    if delta is not None:
        print(f"  paired accuracy change {delta.point:+.4f} [{delta.low:+.4f}, {delta.high:+.4f}]")

    # --- two kinds of no verdict, never summed ------------------------------------------------
    answered = banded & grounded
    table = Counter(zip((predicted == NEI_INDEX).tolist(), (~answered).tolist()))
    print("\n'no verdict' has two sources and they are not the same thing")
    print(f"  {'':<24} {'declined':>10} {'answered':>10}")
    print(f"  {'predicted NEI':<24} {table[(True, True)]:>10,} {table[(True, False)]:>10,}")
    print(f"  {'predicted a verdict':<24} {table[(False, True)]:>10,} {table[(False, False)]:>10,}")

    # --- band promises, under the gate --------------------------------------------------------
    assigned = [policy.assign(float(c)) for c in confidence]
    print("\nband promises among answered claims, thresholds frozen on calibration")
    print(f"  {'band':<10} {'promised':>9} {'measured':>9} {'n':>7}   verdict")
    bands_out = {}
    for band in BAND_ORDER:
        members = np.asarray([a == band for a in assigned]) & answered
        if band not in policy.thresholds or not members.any():
            print(f"  {band.value:<10} {'--':>9} {'--':>9} {0:>7}   not fitted or empty")
            bands_out[band.value] = None
            continue
        measured = float(correct[members].mean())
        kept = measured >= policy.targets[band]
        print(f"  {band.value:<10} {policy.targets[band]:>9.2f} {measured:>9.4f} "
              f"{int(members.sum()):>7,}   {'kept' if kept else 'MISSED'}")
        bands_out[band.value] = {
            "promised": policy.targets[band], "measured": measured,
            "n": int(members.sum()), "kept": bool(kept),
        }

    out = RUNS / f"sufficiency-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "variant": args.variant,
                "base_rate": base_rate,
                "criterion_passed": bool(passed),
                "phase_03": PHASE_03,
                "policies": summary,
                "bands": bands_out,
                "selective": {k: v.to_dict() for k, v in selective.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
