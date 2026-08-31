"""
Apply the frozen calibration to a split it was never fitted on, and report what it bought.

The headline is not ECE. It is whether the band promises hold: "claims shown as strong were right
about nine times in ten" is a commitment made on the calibration split and either kept or broken
here. A band that misses its target is reported as missing it. Retuning a threshold against these
numbers would make every promise in the project circular, which is why the fitting lives in
scripts/calibrate.py and nothing here writes an artifact.

Two things reach a reader as "no verdict" and this script refuses to merge them, following
src/verdict/labels.py:

  not_enough_evidence   a claim about the world -- we looked, and nothing settles it. A verdict
                        the model predicts, scored like any other class.
  abstention            a claim about the model -- its own confidence is too low to show anything.
                        Orthogonal: the system can abstain on a claim whose true label is NEI, and
                        can confidently answer NEI.

They are cross-tabulated rather than summed, because a system that abstains constantly and a
system that predicts NEI constantly fail in completely different ways and a merged "no verdict
rate" would look identical for both.

The subgroup audit exists to test Phase 02's finding from the other side. If confidence falls on
claims whose gold evidence retrieval missed, abstention preferentially drops exactly the claims
the system cannot answer. If it does not, the model cannot perceive its own evidence gap, and the
abstention is running on something other than evidence sufficiency.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.calibration.bands import BAND_ORDER, BandPolicy
from src.calibration.scaling import from_dict as calibrator_from_dict
from src.data.fever import NOT_ENOUGH_INFO, load_claims
from src.data.splits import split_dev
from src.eval.calibration import evaluate_calibration
from src.eval.retrieval import recall_at_k
from src.eval.selective import evaluate_selective
from src.eval.stats import paired_bootstrap
from src.verdict.contract import installed_variants
from src.verdict.encode import LABELS
from src.verdict.labels import Verdict

MODELS = Path("models/verdict")
DEV = "data/fever/shared_task_dev.jsonl"
RUNS = Path("runs")

NEI_INDEX = LABELS.index(Verdict.NOT_ENOUGH_EVIDENCE.value)

# Phase 02, quoted beside every number here so a gain is read against the right yardstick.
RETRIEVAL_CEILING = {"test": 0.8111, "calibration": 0.8009}
EVIDENCE_FREE_BASELINE = {"test": 0.5865, "calibration": 0.5830}


def read_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def per_claim_brier(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(labels)), labels] = 1.0
    return ((probabilities - onehot) ** 2).sum(axis=1)


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    for index, name in enumerate(LABELS):
        tp = int(((predictions == index) & (labels == index)).sum())
        fp = int(((predictions == index) & (labels != index)).sum())
        fn = int(((predictions != index) & (labels == index)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(np.mean(list(per_class.values()))), per_class


def gold_read_flags(split: str, ids: list[int], used: list[int]) -> dict[int, bool | None]:
    """
    Whether each claim's gold evidence was inside what the model actually read.

    None for NOT ENOUGH INFO, which has no gold by construction and would otherwise be counted as
    a retrieval failure. Reuses recall_at_k so this matches scripts/eval_verdict.py exactly.
    """
    calibration, test = split_dev(load_claims(DEV))
    claims = {c.id: c for c in (test if split == "test" else calibration)}
    retrieved: dict[int, list] = {}
    with open(RUNS / f"bm25-{split}" / "retrieved.jsonl", encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            retrieved[row["id"]] = row["evidence"]

    flags: dict[int, bool | None] = {}
    for claim_id, budget in zip(ids, used, strict=True):
        claim = claims.get(claim_id)
        if claim is None or claim.label == NOT_ENOUGH_INFO:
            flags[claim_id] = None
            continue
        refs = [(t, i) for t, i in retrieved[claim_id]]
        flags[claim_id] = recall_at_k(refs, claim.groups, budget)
    return flags


def subgroup(name: str, mask: np.ndarray, probabilities: np.ndarray, labels: np.ndarray) -> dict | None:
    if mask.sum() < 30:      # below this a subgroup ECE is noise wearing a number
        return None
    report = evaluate_calibration(probabilities[mask], labels[mask], label=name)
    return {
        "n": int(mask.sum()),
        "accuracy": report.accuracy,
        "ece": report.ece,
        "mean_confidence": report.mean_confidence,
        "overconfidence": report.overconfidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--models", default=str(MODELS))
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.models)
    available = [
        v for v in installed_variants(args.models)
        if (root / v / f"predictions_{args.split}.jsonl").exists() and (root / v / "calibration.json").exists()
    ]
    if not available:
        raise SystemExit(
            f"no calibrated variants under {root}; fit first:\n  python -m scripts.calibrate"
        )

    label_index = {name: i for i, name in enumerate(LABELS)}
    out: dict[str, object] = {"split": args.split, "variants": {}}

    for variant in available:
        rows = read_predictions(root / variant / f"predictions_{args.split}.jsonl")
        frozen = json.loads((root / variant / "calibration.json").read_text(encoding="utf-8"))
        calibrator = calibrator_from_dict(frozen["calibrator"])
        policy = BandPolicy.from_dict(frozen["bands"])

        logits = np.asarray([r["logits"] for r in rows], dtype=np.float64)
        labels = np.asarray([label_index[r["label"]] for r in rows], dtype=np.int64)
        ids = [r["id"] for r in rows]
        used = [r["n_evidence_used"] for r in rows]

        raw_probabilities = _softmax(logits)
        probabilities = calibrator.transform(logits)
        raw = evaluate_calibration(
            raw_probabilities, labels, label="uncalibrated", resamples=args.resamples
        )
        cal = evaluate_calibration(
            probabilities, labels, label=frozen["selected"], resamples=args.resamples
        )

        predictions = probabilities.argmax(axis=1)
        raw_predictions = logits.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        correct = predictions == labels

        macro_before, class_before = macro_f1(labels, raw_predictions)
        macro_after, class_after = macro_f1(labels, predictions)

        print(f"\n{'=' * 74}\n{variant}  --  {args.split}, {len(labels):,} claims"
              f"   |   calibrator: {frozen['selected']}  (fitted on calibration)")

        print(f"\n  {'':<14} {'accuracy':>9} {'macro F1':>9} {'ECE':>8} {'adaECE':>8} {'MCE':>8} {'Brier':>8}")
        print(f"  {'raw':<14} {raw.accuracy:>9.4f} {macro_before:>9.4f} {raw.ece:>8.4f} "
              f"{raw.adaptive_ece:>8.4f} {raw.mce:>8.4f} {raw.brier:>8.4f}")
        print(f"  {'calibrated':<14} {cal.accuracy:>9.4f} {macro_after:>9.4f} {cal.ece:>8.4f} "
              f"{cal.adaptive_ece:>8.4f} {cal.mce:>8.4f} {cal.brier:>8.4f}")
        print(f"  retrieval ceiling {RETRIEVAL_CEILING[args.split]:.4f}"
              f"   |   evidence-free baseline {EVIDENCE_FREE_BASELINE[args.split]:.4f}")

        # Paired on the same claims, on a proper score. Differencing two ECEs would not be valid:
        # ECE is a biased L1 aggregate, so its bootstrap cannot support a signed comparison.
        delta = paired_bootstrap(
            per_claim_brier(probabilities, labels).tolist(),
            per_claim_brier(raw_probabilities, labels).tolist(),
            seed=0,
        )
        print(f"  Brier change (calibrated - raw)  {delta.point:+.4f} "
              f"[{delta.low:+.4f}, {delta.high:+.4f}]  "
              f"{'significant' if delta.excludes_zero else 'not significant'}")

        print("\n  per-class F1     " + "  ".join(f"{n[:9]:>11}" for n in LABELS))
        print("    raw            " + "  ".join(f"{class_before[n]:>11.4f}" for n in LABELS))
        print("    calibrated     " + "  ".join(f"{class_after[n]:>11.4f}" for n in LABELS))
        print("    predicted mix  " + "  ".join(
            f"{int((predictions == i).sum()):>11,}" for i in range(len(LABELS))))
        print("    true mix       " + "  ".join(
            f"{int((labels == i).sum()):>11,}" for i in range(len(LABELS))))

        # --- bands: the promise, kept or not ---------------------------------------------------
        assigned = [policy.assign(float(p)) for p in confidence]
        print("\n  band promises, thresholds frozen on calibration")
        print(f"  {'band':<10} {'promised':>9} {'measured':>9} {'n':>7}   verdict")
        bands_out = {}
        for band in BAND_ORDER:
            members = np.array([a == band for a in assigned])
            if band not in policy.thresholds:
                print(f"  {band.value:<10} {'--':>9} {'--':>9} {'--':>7}   not fitted (unreachable)")
                bands_out[band.value] = None
                continue
            if members.sum() == 0:
                print(f"  {band.value:<10} {policy.targets[band]:>9.2f} {'--':>9} {0:>7}   empty on this split")
                bands_out[band.value] = {"promised": policy.targets[band], "n": 0, "measured": None}
                continue
            measured = float(correct[members].mean())
            kept = measured >= policy.targets[band]
            print(f"  {band.value:<10} {policy.targets[band]:>9.2f} {measured:>9.4f} "
                  f"{int(members.sum()):>7,}   {'kept' if kept else 'MISSED'}")
            bands_out[band.value] = {
                "promised": policy.targets[band], "measured": measured,
                "n": int(members.sum()), "kept": bool(kept),
            }
        abstained = np.array([a is None for a in assigned])
        print(f"  abstained {int(abstained.sum()):,} of {len(labels):,} "
              f"({abstained.mean():.1%})   coverage {1 - abstained.mean():.4f}")

        # --- abstention is not NEI -------------------------------------------------------------
        predicted_nei = predictions == NEI_INDEX
        table = Counter(zip(predicted_nei.tolist(), abstained.tolist()))
        print("\n  'no verdict' has two sources and they are not the same thing")
        print(f"    {'':<22} {'abstained':>11} {'answered':>11}")
        print(f"    {'predicted NEI':<22} {table[(True, True)]:>11,} {table[(True, False)]:>11,}")
        print(f"    {'predicted a verdict':<22} {table[(False, True)]:>11,} {table[(False, False)]:>11,}")

        # --- selective prediction --------------------------------------------------------------
        selective = evaluate_selective(confidence, correct, label=variant)
        print("\n  selective prediction")
        print(f"    risk at full coverage {selective.full_coverage_risk:.4f}   "
              f"AURC {selective.aurc:.4f}   oracle {selective.optimal_aurc:.4f}   "
              f"E-AURC {selective.excess_aurc:.4f}")
        for target, point in selective.at_risk.items():
            if point is None:
                print(f"    risk <= {target}:  unreachable at any coverage")
            else:
                print(f"    risk <= {target}:  coverage {point.coverage:.4f} "
                      f"({point.answered:,} claims, accuracy {point.accuracy:.4f})")

        # --- subgroups: does confidence know when the evidence was missing? --------------------
        flags = gold_read_flags(args.split, ids, used)
        verifiable = np.array([flags[i] is not None for i in ids])
        gold_in = np.array([flags[i] is True for i in ids])
        gold_out = np.array([flags[i] is False for i in ids])
        # The contract, not the observed maximum: if nothing happened to hit the cap on this split
        # the observed max would mark the longest ordinary row as truncated.
        max_length = json.loads(
            (root / variant / "contract.json").read_text(encoding="utf-8")
        )["max_length"]
        truncated = np.array([r["token_len"] >= max_length for r in rows])

        groups = {
            "gold_read": gold_in,
            "gold_missed": gold_out,
            "verifiable": verifiable,
            "nei": ~verifiable,
            "truncated": truncated,
        }
        for index, name in enumerate(LABELS):
            groups[f"true_{name}"] = labels == index

        print("\n  subgroup audit  (n >= 30)")
        print(f"    {'group':<22} {'n':>6} {'acc':>8} {'ECE':>8} {'conf':>8} {'over':>8}")
        subgroups = {}
        for name, mask in groups.items():
            entry = subgroup(name, mask, probabilities, labels)
            subgroups[name] = entry
            if entry is None:
                continue
            print(f"    {name:<22} {entry['n']:>6,} {entry['accuracy']:>8.4f} {entry['ece']:>8.4f} "
                  f"{entry['mean_confidence']:>8.4f} {entry['overconfidence']:>+8.4f}")

        if gold_in.sum() and gold_out.sum():
            drop = float(confidence[gold_in].mean() - confidence[gold_out].mean())
            share = float(gold_out[abstained].sum() / max(abstained.sum(), 1))
            base = float(gold_out.sum() / max(verifiable.sum(), 1))
            print(f"\n    confidence falls {drop:+.4f} when gold was missed")
            print(f"    abstentions that are gold-missed: {share:.1%} against a {base:.1%} base rate")

        out["variants"][variant] = {
            "calibrator": frozen["selected"],
            "raw": raw.to_dict(),
            "calibrated": cal.to_dict(),
            "macro_f1": {"raw": macro_before, "calibrated": macro_after},
            "per_class_f1": {"raw": class_before, "calibrated": class_after},
            "brier_delta": delta.to_dict(),
            "bands": bands_out,
            "abstained": int(abstained.sum()),
            "coverage": float(1 - abstained.mean()),
            "no_verdict_table": {f"{k[0]}_{k[1]}": v for k, v in table.items()},
            "selective": selective.to_dict(),
            "subgroups": subgroups,
        }

    destination = RUNS / f"calibration-{args.split}"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{'=' * 74}\nwritten to {destination / 'metrics.json'}")
    return 0


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    sys.exit(main())
