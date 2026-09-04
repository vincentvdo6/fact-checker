"""
The detector against both filters and both label sources, on debates it never trained on.

The headline comparison in `scripts/compare_checkworthy.py` carries one stated bound: the 120 SOTU
labels and the hand-written rules share an author. A second transcript labelled by the same person
would add a sample and leave that bound exactly where it is. What actually loosens it is a
different annotator pool, and ClaimBuster ships one.

  crowdsourced   the labels the detector trained on, held-out debates, 3,509 sentences
  groundtruth    the same debates labelled by the dataset's expert annotators, 101 sentences,
                 and -- checked in the build -- not one shared sentence id with the crowd split

So the expert set is a genuinely independent read: different annotators from the training labels,
different annotators from the SOTU rubric, and different sentences from the crowd evaluation. It is
small, and 101 sentences is quoted with everything it implies rather than rounded off.

**The rules are scored here too**, which is the point. If the detector's advantage were an artifact
of the SOTU labels agreeing with ClaimBuster's notion of check-worthy, it would shrink here. If it
holds on expert labels the rules have never been near, it is a fact about the two filters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.eval.intervals import wilson
from src.pipeline.detector import DetectorFilter
from src.pipeline.segment import check_worthy

DATA = Path("data/kaggle/checkworthy-v1")
RUNS = Path("runs/checkworthy-heldout")


def read(name: str) -> list[dict]:
    path = DATA / f"checkworthy_{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prf(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    tp = int((predicted & actual).sum())
    precision = tp / max(int(predicted.sum()), 1)
    recall = tp / max(int(actual.sum()), 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": float((predicted == actual).mean()),
    }




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RUNS))
    parser.add_argument("--binarization", default="factual",
                        choices=("factual", "check_worthy"),
                        help="which definition the positives are read against")
    args = parser.parse_args()

    positive = (
        (lambda label: label in ("unimportant_factual", "check_worthy"))
        if args.binarization == "factual"
        else (lambda label: label == "check_worthy")
    )
    learned = DetectorFilter(args.binarization)
    print(f"detector: {args.binarization} >= {learned.threshold} "
          f"(threshold frozen on the calibration debates)")

    report: dict[str, object] = {
        "binarization": args.binarization,
        "threshold": learned.threshold,
        "min_words": learned.min_words,
        "sources": {},
    }
    for source in ("test", "groundtruth"):
        rows = read(source)
        actual = np.array([positive(r["label"]) for r in rows], dtype=bool)
        texts = [r["text"] for r in rows]
        annotators = "crowdworkers" if source == "test" else "the dataset's expert annotators"

        rules = np.array([check_worthy(t).worthy for t in texts], dtype=bool)
        detector = np.array([d.worthy for d in learned.decide_batch(texts)], dtype=bool)

        print(f"\n{source}: {len(rows):,} sentences over "
              f"{len({r['debate'] for r in rows})} held-out debates, labelled by {annotators}")
        print(f"base rate {actual.mean():.4f}")
        print(f"  {'system':<26} {'precision':>10} {'recall':>8} {'F1':>8} {'accuracy':>9} "
              f"{'95% CI on accuracy':>22}")
        results = {}
        for name, predicted in (("hand-written rules", rules), ("learned detector", detector)):
            got = prf(predicted, actual)
            results[name] = got
            low, high = wilson(int((predicted == actual).sum()), len(actual))
            print(f"  {name:<26} {got['precision']:>10.4f} {got['recall']:>8.4f} "
                  f"{got['f1']:>8.4f} {got['accuracy']:>9.4f} {f'[{low:.4f}, {high:.4f}]':>22}")
        report["sources"][source] = {
            "sentences": len(rows),
            "debates": len({r["debate"] for r in rows}),
            "annotators": annotators,
            "base_rate": float(actual.mean()),
            "systems": results,
        }
        gain = results["learned detector"]["f1"] - results["hand-written rules"]["f1"]
        print(f"  {'':<26} {'':>10} {'':>8} {gain:>+8.4f}   F1, detector over rules")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'metrics.json'}")

    print("\nThe expert set is 101 sentences. It is a different annotator pool from both the "
          "training\nlabels and the SOTU rubric, which is what it is for -- not a larger sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
