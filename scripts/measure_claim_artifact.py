"""
Phase 06's first question: is the claim-only artifact a property of FEVER or of fact-checking?

Every finding in Phases 02 to 05 rests on one measurement -- an evidence-free DeBERTa scoring
0.5865 on FEVER against a 0.3410 majority. That leak is the reason groundedness and correctness
diverge (Phase 04), the reason honest abstention costs coverage rather than buying accuracy, and
the reason relabelling ungrounded rows backfired (Phase 05). If the leak is peculiar to FEVER,
those are findings about a dataset. If it is general, they are findings about the task.

FEVER's claims were written *from* Wikipedia sentences by annotators told to make them true or
false. AVeriTeC's were said by real people and then checked. That is exactly the difference that
should produce or fail to produce lexical leakage, so the two are measured the same way here.

A unigram naive Bayes on claim text alone, no evidence, no GPU, no seed. Whatever it scores above
each dataset's own majority floor is leakage. The lift is the comparable quantity -- the absolute
accuracies are not, since FEVER is a three-class problem and AVeriTeC a four-class one with a much
higher floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.averitec import load_all as load_averitec
from src.data.fever import load_claims as load_fever
from src.data.splits import split_averitec, split_dev
from src.eval.artifact import measure_artifact
from src.verdict.encode import FEVER_TO_LABEL

RUNS = Path("runs")
FEVER_TRAIN = "data/fever/train.jsonl"
FEVER_DEV = "data/fever/shared_task_dev.jsonl"

# The evidence-free DeBERTa from Phase 02, for scale. A bag of words is a far weaker probe, so it
# should land below this -- if it did not, the transformer would be adding nothing at all.
FEVER_NEURAL = {"accuracy": 0.5865, "majority": 0.3410}


def fever_arrays(limit: int) -> tuple[list[str], list[str], list[str], list[str]]:
    train = load_fever(FEVER_TRAIN)[:limit] if limit else load_fever(FEVER_TRAIN)
    _, test = split_dev(load_fever(FEVER_DEV))
    return (
        [c.text for c in train], [FEVER_TO_LABEL[c.label] for c in train],
        [c.text for c in test], [FEVER_TO_LABEL[c.label] for c in test],
    )


def averitec_arrays() -> tuple[list[str], list[str], list[str], list[str]]:
    train, _, test = split_averitec(load_averitec())
    return (
        [c.text for c in train], [c.label for c in train],
        [c.text for c in test], [c.label for c in test],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fever-limit", type=int, default=0,
        help="cap FEVER's train claims; 0 uses all 145k. Matching AVeriTeC's 2,556 answers "
             "whether any gap is leakage or merely 50x more data",
    )
    args = parser.parse_args()

    reports = []
    for name, arrays in (("fever", fever_arrays(args.fever_limit)), ("averitec", averitec_arrays())):
        print(f"measuring {name} ...", flush=True)
        reports.append(measure_artifact(*arrays, dataset=name))

    print(f"\n{'dataset':<12} {'classes':>8} {'train':>9} {'test':>7} "
          f"{'majority':>9} {'accuracy':>9} {'lift':>8}")
    for report in reports:
        print(f"{report.dataset:<12} {report.classes:>8} {report.train_claims:>9,} "
              f"{report.test_claims:>7,} {report.majority:>9.4f} {report.accuracy:>9.4f} "
              f"{report.lift:>+8.4f}")
    print(f"{'fever (nn)':<12} {3:>8} {'145,449':>9} {'2,000':>7} "
          f"{FEVER_NEURAL['majority']:>9.4f} {FEVER_NEURAL['accuracy']:>9.4f} "
          f"{FEVER_NEURAL['accuracy'] - FEVER_NEURAL['majority']:>+8.4f}   Phase 02, evidence-free DeBERTa")

    by_name = {r.dataset: r for r in reports}
    gap = by_name["fever"].lift - by_name["averitec"].lift
    print(f"\nlexical leakage, FEVER minus AVeriTeC: {gap:+.4f}")

    out = RUNS / "claim-artifact"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "probe": "unigram naive Bayes, claim text only",
                "fever_limit": args.fever_limit,
                "reports": [r.to_dict() for r in reports],
                "fever_neural_reference": FEVER_NEURAL,
                "lift_gap": gap,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written to {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
