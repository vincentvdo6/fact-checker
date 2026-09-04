"""
Where the check-worthiness detector fails, broken out by subgroup.

An aggregate F1 says a filter is good. It does not say whether it is good *evenly*, and for a
filter that decides which claims get fact-checked at all, uneven is the interesting failure. A
detector that admits one party's sentences more readily than the other's would put a thumb on the
scale before a single verdict is scored, and the aggregate number would not move.

Three cuts, each chosen because it could plausibly carry a bias rather than because it is easy:

  party      the one with an obvious way to be unfair, and the one a reader will ask about
  length     the detector's known coverage gap -- ClaimBuster holds 3 sentences under four words
             in 22,501, so short input is out of distribution by construction
  speaker    finer than party, and catches a single figure's idiom being systematically missed

**Rates are reported beside errors on purpose.** A group whose sentences genuinely contain more
claims *should* be admitted more often; that is the filter working. What matters is whether the
error rate differs once the base rate is accounted for, so both are printed and neither is
presented alone.

Numbers here are descriptive. With 3,509 sentences split several ways some cells are small, and a
Wilson interval is printed so a gap inside the noise is not read as a finding.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eval.intervals import wilson
from src.pipeline.detector import MIN_WORDS, DetectorFilter

DATA = Path("data/kaggle/checkworthy-v1")
RUNS = Path("runs/checkworthy-audit")
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
MIN_CELL = 30       # below this a rate is noise; the cell is reported but not compared




def length_band(text: str) -> str:
    words = len(WORD.findall(text))
    if words < MIN_WORDS:
        return f"under {MIN_WORDS} words"
    if words < 10:
        return "4-9 words"
    if words < 20:
        return "10-19 words"
    return "20+ words"


def group_rows(rows: list[dict], key) -> dict[str, list[int]]:
    """
    Index every row into its subgroup, keeping every cell however small.

    An earlier version dropped cells below ten, which made the table read as a complete
    decomposition while two speakers and six sentences were absent from the groundtruth split. A
    subgroup too small to compare against is still a subgroup that was audited; `report` marks it
    and excludes it from the pairwise comparison, which is the honest place for that decision.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[key(row)].append(i)
    return dict(grouped)


def report(name: str, groups: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    print(f"\n{name}")
    print(f"  {'group':<26} {'n':>6} {'base rate':>10} {'admitted':>9} {'error':>7} "
          f"{'95% CI on error':>22}")
    rows = []
    for group, (predicted, actual) in sorted(groups.items(), key=lambda kv: -len(kv[1][0])):
        n = len(actual)
        errors = int((predicted != actual).sum())
        low, high = wilson(errors, n)
        flag = "" if n >= MIN_CELL else "   (small)"
        print(f"  {group:<26} {n:>6,} {actual.mean():>10.4f} {predicted.mean():>9.4f} "
              f"{errors / n:>7.4f} {f'[{low:.4f}, {high:.4f}]':>22}{flag}")
        rows.append({
            "group": group, "n": n, "base_rate": float(actual.mean()),
            "admitted_rate": float(predicted.mean()), "error_rate": errors / n,
            "ci": [low, high], "comparable": n >= MIN_CELL,
        })

    comparable = [r for r in rows if r["comparable"]]
    if len(comparable) < 2:
        print(f"  no comparison: {len(comparable)} cell(s) reach n >= {MIN_CELL}")
        return rows

    # Every pair, not just the two extreme point estimates. Wilson width varies with n, so the
    # widest gap in point estimates is not necessarily the most separated pair -- reporting only
    # that pair can print "inside the noise" while a genuinely disjoint pair sits in the table.
    disjoint = [
        (a, b) for i, a in enumerate(comparable) for b in comparable[i + 1:]
        if a["ci"][0] > b["ci"][1] or b["ci"][0] > a["ci"][1]
    ]
    worst = max(comparable, key=lambda r: r["error_rate"])
    best = min(comparable, key=lambda r: r["error_rate"])
    print(f"  widest gap: {worst['group']} {worst['error_rate']:.4f} against "
          f"{best['group']} {best['error_rate']:.4f}")
    if disjoint:
        pairs = ", ".join(f"{a['group']} vs {b['group']}" for a, b in disjoint[:3])
        more = f" and {len(disjoint) - 3} more" if len(disjoint) > 3 else ""
        print(f"  {len(disjoint)} pair(s) with disjoint intervals: {pairs}{more}")
    else:
        print("  no pair has disjoint intervals -- every gap here is inside the noise")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("test", "calibration", "groundtruth"))
    parser.add_argument("--binarization", default="factual", choices=("factual", "check_worthy"))
    parser.add_argument("--out", default=str(RUNS))
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (DATA / f"checkworthy_{args.split}.jsonl").read_text(
            encoding="utf-8").splitlines() if line
    ]
    positive = (
        (lambda label: label in ("unimportant_factual", "check_worthy"))
        if args.binarization == "factual"
        else (lambda label: label == "check_worthy")
    )
    actual = np.array([positive(r["label"]) for r in rows], dtype=bool)

    learned = DetectorFilter(args.binarization)
    print(f"{args.split}: {len(rows):,} sentences, detector {args.binarization} "
          f">= {learned.threshold}, floor {learned.min_words} words")
    predicted = np.array([d.worthy for d in learned.decide_batch([r["text"] for r in rows])],
                         dtype=bool)
    print(f"overall error rate {float((predicted != actual).mean()):.4f}")

    cuts = {
        "by party": lambda r: r["party"] or "unstated",
        "by sentence length": lambda r: length_band(r["text"]),
        "by speaker": lambda r: r["speaker"],
    }
    # The length cut exists to probe the coverage gap, and on every split that can be run here the
    # gap band is empty -- ClaimBuster's 3 short sentences are all in train. Saying so is the whole
    # point: an absent row reads identically to a row that came out fine.
    bands = {length_band(r["text"]) for r in rows}
    missing = [b for b in (f"under {MIN_WORDS} words", "4-9 words", "10-19 words",
                           "20+ words") if b not in bands]
    audit: dict[str, object] = {"split": args.split, "binarization": args.binarization,
                                "threshold": learned.threshold, "cuts": {}, "empty_length_bands": []}
    for name, key in cuts.items():
        groups = {
            g: (predicted[np.array(idx)], actual[np.array(idx)])
            for g, idx in group_rows(rows, key).items()
        }
        audit["cuts"][name] = report(name, groups)
        if name == "by sentence length" and missing:
            print(f"  absent from this split: {', '.join(missing)} -- 0 sentences.")
            print("  The coverage gap cannot be audited here at all, which is different from "
                  "having been checked and found clean.")

    audit["empty_length_bands"] = missing
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics-{args.split}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"\nwrote {out / f'metrics-{args.split}.json'}")
    print(f"Every cell is printed. Those under {MIN_CELL} are marked and excluded from the "
          f"pairwise comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
