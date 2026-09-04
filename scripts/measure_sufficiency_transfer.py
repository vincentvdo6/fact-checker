"""
Does the sufficiency signal still work when retrieval fails for a reason FEVER never contains?

Phase 04's gate rests on one number: retrieval-side features predict whether the model was given
the evidence it needed, AUC 0.7485 on FEVER's calibration split, against 0.5856 for verdict
confidence. That bought the honest abstention Phase 03 could not.

Running the demo raised a doubt the FEVER numbers cannot settle. On the 2016 State of the Union
the gate declines 22.6% of claims -- almost exactly the 24.2% it declines on FEVER -- while the
retrieved evidence is frequently unrelated to the claim. "Our auto industry just had its best year
ever" retrieves *Auto-Owners Insurance* and an album called *Our Newest Album Ever!*, and nothing
in the sufficiency features notices.

**The mechanism, stated so it can be wrong.** The Phase 04 features are BM25 *score shape*: top
score, decay, margin, page concentration. Those describe how confidently BM25 ranked what it found,
not whether what it found is about the claim. On FEVER a confidently-shaped retrieval usually *is*
the right page, because FEVER claims are short statements about Wikipedia entities and the entity
is in the claim. On political speech the same shape arises from a keyword collision. If that is
right, sufficiency should separate relevant from irrelevant retrieval no better than chance here.

So it is measured rather than argued. `labels/relevance-sotu-2016.json` marks, for 40 demo claims,
whether retrieval returned anything that could bear on the claim; this scores sufficiency as a
detector of that. The comparison is against Phase 04's own 0.7485 -- the same signal, the same
gate, a different distribution.

The verdict model is not what is on trial. It answers NOT ENOUGH EVIDENCE on most of these, which
is the correct reading of irrelevant evidence and the one part of the demo working as designed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from src.eval.intervals import wilson
from src.eval.selective import roc_auc
from src.eval.stats import bootstrap_ci

LABELS = Path("labels")
RUNS = Path("runs/sufficiency-transfer")
FEVER_AUC = 0.7485          # Phase 04, retrieval features predicting gold_read on calibration
FEVER_DECLINE = 0.242       # share of FEVER test claims the gate declines on sufficiency alone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", default="sotu-2016")
    parser.add_argument("--run", default="runs/demo/verdicts.json")
    parser.add_argument("--out", default=str(RUNS))
    args = parser.parse_args()

    truth = json.loads(
        (LABELS / f"relevance-{args.transcript}.json").read_text(encoding="utf-8")
    )
    demo = json.loads(Path(args.run).read_text(encoding="utf-8"))
    scored = {r["index"]: r for r in demo["rows"] if r.get("outcome")}

    labelled = {int(i): v for i, v in truth["labels"].items()}
    missing = sorted(set(labelled) - set(scored))
    if missing:
        raise SystemExit(
            f"{len(missing)} labelled claims are not in this run ({missing[:5]}). The demo was "
            "re-run with a different filter or transcript; re-label rather than scoring the "
            "labels against claims they were not written for."
        )

    order = sorted(labelled)
    relevant = np.array([labelled[i] for i in order], dtype=bool)
    sufficiency = np.array([scored[i]["sufficiency"] for i in order], dtype=np.float64)
    threshold = 0.5

    auc = roc_auc(sufficiency, relevant)

    # Resample claim indices through the existing bootstrap rather than reimplementing one: AUC is
    # a rank statistic over paired values, not a mean of per-claim numbers, so there is nothing to
    # average directly. A draw that lands all one class has no AUC to speak of and is skipped.
    def auc_of(rows: np.ndarray) -> float:
        index = rows.astype(int)
        drawn = relevant[index]
        if drawn.all() or not drawn.any():
            return float("nan")
        return roc_auc(sufficiency[index], drawn)

    interval = bootstrap_ci(np.arange(len(order), dtype=np.float64), auc_of, seed=0)
    print(f"{args.transcript}: {len(order)} claims labelled for retrieval relevance")
    print(f"retrieval returned something usable for {int(relevant.sum())}/{len(order)} "
          f"({relevant.mean():.1%})")
    print()
    print(f"{'':<34} {'mean sufficiency':>17} {'n':>5}")
    print(f"{'retrieval was relevant':<34} {sufficiency[relevant].mean():>17.4f} "
          f"{int(relevant.sum()):>5}")
    print(f"{'retrieval was not':<34} {sufficiency[~relevant].mean():>17.4f} "
          f"{int((~relevant).sum()):>5}")
    print(f"{'separation':<34} "
          f"{sufficiency[relevant].mean() - sufficiency[~relevant].mean():>+17.4f}")

    print(f"\n{'sufficiency predicting ...':<40} {'AUC':>8}")
    print(f"{'gold_read, FEVER calibration (Phase 04)':<40} {FEVER_AUC:>8.4f}")
    print(f"{'relevance, this transcript':<40} {auc:>8.4f}   "
          f"95% CI [{interval.low:.4f}, {interval.high:.4f}]")

    # The gate's own behaviour on the two groups, which is what actually reaches a reader.
    declined = sufficiency < threshold
    print(f"\nthe gate declines {declined.mean():.1%} of these claims "
          f"(FEVER test: {FEVER_DECLINE:.1%})")
    for name, mask in (("relevant retrieval", relevant), ("irrelevant retrieval", ~relevant)):
        share = float(declined[mask].mean())
        low, high = wilson(int(declined[mask].sum()), int(mask.sum()))
        print(f"  declined among {name:<22} {share:>7.1%}   95% CI [{low:.3f}, {high:.3f}]")

    # The reading turns on what the interval excludes, not on the point estimate. 0.5797 is above
    # chance, and calling that "sufficiency does not transfer" would overstate exactly as badly as
    # calling it "the gate holds up" -- so the branch that fires is the one the data supports.
    clears_chance = interval.low > 0.5
    below_fever = interval.high < FEVER_AUC
    if clears_chance and below_fever:
        verdict = ("sufficiency retains signal out of domain, but materially less than on FEVER")
    elif below_fever:
        verdict = ("sufficiency is measurably worse here than on FEVER, and this sample cannot "
                   "tell whether it retains any signal at all")
    elif clears_chance:
        verdict = "sufficiency clears chance here; this sample cannot show it degraded"
    else:
        verdict = ("INCONCLUSIVE -- the interval spans both chance and the FEVER number, so these "
                   "40 claims cannot distinguish 'the gate transfers' from 'it does not'")
    print(f"\n{verdict}.")
    print(f"AUC {auc:.4f} [{interval.low:.4f}, {interval.high:.4f}] against {FEVER_AUC:.4f} on "
          f"FEVER; 0.5 is chance.")
    print(f"  the interval {'excludes' if clears_chance else 'includes'} chance and "
          f"{'excludes' if below_fever else 'includes'} FEVER's number")
    if not (clears_chance or below_fever):
        needed = int(len(order) * (0.2579 / 0.08) ** 2)
        print(f"  narrowing this to a decision needs roughly {needed:,} labelled claims at this "
              f"effect size, not {len(order)}")

    payload = {
        "transcript": args.transcript,
        "claims": len(order),
        "relevant": int(relevant.sum()),
        "relevance_rate": float(relevant.mean()),
        "mean_sufficiency": {
            "relevant": float(sufficiency[relevant].mean()),
            "irrelevant": float(sufficiency[~relevant].mean()),
        },
        "auc": {
            "relevance_here": auc,
            "relevance_here_ci": [interval.low, interval.high],
            "gold_read_fever_phase04": FEVER_AUC,
        },
        "declined": {
            "overall": float(declined.mean()),
            "relevant": float(declined[relevant].mean()),
            "irrelevant": float(declined[~relevant].mean()),
            "fever_test": FEVER_DECLINE,
        },
        "sufficiency_deciles": [
            float(x) for x in statistics.quantiles(sufficiency.tolist(), n=10)
        ],
        "limitation": truth["limitation"],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out / 'metrics.json'}")
    print(f"\nlimitation: {truth['limitation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
