"""
Report the three variants beside the bound their evidence allowed.

Accuracy alone cannot be read here. The claim-only baseline says how much of any score is FEVER
recognising its own claims rather than the model checking them; the gold oracle says what
reasoning is worth when the evidence is present; and the packed ceiling says what was reachable
at all. A single number without those three is not interpretable, so this prints them together
or not at all.

Differences between variants are paired: all three saw the same claims, so a claim that is hard
for one is usually hard for the others, and that shared difficulty cancels. Comparing them with
independent intervals would report an uncertainty the data does not have.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.fever import NOT_ENOUGH_INFO, load_claims
from src.data.splits import split_dev
from src.eval.retrieval import Ceiling, recall_at_k
from src.eval.stats import mcnemar, paired_bootstrap
from src.eval.verdict import evaluate_verdicts
from src.verdict.encode import FEVER_TO_LABEL

MODELS = Path("models/verdict")
DEV = "data/fever/shared_task_dev.jsonl"
RUNS = Path("runs")
VARIANTS = ("retrieved", "claim_only", "gold")


def read_predictions(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as handle:
        return {row["id"]: row for row in map(json.loads, handle)}


def ceiling_for(claims, retrieved: dict[int, list], predictions: dict[int, dict]) -> Ceiling:
    """
    The bound the model could actually reach: gold inside the evidence it read, not merely
    inside what retrieval returned. n_evidence_used comes back from the notebook per claim, so
    this is a measurement rather than an assumption.
    """
    verifiable = [c for c in claims if c.label != NOT_ENOUGH_INFO and c.id in predictions]
    hits = sum(
        recall_at_k(
            [(t, i) for t, i in retrieved[c.id]], c.groups, predictions[c.id]["n_evidence_used"]
        )
        for c in verifiable
    )
    reachable = hits / len(verifiable)
    nei_share = sum(1 for c in claims if c.label == NOT_ENOUGH_INFO and c.id in predictions) / len(predictions)
    return Ceiling(
        verifiable=reachable,
        overall=1 - (1 - nei_share) * (1 - reachable),
        nei_share=nei_share,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--models", default=str(MODELS))
    args = parser.parse_args()

    calibration, test = split_dev(load_claims(DEV))
    claims = {c.id: c for c in (calibration if args.split == "calibration" else test)}

    retrieved: dict[int, list] = {}
    with open(RUNS / f"bm25-{args.split}" / "retrieved.jsonl", encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            retrieved[row["id"]] = row["evidence"]

    available = {}
    for variant in VARIANTS:
        path = Path(args.models) / variant / f"predictions_{args.split}.jsonl"
        if path.exists():
            available[variant] = read_predictions(path)
    if not available:
        raise SystemExit(
            f"no predictions under {args.models}; install an artifact first:\n"
            "  python -m scripts.install_artifacts artifacts_<variant>_v1.zip"
        )

    reports, correctness = {}, {}
    for variant, predictions in available.items():
        ordered = [claims[i] for i in predictions if i in claims]
        labels = [FEVER_TO_LABEL[c.label] for c in ordered]
        guesses = [predictions[c.id]["pred"] for c in ordered]
        read = [
            None if c.label == NOT_ENOUGH_INFO
            else recall_at_k(
                [(t, i) for t, i in retrieved[c.id]], c.groups, predictions[c.id]["n_evidence_used"]
            )
            for c in ordered
        ]
        reports[variant] = evaluate_verdicts(
            labels, guesses, variant=variant,
            ceiling=ceiling_for(ordered, retrieved, predictions), gold_read=read,
        )
        correctness[variant] = [g == p for g, p in zip(labels, guesses, strict=True)]

    print(f"{args.split}: {len(next(iter(available.values()))):,} claims\n")
    print(f"{'variant':<12} {'accuracy':>9} {'macro F1':>9}   " + "  ".join(f"{p.label[:9]:>9}" for p in next(iter(reports.values())).per_class))
    for variant, report in reports.items():
        cells = "  ".join(f"{c.f1:>9.3f}" for c in report.per_class)
        print(f"{variant:<12} {report.accuracy:>9.4f} {report.macro_f1:>9.4f}   {cells}")

    prior = max(c.support for c in next(iter(reports.values())).per_class) / next(iter(reports.values())).claims
    print(f"{'majority':<12} {prior:>9.4f}")

    main_report = reports.get("retrieved") or next(iter(reports.values()))
    print(f"\npacked ceiling  {main_report.ceiling.verifiable:.4f} verifiable"
          f"   |   nei share {main_report.ceiling.nei_share:.3f}")
    if main_report.by_retrieval:
        b = main_report.by_retrieval
        print(f"  gold read      {b.with_gold:.4f}  (n={b.n_with_gold:,})")
        print(f"  gold missed    {b.without_gold:.4f}  (n={b.n_without_gold:,})"
              "   <- near the prior means the model is reading artifacts, not evidence")

    if len(correctness) > 1:
        print("\npaired differences")
        base = "retrieved" if "retrieved" in correctness else next(iter(correctness))
        for variant in correctness:
            if variant == base:
                continue
            a = [float(x) for x in correctness[base]]
            b = [float(x) for x in correctness[variant]]
            interval = paired_bootstrap(a, b, seed=0)
            test_result = mcnemar(correctness[base], correctness[variant])
            sign = "significant" if interval.excludes_zero else "not significant"
            print(f"  {base} - {variant:<12} {interval.point:+.4f} "
                  f"[{interval.low:+.4f}, {interval.high:+.4f}]  "
                  f"McNemar p={test_result.p_value:.2e}  {sign}")

    out = RUNS / f"verdict-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps({v: r.to_dict() for v, r in reports.items()}, indent=2), encoding="utf-8"
    )
    print(f"\nwritten to {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
