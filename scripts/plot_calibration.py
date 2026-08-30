"""
Render the figures from the metrics file, without recomputing anything.

Plotting is deliberately downstream of every number. This script reads
`runs/calibration-<split>/metrics.json` and draws it; it cannot disagree with the reported table
because it has no arithmetic of its own beyond laying out coordinates. That also keeps matplotlib
out of the metric path, so the code that produces reported numbers stays importable and testable
without a plotting stack.

Two figures per variant, because they answer different questions. The reliability diagram asks
whether a stated probability means what it says -- bars below the diagonal are overconfidence, the
direction that misleads a reader. The risk-coverage curve asks whether abstention buys anything,
and the oracle line beneath it is the honest reference: the gap between them, not the curve's
height, is what the confidence ranking contributed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # no display on a headless run; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("runs")


def reliability(axis, bins: list[dict], title: str) -> None:
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#999", label="perfect", zorder=1)
    occupied = [b for b in bins if b["count"] > 0]
    if occupied:
        centres = [(b["low"] + b["high"]) / 2 for b in occupied]
        width = min(b["high"] - b["low"] for b in occupied) * 0.85
        axis.bar(centres, [b["accuracy"] for b in occupied], width=width,
                 color="#4C72B0", edgecolor="white", linewidth=0.5, label="accuracy", zorder=2)
        axis.plot(centres, [b["confidence"] for b in occupied], marker="o", markersize=3,
                  linewidth=1.2, color="#C44E52", label="confidence", zorder=3)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("confidence")
    axis.set_ylabel("accuracy")
    axis.set_title(title, fontsize=10)
    axis.legend(fontsize=7, loc="upper left")


def risk_coverage(axis, curve: list[dict], optimal: float, excess: float, title: str) -> None:
    coverage = [p["coverage"] for p in curve]
    risk = [p["risk"] for p in curve]
    axis.plot(coverage, risk, linewidth=1.6, color="#4C72B0", label="model")
    if curve:
        # Random abstention: declining claims at random leaves risk where it started.
        axis.axhline(curve[-1]["risk"], linestyle="--", linewidth=1, color="#999",
                     label="random abstention")
    axis.set_xlim(0, 1)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("coverage")
    axis.set_ylabel("risk among answered")
    axis.set_title(f"{title}\nE-AURC {excess:.4f}  (oracle AURC {optimal:.4f})", fontsize=10)
    axis.legend(fontsize=7, loc="upper left")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    source = RUNS / f"calibration-{args.split}" / "metrics.json"
    if not source.exists():
        raise SystemExit(
            f"{source} does not exist; produce it first:\n"
            f"  python -m scripts.eval_calibration --split {args.split}"
        )
    metrics = json.loads(source.read_text(encoding="utf-8"))
    destination = source.parent

    for variant, payload in metrics["variants"].items():
        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        reliability(axes[0], payload["raw"]["bins"], f"{variant} raw\nECE {payload['raw']['ece']:.4f}")
        reliability(
            axes[1], payload["calibrated"]["bins"],
            f"{variant} calibrated ({payload['calibrator']})\nECE {payload['calibrated']['ece']:.4f}",
        )
        risk_coverage(
            axes[2], payload["selective"]["curve"], payload["selective"]["optimal_aurc"],
            payload["selective"]["excess_aurc"], f"{variant} risk-coverage",
        )
        figure.suptitle(f"{variant} -- {args.split} split, {payload['calibrated']['claims']:,} claims",
                        fontsize=11)
        figure.tight_layout()
        out = destination / f"calibration_{variant}.png"
        figure.savefig(out, dpi=args.dpi)
        plt.close(figure)
        print(f"  wrote {out}")

    print(f"\n{len(metrics['variants'])} figure(s) in {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
