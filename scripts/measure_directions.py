"""Measure what the viewer sees: when the reading points a direction on an unseen claim, is it right?

The judge's per-sentence counting error (`eval_pair_judge`) is the wrong unit for the composed
verdict. A wrong count usually lands on a page that is about the claim being true or false, so
the direction the rules compose from it is right far more often than the sentence is. This
script reads clicks recorded with FACT_CHECKER_RECORD_CLICKS, takes every assertion whose
status has a direction (supported / qualified_support / contradicted / qualified_contradiction),
and scores it against a truth file written for the claims. Claims whose truth is null are
listed and left unscored.

The direction of an assertion does not depend on the single-source rule: that rule only decides
whether a directional reading is "established" or "not established", so a record measured
under either rule scores the same readings. The record is written beside the active judge
(`direction_measurement.json`) with `--record`, and composition shows it on every page that
points a direction. The measurement is in-sample for the rule decided on it; the next batch of
recorded clicks is its test.

    python -m scripts.measure_directions --clicks runs/clicks-probe --clicks runs/clicks-count \
        --truth labels/claim-truth-2026-09-13.json [--record]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from src.verdict.composition import AGAINST, FOR
from src.verdict.pair_judge import MODEL_DIR

DIRECTIONAL = set(FOR) | set(AGAINST)


def _publisher(url: str) -> str:
    host = urlsplit(url or "").hostname or ""
    return host.removeprefix("www.")


def readings(click_dirs: list[Path]) -> tuple[list[dict], int]:
    """Every directional assertion in the recorded clicks, and the number of claims read."""
    rows, claims = [], 0
    for directory in click_dirs:
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            for row in record["result"]["rows"]:
                decomposed = (row.get("result") or {}).get("decomposed") or {}
                verdict = decomposed.get("verdict")
                if not verdict:
                    continue
                claims += 1
                for assertion in verdict["assertions"]:
                    if assertion["status"] not in DIRECTIONAL:
                        continue
                    rows.append({"click": path.name, "claim": decomposed["claim"], "assertion": assertion["text"],
                                 "status": assertion["status"], "direction": assertion["direction"],
                                 "limits": list(assertion.get("limits", [])),
                                 "sources": sorted({_publisher(item.get("url", "")) for item in assertion["evidence"]})})
    return rows, claims


def score(rows: list[dict], truth: dict[str, dict]) -> dict:
    scored, unscored = [], []
    for row in rows:
        entry = truth.get(row["claim"])
        if entry is None:
            raise KeyError(f"no truth value for claim: {row['claim']!r}")
        if entry["true"] is None:
            unscored.append({**row, "note": entry.get("note", "")})
            continue
        right = (row["direction"] == "for") == bool(entry["true"])
        scored.append({**row, "truth": entry["true"], "right": right})
    wrong = [row for row in scored if not row["right"]]
    limits = Counter(limit for row in scored for limit in row["limits"])
    return {"directional": len(scored), "right": len(scored) - len(wrong), "wrong": len(wrong),
            "established": sum(row["status"] in ("supported", "contradicted") for row in scored),
            "held_by_limit": dict(limits), "unscored": unscored, "wrong_readings": wrong, "readings": scored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clicks", type=Path, action="append", required=True, help="a directory of recorded clicks; repeatable")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--measured-on", default=None, help="date label for the record; default: the truth file's date")
    parser.add_argument("--record", action="store_true", help=f"write direction_measurement.json under {MODEL_DIR}")
    parser.add_argument("--out", type=Path, default=Path("runs/direction-measurement/report.json"))
    args = parser.parse_args()
    truth_file = json.loads(args.truth.read_text(encoding="utf-8"))
    rows, claims = readings(args.clicks)
    report = score(rows, truth_file["claims"])
    measured_on = args.measured_on or truth_file.get("written", "")
    record = {"claims": claims, "directional": report["directional"], "right": report["right"], "wrong": report["wrong"],
              "measured_on": measured_on, "truth": str(args.truth), "truth_labeller": truth_file.get("labeller", ""),
              "clicks": [str(path) for path in args.clicks],
              "note": "directions are the same under the two-source rule these clicks were recorded with and the single-source "
                      "rule decided on them; in-sample for that decision"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"record": record, **report}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items() if key not in ("truth_labeller", "note")}))
    print("held by limit:", json.dumps(report["held_by_limit"]))
    for row in report["wrong_readings"]:
        print("WRONG", row["status"], "|", row["claim"], "|", row["sources"])
    for row in report["unscored"]:
        print("unscored", row["status"], "|", row["claim"], "|", row["note"])
    if args.record:
        target = MODEL_DIR / "direction_measurement.json"
        target.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", target)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
