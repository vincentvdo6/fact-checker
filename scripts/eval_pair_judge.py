"""
Score a pair judge on hand-labelled pairs from `labels/pairs-*.json`.

Only items with a filled `relation` are scored; the file is written by `scripts.harvest_pairs`
with every label empty, so a fresh harvest scores nothing rather than something misleading.
Each judge is the same class the decomposed chain runs, so what is measured is what the
extension would do. The numbers are printed with their n, and the file records which judge,
which labels and how many items were unlabelled.

    python -m scripts.eval_pair_judge --labels labels/pairs-frozen-2026-09-10.json --judge fever
    python -m scripts.eval_pair_judge --labels labels/pairs-frozen-2026-09-10.json --judge mnli-xsmall
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from src.calibration.bands import Band
from src.eval.pairs import report
from src.verdict.assertions import claim_assertions
from src.verdict.mnli_judge import MnliJudge
from src.verdict.nli_judge import NliJudge
from src.verdict.pair_judge import PairJudge

OUT = Path("runs/pair-judge-eval")


def units_and_assertions(items: list[dict]) -> tuple[list[dict], list[dict], list[tuple[str, str]]]:
    """One assertion per distinct text and one unit per distinct sentence, keyed for the judge."""
    assertions: dict[str, dict] = {}
    units: dict[str, dict] = {}
    keys = []
    for item in items:
        if item["assertion"] not in assertions:
            built = claim_assertions(item["assertion"])
            if len(built) != 1:
                raise SystemExit(f"a labelled assertion split into {len(built)} parts: {item['assertion']!r}")
            assertions[item["assertion"]] = built[0] | {"id": f"assertion-{len(assertions) + 1}"}
        key = f"{item['url']}|{item['sentence']}"
        if key not in units:
            units[key] = {"id": f"u{len(units) + 1}", "passage_id": f"p{len(units) + 1}", "source_id": item["url"] or "source",
                          "text": item["sentence"], "roles": ["reported_observation"], "eligible": True,
                          "definitions": [{"id": f"d{index}", "text": text} for index, text in enumerate(item["definitions"])]}
        keys.append((assertions[item["assertion"]]["id"], units[key]["id"]))
    return list(assertions.values()), list(units.values()), keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--judge", choices=["fever", "mnli-xsmall", "mnli-base", "pair"], default="fever")
    parser.add_argument("--probe-negation", action="store_true")
    parser.add_argument("--probe-hedges", action="store_true")
    parser.add_argument("--model-dir", default=None, help="pair only: an installed judge other than the default")
    parser.add_argument("--record", action="store_true",
                        help="pair only: write the counting record at the shipped band to <model dir>/news_measurement.json, "
                             "which the chain then prints beside every count; use only for labels on claims the judge never saw")
    args = parser.parse_args()
    payload = json.loads(args.labels.read_text(encoding="utf-8"))
    labelled = [item for item in payload["items"] if item.get("relation")]
    if not labelled:
        raise SystemExit(f"{args.labels} has no labelled items yet ({len(payload['items'])} unlabelled)")
    assertions, units, keys = units_and_assertions(labelled)
    sources = [{"id": unit["source_id"], "url": unit["source_id"], "reading_context": []} for unit in units]
    # A judge scores every assertion against every unit it is handed; each assertion is handed only the
    # sentences labelled against it, as the chain does, so a many-claim file is not a cross product.
    by_assertion: dict[str, list[dict]] = {}
    units_by_id = {unit["id"]: unit for unit in units}
    for assertion_id, unit_id in dict.fromkeys(keys):
        by_assertion.setdefault(assertion_id, []).append(units_by_id[unit_id])
    if args.judge == "fever":
        judge = NliJudge(sources, min_band=Band.WEAK)
    elif args.judge == "pair":
        judge = PairJudge(**({"model_dir": args.model_dir} if args.model_dir else {}),
                          probe_negation=args.probe_negation, probe_hedges=args.probe_hedges)
    else:
        judge = MnliJudge(model_dir=Path("models/nli") / f"nli-deberta-v3-{args.judge.split('-')[1]}")
    judged = {(row["assertion_id"], row["unit_id"]): row
              for assertion in assertions for row in judge([assertion], by_assertion[assertion["id"]])}
    predictions = [judged[key]["relation"] for key in keys]
    labels = [item["relation"] for item in labelled]
    scored = report(labels, predictions)
    sweep = threshold_sweep(labels, [judged[key] for key in keys])
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"labels": str(args.labels), "labeller": payload.get("labeller", ""), "judge": args.judge,
              "labelled": len(labelled), "unlabelled": len(payload["items"]) - len(labelled), "report": scored,
              "threshold_sweep": sweep,
              "rows": [{"id": item["id"], "label": item["relation"], "predicted": judged[key]["relation"],
                        "confidence": judged[key].get("confidence")} for item, key in zip(labelled, keys, strict=True)]}
    judge_name = f"{args.judge}-{Path(args.model_dir).name}" if args.model_dir else args.judge
    path = OUT / f"{judge_name}-{args.labels.stem}-{date.today().isoformat()}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in scored.items() if key != "confusion"}, indent=1))
    print("confusion (rows actual, columns predicted):")
    for actual, row in scored["confusion"].items():
        print(f"  {actual:<16}", " ".join(f"{row[predicted]:>4}" for predicted in row))
    if sweep:
        print("counting threshold sweep on the judge's raw counted labels (where a news band would sit):")
        for row in sweep:
            print(f"  >= {row['threshold']:.2f}  counted {row['counted']:>4}  wrong {row['wrong']:>3}  "
                  f"error {row['error']:.3f}  recall of true counts {row['recall']:.3f}")
    print(f"saved {path}")
    if args.record:
        if args.judge != "pair":
            raise SystemExit("--record applies to the pair judge only")
        record = {"labels": str(args.labels), "labeller": payload.get("labeller", ""), "measured_on": date.today().isoformat(),
                  "claims": len({item["case"] for item in labelled}), "pairs": len(labelled),
                  "counted": scored["counting_error"]["counted"], "wrong": scored["counting_error"]["wrong"],
                  "band": "shipped counting band", "threshold_sweep": sweep}
        target = judge.root / "news_measurement.json"
        target.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"recorded {target}: wrong {record['wrong']} of {record['counted']} counts on {record['claims']} claims")


def threshold_sweep(labels: list[str], rows: list[dict]) -> list[dict]:
    """Counting error and recall if every raw count at or above a threshold were admitted; nothing is fitted."""
    counted = {"states", "states_negation"}
    if not any("read_as" in row and "confidence" in row for row in rows):
        return []
    true_counts = sum(label in counted for label in labels)
    sweep = []
    for threshold in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        admitted = [(label, row) for label, row in zip(labels, rows, strict=True)
                    if row.get("read_as") in counted and float(row.get("confidence", 0)) >= threshold]
        wrong = sum(1 for label, row in admitted if label != row["read_as"])
        right = len(admitted) - wrong
        sweep.append({"threshold": threshold, "counted": len(admitted), "wrong": wrong,
                      "error": wrong / len(admitted) if admitted else 0.0,
                      "recall": right / true_counts if true_counts else 0.0})
    return sweep


if __name__ == "__main__":
    main()
