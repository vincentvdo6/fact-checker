"""
Replay the seven frozen cases through the decomposed chain with the local NLI judge.

Reads the archived claim packets and source-only role annotations straight from the
retired-experiment archive, so nothing here depends on a model that has been deleted.
The role annotations are the saved medium-effort readings. The judge is one of the
small local cross-encoders behind the pair contract: the calibrated FEVER verdict model,
or an off-the-shelf SNLI/MNLI model fetched by `scripts.fetch_nli_judge`. The output
records every stage for every case beside the frozen oracle's allowed relationships.
Agreement with the oracle on seven development cases is a diagnostic, not accuracy.

    python -m scripts.replay_decomposed                       # FEVER model, any band
    python -m scripts.replay_decomposed --min-band moderate
    python -m scripts.replay_decomposed --judge mnli-xsmall
    python -m scripts.replay_decomposed --judge mnli-base
    python -m scripts.replay_decomposed --judge pair          # the trained judge, once installed
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from datetime import date
from pathlib import Path

from src.calibration.bands import Band
from src.pipeline.segment import segment
from src.verdict.chain import run_chain
from src.verdict.mnli_judge import MnliJudge
from src.verdict.nli_judge import NliJudge
from src.verdict.pair_judge import PairJudge
from src.verdict.reading import reading_packet
from src.verdict.role_rules import annotate

ARCHIVE = Path("runs/retired-experiments/records-before-cleanup.zip")
CASES = "runs/local-reasoner/brief-answer-gpt-oss-sampler-input-2026-09-10.json"
ROLES = "runs/local-reasoner/source-bundle-gpt-oss-medium-2026-09-10.json"
OUT = Path("runs/decomposed-replay")


def project_annotations(packet: dict, archived: list[dict]) -> list[dict]:
    """Archived roles onto the current reading's units.

    The archive annotated sentences segmented without line breaks, so a heading kept with its
    paragraph was one sentence there and is two units now, and every later number in that
    paragraph has shifted. Units are therefore matched by span, never by identity: each current
    unit inherits roles only when its entire span fits in one archived sentence.
    Repaired citation boundaries can merge old fragments; those need fresh labels.
    """
    roles = {row["unit_id"]: row["roles"] for row in archived}
    projected = []
    for passage in reading_packet(packet)["passages"]:
        old = [(f"{passage['id']}:u{sentence.index + 1}", sentence.start, sentence.end)
               for sentence in segment(passage["text"])]
        for unit in passage["units"]:
            holder = next((identity for identity, start, end in old
                           if start <= unit["start"] < unit["end"] <= end), None)
            if holder is None:
                raise ValueError(f"Sentence boundaries changed for {unit['id']}; archived roles require relabelling.")
            projected.append({"unit_id": unit["id"], "roles": roles[holder]})
    return projected


def load_records(archive: Path, *, with_archived_roles: bool = True) -> tuple[list[dict], dict[int, list[dict]]]:
    with zipfile.ZipFile(archive) as bundle:
        cases = json.loads(bundle.read(CASES).decode("utf-8"))["frozen_cases"]
        if not with_archived_roles:
            return cases, {}
        readings = json.loads(bundle.read(ROLES).decode("utf-8"))["runs"]
    annotations: dict[int, list[dict]] = {}
    for run in readings:
        index = int(run["id"].split(":")[1])
        content = json.loads(run["response"]["choices"][0]["message"]["content"])
        annotations.setdefault(index, []).extend(
            {"unit_id": row["unit_id"], "roles": row["roles"]} for row in content["annotations"])
    for index, case in enumerate(cases, 1):
        annotations[index] = project_annotations(case["original_packet"], annotations[index])
    return cases, annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--judge", choices=["fever", "mnli-xsmall", "mnli-base", "pair"], default="fever")
    parser.add_argument("--min-band", choices=[band.value for band in Band], default=Band.WEAK.value,
                        help="fever and pair: weakest calibrated band a pair needs before its label counts")
    parser.add_argument("--probe-negation", action="store_true",
                        help="pair only: judge the positive form of a denial and invert, as the fever judge does")
    parser.add_argument("--probe-hedges", action="store_true",
                        help="pair only: read a hedged figure against the sentence's own fitting figure")
    parser.add_argument("--model-dir", default="models/pair_judge/v4", help="pair only: which installed judge")
    parser.add_argument("--roles", choices=["archived", "rules"], default="archived",
                        help="archived model annotations, or the closed-class rule typer the live path uses")
    args = parser.parse_args()
    cases, annotations = load_records(args.archive, with_archived_roles=args.roles == "archived")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"archive": str(args.archive), "judge": args.judge, "min_band": args.min_band if args.judge in ("fever", "pair") else None,
              "probe_negation": args.probe_negation, "probe_hedges": args.probe_hedges,
              "roles": "saved medium-effort source-only annotations" if args.roles == "archived" else "closed-class rules",
              "cases": []}
    started = time.monotonic()
    for index, case in enumerate(cases, 1):
        packet = case["original_packet"]
        if args.judge == "fever":
            judge = NliJudge(packet["sources"], min_band=Band(args.min_band))
        elif args.judge == "pair":
            judge = PairJudge(model_dir=args.model_dir, min_band=Band(args.min_band), probe_negation=args.probe_negation,
                              probe_hedges=args.probe_hedges)
        else:
            judge = MnliJudge(model_dir=Path("models/nli") / f"nli-deberta-v3-{args.judge.split('-')[1]}")
        clock = time.monotonic()
        roles = annotations[index] if args.roles == "archived" else annotate(reading_packet(packet))
        result = run_chain(packet, roles, judge)
        expected = case["case"].get("expected", {}).get("allowed_overall_relationships")
        # The oracle predates the dated-source rule: a control whose synthetic sources carry no publication date
        # can reach qualified at most, so "supported"/"contradicted" there is read as "qualified" -- stated, not
        # tuned. (Between 2026-09-11 and 2026-09-14 the same mapping applied for a single source; one dated
        # independent source establishes again.)
        undated = not any(source.get("published_at") for source in packet["sources"])
        allowed = None if expected is None else sorted({("qualified" if undated and value in ("supported", "contradicted")
                                                          else value) for value in expected})
        summary = {"id": case["case"]["id"], "relationship": result["verdict"]["relationship"],
                   "summary": result["verdict"]["summary"], "allowed": expected,
                   "allowed_under_dated_source_rule": allowed,
                   "agrees": None if allowed is None else result["verdict"]["relationship"] in allowed,
                   "eligible": result["verdict"]["eligible"], "withheld": result["verdict"]["withheld"],
                   "pairs": len(result["judgments"]),
                   "counted": sum(1 for row in result["judgments"] if row["relation"] in ("states", "states_negation")),
                   "seconds": round(time.monotonic() - clock, 3)}
        report["cases"].append(summary | {"result": result})
        print(json.dumps({key: value for key, value in summary.items() if key != "withheld"}), flush=True)
    report["seconds"] = round(time.monotonic() - started, 3)
    tag = f"{args.judge}-{args.min_band}" if args.judge in ("fever", "pair") else args.judge
    if args.judge == "pair":
        tag += "-" + Path(args.model_dir).name
    if args.probe_negation:
        tag += "-probe"
    if args.probe_hedges:
        tag += "-hedges"
    if args.roles == "rules":
        tag += "-rules"
    path = OUT / f"{tag}-{date.today().isoformat()}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for row in report["cases"]:
        print(f"\n===== {row['id']}\n{row['result']['text']}")
    print(f"\nsaved {path} in {report['seconds']} s")


if __name__ == "__main__":
    main()
