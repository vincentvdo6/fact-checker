"""
Add labelled news pairs to a pair-judge dataset's train split, as the judge will see them at run time.

The pair judge was trained on FEVER and MNLI and, measured on labelled news pairs, is wrong
about half the time when it counts a sentence. This writes a new dataset directory from an
existing one plus one or more label files (`labels/pairs-*.json`, only items with a relation),
touching the train split only: the calibration, trainval and test splits are copied byte for
byte, so every held-out number stays comparable across versions. A news pair uses the negation
probe's input form -- the premise is the sentence followed by its own-paragraph
definitions, and a negated assertion is asked in its positive form with the label inverted,
which is what the negation probe does at run time. Runtime hedge/figure transformation is not
applied: the labels describe the original assertion. A label file named with `--exclude` (the
measurement set) must share no (premise, hypothesis) with the training pairs, or nothing is
written. News pairs are repeated `--repeat` times after validation so a few thousand of them
weigh against the six hundred thousand base pairs; the manifest records the repeat, the label
files, their hashes and labeller, and the relation counts.

    python -m scripts.add_news_pairs --labels labels/pairs-train-2026-09-13.json \\
        --exclude labels/pairs-probe-2026-09-12.json --dest data/kaggle/pair-judge-v3 --repeat 16
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import time
from pathlib import Path

from src.verdict.assertions import claim_assertions, positive_form
from src.verdict.dataset import sha256
from src.verdict.pair_judgment import RELATIONS
from src.verdict.pairs import Pair, negated, validate

BASE = Path("data/kaggle/pair-judge-v2")
INVERTED = {"states": "states_negation", "states_negation": "states"}


def news_pair(item: dict, label_file: str) -> Pair:
    """One labelled news item as the judge would be asked it."""
    built = claim_assertions(item["assertion"])
    if len(built) != 1:
        raise ValueError(f"{item['id']}: the assertion splits into {len(built)} parts")
    hypothesis, relation = item["assertion"], item["relation"]
    if built[0]["negated"] and (positive := positive_form(item["assertion"])):
        hypothesis, relation = positive, INVERTED.get(relation, relation)
    premise = " ".join([item["sentence"], *item.get("definitions", [])])
    return Pair(id=f"news:{item['id']}", source="news", relation=relation, premise=premise, hypothesis=hypothesis,
                group=f"news:{item['case']}", negated_hypothesis=negated(hypothesis),
                provenance={"label_file": label_file, "qualifiers": list(item.get("qualifiers", [])),
                            "url": item.get("url", ""), "labelled_relation": item["relation"]})


def load_pairs(path: Path) -> list[Pair]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [Pair(**json.loads(line)) for line in handle]


def write_pairs(pairs: list[Pair], path: Path) -> str:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return sha256(path)


def fresh_pairs(news: list[Pair], base: list[Pair]) -> tuple[list[Pair], int, int]:
    """Count each news input once; a conflicting label must never be silently discarded."""
    unique: dict[tuple[str, str], Pair] = {}
    for pair in news:
        previous = unique.setdefault((pair.premise, pair.hypothesis), pair)
        if previous.relation != pair.relation:
            raise ValueError(f"{pair.id}: conflicting relation with {previous.id}")
    existing = {(pair.premise, pair.hypothesis): pair for pair in base}
    fresh = []
    for key, pair in unique.items():
        if key not in existing:
            fresh.append(pair)
        elif existing[key].relation != pair.relation:
            raise ValueError(f"{pair.id}: conflicting relation with {existing[key].id}")
    return fresh, len(news) - len(unique), len(unique) - len(fresh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="label files whose pairs must not enter training (the measurement set)")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.dest.exists():
        parser.error("--dest must be a new directory")
    started = time.monotonic()

    news, labellers = [], set()
    for path in args.labels:
        payload = json.loads(path.read_text(encoding="utf-8"))
        labellers.add(payload.get("labeller", ""))
        items = [item for item in payload["items"] if item.get("relation")]
        news.extend(news_pair(item, str(path)) for item in items)
        print(f"{path}: {len(items):,} labelled of {len(payload['items']):,}", flush=True)
    excluded = set()
    for path in args.exclude:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["items"]:
            pair = news_pair(item | {"relation": "unrelated"}, str(path))
            excluded.add((pair.premise, pair.hypothesis))
    clash = [pair.id for pair in news if (pair.premise, pair.hypothesis) in excluded]
    if clash:
        raise SystemExit(f"{len(clash)} training pairs also sit in an excluded label file, e.g. {clash[0]}")

    by_split = {split: load_pairs(args.base / f"pairs_{split}.jsonl.gz") for split in ("train", "trainval", "calibration", "test")}
    fresh, duplicates, already_in_base = fresh_pairs(news, by_split["train"])
    by_split["train"] = by_split["train"] + fresh
    counts = validate(by_split)         # unique pairs, every split; the repeats below are deliberate copies
    rng = random.Random(f"{args.seed}:news")
    repeated = [Pair(**(pair.to_dict() | {"id": f"{pair.id}#{copy}"})) for copy in range(1, args.repeat) for pair in fresh]
    train = by_split["train"] + repeated
    rng.shuffle(train)

    args.dest.mkdir(parents=True)
    files = {}
    for split in ("trainval", "calibration", "test"):
        name = f"pairs_{split}.jsonl.gz"
        shutil.copyfile(args.base / name, args.dest / name)
        files[name] = {"rows": len(by_split[split]), "sha256": sha256(args.dest / name), "copied_from": str(args.base / name)}
    files["pairs_train.jsonl.gz"] = {"rows": len(train), "sha256": write_pairs(train, args.dest / "pairs_train.jsonl.gz")}
    relations = {relation: sum(pair.relation == relation for pair in fresh) for relation in RELATIONS}
    base_manifest = json.loads((args.base / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest = base_manifest | {
        "base": {"dir": str(args.base), "manifest_sha256": sha256(args.base / "dataset_manifest.json")},
        "news": {"label_files": {str(path): sha256(path) for path in args.labels}, "labellers": sorted(labellers),
                 "excluded": {str(path): sha256(path) for path in args.exclude}, "unique_pairs": len(fresh),
                 "duplicate_rows": duplicates, "already_in_base": already_in_base,
                 "repeat": args.repeat, "relations": relations,
                 "shape": "premise = sentence + own-paragraph definitions; a negated assertion is asked in its "
                          "positive form with the label inverted; runtime hedge/figure transformation is not applied"},
        "counts": counts, "files": files, "seconds": round(time.monotonic() - started, 1)}
    (args.dest / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.dest / "dataset-metadata.json").write_text(json.dumps(
        {"title": args.dest.name, "id": f"vincentvdo6/{args.dest.name}", "licenses": [{"name": "CC-BY-SA-4.0"}]},
        indent=2), encoding="utf-8")
    print(json.dumps({"news_unique": len(fresh), "repeat": args.repeat, "train_rows": len(train), "relations": relations,
                      "seconds": manifest["seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
