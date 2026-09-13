"""
Assemble the four-relation pair dataset the pair-judge notebook reads, and refuse to write a bad one.

FEVER pairs follow the project's claim split exactly -- train / trainval from train.jsonl after
`drop_leaked` and `holdout`, calibration / test as the two halves of dev -- so no claim key is on
two sides of any evaluation, and the retrieval those pairs are drawn from is the same
`runs/evidence-*` and `runs/bm25-*` output every verdict model was trained on. MNLI pairs are
assigned by a salted hash of promptID: the two dev files become calibration and test, and a fixed
number of train prompts are held out as trainval. Unrelated pairs are drawn within a split, so a
donor sentence never travels across one.

Every check runs before a byte lands. The manifest records the quotas, the seeds, the label
histograms per split and source, the negated-hypothesis rate per relation, and the hashes of
everything read and written, so a number a notebook reports can be traced to the data it read.

    python -m scripts.build_pair_dataset                                                   # v1
    python -m scripts.build_pair_dataset --dest data/kaggle/pair-judge-v2 --negation-augment 0.3   # v2
    python -m scripts.build_pair_dataset --mnli-train-limit 60000 --dest data/kaggle/pair-judge-smoke --limit-train 2000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from src.data.fever import load_claims
from src.data.mnli import load_pairs
from src.data.splits import drop_leaked, holdout, split_dev
from src.retrieval import wiki
from src.verdict.dataset import SentenceStore, sha256
from src.verdict.pair_judgment import RELATIONS
from src.verdict.pairs import Pair, SentenceLookup, fever_pairs, mnli_pairs, negation_augment, validate

TRAIN = Path("data/fever/train.jsonl")
DEV = Path("data/fever/shared_task_dev.jsonl")
MNLI = Path("data/mnli")
RUNS = Path("runs")
DEST = Path("data/kaggle/pair-judge-v1")
MNLI_SALT = "pair-judge-mnli-v1"
MNLI_TRAINVAL_PROMPTS = 2000


def load_retrieved(path: Path) -> dict[int, list]:
    if not path.exists():
        raise SystemExit(f"{path} is missing; the verdict retrieval runs are a prerequisite")
    rows: dict[int, list] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["id"]] = row["evidence"]
    return rows


def _rank(prompt_id: str) -> int:
    return int.from_bytes(hashlib.sha1(f"{MNLI_SALT}:{prompt_id}".encode()).digest()[:8], "big")


def mnli_splits(limit: int) -> dict[str, list]:
    """Dev matched + mismatched cut in half by prompt hash; train minus a held-out prompt set."""
    dev = load_pairs(MNLI / "multinli_1.0_dev_matched.jsonl") + load_pairs(MNLI / "multinli_1.0_dev_mismatched.jsonl")
    prompts = sorted({row.prompt_id for row in dev}, key=_rank)
    calibration_prompts = set(prompts[: len(prompts) // 2])
    train_rows = load_pairs(MNLI / "multinli_1.0_train.jsonl")
    train_prompts = sorted({row.prompt_id for row in train_rows}, key=_rank)
    trainval_prompts = set(train_prompts[:MNLI_TRAINVAL_PROMPTS])
    kept_prompts: set[str] = set()
    budget = 0
    per_prompt = Counter(row.prompt_id for row in train_rows)
    for prompt in train_prompts[MNLI_TRAINVAL_PROMPTS:]:
        if budget + per_prompt[prompt] > limit:
            break
        kept_prompts.add(prompt)
        budget += per_prompt[prompt]
    return {"train": [row for row in train_rows if row.prompt_id in kept_prompts],
            "trainval": [row for row in train_rows if row.prompt_id in trainval_prompts],
            "calibration": [row for row in dev if row.prompt_id in calibration_prompts],
            "test": [row for row in dev if row.prompt_id not in calibration_prompts]}


def write(pairs: list[Pair], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--train-run", default="evidence-train")
    parser.add_argument("--limit-train", type=int, default=0, help="FEVER train claims to use (0 = all)")
    parser.add_argument("--mnli-train-limit", type=int, default=120_000)
    parser.add_argument("--gold-per-claim", type=int, default=2)
    parser.add_argument("--bears-per-claim", type=int, default=1)
    parser.add_argument("--unrelated-per-claim", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negation-augment", type=float, default=0.0,
                        help="train only: share of eligible pairs whose hypothesis is negated under a matching label")
    args = parser.parse_args()
    started = time.monotonic()

    calibration_claims, test_claims = split_dev(load_claims(DEV))
    train_all = drop_leaked(load_claims(TRAIN), calibration_claims, test_claims)
    train_claims, trainval_claims = holdout(train_all)
    if args.limit_train:
        train_claims = train_claims[: args.limit_train]
    fever_sources = {
        "train": (train_claims, RUNS / args.train_run / "retrieved.jsonl"),
        "trainval": (trainval_claims, RUNS / "evidence-trainval" / "retrieved.jsonl"),
        "calibration": (calibration_claims, RUNS / "bm25-calibration" / "retrieved.jsonl"),
        "test": (test_claims, RUNS / "bm25-test" / "retrieved.jsonl"),
    }
    lookup = SentenceLookup(SentenceStore(wiki.connect()))
    mnli = mnli_splits(args.mnli_train_limit)
    by_split: dict[str, list[Pair]] = {}
    inputs: dict[str, str] = {}
    for split, (claims, path) in fever_sources.items():
        retrieved = load_retrieved(path)
        inputs[str(path)] = sha256(path)
        present = [claim for claim in claims if claim.id in retrieved]
        print(f"{split}: {len(present):,} of {len(claims):,} FEVER claims have retrieval; {len(mnli[split]):,} MNLI rows",
              flush=True)
        rng = random.Random(f"{args.seed}:{split}")
        pairs = fever_pairs(present, retrieved, lookup, rng=rng, gold_per_claim=args.gold_per_claim,
                            bears_per_claim=args.bears_per_claim, unrelated_per_claim=args.unrelated_per_claim)
        pairs += mnli_pairs(mnli[split], rng=rng)
        rng.shuffle(pairs)
        by_split[split] = pairs
    if args.negation_augment:
        # After every split exists: an augmented train pair must not coincide with a held-out one.
        held_out = {(pair.premise, pair.hypothesis) for split, pairs in by_split.items() if split != "train"
                    for pair in pairs}
        rng = random.Random(f"{args.seed}:augment")
        added = negation_augment(by_split["train"], rng=rng, rate=args.negation_augment, exclude=held_out)
        by_split["train"] = by_split["train"] + added
        rng.shuffle(by_split["train"])
        print(f"negation augmentation added {len(added):,} train pairs", flush=True)
    counts = validate(by_split)

    negation = {split: {relation: [0, 0] for relation in RELATIONS} for split in by_split}
    for split, pairs in by_split.items():
        for pair in pairs:
            negation[split][pair.relation][0] += pair.negated_hypothesis
            negation[split][pair.relation][1] += 1
    files = {}
    for split, pairs in by_split.items():
        name = f"pairs_{split}.jsonl.gz"
        files[name] = {"rows": len(pairs), "sha256": write(pairs, args.dest / name)}
        print(f"wrote {name}: {len(pairs):,} pairs", flush=True)
    for name in ("manifest.json",):
        inputs[str(MNLI / name)] = sha256(MNLI / name)
    manifest = {
        "task": "pair_judge", "relations": list(RELATIONS), "contract_version": 1,
        "fields": {"premise": "the source sentence", "hypothesis": "the assertion",
                   "relation": "one of relations", "group": "leakage key, never crosses a split"},
        "quotas": {"gold_per_claim": args.gold_per_claim, "bears_per_claim": args.bears_per_claim,
                   "unrelated_per_claim": args.unrelated_per_claim, "mnli_train_limit": args.mnli_train_limit,
                   "mnli_trainval_prompts": MNLI_TRAINVAL_PROMPTS, "mnli_unrelated_every": 3, "limit_train": args.limit_train,
                   "negation_augment": args.negation_augment},
        "seed": args.seed, "mnli_salt": MNLI_SALT, "counts": counts,
        "negated_hypothesis": {split: {relation: {"negated": row[0], "total": row[1]} for relation, row in rows.items()}
                               for split, rows in negation.items()},
        "known_label_noise": ["FEVER annotation is not exhaustive: a retrieved sentence labelled bears_on may entail.",
                              "FEVER REFUTES claims carry negation words far more often than SUPPORTS claims."],
        "inputs": inputs, "files": files, "seconds": round(time.monotonic() - started, 1),
    }
    (args.dest / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.dest / "dataset-metadata.json").write_text(json.dumps(
        {"title": args.dest.name, "id": f"vincentvdo6/{args.dest.name}", "licenses": [{"name": "CC-BY-SA-4.0"}]},
        indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "seconds": manifest["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
