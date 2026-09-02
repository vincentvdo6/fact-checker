"""
Assemble ClaimBuster into train / calibration / test for a check-worthiness head.

**Split by debate, never by sentence.** Adjacent sentences in a debate share speaker, topic and
often clause structure, so a random sentence split lets the model memorise a debate and score it
back. That inflates the number and would be invisible: the leak looks like skill. The 33 files are
partitioned instead, hashed by name so the assignment is stable when the data is refetched.

**Three classes, kept.** ClaimBuster separates factual-but-unimportant (UFS) from check-worthy
factual (CFS), and the difference is *importance* rather than verifiability. The Phase 07 rubric
labelled the 120 SOTU sentences by verifiability alone, so the two definitions disagree on exactly
the UFS band. Binarizing here would bury that disagreement; keeping three classes lets the
evaluation report both readings and say which one the demo should act on:

  factual      = UFS + CFS   the Phase 07 rubric's notion, comparable to the hand labels
  check-worthy = CFS         ClaimBuster's own task, stricter, what a fact-checker would triage

**The 120 SOTU labels are not in here and must never be.** They are the out-of-domain test for
whatever this trains, and their independence -- different annotators, different genre, different
decade -- is the only thing that makes the comparison against the hand-written filter meaningful.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

SOURCE = Path("data/claimbuster")
DEST = Path("data/kaggle/checkworthy-v1")
CONTRACT_VERSION = 1
SALT = "checkworthy-split-v1"

# Class index order is part of the on-disk contract for any trained head: append, never reorder.
LABELS = ("non_factual", "unimportant_factual", "check_worthy")
FROM_VERDICT = {"-1": "non_factual", "0": "unimportant_factual", "1": "check_worthy"}

# Of 33 debates: roughly two thirds to train, the rest split between fitting thresholds and the
# held-out report. Small in absolute terms, but a debate is 320-1012 sentences.
CALIBRATION_FILES = 5
TEST_FILES = 6


def rank(name: str) -> str:
    return hashlib.sha256(f"{SALT}:{name}".encode()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_row(record: dict) -> dict:
    return {
        "id": f"cb-{record['Sentence_id']}",
        "label": FROM_VERDICT[record["Verdict"]],
        "text": record["Text"].strip(),
        "debate": record["File_id"],
        "speaker": record["Speaker"],
        "party": record["Speaker_party"],
    }


def summarise(rows: list[dict]) -> dict[str, object]:
    counts = Counter(r["label"] for r in rows)
    return {
        "rows": len(rows),
        "debates": len({r["debate"] for r in rows}),
        "labels": {name: counts.get(name, 0) for name in LABELS},
        "prior": {name: counts.get(name, 0) / len(rows) for name in LABELS},
        # Both binarizations, so the evaluation can report either without recomputing.
        "factual_rate": (counts.get("unimportant_factual", 0)
                         + counts.get("check_worthy", 0)) / len(rows),
        "check_worthy_rate": counts.get("check_worthy", 0) / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--dest", default=str(DEST))
    args = parser.parse_args()

    source = Path(args.source)
    crowd = [to_row(r) for r in read_rows(source / "crowdsourced.csv")]
    truth = [to_row(r) for r in read_rows(source / "groundtruth.csv")]

    debates = sorted({r["debate"] for r in crowd}, key=rank)
    held_test = set(debates[:TEST_FILES])
    held_calibration = set(debates[TEST_FILES:TEST_FILES + CALIBRATION_FILES])
    print(f"{len(crowd):,} crowdsourced sentences over {len(debates)} debates")

    by_split = {
        "train": [r for r in crowd if r["debate"] not in held_test | held_calibration],
        "calibration": [r for r in crowd if r["debate"] in held_calibration],
        "test": [r for r in crowd if r["debate"] in held_test],
    }

    # A debate in two splits is the leak this whole scheme exists to prevent, so it is asserted
    # rather than assumed.
    seen = {name: {r["debate"] for r in rows} for name, rows in by_split.items()}
    for a in seen:
        for b in seen:
            if a < b and seen[a] & seen[b]:
                raise SystemExit(f"{a} and {b} share debates: {sorted(seen[a] & seen[b])}")
    if sum(len(v) for v in by_split.values()) != len(crowd):
        raise SystemExit("splits do not partition the crowdsourced rows")

    # groundtruth is a re-labelling of sentences from the same debates, so it cannot be a fourth
    # split without leaking. It travels as a label-quality probe, restricted to the test debates.
    probe = [r for r in truth if r["debate"] in held_test]

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "labels": list(LABELS),
        "source": "claimbuster",
        "licence": "cc-by-4.0",
        "citation": ("Arslan, Hassan, Li, Tremayne. A Benchmark Dataset of Check-worthy "
                     "Factual Claims. ICWSM 2020."),
        "split_by": "debate file, hashed by name",
        "splits": {},
        "files": {},
    }
    for name, rows in by_split.items():
        path = dest / f"checkworthy_{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][name] = summarise(rows)
        manifest["files"][path.name] = {"bytes": path.stat().st_size}

    with open(dest / "checkworthy_groundtruth.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in probe:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["groundtruth_probe"] = summarise(probe) if probe else {"rows": 0}

    manifest["train_prior"] = manifest["splits"]["train"]["prior"]
    (dest / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'split':<14} {'rows':>7} {'debates':>8}  " + "  ".join(f"{n[:11]:>12}" for n in LABELS))
    for name, rows in by_split.items():
        s = manifest["splits"][name]
        print(f"{name:<14} {s['rows']:>7,} {s['debates']:>8}  "
              + "  ".join(f"{s['labels'][n]:>12,}" for n in LABELS))
    print(f"\n{'':<14} {'factual':>10} {'check-worthy':>14}   (the two binarizations)")
    for name in by_split:
        s = manifest["splits"][name]
        print(f"{name:<14} {s['factual_rate']:>10.4f} {s['check_worthy_rate']:>14.4f}")
    print(f"\ngroundtruth probe on test debates: {manifest['groundtruth_probe']['rows']:,} sentences")
    print(f"written to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
