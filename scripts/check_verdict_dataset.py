"""
Re-validate the written dataset, reading it exactly the way the notebook will.

The builder validates what it is about to write. This validates what it wrote. Those are
different claims, and only the second one is what a training run depends on -- a truncated
write, a compression error or a stale file from an earlier build would all pass the first check
and fail the second.

Run this, read the output, and only then upload. It is the last check that costs minutes rather
than GPU hours.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.data.fever import load_claims
from src.data.splits import drop_leaked, holdout, split_dev
from src.verdict.dataset import assert_splits_disjoint, read_rows, sha256, validate
from src.verdict.encode import LABELS, TEMPLATE_ID

TRAIN = "data/fever/train.jsonl"
DEV = "data/fever/shared_task_dev.jsonl"
DEST = Path("data/kaggle/fever-verdict-v1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DEST))
    args = parser.parse_args()

    dest = Path(args.dest)
    manifest_path = dest / "dataset_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} is missing; run scripts/build_verdict_dataset.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: list[str] = []

    if manifest["template_id"] != TEMPLATE_ID:
        failures.append(f"template {manifest['template_id']} != {TEMPLATE_ID}")
    if manifest["labels"] != list(LABELS):
        failures.append(f"labels {manifest['labels']} != {list(LABELS)}")

    print("=== files ===")
    for name, entry in manifest["files"].items():
        path = dest / name
        if not path.exists():
            failures.append(f"{name} is missing")
            continue
        digest = sha256(path)
        ok = digest == entry["sha256"]
        print(f"  {'ok  ' if ok else 'BAD '} {name}")
        if not ok:
            failures.append(f"{name} sha256 {digest[:12]} != manifest {entry['sha256'][:12]}")

    calibration_claims, test_claims = split_dev(load_claims(DEV))
    train_all = drop_leaked(load_claims(TRAIN), calibration_claims, test_claims)
    train_claims, trainval_claims = holdout(train_all)
    expected = {
        "train": {c.id: c for c in train_claims},
        "trainval": {c.id: c for c in trainval_claims},
        "calibration": {c.id: c for c in calibration_claims},
        "test": {c.id: c for c in test_claims},
    }

    # Which splits the builder relabelled, taken from the manifest rather than from a flag: the
    # checker cannot then be told to tolerate a relabel the build did not actually record, and an
    # unrecorded relabel stays a failure.
    regrounded = manifest.get("regrounded") or {}
    if regrounded:
        moved = ", ".join(
            f"{split} {stats['moved']:,} ({stats['share_of_verifiable']:.1%} of verifiable)"
            for split, stats in regrounded.items()
        )
        print(f"\n=== regrounded: {moved} ===")

    print("\n=== splits, as the notebook reads them ===")
    by_split = {}
    claims_by_split = {}
    for split in expected:
        path = dest / f"verdict_{split}.jsonl.gz"
        if not path.exists():
            failures.append(f"{path.name} is missing")
            continue
        rows = list(read_rows(path))
        present = {r.id: expected[split][r.id] for r in rows if r.id in expected[split]}
        stray = [r.id for r in rows if r.id not in expected[split]]
        if stray:
            failures.append(f"{split}: {len(stray)} rows are not in the split, first {stray[:3]}")
            continue

        try:
            validate(rows, present, split=split, regrounded=split in regrounded)
        except ValueError as error:
            failures.append(str(error))
            continue

        by_split[split] = rows
        claims_by_split[split] = present
        counts = Counter(r.label for r in rows)
        declared = manifest["splits"][split]["rows"]
        match = "ok  " if declared == len(rows) else "BAD "
        if declared != len(rows):
            failures.append(f"{split}: {len(rows)} rows on disk, manifest says {declared}")
        print(f"  {match} {split:<12} {len(rows):>8,}  " + "  ".join(f"{label[:4]} {counts.get(label, 0):>7,}" for label in LABELS))

    try:
        assert_splits_disjoint(by_split, claims_by_split)
        print("\n  ok   no claim id or key spans two splits")
    except ValueError as error:
        failures.append(str(error))

    if failures:
        print(f"\nFAILED, {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    total = sum(len(rows) for rows in by_split.values())
    print(f"\nall checks passed: {total:,} rows across {len(by_split)} splits")
    print("next: python -m scripts.upload_verdict_dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
