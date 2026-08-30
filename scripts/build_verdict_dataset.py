"""
Assemble the four splits the training notebooks read, and refuse to write a bad one.

Every check runs before a byte lands on disk. The failure this exists to prevent is a bad
dataset discovered on a GPU: by then the wall clock is spent, the quota is spent, and the
numbers have to be thrown away. Ten minutes here is the cheapest place to catch it.

The manifest carries more than provenance. It holds the ceiling curve, the label histograms and
the training prior, so a notebook can print accuracy beside its bound without recomputing
anything -- and so the numbers a run reports can be traced to the data it actually read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.data.fever import NOT_ENOUGH_INFO, Claim, load_claims
from src.data.splits import drop_leaked, holdout, split_dev
from src.eval.retrieval import recall_at_k
from src.retrieval import wiki
from src.verdict.dataset import (
    Row,
    SentenceStore,
    assert_splits_disjoint,
    build_rows,
    sha256,
    validate,
    write_rows,
)
from src.verdict.encode import FEVER_TO_LABEL, LABELS, TEMPLATE_ID

TRAIN = "data/fever/train.jsonl"
DEV = "data/fever/shared_task_dev.jsonl"
RUNS = Path("runs")
DEST = Path("data/kaggle/fever-verdict-v1")

CONTRACT_VERSION = 1
CEILING_KS = tuple(range(1, 26))


def load_retrieved(path: Path) -> dict[int, list]:
    if not path.exists():
        raise SystemExit(f"{path} is missing; run scripts/retrieve_evidence.py --merge first")
    rows: dict[int, list] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["id"]] = row["evidence"]
    return rows


def ceiling_curve(claims: list[Claim], retrieved: dict[int, list]) -> dict[str, float]:
    """Strict recall at every k, verifiable only, so a notebook never recomputes it."""
    verifiable = [c for c in claims if c.label != NOT_ENOUGH_INFO]
    curve: dict[str, float] = {}
    for k in CEILING_KS:
        hits = sum(
            recall_at_k([(t, i) for t, i in retrieved[c.id]], c.groups, k) for c in verifiable
        )
        curve[str(k)] = hits / len(verifiable)
    return curve


def summarise(rows: list[Row]) -> dict[str, object]:
    labels = Counter(r.label for r in rows)
    return {
        "rows": len(rows),
        "labels": {label: labels.get(label, 0) for label in LABELS},
        "prior": {label: labels.get(label, 0) / len(rows) for label in LABELS},
        "gold_unresolved": sum(1 for r in rows if not r.gold_resolved),
        "mean_evidence": sum(len(r.evidence) for r in rows) / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", default="evidence-train")
    parser.add_argument("--dest", default=str(DEST))
    parser.add_argument("--limit-train", type=int, default=0, help="smaller train split, for a dry run")
    args = parser.parse_args()

    calibration_claims, test_claims = split_dev(load_claims(DEV))
    train_all = drop_leaked(load_claims(TRAIN), calibration_claims, test_claims)
    train_claims, trainval_claims = holdout(train_all)
    if args.limit_train:
        train_claims = train_claims[: args.limit_train]

    sources = {
        "train": (train_claims, RUNS / args.train_run / "retrieved.jsonl"),
        "trainval": (trainval_claims, RUNS / args.train_run / "retrieved.jsonl"),
        "calibration": (calibration_claims, RUNS / "bm25-calibration" / "retrieved.jsonl"),
        "test": (test_claims, RUNS / "bm25-test" / "retrieved.jsonl"),
    }

    conn = wiki.connect()
    store = SentenceStore(conn)
    by_split: dict[str, list[Row]] = {}
    claims_by_split: dict[str, dict[int, Claim]] = {}
    cached: dict[Path, dict[int, list]] = {}

    for split, (claims, path) in sources.items():
        if path not in cached:
            cached[path] = load_retrieved(path)
        retrieved = cached[path]
        # calibration and test were retrieved for a 2,000-claim sample, not the whole split.
        present = [c for c in claims if c.id in retrieved]
        if not present:
            raise SystemExit(f"{split}: no claims have retrieved evidence in {path}")
        print(f"{split}: {len(present):,} of {len(claims):,} claims have evidence", flush=True)

        rows = list(build_rows(present, retrieved, store))
        validate(rows, {c.id: c for c in present}, split=split)
        by_split[split] = rows
        claims_by_split[split] = {c.id: c for c in present}

    assert_splits_disjoint(by_split, claims_by_split)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "template_id": TEMPLATE_ID,
        "labels": list(LABELS),
        "fever_to_label": FEVER_TO_LABEL,
        "splits": {},
        "files": {},
    }

    for split, rows in by_split.items():
        name = f"verdict_{split}.jsonl.gz"
        digest = write_rows(rows, dest / name)
        manifest["splits"][split] = summarise(rows)
        manifest["files"][name] = {"sha256": digest, "bytes": (dest / name).stat().st_size}
        print(f"  wrote {name}  {(dest / name).stat().st_size / 1e6:.1f} MB", flush=True)

    manifest["train_prior"] = manifest["splits"]["train"]["prior"]
    for split in ("calibration", "test"):
        manifest["splits"][split]["ceiling_curve"] = ceiling_curve(
            list(claims_by_split[split].values()), cached[sources[split][1]]
        )

    spec = Path("notebooks/encode_spec.py")
    (dest / spec.name).write_bytes(spec.read_bytes())
    manifest["files"][spec.name] = {"sha256": sha256(dest / spec.name)}

    (dest / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'split':<12} {'rows':>9}  " + "  ".join(f"{label[:9]:>10}" for label in LABELS))
    for split, rows in by_split.items():
        counts = manifest["splits"][split]["labels"]
        print(f"{split:<12} {len(rows):>9,}  " + "  ".join(f"{counts[label]:>10,}" for label in LABELS))
    unresolved = sum(manifest["splits"][s]["gold_unresolved"] for s in by_split)
    print(f"\ngold unresolved across all splits: {unresolved:,}")
    print(f"written to {dest}\nnext: python -m scripts.check_verdict_dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
