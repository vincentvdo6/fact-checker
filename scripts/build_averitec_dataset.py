"""
Assemble AVeriTeC into the row shape the training notebooks already read.

The question this dataset exists to answer: Phase 02 measured an evidence-free DeBERTa scoring
0.5865 on FEVER against a 0.3410 majority, and every finding since rests on that leak. A unigram
probe (scripts/measure_claim_artifact.py) found FEVER and AVeriTeC leak about equally at matched
training size, but could not rule out that FEVER's keeps growing with data while AVeriTeC's does
not -- a bag of words cannot tell us what a transformer would extract. Only training one does.

**MIXED is dropped, and that is a stated scope rather than a convenience.** The notebook's encoder
declares FEVER's three verdicts, and `Conflicting Evidence/Cherrypicking` has no place in them.
Keeping it would mean a four-class head, a new template id and a contract version bump, and would
also make the comparison a four-class problem against a three-class one. Dropping it costs 233
claims of 3,568 (6.5%) and buys a like-for-like measurement: same label space, same encoder, same
notebook. The four-class problem is a separate question and needs its own contract.

Evidence is the annotator's question-answer pairs, one per synthetic "page", carrying the shape
`(title, index, text)` that `Row` expects. There is no retrieval here and no corpus -- these pairs
are what a human found, so they are gold by construction and every row is marked resolved. That
makes this dataset usable for the `gold` variant as well as `claim_only`; what it cannot support
is `retrieved`, because nothing was retrieved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.data.averitec import load_all
from src.data.splits import _rank, split_averitec
from src.verdict.dataset import Row, write_rows
from src.verdict.encode import LABELS, TEMPLATE_ID

DEST = Path("data/kaggle/averitec-v1")
CONTRACT_VERSION = 1
TRAINVAL_SALT = "averitec-trainval-v1"
TRAINVAL_FRACTION = 0.12


def to_row(claim) -> Row:
    """
    One AVeriTeC claim in the notebook's row shape.

    Each question-answer pair becomes its own single-sentence page. The synthetic title keeps the
    encoder's per-page grouping meaningful -- pairs are independent findings, and rendering them
    under one title would let `render` merge unrelated evidence into a single run.
    """
    evidence = tuple(
        (f"Question {i + 1}", i, f"{question} {answer}".strip())
        for i, (question, answer) in enumerate(claim.questions)
    )
    return Row(
        id=claim.id,
        label=claim.label,
        claim=claim.text,
        evidence=evidence,
        gold=evidence,          # a human assembled these; there is nothing better to reserve
        gold_resolved=bool(evidence),
    )


def carve_trainval(train: list) -> tuple[list, list]:
    """Hold out a model-selection split, by claim key, stratified by verdict."""
    keys_by_label: dict[str, list[str]] = {}
    for claim in train:
        keys_by_label.setdefault(claim.label, []).append(claim.key)

    held: set[str] = set()
    for label, keys in sorted(keys_by_label.items()):
        ordered = sorted(set(keys), key=lambda k: _rank(k, TRAINVAL_SALT))
        held.update(ordered[: round(len(ordered) * TRAINVAL_FRACTION)])
    return [c for c in train if c.key not in held], [c for c in train if c.key in held]


def summarise(rows: list[Row]) -> dict[str, object]:
    counts = Counter(row.label for row in rows)
    return {
        "rows": len(rows),
        "labels": {label: counts.get(label, 0) for label in LABELS},
        "prior": {label: counts.get(label, 0) / len(rows) for label in LABELS},
        "gold_unresolved": sum(1 for row in rows if not row.gold_resolved),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DEST))
    args = parser.parse_args()

    claims = load_all()
    kept = [c for c in claims if c.label in LABELS]
    dropped = len(claims) - len(kept)
    print(f"{len(claims):,} claims, dropped {dropped:,} MIXED, {len(kept):,} remain")

    train_all, calibration, test = split_averitec(kept)
    train, trainval = carve_trainval(train_all)

    by_split = {
        "train": [to_row(c) for c in train],
        "trainval": [to_row(c) for c in trainval],
        "calibration": [to_row(c) for c in calibration],
        "test": [to_row(c) for c in test],
    }

    keys = {name: {c.key for c in part} for name, part in
            (("train", train), ("trainval", trainval), ("calibration", calibration), ("test", test))}
    for a in keys:
        for b in keys:
            if a < b and keys[a] & keys[b]:
                raise SystemExit(f"{a} and {b} share {len(keys[a] & keys[b])} claim keys")

    for name, rows in by_split.items():
        if not rows:
            raise SystemExit(f"{name} is empty")
        if any(not row.evidence for row in rows):
            raise SystemExit(f"{name} has rows with no evidence")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "template_id": TEMPLATE_ID,
        "labels": list(LABELS),
        "fever_to_label": {},          # not FEVER; the loader maps AVeriTeC's own strings
        "source": "averitec",
        "mixed_dropped": dropped,
        "splits": {},
        "files": {},
    }
    for name, rows in by_split.items():
        filename = f"verdict_{name}.jsonl.gz"
        digest = write_rows(rows, dest / filename)
        manifest["splits"][name] = summarise(rows)
        manifest["files"][filename] = {"sha256": digest, "bytes": (dest / filename).stat().st_size}
        print(f"  wrote {filename}  {(dest / filename).stat().st_size / 1e6:.2f} MB")

    manifest["train_prior"] = manifest["splits"]["train"]["prior"]
    spec = Path("notebooks/encode_spec.py")
    (dest / spec.name).write_bytes(spec.read_bytes())
    from src.verdict.dataset import sha256

    manifest["files"][spec.name] = {"sha256": sha256(dest / spec.name)}
    (dest / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'split':<12} {'rows':>7}  " + "  ".join(f"{label[:9]:>10}" for label in LABELS))
    for name, rows in by_split.items():
        counts = manifest["splits"][name]["labels"]
        print(f"{name:<12} {len(rows):>7,}  " + "  ".join(f"{counts[label]:>10,}" for label in LABELS))
    prior = manifest["train_prior"]
    print("\ntrain prior  " + "  ".join(f"{label[:9]} {prior[label]:.3f}" for label in LABELS))
    print(f"written to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
