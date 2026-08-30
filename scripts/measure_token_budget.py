"""
How much retrieved evidence actually fits the model's context, and what ceiling that leaves.

Phase 01 reports a retrieval ceiling of strict recall@25. The model never sees 25 sentences:
they average about 20 words, so 25 of them overrun a 512-token sequence before the claim is
added. The ceiling a verdict model can actually reach is therefore set by the context window,
not by what retrieval returned, and this measures the difference rather than assuming it.

Packing is per claim, not a fixed k. A fixed k would have to be chosen for the longest claims
and would waste context on every shorter one. What the sweep decides is `max_length` and the
template, both of which enter the artifact contract and cannot change after the training dataset
is uploaded without invalidating every checkpoint.

Sentences are resolved by dict lookup on the stored index, never by list position. The store
drops empty sentences, so position i is not index i -- resolving positionally returns a
different, entirely plausible sentence, and the model would train on mismatched evidence with
nothing raising.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.data.fever import NOT_ENOUGH_INFO, load_claims
from src.data.splits import split_dev
from src.eval.retrieval import recall_at_k
from src.retrieval import wiki
from src.verdict.encode import pack

DEV = "data/fever/shared_task_dev.jsonl"
RUNS = Path("runs")
OUT = RUNS / "token-budget"

BASE_MODEL = "microsoft/deberta-v3-base"
BUDGETS = (256, 384, 512)
FIXED_KS = (5, 10, 15, 20, 25)


def load_retrieved(path: Path) -> dict[int, list[list]]:
    rows: dict[int, list[list]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["id"]] = [[title, index] for title, index in row["evidence"]]
    return rows


def resolve(conn, refs: list[list]) -> list[list]:
    """Attach sentence text, by index lookup rather than list position."""
    out: list[list] = []
    cache: dict[str, dict[int, str]] = {}
    for title, index in refs:
        if title not in cache:
            doc_id = wiki.doc_id(conn, title)
            cache[title] = dict(wiki.sentences(conn, doc_id)) if doc_id is not None else {}
        text = cache[title].get(index)
        if text:
            out.append([title, index, text])
    return out


def refs_only(rows: list[list]) -> list[tuple[str, int]]:
    """Scoring compares (title, index); the sentence text is for the tokenizer, not the metric."""
    return [(title, index) for title, index, _ in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def measure(first: str, second: str) -> int:
        return len(tokenizer(first, second)["input_ids"])

    calibration, test = split_dev(load_claims(DEV))
    claims = {c.id: c for c in (calibration if args.split == "calibration" else test)}
    retrieved = load_retrieved(RUNS / f"bm25-{args.split}" / "retrieved.jsonl")

    conn = wiki.connect()
    verifiable = [c for c in claims.values() if c.label != NOT_ENOUGH_INFO and c.id in retrieved]
    if args.limit:
        verifiable = verifiable[: args.limit]

    resolved = {c.id: resolve(conn, retrieved[c.id]) for c in verifiable}
    print(f"{args.split}: {len(verifiable):,} verifiable claims with evidence")

    report: dict[str, object] = {"split": args.split, "claims": len(verifiable), "budgets": {}}
    print(f"\n{'budget':>7} {'fits p10':>9} {'median':>7} {'p90':>5} {'packed':>8} {'@25':>7} {'loss':>7}")
    for budget in BUDGETS:
        fitted = {c.id: pack(c.text, resolved[c.id], budget, measure) for c in verifiable}
        counts = sorted(fitted.values())
        packed = sum(
            recall_at_k(refs_only(resolved[c.id]), c.groups, fitted[c.id]) for c in verifiable
        ) / len(verifiable)
        full = sum(recall_at_k(refs_only(resolved[c.id]), c.groups, 25) for c in verifiable) / len(verifiable)
        p10, median, p90 = (counts[int(len(counts) * q)] for q in (0.10, 0.50, 0.90))
        print(f"{budget:>7} {p10:>9} {median:>7} {p90:>5} {packed:>8.3f} {full:>7.3f} {full - packed:>7.3f}")
        report["budgets"][str(budget)] = {
            "fits": {"p10": p10, "median": median, "p90": p90, "max": counts[-1]},
            "share_fitting_k": {str(k): sum(v >= k for v in counts) / len(counts) for k in FIXED_KS},
            "packed_ceiling": packed,
            "recall_at_25": full,
            "truncation_loss": full - packed,
            "histogram": dict(sorted(Counter(counts).items())),
        }

    print(f"\nfixed-k ceilings for comparison, {args.split}")
    for k in FIXED_KS:
        value = sum(recall_at_k(refs_only(resolved[c.id]), c.groups, k) for c in verifiable) / len(verifiable)
        print(f"  @{k:<3} {value:.3f}")
        report.setdefault("fixed_k", {})[str(k)] = value

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT / f'{args.split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
