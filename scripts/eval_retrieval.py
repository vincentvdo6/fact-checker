"""
Score the retriever on a split and write the run to runs/<name>/.

Sentences are stored at k=25 so later phases can sweep smaller cutoffs, and Phase 02 can read
retrieved.jsonl as its evidence without paying for retrieval again.

Tuning happens on a train sample. Choosing n, k, k1 or b by looking at calibration or test would
leak into the split that the whole calibration story rests on, and it would leak invisibly --
nothing downstream can tell that a threshold was picked with the answers in view.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.data.fever import Claim, load_claims
from src.data.splits import drop_leaked, split_dev
from src.eval.retrieval import evaluate_retrieval
from src.retrieval import wiki
from src.retrieval.search import DEFAULT_MAX_TITLES, INDEX_DIR, load_index, retrieve

TRAIN = "data/fever/train.jsonl"
DEV = "data/fever/shared_task_dev.jsonl"
RUNS = Path("runs")

STORED_K = 25
REPORT_KS = (1, 5, 10, 25)


def load_split(name: str) -> list[Claim]:
    calibration, test = split_dev(load_claims(DEV))
    if name == "calibration":
        return calibration
    if name == "test":
        return test
    return drop_leaked(load_claims(TRAIN), calibration, test)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        sha = out.stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        # A clean sha on a dirty tree names code that did not produce the run.
        return f"{sha}-dirty" if dirty.stdout.strip() else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "calibration", "test"), default="train")
    parser.add_argument("--limit", type=int, default=2000, help="claims to score; 0 for all")
    parser.add_argument("--pages", type=int, default=10, help="stage-1 pages per claim")
    parser.add_argument("--name", default=None, help="run directory under runs/")
    parser.add_argument("--no-title-injection", action="store_true")
    parser.add_argument("--no-title-prefix", action="store_true")
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    args = parser.parse_args()

    claims = load_split(args.split)
    if args.limit:
        claims = claims[: args.limit]

    conn = wiki.connect()
    index = load_index(conn, args.index_dir)

    retrieved, pages, scores = [], [], []
    start = time.time()
    for done, claim in enumerate(claims, start=1):
        result = retrieve(
            conn,
            index,
            claim.text,
            n=args.pages,
            k=STORED_K,
            inject_titles=not args.no_title_injection,
            prepend_title=not args.no_title_prefix,
        )
        retrieved.append(result.refs)
        pages.append(result.pages)
        scores.append(result.scores)
        if done % 250 == 0:
            rate = done / (time.time() - start)
            print(f"  {done:,}/{len(claims):,}  {rate:.1f} claims/s", flush=True)
    elapsed = time.time() - start

    # A page cutoff above the number retrieved is not a measurement -- it just repeats the
    # deepest real one, which reads as a plateau that was never observed.
    doc_ns = tuple(n for n in REPORT_KS if n <= args.pages) or (args.pages,)
    report = evaluate_retrieval(
        claims, retrieved, ks=REPORT_KS, pages_retrieved=pages, ns=doc_ns
    )

    name = args.name or f"bm25-{args.split}"
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)

    (out / "config.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "claims": len(claims),
                "pages_per_claim": args.pages,
                "stored_k": STORED_K,
                "title_injection": not args.no_title_injection,
                "max_titles": DEFAULT_MAX_TITLES,
                "title_prefix": not args.no_title_prefix,
                "k1": index.k1,
                "b": index.b,
                "corpus_pages": wiki.page_count(conn),
                "index_terms": index.n_terms,
                "index_postings": index.n_postings,
                "git_sha": git_sha(),
                "seconds": round(elapsed, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "metrics.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    with open(out / "retrieved.jsonl", "w", encoding="utf-8") as handle:
        for claim, refs, claim_scores in zip(claims, retrieved, scores, strict=True):
            row = {
                "id": claim.id,
                "evidence": [[title, idx] for title, idx in refs],
                "scores": [round(float(score), 4) for score in claim_scores],
            }
            handle.write(json.dumps(row) + "\n")

    print(f"\n{args.split}: {len(claims):,} claims in {elapsed / 60:.1f} min")
    print(f"  verifiable {report.verifiable:,}   nei {report.nei:,} ({report.nei_share:.1%})")
    print(f"\n{'':>4} {'doc_rec':>8} {'recall':>8} {'r_any':>8} {'mrr':>8} {'ndcg':>8} {'ceil_all':>9}")
    for k in REPORT_KS:
        cut = report.at(k)
        doc = f"{report.doc_at(k).doc_recall:.3f}" if k in doc_ns else "-"
        print(
            f"@{k:<3} {doc:>8} {cut.recall:>8.3f} {cut.recall_any:>8.3f} "
            f"{cut.mrr:>8.3f} {cut.ndcg_any:>8.3f} {cut.ceiling.overall:>9.3f}"
        )
    print(f"\nceiling_overall assumes an oracle answers all {report.nei_share:.1%} NEI for free")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
