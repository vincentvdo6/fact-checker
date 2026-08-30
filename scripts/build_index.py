"""
Build the BM25 index over the wiki corpus.

Two full tokenization passes over 5.4M pages, so this is a long job and a checkpoint rather
than something to run casually. Run it first with --limit-docs on a few thousand pages: the
numbers it prints are what the full build gets extrapolated from, and a tf-clip count above
a large count means the tokenizer is broken. A trickle is normal: 330 postings clipped on
the full corpus, across the ~191 pages long enough to use one term 255 times against a corpus
mean of 86. About a quarter are stopwords; most of the rest is wiki table markup (align,
center, style) and genus names on taxonomic list pages. Clipping there is harmless because
BM25 saturates tf well below it.

A page's indexed text is its title plus every one of its sentences, with the title counted
once. Titles are underscore-joined page ids, which is why tokenize splits on underscores --
without that, "Soul_Food_-LRB-film-RRB-" contributes one useless term instead of two useful
ones.

The arrays land in data/fever/index/. The vocabulary goes into the corpus SQLite instead, as a
terms table: 3.4M terms is several hundred MB of Python strings held whole, and a query needs
eight of them. df is stored there beside each term even though it is recoverable from indptr,
so the table answers "how common is this word" without opening the index at all.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.retrieval import wiki
from src.retrieval.bm25 import MAX_TF, BuildStats, build_index
from src.retrieval.text import tokenize

DEFAULT_DB = Path("data/fever/wiki.sqlite3")
DEFAULT_INDEX_DIR = Path("data/fever/index")

TERMS_SCHEMA = "CREATE TABLE terms (term TEXT PRIMARY KEY, term_id INTEGER NOT NULL, df INTEGER NOT NULL)"

MISSING_DB = """\
{path} not found.

The BM25 index is built from the corpus database, not from the raw dump. Build it first:

    python -m scripts.build_wiki
"""


class PageStream:
    """
    Replayable source of tokenized pages, and the sentence-length counters taken along the way.

    build_index calls this once per pass, so the sentence counters are reset at the top of every
    pass rather than accumulated across both, where they would simply double. Two running counters
    is the whole cost of knowing the sentence-level avgdl, and sentence selection cannot use the
    page-level one: sentences are an order of magnitude shorter, so the page norm would rate every
    sentence as short and the length term would stop discriminating.
    """

    def __init__(self, conn: sqlite3.Connection, limit: int | None = None) -> None:
        self.conn = conn
        self.limit = limit
        self.sentences = 0
        self.sentence_tokens = 0

    def __call__(self) -> Iterator[tuple[int, list[str]]]:
        self.sentences = 0
        self.sentence_tokens = 0
        for seen, (doc_id, title, lines) in enumerate(wiki.iter_pages(self.conn)):
            if self.limit is not None and seen >= self.limit:
                break
            tokens = tokenize(title)
            for _, sentence in wiki.parse_lines(lines):
                sentence_tokens = tokenize(sentence)
                self.sentences += 1
                self.sentence_tokens += len(sentence_tokens)
                tokens.extend(sentence_tokens)
            yield doc_id, tokens

    def sentence_avgdl(self) -> float:
        return self.sentence_tokens / self.sentences if self.sentences else 0.0


def write_terms(conn: sqlite3.Connection, vocab: dict[str, int], indptr: np.ndarray) -> None:
    """Replace the terms table with this build's vocabulary. df is the column length in indptr."""
    conn.execute("DROP TABLE IF EXISTS terms")
    conn.execute(TERMS_SCHEMA)
    conn.executemany(
        "INSERT INTO terms (term, term_id, df) VALUES (?, ?, ?)",
        # A generator, not a list: 3.4M tuples materialized would cost more than the index arrays.
        ((term, term_id, int(indptr[term_id + 1] - indptr[term_id])) for term, term_id in vocab.items()),
    )
    conn.commit()
    written = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
    if written != len(vocab):
        raise SystemExit(f"terms table holds {written} rows for a vocabulary of {len(vocab)}")


def peak_rss_bytes() -> int | None:
    """
    Peak working set, or None where the platform will not say.

    On Windows this counts memory-mapped pages that are still resident, so the streamed posting
    arrays inflate it. Read it as an upper bound on what the build actually needed.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)

    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024   # ru_maxrss is KiB on Linux


def report(
    stats: BuildStats,
    *,
    index_dir: Path,
    sentence_avgdl: float,
    avgdl: float,
    elapsed: float,
    corpus_pages: int,
) -> None:
    """Print the checkpoint numbers, then extrapolate them if this was a limited build."""
    files = sorted(path for path in index_dir.iterdir() if path.is_file())
    on_disk = sum(path.stat().st_size for path in files)
    rss = peak_rss_bytes()

    print()
    print(f"pages           {stats.docs:,} of {corpus_pages:,}")
    print(f"tokens          {stats.tokens:,}  ({stats.tokens_per_doc:.1f} per page, avgdl {avgdl:.2f})")
    print(f"sentence avgdl  {sentence_avgdl:.2f}")
    print(f"vocabulary      {stats.terms:,} terms")
    print(f"postings        {stats.postings:,}  ({stats.postings_per_doc:.1f} per page)")

    clipped = f"tf clipped      {stats.tf_clipped:,} postings at {MAX_TF}"
    # ~330 is the measured full-corpus figure; orders of magnitude more means the tokenizer.
    print(clipped if stats.tf_clipped < 5_000 else f"{clipped}   <-- far above the expected trickle")

    print(f"wall time       {elapsed:.1f} s")
    print(f"peak RSS        {'unavailable' if rss is None else f'{rss / 1e9:.2f} GB'}")
    print(f"on disk         {on_disk / 1e6:.1f} MB")
    for path in files:
        print(f"  {path.name:<12} {path.stat().st_size / 1e6:>10.1f} MB")

    if stats.docs >= corpus_pages:
        return

    factor = corpus_pages / stats.docs
    print(f"\nextrapolated to the full {corpus_pages:,} pages (x{factor:.1f})")
    print(f"  postings      {stats.postings * factor:,.0f}")
    print(f"  on disk       {on_disk * factor / 1e9:.2f} GB")
    print(f"  wall time     {elapsed * factor / 60:.1f} min")
    print("  vocabulary and peak RSS do not scale linearly: vocabulary growth is sublinear")
    print("  (Heaps' law) and the posting arrays are streamed to disk rather than held.")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="corpus database")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="where the arrays go")
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        metavar="N",
        help="index only the first N pages, for a smoke build",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.db.exists():
        print(MISSING_DB.format(path=args.db), file=sys.stderr)
        return 1
    conn = wiki.connect(args.db)
    corpus_pages = wiki.page_count(conn)
    pages = PageStream(conn, args.limit_docs)

    started = time.perf_counter()
    index, stats = build_index(pages, directory=args.index_dir, sentence_avgdl=pages.sentence_avgdl)
    write_terms(conn, dict(index.vocab), index.indptr)
    elapsed = time.perf_counter() - started
    conn.close()

    report(
        stats,
        index_dir=args.index_dir,
        sentence_avgdl=index.sentence_avgdl,
        avgdl=index.avgdl,
        elapsed=elapsed,
        corpus_pages=corpus_pages,
    )
    if args.limit_docs is not None:
        print(f"\nsmoke build: the index and the terms table now describe {stats.docs:,} pages, not the corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
