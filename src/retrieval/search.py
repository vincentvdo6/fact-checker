"""
Two-stage retrieval: pages first, then sentences within them.

Searching 25M sentences directly would mean 4.7x the documents whose lengths average
twenty tokens, where BM25's length normalization is mostly noise. It would also discard the
strongest signal FEVER offers: claims are written by mutating a sentence from a named page, so
the page title carries much of the answer, and a title is a page-level feature.

Splitting the stages is a measurement as much as an optimization. doc_recall@N is a hard
ceiling on recall@k, so when a claim is missed the two numbers say whether the page was never
found or the right sentence was never picked out of it. Those need opposite fixes.

The vocabulary stays in SQLite rather than a dict. A claim is about eight tokens, so a query
costs a handful of indexed lookups, against several hundred MB resident for the full term
table -- and every process that touches the index would pay that.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.retrieval import wiki
from src.retrieval.bm25 import BM25Index
from src.retrieval.text import title_norm, tokenize

INDEX_DIR = Path("data/fever/index")

# Longest title, in tokens, the injector will look for inside a claim. Gold titles run p95 5
# and p99 7, and 99.76% fit under this, so going higher costs lookups and finds nothing.
MAX_TITLE_TOKENS = 8

# How many injected pages may take slots from BM25. Title matching finds the right page far
# more often than BM25 does, but its candidates carry no relevance order beyond match length,
# and a one-token match hits pages like "House" or "Shooter" -- a median of 13 per claim and
# up to 43. Uncapped it fills the budget and evicts BM25 outright. doc_recall@10 over 600
# verifiable train claims: off 0.558, cap 2 0.820, cap 3 0.835, cap 5 0.830, cap 12 0.763,
# uncapped 0.763 -- uncapped collapses to cap 12 because the budget is full either way.
DEFAULT_MAX_TITLES = 3

Ref = tuple[str, int]


class SqliteVocab(Mapping[str, int]):
    """The terms table as a read-only mapping, so no process holds the vocabulary in memory."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._size = conn.execute("SELECT count(*) FROM terms").fetchone()[0]

    def __getitem__(self, term: str) -> int:
        row = self._conn.execute("SELECT term_id FROM terms WHERE term = ?", (term,)).fetchone()
        if row is None:
            raise KeyError(term)
        return int(row[0])

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[str]:
        for (term,) in self._conn.execute("SELECT term FROM terms"):
            yield term


def load_index(conn: sqlite3.Connection, directory: str | Path = INDEX_DIR) -> BM25Index:
    return BM25Index.load(directory, SqliteVocab(conn))


@dataclass(frozen=True, slots=True)
class Result:
    """Both stages' output. `pages` scores doc_recall@N, `refs` scores recall@k."""

    pages: tuple[str, ...]
    refs: tuple[Ref, ...]
    scores: tuple[float, ...]


def title_candidates(conn: sqlite3.Connection, claim: str) -> list[int]:
    """
    Pages whose whole title appears verbatim in the claim, longest match first.

    Some gold title appears as a contiguous token run in 83.5% of verifiable dev claims, so
    this finds most of what stage 1 needs and never all of it. Longest match first, the only
    relevance signal available here -- a four-token match is far likelier to be the subject
    than a one-token one, and the cap in retrieve_pages relies on that ordering.
    """
    tokens = tokenize(claim)
    seen: dict[int, None] = {}
    for size in range(min(MAX_TITLE_TOKENS, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            for doc_id in wiki.find_by_norm(conn, " ".join(tokens[start : start + size])):
                seen.setdefault(doc_id, None)
    return list(seen)


def retrieve_pages(
    conn: sqlite3.Connection,
    index: BM25Index,
    claim: str,
    *,
    n: int = 10,
    inject_titles: bool = True,
    max_titles: int = DEFAULT_MAX_TITLES,
) -> list[int]:
    """
    Candidate pages, title matches first, then BM25, truncated to n.

    The cap is what keeps this a merge. Injected candidates go ahead of BM25 because a verbatim
    title match beats a lexical score, but capping them leaves at least n - max_titles slots
    for BM25 rather than letting a long candidate list take the budget whole.
    """
    ranked = [doc_id for doc_id, _ in index.search(claim, k=n)]
    if not inject_titles or max_titles <= 0:
        return ranked[:n]

    merged: dict[int, None] = dict.fromkeys(title_candidates(conn, claim)[:max_titles])
    for doc_id in ranked:
        merged.setdefault(doc_id, None)
    return list(merged)[:n]


def retrieve_sentences(
    conn: sqlite3.Connection,
    index: BM25Index,
    claim: str,
    pages: Sequence[int],
    *,
    k: int = 25,
    prepend_title: bool = True,
) -> tuple[list[Ref], list[float]]:
    """
    Rank every sentence of the candidate pages against the claim.

    Titles are prepended because FEVER sentences lean on the page for their subject -- "He was
    born in ..." -- which is invisible within a page but decides the ordering once sentences
    from several pages are merged into one list.
    """
    refs: list[Ref] = []
    texts: list[str] = []
    order: list[tuple[int, int]] = []
    for rank, doc_id in enumerate(pages):
        title = wiki.titles(conn, [doc_id])[doc_id]
        prefix = f"{title_norm(title)} " if prepend_title else ""
        for sentence_index, sentence in wiki.sentences(conn, doc_id):
            refs.append((title, sentence_index))
            texts.append(prefix + sentence)
            order.append((rank, sentence_index))

    if not texts:
        return [], []

    if index.sentence_avgdl <= 0.0:
        # Falling back to the page norm here would silently swap 17.9 for 86.2 and reorder
        # every sentence -- the mistake score_texts exists to prevent.
        raise ValueError("index has no sentence_avgdl; rebuild it with scripts/build_index.py")
    scores = index.score_texts(claim, texts, avgdl=index.sentence_avgdl)
    # Ties resolve by the page's own rank then sentence order, so a rerun cannot reshuffle them.
    ranking = np.lexsort(
        (
            np.fromiter((s for _, s in order), dtype=np.int64, count=len(order)),
            np.fromiter((p for p, _ in order), dtype=np.int64, count=len(order)),
            -scores,
        )
    )

    chosen: dict[Ref, float] = {}
    for row in ranking:
        if len(chosen) == k:
            break
        chosen.setdefault(refs[row], float(scores[row]))
    return list(chosen), list(chosen.values())


def retrieve(
    conn: sqlite3.Connection,
    index: BM25Index,
    claim: str,
    *,
    n: int = 10,
    k: int = 25,
    inject_titles: bool = True,
    prepend_title: bool = True,
    max_titles: int = DEFAULT_MAX_TITLES,
) -> Result:
    """Both stages for one claim: n candidate pages, then the k best sentences within them."""
    pages = retrieve_pages(
        conn, index, claim, n=n, inject_titles=inject_titles, max_titles=max_titles
    )
    refs, scores = retrieve_sentences(conn, index, claim, pages, k=k, prepend_title=prepend_title)
    titles = wiki.titles(conn, pages)
    return Result(
        pages=tuple(titles[doc_id] for doc_id in pages),
        refs=tuple(refs),
        scores=tuple(scores),
    )
