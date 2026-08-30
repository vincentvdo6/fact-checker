"""
The FEVER Wikipedia corpus as a queryable store.

The dump is 109 JSONL shards, 7.6 GB, ~5.4M pages, and every stage downstream needs random
access into it: BM25 over the sentences, a title lookup to resolve gold evidence, a sentence
fetch for whatever the retriever returns. One SQLite file covers all three, needs no server,
and is reproducible from the shards with a single command.

`doc_id` is dense from zero because it doubles as the column index of the term-document
matrix. A gap would silently misalign an index that was already built, so a page is either
ingested or its id is never allocated -- nothing renumbers afterwards.

`sentences` holds the dump's own `lines` format with the noise removed: "index\\tsentence"
records joined by newlines, hyperlink columns and empty sentences dropped. Not JSON, because
the indices are the identifiers gold evidence cites and have to survive a gap -- a page can be
missing sentence 3 -- and this round-trips through parse_lines with no per-page decode cost
across 5.4M rows.

Only `lines` is read. A record's `text` field is the same sentences concatenated, so it adds
nothing the indices don't already carry and would double the ingest.

Titles are stored under title_key(); src/retrieval/text.py has the reason that matters, which
is that the dump and the gold evidence disagree about Unicode form. `norm` is the loose form
title_norm() produces, kept as its own indexed column because matching a claim's text against
a page title is a retrieval strategy in its own right.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from src.retrieval.text import title_key, title_norm

DB_PATH = Path("data/fever/wiki.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    doc_id    INTEGER PRIMARY KEY,   -- also the BM25 column index, so dense from 0
    title     TEXT NOT NULL,         -- title_key() applied; the exact join key
    norm      TEXT NOT NULL,         -- title_norm(); for title-vs-claim matching
    sentences TEXT NOT NULL          -- "idx\\tsentence\\n..." with empties dropped
);
"""

INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS pages_title ON pages(title)",
    "CREATE INDEX IF NOT EXISTS pages_norm ON pages(norm)",
)

INSERT = "INSERT INTO pages (doc_id, title, norm, sentences) VALUES (?, ?, ?, ?)"

# SQLite's compiled-in ceiling on host parameters is 999 on builds before 3.32.
_PARAM_CHUNK = 900


def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    """
    Open the store, creating an empty file if it is absent.

    An empty file is a valid outcome: create_tables() is what puts the schema in, and the
    build script owns that. Querying a store that was never built raises "no such table".
    """
    conn = sqlite3.connect(path)
    # The store is derived data, rebuilt from the shards in one command, so durability is not
    # worth an fsync per transaction across a 5.4M-row load.
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = -65536")  # negative is KiB, so 64 MB
    # Default is 0, which means a single reader opening the store mid-rebuild fails the
    # build's CREATE INDEX outright rather than waiting for it.
    conn.execute("PRAGMA busy_timeout = 30000")  # ms
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def create_indexes(conn: sqlite3.Connection) -> None:
    """Build the indexes. Called after the load: see scripts/build_wiki.py for why."""
    for statement in INDEXES:
        conn.execute(statement)


def analyze(conn: sqlite3.Connection) -> None:
    conn.execute("ANALYZE")


def scan_lines(lines: str) -> tuple[list[tuple[int, str]], int]:
    """
    Parsed rows, plus the count of records dropped for a non-integer index.

    Roughly two records per 19,000 pages, about 500 corpus-wide, are a stray annotation row:
    the hyperlink columns of a sentence that itself contained a newline, arriving as their own
    record with a page title where the index belongs. The record is dropped and the rest of the
    page kept; aborting the page would lose it whole.

    The index is read from the record, never inferred from position. Indices were contiguous
    everywhere sampled, but enumerate() agrees with the dump right up until a dropped record
    shifts every sentence after it, and gold evidence cites these numbers.
    """
    rows: list[tuple[int, str]] = []
    malformed = 0
    if not lines:
        return rows, malformed

    for record in lines.split("\n"):
        if not record:
            continue
        fields = record.split("\t")
        head = fields[0]
        # isdigit over int(): int("1_0") is 10, and a stray field can look like that.
        if not (head.isascii() and head.isdigit()):
            malformed += 1
            continue
        # fields[2:] are (anchor, target) hyperlink pairs and are not part of the sentence.
        sentence = fields[1] if len(fields) > 1 else ""
        if sentence.strip():
            rows.append((int(head), sentence))
    return rows, malformed


def parse_lines(lines: str) -> list[tuple[int, str]]:
    """(index, sentence) for every non-empty sentence, index as written in the dump."""
    return scan_lines(lines)[0]


def format_sentences(rows: Iterable[tuple[int, str]]) -> str:
    """The stored form of a page's sentences; inverse of parse_lines."""
    return "\n".join(f"{index}\t{sentence}" for index, sentence in rows)


def page_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT count(*) FROM pages").fetchone()[0]


def iter_pages(conn: sqlite3.Connection) -> Iterator[tuple[int, str, str]]:
    """
    Every page as (doc_id, title, sentences), in doc_id order, streamed.

    doc_id is the rowid, so this is a table scan with no sort and no materialized result set.
    The inverted index build walks the whole corpus twice and cannot hold it in memory.
    """
    cursor = conn.execute("SELECT doc_id, title, sentences FROM pages ORDER BY doc_id")
    try:
        yield from cursor
    finally:
        cursor.close()


def doc_id(conn: sqlite3.Connection, title: str) -> int | None:
    """doc_id for a page title, or None if the dump has no such page. Applies title_key."""
    row = conn.execute("SELECT doc_id FROM pages WHERE title = ?", (title_key(title),)).fetchone()
    return None if row is None else row[0]


def sentences(conn: sqlite3.Connection, doc_id: int) -> list[tuple[int, str]]:
    """Sentences of a page. Empty for an unknown doc_id and for the pages the dump left blank."""
    row = conn.execute("SELECT sentences FROM pages WHERE doc_id = ?", (doc_id,)).fetchone()
    return [] if row is None else parse_lines(row[0])


def titles(conn: sqlite3.Connection, doc_ids: Iterable[int]) -> dict[int, str]:
    """Stored titles for the given ids. An id with no page is absent from the result."""
    wanted = list(dict.fromkeys(doc_ids))
    found: dict[int, str] = {}
    for start in range(0, len(wanted), _PARAM_CHUNK):
        chunk = wanted[start : start + _PARAM_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        query = f"SELECT doc_id, title FROM pages WHERE doc_id IN ({placeholders})"
        found.update(conn.execute(query, chunk))
    return found


def find_by_norm(conn: sqlite3.Connection, norm: str) -> list[int]:
    """
    doc_ids whose loose title form matches exactly, in doc_id order.

    title_norm is idempotent over its own output, so an already-normalized argument and a raw
    title both work.
    """
    rows = conn.execute("SELECT doc_id FROM pages WHERE norm = ? ORDER BY doc_id", (title_norm(norm),))
    return [row[0] for row in rows]
