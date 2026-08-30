"""
Build data/fever/wiki.sqlite3 from the extracted wiki-pages shards.

Roughly 20 minutes and 4.2 GB for the full 109 shards. Indexes are created after the load
rather than before it: maintaining a unique b-tree over 5.4M unsorted titles while inserting
costs far more than sorting once at the end.

The run ends with a verification report, and that report is the point of the script. The
figure that matters is gold title resolution: every one of the 14,533 titles FEVER's gold
evidence names must find a page. Anything short of that is a normalization bug, and it does
not announce itself later -- it arrives as a few points of missing recall that look like
ordinary retrieval difficulty and get blamed on the retriever or the model. So a full build
that resolves less than everything exits non-zero and says so.

Titles are printed through ascii(). The Windows console is cp1252 and raises on a combining
accent, which would kill the report on exactly the non-ASCII titles worth showing; ascii()
also makes an NFD title visibly different from its NFC twin, which two identical-looking
glyphs would not.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.data.fever import load_claims
from src.retrieval.text import title_key, title_norm
from src.retrieval.wiki import (
    DB_PATH,
    INSERT,
    analyze,
    connect,
    create_indexes,
    create_tables,
    doc_id,
    format_sentences,
    page_count,
    scan_lines,
    sentences,
)

SHARD_DIR = Path("data/fever/wiki-pages")
CLAIM_FILES = (Path("data/fever/train.jsonl"), Path("data/fever/shared_task_dev.jsonl"))

EXPECTED_SHARDS = 109
EXPECTED_GOLD_TITLES = 14_533
BATCH = 20_000


@dataclass(slots=True)
class Stats:
    shards: int = 0
    pages: int = 0
    empty_id: int = 0
    duplicate_titles: int = 0
    malformed_rows: int = 0
    blank_pages: int = 0
    sentences: int = 0
    tokens: int = 0
    load_seconds: float = 0.0
    index_seconds: float = 0.0


def build(shards: Sequence[Path], db_path: Path = DB_PATH) -> Stats:
    """
    Load the shards into a fresh store at db_path, replacing any existing one.

    Rebuilt rather than appended to because doc_id has to stay dense from zero: a second pass
    over the same shards would either duplicate every title or leave gaps behind.
    """
    for suffix in ("", "-journal", "-wal", "-shm"):
        stale = db_path.with_name(db_path.name + suffix)
        if stale.exists():
            stale.unlink()

    stats = Stats(shards=len(shards))
    # Keep-first dedup, held in Python because the unique index does not exist yet. Around
    # 5.4M short strings, on the order of 600 MB, for the length of the build.
    seen: set[str] = set()
    batch: list[tuple[int, str, str, str]] = []

    conn = connect(db_path)
    try:
        create_tables(conn)
        started = time.perf_counter()
        for shard in shards:
            shard_started = time.perf_counter()
            shard_pages = 0
            with open(shard, encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    page_id = record["id"]
                    # The first record of shard 001 has an empty id, so this is not defensive.
                    if not page_id:
                        stats.empty_id += 1
                        continue

                    title = title_key(page_id)
                    if title in seen:
                        stats.duplicate_titles += 1
                        continue
                    seen.add(title)

                    rows, malformed = scan_lines(record["lines"])
                    stats.malformed_rows += malformed
                    stats.sentences += len(rows)
                    stats.blank_pages += not rows
                    for _, sentence in rows:
                        stats.tokens += len(sentence.split())

                    batch.append((stats.pages, title, title_norm(title), format_sentences(rows)))
                    stats.pages += 1
                    shard_pages += 1
                    if len(batch) >= BATCH:
                        conn.executemany(INSERT, batch)
                        batch.clear()

            if batch:
                conn.executemany(INSERT, batch)
                batch.clear()
            conn.commit()
            print(f"  {shard.name}  {shard_pages:>7,} pages  {time.perf_counter() - shard_started:6.1f}s")

        stats.load_seconds = time.perf_counter() - started

        started = time.perf_counter()
        create_indexes(conn)
        analyze(conn)
        conn.commit()
        stats.index_seconds = time.perf_counter() - started
    finally:
        conn.close()

    return stats


def gold(paths: Iterable[Path]) -> tuple[set[str], set[tuple[str, int]]]:
    """
    Distinct gold titles and (title, sentence index) references, under title_key.

    Normalizing here and in the store is the only reason the 170 gold titles whose NFD form
    differs from the dump's NFC form resolve at all.
    """
    titles: set[str] = set()
    pairs: set[tuple[str, int]] = set()
    for path in paths:
        for claim in load_claims(path):
            titles.update(title_key(page) for page in claim.pages)
            for group in claim.groups:
                pairs.update((title_key(page), index) for page, index in group)
    return titles, pairs


def resolve(
    conn: sqlite3.Connection,
    titles: set[str],
    pairs: set[tuple[str, int]],
) -> tuple[dict[str, int], set[tuple[str, int]]]:
    """Gold titles that found a page, and gold references that found a non-empty sentence."""
    found = {title: did for title in sorted(titles) if (did := doc_id(conn, title)) is not None}

    indices: dict[str, set[int]] = {}
    resolved: set[tuple[str, int]] = set()
    for title, index in sorted(pairs):
        did = found.get(title)
        if did is None:
            continue
        if title not in indices:
            indices[title] = {i for i, _ in sentences(conn, did)}
        if index in indices[title]:
            resolved.add((title, index))
    return found, resolved


def examples(titles: Iterable[str], limit: int = 5) -> str:
    return ", ".join(ascii(title) for title in sorted(titles)[:limit])


def report(conn: sqlite3.Connection, stats: Stats, *, partial: bool) -> bool:
    """Print the verification report. False means a full build failed to resolve gold evidence."""
    stored = page_count(conn)
    print("\ningest")
    print(f"  shards read            {stats.shards:>12,}")
    print(f"  pages ingested         {stats.pages:>12,}")
    print(f"  rows in store          {stored:>12,}")
    print(f"  skipped, empty id      {stats.empty_id:>12,}")
    print(f"  skipped, duplicate     {stats.duplicate_titles:>12,}   first occurrence kept")
    print(f"  skipped, bad index     {stats.malformed_rows:>12,}   lines rows, page kept")
    print(f"  load                   {stats.load_seconds:>12.1f}s")
    print(f"  index and analyze      {stats.index_seconds:>12.1f}s")

    print("\ncorpus")
    print(f"  sentences              {stats.sentences:>12,}")
    print(f"  pages with none        {stats.blank_pages:>12,}")
    if stats.pages:
        print(f"  sentences per page     {stats.sentences / stats.pages:>12.2f}")
    if stats.sentences:
        print(f"  tokens per sentence    {stats.tokens / stats.sentences:>12.2f}")

    if stored != stats.pages:
        print(f"\n  FAIL  store holds {stored:,} rows against {stats.pages:,} pages ingested")
        return False

    absent = [path for path in CLAIM_FILES if not path.exists()]
    if absent:
        print("\ngold evidence")
        print(f"  skipped, missing {', '.join(path.name for path in absent)}")
        print("  run python -m scripts.fetch_fever, then build again for the report that matters")
        return True

    titles, pairs = gold(CLAIM_FILES)
    found, resolved = resolve(conn, titles, pairs)

    print("\ngold evidence")
    if len(titles) != EXPECTED_GOLD_TITLES:
        print(f"  note: {len(titles):,} distinct gold titles, expected {EXPECTED_GOLD_TITLES:,}")
    print(f"  titles resolved        {len(found):>12,} / {len(titles):,}   {len(found) / len(titles):.2%}")
    print(f"  references resolved    {len(resolved):>12,} / {len(pairs):,}   {len(resolved) / len(pairs):.2%}")

    if partial:
        print(f"\n  partial build: {stats.shards} of {EXPECTED_SHARDS} shards, so both figures fall short by")
        print("  construction. Only a full build verifies normalization.")
        return True

    if len(found) == len(titles) and len(resolved) == len(pairs):
        print("\n  ok: every gold title and every gold sentence reference resolves.")
        return True

    print("\n  FAIL: normalization is broken. Fix it before building anything on this store.")
    print("  A shortfall here becomes a silent recall loss that gets blamed on the model.")
    if len(found) < len(titles):
        print(f"  unresolved titles: {examples(titles - found.keys())}")
    if len(resolved) < len(pairs):
        stranded = {title for title, _ in pairs - resolved if title in found}
        if stranded:
            print(f"  resolved but missing the cited sentence: {examples(stranded)}")
    return False


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FEVER wiki store from the extracted shards.")
    parser.add_argument("--shards", type=Path, default=SHARD_DIR, help="directory of wiki-NNN.jsonl shards")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="store to write")
    parser.add_argument("--limit-shards", type=int, metavar="N", help="read only the first N shards, for a smoke test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # A twenty-minute build is usually redirected to a log, where the default block buffering
    # hides progress until it finishes. cp1252 is the console default and raises on the
    # combining accents in gold titles, so the report's own encoding is fixed here too.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = parse_args(argv)

    available = sorted(args.shards.glob("wiki-*.jsonl"))
    if not available:
        raise SystemExit(f"no shards under {args.shards}; run python -m scripts.fetch_wiki")
    shards = available if args.limit_shards is None else available[: args.limit_shards]

    print(f"building {args.db} from {len(shards)} of {len(available)} shards")
    stats = build(shards, args.db)

    conn = connect(args.db)
    try:
        ok = report(conn, stats, partial=len(shards) < EXPECTED_SHARDS)
    finally:
        conn.close()

    print(f"\n{args.db}  {args.db.stat().st_size / 1e9:.2f} GB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
