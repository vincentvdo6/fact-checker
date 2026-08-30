"""
Retrieve evidence for a whole split, resumably.

The train split is 145,446 claims at roughly 300 ms each, so this runs for hours and will be
interrupted -- by a reboot, a killed shell, or a machine that goes to sleep. Everything here
exists to make that survivable.

The output file is the progress file. Rows are appended as they are produced and a resume reads
the ids already present, so there is no sidecar state that can disagree with the data. A kill
during a write leaves a partial final line; that line is truncated before appending, because the
alternative is worse than a crash -- a half-written row can parse as valid JSON with a short
evidence list, and a claim that silently retrieved three sentences instead of twenty-five looks
like a retrieval failure rather than a torn file.

Sharding partitions by `claim_id % shards`, which needs no coordination between processes: each
one knows its own work from its own arguments. The index is memory-mapped, so parallel shards
share one page cache rather than each paying for their own.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.data.fever import Claim, load_claims
from src.data.splits import drop_leaked, split_dev
from src.retrieval import wiki
from src.retrieval.search import INDEX_DIR, load_index, retrieve

TRAIN = "data/fever/train.jsonl"
DEV = "data/fever/shared_task_dev.jsonl"
RUNS = Path("runs")

STORED_K = 25
FLUSH_EVERY = 250


def load_split(name: str) -> list[Claim]:
    calibration, test = split_dev(load_claims(DEV))
    if name == "calibration":
        return calibration
    if name == "test":
        return test
    return drop_leaked(load_claims(TRAIN), calibration, test)


def repair(path: Path) -> int:
    """
    Drop a torn final line. Returns the number of bytes removed.

    Only the last line can be torn -- everything before it was followed by a newline that the
    kernel had already accepted.
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return 0
        # Walk back to the last newline; everything after it is a partial row.
        block = 1 << 16
        position = size
        while position > 0:
            step = min(block, position)
            position -= step
            handle.seek(position)
            chunk = handle.read(step)
            cut = chunk.rfind(b"\n")
            if cut != -1:
                keep = position + cut + 1
                handle.truncate(keep)
                return size - keep
        handle.truncate(0)
        return size


def completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            done.add(json.loads(line)["id"])
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "calibration", "test"), default="train")
    parser.add_argument("--name", default=None, help="run directory under runs/")
    parser.add_argument("--pages", type=int, default=25)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="claims to score; 0 for all")
    parser.add_argument("--merge", action="store_true", help="concatenate shard parts and exit")
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    args = parser.parse_args()

    if not 0 <= args.shard < args.shards:
        raise SystemExit(f"--shard must lie in [0, {args.shards})")

    out = RUNS / (args.name or f"evidence-{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    if args.merge:
        return merge(out, load_split(args.split), args.shards)

    claims = load_split(args.split)
    if args.limit:
        claims = claims[: args.limit]
    mine = [c for c in claims if c.id % args.shards == args.shard]

    part = out / f"retrieved.part{args.shard}.jsonl"
    dropped = repair(part)
    if dropped:
        print(f"repaired {part.name}: dropped {dropped} bytes of a torn final row", flush=True)
    done = completed(part)
    todo = [c for c in mine if c.id not in done]
    print(
        f"shard {args.shard}/{args.shards}: {len(mine):,} claims, "
        f"{len(done):,} done, {len(todo):,} to go",
        flush=True,
    )
    if not todo:
        return 0

    conn = wiki.connect()
    index = load_index(conn, args.index_dir)

    start = time.time()
    with open(part, "a", encoding="utf-8") as handle:
        for position, claim in enumerate(todo, start=1):
            result = retrieve(conn, index, claim.text, n=args.pages, k=STORED_K)
            row = {"id": claim.id, "evidence": [[title, idx] for title, idx in result.refs]}
            handle.write(json.dumps(row) + "\n")
            if position % FLUSH_EVERY == 0:
                handle.flush()
                os.fsync(handle.fileno())
                rate = position / (time.time() - start)
                remaining = (len(todo) - position) / rate / 3600
                print(f"  {position:,}/{len(todo):,}  {rate:.1f}/s  {remaining:.1f}h left", flush=True)
        handle.flush()
        os.fsync(handle.fileno())

    elapsed = time.time() - start
    print(f"shard {args.shard}: {len(todo):,} claims in {elapsed / 60:.1f} min")
    (out / f"config.part{args.shard}.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "shard": args.shard,
                "shards": args.shards,
                "pages_per_claim": args.pages,
                "stored_k": STORED_K,
                "claims": len(mine),
                "seconds": round(elapsed, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def merge(out: Path, claims: list[Claim], shards: int) -> int:
    """Concatenate shard parts in split order, refusing to write a partial or duplicated set."""
    rows: dict[int, list] = {}
    for shard in range(shards):
        part = out / f"retrieved.part{shard}.jsonl"
        if not part.exists():
            raise SystemExit(f"{part} is missing; run --shard {shard} --shards {shards} first")
        with open(part, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] in rows:
                    raise SystemExit(f"claim {row['id']} appears in more than one shard")
                rows[row["id"]] = row["evidence"]

    missing = [c.id for c in claims if c.id not in rows]
    if missing:
        raise SystemExit(f"{len(missing):,} claims have no evidence, first {missing[:5]}")

    target = out / "retrieved.jsonl"
    with open(target, "w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(json.dumps({"id": claim.id, "evidence": rows[claim.id]}) + "\n")
    print(f"merged {len(claims):,} claims into {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
