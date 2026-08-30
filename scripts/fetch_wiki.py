"""
Download and extract the FEVER Wikipedia corpus.

The archive expands to 7.6 GB across 109 shards. It also carries 109 AppleDouble resource
forks under __MACOSX/ whose names end in .jsonl, so extractall() plus a *.jsonl glob picks
up 109 binary decoys. Members are extracted by explicit name instead.

Ends with a format report rather than trusting the layout: shard count, the first record's
keys, and a raw repr of one lines field. Confirm that output before relying on the parser.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://fever.ai/download/fever/wiki-pages.zip"
ARCHIVE = Path("data/fever/wiki-pages.zip")
DEST = Path("data/fever/wiki-pages")

EXPECTED_BYTES = 1_713_485_474
SHARD = re.compile(r"^wiki-pages/wiki-\d{3}\.jsonl$")
EXPECTED_SHARDS = 109


def download() -> None:
    if ARCHIVE.exists() and ARCHIVE.stat().st_size == EXPECTED_BYTES:
        print(f"{ARCHIVE.name}: already present")
        return

    print(f"{ARCHIVE.name}: downloading {EXPECTED_BYTES / 1e9:.2f} GB")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, ARCHIVE)

    actual = ARCHIVE.stat().st_size
    if actual != EXPECTED_BYTES:
        ARCHIVE.unlink()
        raise SystemExit(f"{ARCHIVE.name}: got {actual} bytes, expected {EXPECTED_BYTES}")
    print(f"{ARCHIVE.name}: ok")


def extract() -> list[Path]:
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = sorted(name for name in archive.namelist() if SHARD.match(name))
        if len(members) != EXPECTED_SHARDS:
            raise SystemExit(f"expected {EXPECTED_SHARDS} shards, matched {len(members)}")

        for name in members:
            target = DEST / Path(name).name
            if target.exists():
                continue
            with archive.open(name) as source, open(target, "wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
    return sorted(DEST.glob("wiki-*.jsonl"))


def report(shards: list[Path]) -> None:
    total = sum(path.stat().st_size for path in shards)
    print(f"\nshards      {len(shards)}  ({shards[0].name} .. {shards[-1].name})")
    print(f"extracted   {total / 1e9:.2f} GB")

    with open(shards[0], encoding="utf-8") as handle:
        first = json.loads(handle.readline())
        # The first record of shard 001 has an empty id; read on for a real one.
        sample = first
        while sample["id"] == "":
            sample = json.loads(handle.readline())

    print(f"record keys {sorted(first)}")
    print(f"first id    {first['id']!r}")
    print(f"sample id   {sample['id']!r}")
    print(f"sample lines prefix\n    {sample['lines'][:220]!r}")


def main() -> int:
    download()
    shards = extract()
    report(shards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
