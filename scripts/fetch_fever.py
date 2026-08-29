"""
Download the FEVER claim files into data/fever/.

Only the claim sets are fetched here. The 1.7 GB wiki-pages dump is a retrieval
dependency and is left to the retrieval phase, so a fresh clone can build splits
without it.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://fever.ai/download/fever"
DEST = Path("data/fever")

# Sizes are checked rather than hashed: the release is static, and a truncated
# download is the failure that actually happens.
FILES = {
    "train.jsonl": 33_024_303,
    "shared_task_dev.jsonl": 4_349_935,
}


def fetch(name: str, expected_bytes: int) -> None:
    target = DEST / name
    if target.exists() and target.stat().st_size == expected_bytes:
        print(f"{name}: already present")
        return

    print(f"{name}: downloading {expected_bytes / 1e6:.1f} MB")
    urllib.request.urlretrieve(f"{BASE_URL}/{name}", target)

    actual = target.stat().st_size
    if actual != expected_bytes:
        target.unlink()
        raise SystemExit(f"{name}: got {actual} bytes, expected {expected_bytes}")
    print(f"{name}: ok")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        fetch(name, size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
