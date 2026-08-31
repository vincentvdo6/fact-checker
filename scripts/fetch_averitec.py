"""
Download the AVeriTeC claim files into data/averitec/.

AVeriTeC is the external-validity check for everything Phases 02 to 05 found. Every result there
traces back to one property of FEVER: its claims were written *from* Wikipedia sentences, so the
wording leaks the verdict and an evidence-free model scores 0.5865 against a 0.3410 majority. That
artifact is why groundedness and correctness diverge, why honest abstention costs coverage, and
why relabelling ungrounded rows backfired. If those findings are about fact-checking, they should
survive on claims nobody wrote from a corpus. If they are about FEVER, they should not.

AVeriTeC claims are real: collected from fact-checking organisations, dated, attributed to a
speaker, with evidence assembled by annotators as question-answer pairs over web sources rather
than as sentence references into a fixed dump. That difference matters and is not incidental --
the retrieval stack built in Phase 01 does not transfer, because there is no corpus to retrieve
from. What does transfer is the verdict model, the calibration, and every measurement.

The test split ships without labels: it is a shared-task leaderboard file, 14 bytes of empty JSON.
So this fetches train and dev only, and the project's own three-way split is carved from those.

Sizes are checked rather than hashed. The release is static, and a truncated download is the
failure that actually happens -- a partial JSON array raises on parse, but a partial *file* that
happens to close cleanly does not.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data"
DEST = Path("data/averitec")

FILES = {
    "train.json": 10_184_813,
    "dev.json": 1_785_475,
}

# What the release should contain, so a silently changed upstream file is caught here rather than
# by a model trained on something else.
EXPECTED_CLAIMS = {"train.json": 3068, "dev.json": 500}


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

    # Parse it here rather than trusting the byte count alone: the count catches truncation, this
    # catches a file that downloaded whole and is not the release we think it is.
    claims = json.loads(target.read_text(encoding="utf-8"))
    if len(claims) != EXPECTED_CLAIMS[name]:
        target.unlink()
        raise SystemExit(
            f"{name}: got {len(claims):,} claims, expected {EXPECTED_CLAIMS[name]:,}; "
            "the upstream release has changed and every split keyed to it is now wrong"
        )
    print(f"{name}: ok, {len(claims):,} claims")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        fetch(name, size)
    print("\nnext: python -m scripts.eval_averitec_baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
