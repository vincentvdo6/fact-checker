"""
Fetch ClaimBuster: labelled check-worthiness data for US presidential debates.

Checkpoint 3 measured the hand-written filter at precision 0.2571 and recall 0.3913, and the
error table said the causes were structural rather than tunable -- an anchor requirement that
belongs to the retrieval stack, a closed-class verb test that misses present-tense lexical verbs,
and an `-ed` rule that fires on "United" inside a proper noun. Patching those against the same 120
sentences they were scored on would be fitting on the test set. Learning the task from data that
has nothing to do with that transcript is the alternative.

**The labels here are not mine, which is the entire point.** The Checkpoint 3 measurement is
bounded by the heuristic and the hand labels sharing an author. ClaimBuster was labelled by
crowdworkers with no knowledge of this project, so a model trained on it and scored against those
120 SOTU sentences is a genuinely independent comparison -- different annotators, different genre
(debate exchanges against a prepared address), different decade.

Three native labels, kept rather than binarized here:

  -1  NFS  non-factual: opinion, rhetoric, questions, greetings
   0  UFS  factual but unimportant -- "I saw a movie, Crocodile Dundee"
   1  CFS  check-worthy factual -- "We're consuming 50 percent of the world's cocaine"

The distinction between UFS and CFS is *importance*, not verifiability, and which one the demo
should act on is an open question the Phase 07 rubric and ClaimBuster answer differently. Keeping
three classes means that question gets measured instead of guessed; see
`scripts/build_checkworthy_dataset.py`.

CC-BY-4.0, so redistribution is fine with attribution. Cite Arslan, Hassan, Li and Tremayne,
"A Benchmark Dataset of Check-worthy Factual Claims" (ICWSM 2020).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DEST = Path("data/claimbuster")
RECORD = "https://zenodo.org/api/records/3609356"
WANTED = ("crowdsourced.csv", "groundtruth.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DEST))
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(RECORD, timeout=60) as response:      # noqa: S310 - fixed host
        record = json.loads(response.read())
    print(f"{record['metadata']['title']}")
    print(f"licence {record['metadata']['license']['id']}  access {record['metadata']['access_right']}")

    manifest: dict[str, object] = {
        "source": RECORD,
        "title": record["metadata"]["title"],
        "licence": record["metadata"]["license"]["id"],
        "files": {},
    }
    for entry in record["files"]:
        name = entry["key"]
        if name not in WANTED:
            continue
        path = dest / name
        print(f"  fetching {name} ...", flush=True)
        urllib.request.urlretrieve(entry["links"]["self"], path)      # noqa: S310 - fixed host
        manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        print(f"    {path.stat().st_size / 1e6:.1f} MB  sha256 {manifest['files'][name]['sha256'][:16]}...")

    missing = [n for n in WANTED if n not in manifest["files"]]
    if missing:
        raise SystemExit(f"the Zenodo record no longer carries {missing}")

    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwritten to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
