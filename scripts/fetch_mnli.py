"""
Fetch MultiNLI 1.0 as released by NYU: 392,702 train pairs plus two matched/mismatched dev sets.

The pair judge needs entailment over ordinary prose with negation on both sides, which FEVER
alone cannot teach: 664 of its 6,666 REFUTES dev claims carry a negation word against 68
SUPPORTS, so a FEVER-only model learns "not" as a REFUTES tell (Phase 02's claim-only artifact,
seen again in the decomposed replay). MNLI's neutral class is also the one public source of
"same topic, no entailment" pairs, which is exactly the `bears_on` relation.

The archive is ~227 MB and unpacks to ~430 MB of JSONL. Its SHA-256 is recorded in the manifest on
first fetch and checked on every later run. Licence: the corpus mixes OANC, fiction and other
sources under terms permitting research use; see the README inside the archive.

    python -m scripts.fetch_mnli
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
DEST = Path("data/mnli")
FILES = ("multinli_1.0_train.jsonl", "multinli_1.0_dev_matched.jsonl", "multinli_1.0_dev_mismatched.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "multinli_1.0.zip"
    manifest_path = DEST / "manifest.json"
    if not archive.exists():
        print(f"fetching {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=300) as response, archive.open("wb") as handle:
            done = 0
            while block := response.read(1 << 22):
                handle.write(block)
                done += len(block)
                if done % (1 << 25) < (1 << 22):
                    print(f"  {done / (1 << 20):,.0f} MB", flush=True)
    digest = sha256(archive)
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["archive_sha256"]
        if recorded != digest:
            raise SystemExit(f"archive hash changed: recorded {recorded}, found {digest}")
    with zipfile.ZipFile(archive) as bundle:
        members = {Path(name).name: name for name in bundle.namelist()}
        for name in FILES:
            target = DEST / name
            if not target.exists():
                with bundle.open(members[name]) as source, target.open("wb") as handle:
                    while block := source.read(1 << 22):
                        handle.write(block)
                print(f"unpacked {name}: {target.stat().st_size:,} bytes", flush=True)
    manifest = {"url": URL, "archive_bytes": archive.stat().st_size, "archive_sha256": digest,
                "files": {name: {"bytes": (DEST / name).stat().st_size, "sha256": sha256(DEST / name)} for name in FILES}}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"archive_sha256": digest, **{k: v["bytes"] for k, v in manifest["files"].items()}}, indent=1))


if __name__ == "__main__":
    main()
