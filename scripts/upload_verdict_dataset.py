"""
Push the built dataset to Kaggle as a private dataset the notebooks attach.

The version lives in the slug rather than in Kaggle's version history, so a breaking rebuild
cannot silently replace a dataset that a notebook's checkpoint was keyed against. A compatible
refresh -- more claims, same contract -- goes out as a new version of the same slug; anything
that changes the template, the label order or the row shape gets a new slug and a new
contract_version.

Refuses to upload a dataset that has not passed the checker, because the whole point of the
checker is to be the last thing that runs before the data leaves the machine.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from src.verdict.dataset import sha256

DEST = Path("data/kaggle/fever-verdict-v1")
SLUG = "fever-verdict-v1"
TITLE = "FEVER verdict training data"


def kaggle(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def owner() -> str:
    """Read the username from the credentials rather than hardcoding it."""
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        raise SystemExit("~/.kaggle/kaggle.json is missing; configure the Kaggle CLI first")
    return json.loads(path.read_text(encoding="utf-8"))["username"]


def verify(dest: Path) -> dict:
    """Refuse to upload anything the checker has not seen in its current state."""
    manifest_path = dest / "dataset_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} is missing; run scripts/build_verdict_dataset.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for name, entry in manifest["files"].items():
        path = dest / name
        if not path.exists():
            raise SystemExit(f"{name} is missing from {dest}")
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"{name} does not match the manifest; rebuild and re-check")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DEST))
    parser.add_argument("--slug", default=SLUG)
    parser.add_argument("--message", default=None, help="version note; creates the dataset if absent")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    manifest = verify(dest)
    ref = f"{owner()}/{args.slug}"

    # The version belongs in the title as well as the slug. Hard-coding it meant every later
    # dataset went out titled v1, which is exactly the ambiguity the versioned slug exists to
    # prevent -- two datasets with the same name and different contents.
    # A full match on v<digits>, not a "starts with v": the bare slug "fever-verdict" ends in
    # "verdict", which does start with a v, and would have produced "... training data verdict".
    version = args.slug.rsplit("-", 1)[-1]
    title = f"{TITLE} {version}" if re.fullmatch(r"v\d+", version) else TITLE
    (dest / "dataset-metadata.json").write_text(
        json.dumps({"title": title, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, indent=2),
        encoding="utf-8",
    )

    total = sum(entry.get("bytes", 0) for entry in manifest["files"].values())
    rows = sum(split["rows"] for split in manifest["splits"].values())
    print(f"{ref}: {rows:,} rows across {len(manifest['splits'])} splits, {total / 1e6:.0f} MB")
    if args.dry_run:
        print("dry run; nothing uploaded")
        return 0

    existing = kaggle("datasets", "list", "--mine", "-s", args.slug)
    if ref in existing.stdout:
        note = args.message or "rebuild"
        print(f"versioning existing dataset: {note}")
        result = kaggle("datasets", "version", "-p", str(dest), "-m", note, "--dir-mode", "zip")
    else:
        print("creating a new private dataset")
        result = kaggle("datasets", "create", "-p", str(dest), "--dir-mode", "zip")

    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        return result.returncode

    print(f"\nattach in a notebook as: /kaggle/input/{args.slug}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
