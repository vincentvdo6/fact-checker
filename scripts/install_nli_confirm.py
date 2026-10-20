"""Install the NLI model that confirms near-miss supports, pinned to the tested revision and file hashes.

Downloads the six files from Hugging Face at the pinned revision (or copies them from --source),
verifies every SHA-256 against `src.verdict.nli_confirm.FILES`, writes contract.json, and
publishes the directory atomically. An existing destination is never replaced.

    python -m scripts.install_nli_confirm [--source DIR] [--dest DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from src.verdict.nli_confirm import FILES, MODEL_DIR, REPO, REVISION, SUPPORT_LOWER, SUPPORT_THRESHOLD


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install(destination: Path, source: Path | None = None) -> None:
    if destination.exists():
        raise FileExistsError(f"NLI model destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nli-confirm-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        for name, expected in FILES.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source is not None:
                shutil.copy2(source / name, target)
            else:
                from huggingface_hub import hf_hub_download

                shutil.copy2(hf_hub_download(REPO, name, revision=REVISION, cache_dir=temporary), target)
            if (actual := sha256(target)) != expected:
                raise ValueError(f"{name}: SHA-256 {actual} is not the tested {expected}")
        contract = {"repo": REPO, "revision": REVISION, "license": "MIT", "files": FILES,
                    "use": "confirms the pair judge's near-miss supports only; never contradictions",
                    "support_lower": SUPPORT_LOWER, "support_threshold": SUPPORT_THRESHOLD}
        (staging / "contract.json").write_text(json.dumps(contract, indent=1), encoding="utf-8")
        staging.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="a directory already holding the six files")
    parser.add_argument("--dest", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    install(args.dest, args.source)
    print(f"Installed the support-confirming NLI model to {args.dest}.")


if __name__ == "__main__":
    main()
