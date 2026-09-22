"""Install the validated full-precision ordering bundle without replacing another model.

The bundle contains contract.json, source-copy.json, tokenizer.json, ranker.onnx
and the external tensors named by the manifest. Hashes pin the tested weights and
tokenizer; compressed and differently prompted experiments cannot be substituted.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from src.verdict.context_ranker import (
    CONTRACT_SHA256,
    MANIFEST_SHA256,
    MODEL_DIR,
    TOKENIZER_SHA256,
    graph_files,
    verify_file,
)


def install(bundle: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Context ranker destination already exists: {destination}")
    files = {"contract.json": CONTRACT_SHA256, "tokenizer.json": TOKENIZER_SHA256,
             "source-copy.json": MANIFEST_SHA256}
    for name, expected in files.items():
        verify_file(bundle / name, expected)
    files.update(graph_files(bundle))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="context-ranker-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        for name, expected in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle / name, target)
            verify_file(target, expected)
        staging.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dest", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    install(args.bundle, args.dest)
    print(f"Installed the local context ranker to {args.dest}; ordering only, never verdicts.")


if __name__ == "__main__":
    main()
