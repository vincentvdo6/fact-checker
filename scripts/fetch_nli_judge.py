"""
Fetch a small general-domain NLI cross-encoder, as ONNX, to judge (assertion, sentence) pairs.

The FEVER-tuned verdict model reads a negation word in the claim as a REFUTES tell and cannot
read "no shutdowns" as denying a shutdown; it was trained on packed Wikipedia evidence, not
one sentence at a time. `cross-encoder/nli-deberta-v3-xsmall` is a 22M-parameter DeBERTa
trained on SNLI and MNLI -- entailment over ordinary prose, negation included -- and ships an
ONNX graph, so it runs under onnxruntime with no torch and at a size an extension can carry.

Pinned to one revision and hashed on arrival. Nothing here is calibrated on this project's
data: the model gives a label and a softmax, and what those are worth on news sentences is
measured, not assumed.

    python -m scripts.fetch_nli_judge            # xsmall, 22M parameters
    python -m scripts.fetch_nli_judge --size base   # 184M, the size the verdict model already ships at
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

# Pinned main revisions as of 2025-04-11.
SIZES = {"xsmall": ("cross-encoder/nli-deberta-v3-xsmall", "a150876415327c80daeff35ca6f68f5ed8cf5c24"),
         "base": ("cross-encoder/nli-deberta-v3-base", "6c749ce3425cd33b46d187e45b92bbf96ee12ec7")}
LICENSE = "apache-2.0"
FILES = ("onnx/model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json")
OPTIONAL = ("special_tokens_map.json", "added_tokens.json", "spm.model")
LABELS = ("contradiction", "entailment", "neutral")       # id2label order in config.json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def fetch(repo: str, revision: str, target: Path, name: str, *, optional: bool = False) -> Path | None:
    destination = target / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
                while block := response.read(1 << 20):
                    handle.write(block)
        except urllib.error.HTTPError as error:
            if optional and error.code == 404:
                return None
            raise
        print(f"fetched {name}: {destination.stat().st_size:,} bytes", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=sorted(SIZES), default="xsmall")
    args = parser.parse_args()
    repo, revision = SIZES[args.size]
    target = Path("models/nli") / repo.split("/")[1]
    manifest = {"repo": repo, "revision": revision, "license": LICENSE, "labels": list(LABELS), "files": {}}
    for name in FILES + OPTIONAL:
        path = fetch(repo, revision, target, name, optional=name in OPTIONAL)
        if path is not None:
            manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    labels = tuple(config["id2label"][str(index)] for index in range(len(config["id2label"])))
    if labels != LABELS:
        raise SystemExit(f"label order changed upstream: {labels}")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({name: row["sha256"][:16] for name, row in manifest["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
