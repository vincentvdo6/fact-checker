"""
Install a trained artifact, refusing anything whose contract does not match the repository.

Everything is checked before a byte is written. A half-installed artifact is worse than a
rejected one: the contract file would say one thing and the weights beside it another, and the
next run would score against a model it was not told about.

What gets checked is what can produce plausible wrong numbers rather than a crash. A permuted
label order is the worst of them -- every metric still computes, against the wrong classes. A
changed template or sequence budget means the model was shown different text than the repository
would show it. A different tokenizer means the ids differ even when the text does not.

Nothing here loads the weights. Phase 02 runs no inference locally, so validation compares
declarations; the check that the declarations describe reality is the token-length assertion in
the slow tests, which recomputes the notebook's own exported lengths from the repository code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from src.verdict.contract import CONTRACT_FILE, EncoderContract
from src.verdict.encode import LABELS, TEMPLATE_ID

MODELS = Path("models/verdict")
REQUIRED = (
    CONTRACT_FILE,
    "metrics.json",
    "model_v1/model.safetensors",
    "predictions_calibration.jsonl",
    "predictions_test.jsonl",
)


def check(archive: zipfile.ZipFile) -> tuple[EncoderContract, dict]:
    """Every reason to refuse, gathered before anything is written."""
    names = set(archive.namelist())
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        raise SystemExit(f"archive is missing: {missing}")

    contract = EncoderContract.from_dict(json.loads(archive.read(CONTRACT_FILE)))
    metrics = json.loads(archive.read("metrics.json"))

    if list(contract.labels) != list(LABELS):
        raise SystemExit(
            f"label order mismatch: artifact {list(contract.labels)}, repository {list(LABELS)}.\n"
            "Every metric would still compute, against the wrong classes. Retrain."
        )
    if contract.template_id != TEMPLATE_ID:
        raise SystemExit(
            f"template mismatch: artifact {contract.template_id}, repository {TEMPLATE_ID}.\n"
            "The model was shown different text than this repository would show it. Retrain."
        )

    # The tokenizer files travel in the zip, so its identity is checkable rather than asserted.
    digest = hashlib.sha256()
    for name in sorted(n for n in names if n.startswith("model_v1/") and n.endswith((".json", ".model"))):
        digest.update(archive.read(name))
    if digest.hexdigest() != contract.tokenizer_sha256:
        raise SystemExit(
            f"tokenizer mismatch: files hash to {digest.hexdigest()[:12]}, "
            f"contract declares {contract.tokenizer_sha256[:12]}"
        )

    if metrics.get("smoke"):
        raise SystemExit("this is a smoke run; clear CFG['smoke'] and retrain before installing")
    return contract, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="artifacts_<variant>_v1.zip downloaded from Kaggle")
    parser.add_argument("--models", default=str(MODELS))
    parser.add_argument(
        "--as", dest="install_as", default=None,
        help="install under this name instead of the contract's variant, so a second model "
             "trained on different data sits beside the control rather than over it",
    )
    args = parser.parse_args()

    path = Path(args.archive)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")

    with zipfile.ZipFile(path) as archive:
        contract, metrics = check(archive)
        # The contract still says which encoder variant this is; the directory says which run.
        # Phase 05 trains the same "retrieved" variant on regrounded data, and installing it
        # over the Phase 02 control would destroy the only baseline it is measured against.
        dest = Path(args.models) / (args.install_as or contract.variant)
        dest.mkdir(parents=True, exist_ok=True)
        archive.extractall(dest)

    contract.save(dest / CONTRACT_FILE)
    print(f"installed {contract.variant} to {dest}")
    if args.install_as and args.install_as != contract.variant:
        print(f"  installed as {args.install_as}; the contract still declares {contract.variant}")
    print(f"  base       {contract.base_model} @ {contract.base_revision}")
    print(f"  max_length {contract.max_length}   template {contract.template_id}")
    print(f"  steps      {metrics['steps']:,} of {metrics['planned_steps']:,}"
          + ("  (TRUNCATED by the time budget)" if metrics.get("truncated") else ""))
    for split, accuracy in metrics.get("accuracy", {}).items():
        print(f"  {split:<12} {accuracy:.4f}")
    print("\nnext: python -m scripts.eval_verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
