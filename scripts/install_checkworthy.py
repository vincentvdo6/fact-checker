"""
Install the check-worthiness artifacts from their two Kaggle kernels.

The detector arrives in two pieces from two runs: `checkworthy-detector` trains and reports, and
`checkworthy-onnx` converts the weights to a graph this machine can execute. Assembling them by
hand worked once and is exactly the kind of step that stops being reproducible the moment anyone
looks away -- the paths were only in a shell history.

**Everything is validated before a byte is written**, the same posture as
`scripts/install_artifacts.py`. A half-installed detector is worse than a refused one: the contract
would describe one model and the graph beside it another, and every probability downstream would
still compute.

The checks are the ones that produce plausible wrong numbers rather than a crash. A permuted label
order is the worst, because `factual` and `check_worthy` are read off fixed class indices -- swap
two and the demo gates on the wrong quantity while every metric still returns a number. A
mismatched `max_length` or template means the model is being shown different text than it was
trained on. The exported ONNX file is named `verdict.onnx` by the shared export notebook regardless
of task, so it is renamed on the way in rather than left to imply the wrong model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from scripts.build_checkworthy_dataset import LABELS

DEST = Path("models/checkworthy/v1")
EXPORT_NAME = "verdict.onnx"        # the shared export notebook's fixed output name
ONNX_NAME = "onnx/model.onnx"
TEMPLATE_ID = "sentence_v1"
MAX_LENGTH = 128

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spm.model")


def validate(contract: dict) -> None:
    """Refuse anything the repository would then read wrongly."""
    if tuple(contract["labels"]) != LABELS:
        raise SystemExit(
            f"label order is {contract['labels']}, repository expects {list(LABELS)}. "
            "factual and check_worthy are read off fixed class indices, so a permutation "
            "silently gates the demo on the wrong quantity."
        )
    if contract["template_id"] != TEMPLATE_ID:
        raise SystemExit(f"template {contract['template_id']!r}, expected {TEMPLATE_ID!r}")
    if contract["max_length"] != MAX_LENGTH:
        raise SystemExit(f"max_length {contract['max_length']}, expected {MAX_LENGTH}")
    if contract["variant"] != "checkworthy":
        raise SystemExit(f"variant {contract['variant']!r}; this is not the check-worthiness head")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", required=True,
                        help="kaggle kernels output vincentvdo6/checkworthy-detector -p <dir>")
    parser.add_argument("--onnx-output", required=True,
                        help="kaggle kernels output vincentvdo6/checkworthy-onnx -p <dir>")
    parser.add_argument("--dest", default=str(DEST))
    args = parser.parse_args()

    train = Path(args.train_output)
    export = Path(args.onnx_output)
    dest = Path(args.dest)

    contract_path = export / "contract.json"
    if not contract_path.exists():
        raise SystemExit(f"{contract_path} is missing; is that the checkworthy-onnx output?")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate(contract)

    graph = export / EXPORT_NAME
    if not graph.exists():
        raise SystemExit(f"{graph} is missing; the export kernel writes {EXPORT_NAME}")
    metrics_path = train / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} is missing; is that the checkworthy-detector output?")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Nothing is written until every check above has passed.
    (dest / "onnx").mkdir(parents=True, exist_ok=True)
    (dest / "model_v1").mkdir(parents=True, exist_ok=True)
    shutil.copy(contract_path, dest / "contract.json")
    shutil.copy(metrics_path, dest / "metrics.json")
    shutil.copy(graph, dest / ONNX_NAME)
    for name in TOKENIZER_FILES:
        if (export / name).exists():
            shutil.copy(export / name, dest / "model_v1" / name)
    for split in ("calibration", "test"):
        source = train / f"predictions_{split}.jsonl"
        if source.exists():
            shutil.copy(source, dest / f"predictions_{split}.jsonl")

    test = metrics["model"]["test"]
    floor = metrics["baselines"]["naive_bayes/test"]
    print(f"installed to {dest}")
    print(f"  labels          {contract['labels']}")
    print(f"  max_length      {contract['max_length']}  template {contract['template_id']}")
    print(f"  graph           {(dest / ONNX_NAME).stat().st_size / 1e6:.0f} MB")
    print(f"\n{'':<20} {'accuracy':>9} {'factual F1':>11} {'worthy F1':>10}")
    print(f"{'lexical floor':<20} {floor['accuracy']:>9.4f} {floor['factual']['f1']:>11.4f} "
          f"{floor['check_worthy']['f1']:>10.4f}")
    print(f"{'detector':<20} {test['accuracy']:>9.4f} {test['factual']['f1']:>11.4f} "
          f"{test['check_worthy']['f1']:>10.4f}")
    print("\nthose are ClaimBuster's held-out debates. The out-of-domain number is "
          "scripts/compare_checkworthy.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
