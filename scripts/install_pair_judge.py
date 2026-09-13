"""
Install the pair judge from its two Kaggle kernels, validating before a byte is written.

`pair-judge` trains and writes the contract, predictions and metrics; `pair-judge-onnx` turns
the weights into the graph this machine runs. The checks are the ones that produce plausible
wrong numbers rather than a crash: a permuted label order would turn `states` into `unrelated`
with every probability still computing, and a changed template or max_length would show the
model text it was never trained on. The shared export notebook names its output `verdict.onnx`
whatever it exported, so the file is renamed on the way in.

    kaggle kernels output vincentvdo6/pair-judge -p runs/kaggle/pair-judge
    kaggle kernels output vincentvdo6/pair-judge-onnx -p runs/kaggle/pair-judge-onnx
    python -m scripts.install_pair_judge --train-output runs/kaggle/pair-judge --onnx-output runs/kaggle/pair-judge-onnx
    python -m scripts.calibrate_pair_judge
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.verdict.pair_judge import MODEL_DIR, TEMPLATE_ID
from src.verdict.pair_judgment import RELATIONS

EXPORT_NAME = "verdict.onnx"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spm.model", "added_tokens.json")
TRAIN_FILES = ("contract.json", "metrics.json", "train_log.jsonl",
               "predictions_trainval.jsonl", "predictions_calibration.jsonl", "predictions_test.jsonl")


def validate(contract: dict) -> None:
    if tuple(contract["labels"]) != RELATIONS:
        raise SystemExit(f"label order is {contract['labels']}, the judge expects {list(RELATIONS)}")
    if contract["template_id"] != TEMPLATE_ID:
        raise SystemExit(f"template {contract['template_id']!r}, expected {TEMPLATE_ID!r}")
    if contract["variant"] != "pair_judge":
        raise SystemExit(f"variant {contract['variant']!r}; this is not the pair judge")
    if not isinstance(contract["max_length"], int) or contract["max_length"] < 32:
        raise SystemExit(f"max_length {contract['max_length']!r} is not a usable length")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--onnx-output", type=Path, required=True)
    parser.add_argument("--dest", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    contract = json.loads((args.train_output / "contract.json").read_text(encoding="utf-8"))
    validate(contract)
    model_dir = args.train_output / "model_v1"
    missing = [name for name in TRAIN_FILES if not (args.train_output / name).exists()]
    missing += [f"model_v1/{name}" for name in TOKENIZER_FILES[:2] if not (model_dir / name).exists()]
    graph = args.onnx_output / EXPORT_NAME
    if not graph.exists():
        missing.append(str(graph))
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    exported = json.loads((args.onnx_output / "contract.json").read_text(encoding="utf-8")) \
        if (args.onnx_output / "contract.json").exists() else contract
    if exported.get("tokenizer_sha256") != contract["tokenizer_sha256"]:
        raise SystemExit("the exported graph's contract names a different tokenizer than the trained model")

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "onnx").mkdir(exist_ok=True)
    (args.dest / "model_v1").mkdir(exist_ok=True)
    for name in TRAIN_FILES:
        shutil.copy2(args.train_output / name, args.dest / name)
    for name in TOKENIZER_FILES:
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, args.dest / "model_v1" / name)
    shutil.copy2(graph, args.dest / "onnx" / "model.onnx")
    print(f"installed pair judge to {args.dest}: {contract['base_model']} @ {contract['base_revision'][:12]}, "
          f"max_length {contract['max_length']}, graph {graph.stat().st_size / 1e6:.0f} MB")
    print("next: python -m scripts.calibrate_pair_judge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
