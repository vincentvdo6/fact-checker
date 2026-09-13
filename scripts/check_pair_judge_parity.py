"""
Checkpoint: the local pair-judge graph is the model whose logits were saved on Kaggle.

The calibrator and bands are fitted on `predictions_calibration.jsonl`, which torch produced on
Kaggle. If the exported graph drifts, every band promise is voided silently, so this re-scores a
sample of the test pairs through onnxruntime and compares logits, argmax and calibrated band
against the saved rows. Tolerance 1e-3 on logits, as for the verdict model; argmax and band must
agree exactly. Resumable is unnecessary at this size -- 2,000 pairs score in a couple of minutes.

    python -m scripts.check_pair_judge_parity --limit 2000
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np

from src.verdict.pair_judge import MODEL_DIR, PairJudge
from src.verdict.pair_judgment import RELATIONS

DATASET = Path("data/kaggle/pair-judge-v1/pairs_test.jsonl.gz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    saved = {}
    with (args.models / "predictions_test.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            saved[row["id"]] = row
    with gzip.open(DATASET, "rt", encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle]
    missing = [pair["id"] for pair in pairs if pair["id"] not in saved]
    if missing:
        raise SystemExit(f"{len(missing)} test pairs have no saved logits; the dataset and the run disagree")
    sample = random.Random(args.seed).sample(pairs, min(args.limit, len(pairs)))
    judge = PairJudge(model_dir=args.models, min_band=None)
    started = time.monotonic()
    local = judge.logits([(pair["premise"], pair["hypothesis"]) for pair in sample])
    seconds = time.monotonic() - started
    remote = np.asarray([saved[pair["id"]]["logits"] for pair in sample], dtype=np.float64)
    difference = float(np.abs(local - remote).max())
    argmax = int((local.argmax(axis=1) == remote.argmax(axis=1)).sum())
    bands = None
    if judge.bands is not None and judge.calibrator is not None:
        confident = judge.calibrator.transform(local).max(axis=1), judge.calibrator.transform(remote).max(axis=1)
        bands = int(sum(judge.bands.assign(float(a)) == judge.bands.assign(float(b)) for a, b in zip(*confident)))
    report = {"pairs": len(sample), "max_abs_logit_difference": difference, "tolerance": args.tolerance,
              "argmax_agreement": f"{argmax}/{len(sample)}", "band_agreement": None if bands is None else f"{bands}/{len(sample)}",
              "seconds": round(seconds, 1), "labels": list(RELATIONS),
              "passed": difference <= args.tolerance and argmax == len(sample) and (bands is None or bands == len(sample))}
    (args.models / "parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
