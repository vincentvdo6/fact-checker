"""
Checkpoint 1: does the local ONNX model reproduce the logits every calibration artifact was fitted on?

This is the gate for the whole demo, and it is not a formality. The calibrator's temperature and
three per-class biases, the three band thresholds, and the sufficiency gate were all fitted against
the logits in `predictions_test.jsonl`, produced by torch on Kaggle. If the exported graph drifts,
nothing raises: the pipeline runs, the page renders, and a claim shown as "strong -- right about
nine times in ten" carries a promise that was measured against different numbers.

Three comparisons, in increasing order of what they actually protect:

  max |logit difference|   the numerical question. fp32 kernels differ between runtimes, so exact
                           equality is not the bar; 1e-3 is far below what could move a decision.
  argmax agreement         the verdict itself. A single flip here is a claim the demo would label
                           differently from the model that was evaluated.
  band agreement           the operative one. Bands are thresholds on *calibrated* confidence, so
                           this is the only check that covers the whole path the sidebar depends
                           on -- logits, calibrator, and thresholds together.

Inputs are rebuilt through the shipped encoder with one Random advanced across the split in file
order, exactly as the training notebook's `encode()` did. Reconstructing them any other way would
compare the ONNX model against logits computed from different text, and disagreement would be
uninterpretable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from src.calibration.bands import BandPolicy
from src.calibration.scaling import from_dict as calibrator_from_dict
from src.verdict.encode import select_evidence
from src.verdict.runtime import VerdictRuntime

DATA = Path("data/kaggle/fever-verdict-v1")
MODELS = Path("models/verdict")

LOGIT_TOLERANCE = 1e-3

# Scored logits are appended here as they are produced, and fsynced per batch. Scoring 2,000
# claims takes long enough that this machine went down under the load twice before finishing,
# and losing an hour to a power cut is not a reason to weaken the check -- so a crash costs one
# batch. Delete the file to force a clean re-score.
CACHE = "onnx_logits_{split}.jsonl"


def read_rows(split: str) -> list[dict]:
    path = DATA / f"verdict_{split}.jsonl.gz"
    opener, target = (gzip.open, path) if path.exists() else (open, DATA / f"verdict_{split}.jsonl")
    with opener(target, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_cache(path: Path) -> dict[str, list[float]]:
    """
    Logits already scored, by claim id.

    A truncated final line is expected rather than exceptional: the failure this cache exists to
    survive is a hard reset, which can land mid-write. That line is dropped and the claim is
    re-scored, which costs one forward pass; parsing it as valid JSON is what would be dangerous.
    """
    if not path.exists():
        return {}
    done: dict[str, list[float]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["id"]] = row["logits"]
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="retrieved")
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="fewer claims, for a quick look")
    args = parser.parse_args()

    runtime = VerdictRuntime(args.variant)
    root = MODELS / args.variant
    frozen = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    calibrator = calibrator_from_dict(frozen["calibrator"])
    policy = BandPolicy.from_dict(
        json.loads((root / "sufficiency.json").read_text(encoding="utf-8"))["bands"]
    )

    rows = read_rows(args.split)
    exported = {
        row["id"]: row
        for row in map(json.loads, open(root / f"predictions_{args.split}.jsonl", encoding="utf-8"))
    }

    # One Random across the split in file order, as the notebook's encode() did. Every row must be
    # encoded to keep the stream in step, even when --limit cuts the comparison short.
    rng = random.Random(runtime.contract.seed)
    pairs, ids, budgets = [], [], []
    for row in rows:
        evidence = select_evidence(
            row, args.variant, runtime.contract.max_length, runtime.measure, rng
        )
        if args.limit and len(pairs) >= args.limit:
            continue
        pairs.append((row["claim"], evidence))
        ids.append(row["id"])
        budgets.append(len(evidence))

    cache_path = root / CACHE.format(split=args.split)
    done = read_cache(cache_path)
    if done:
        print(f"resuming: {len(done):,} of {len(pairs):,} claims already scored", flush=True)

    print(f"scoring {len(pairs):,} {args.split} claims through onnxruntime "
          f"on {runtime.threads} threads ...", flush=True)
    with open(cache_path, "a", encoding="utf-8") as cache:
        for start in range(0, len(pairs), args.batch):
            chunk = ids[start:start + args.batch]
            if all(i in done for i in chunk):
                continue
            for claim_id, s in zip(chunk, runtime.score_batch(pairs[start:start + args.batch]),
                                   strict=True):
                done[claim_id] = [float(x) for x in s.logits]
                cache.write(json.dumps({"id": claim_id, "logits": done[claim_id]}) + "\n")
            # Flushed and fsynced per batch: the failure this survives is a hard reset, which
            # takes the OS page cache with it.
            cache.flush()
            os.fsync(cache.fileno())
            if (start // args.batch) % 20 == 0 and start:
                print(f"  {start:,}/{len(pairs):,}", flush=True)

    local = np.asarray([done[i] for i in ids], dtype=np.float64)
    kaggle = np.asarray([exported[i]["logits"] for i in ids], dtype=np.float64)

    difference = float(np.abs(local - kaggle).max())
    argmax_same = int((local.argmax(1) == kaggle.argmax(1)).sum())

    local_bands = [policy.assign(float(p)) for p in calibrator.transform(local).max(axis=1)]
    kaggle_bands = [policy.assign(float(p)) for p in calibrator.transform(kaggle).max(axis=1)]
    band_same = sum(a == b for a, b in zip(local_bands, kaggle_bands, strict=True))

    used_same = sum(b == exported[i]["n_evidence_used"] for b, i in zip(budgets, ids, strict=True))

    n = len(ids)
    print(f"\n{'check':<34} {'result':>16}   verdict")
    print(f"{'max |logit difference|':<34} {difference:>16.3e}   "
          f"{'ok' if difference < LOGIT_TOLERANCE else 'FAIL'}")
    print(f"{'argmax agreement':<34} {f'{argmax_same:,}/{n:,}':>16}   "
          f"{'ok' if argmax_same == n else 'FAIL'}")
    print(f"{'band agreement':<34} {f'{band_same:,}/{n:,}':>16}   "
          f"{'ok' if band_same == n else 'FAIL'}")
    print(f"{'evidence count agreement':<34} {f'{used_same:,}/{n:,}':>16}   "
          f"{'ok' if used_same == n else 'FAIL'}")

    passed = difference < LOGIT_TOLERANCE and argmax_same == n and band_same == n and used_same == n
    print(f"\nCheckpoint 1: {'PASS -- the local model is the model that was calibrated' if passed else 'FAIL'}")
    if not passed:
        worst = int(np.abs(local - kaggle).max(axis=1).argmax())
        print(f"  worst row: claim {ids[worst]}")
        print(f"    kaggle {kaggle[worst].round(5)}")
        print(f"    local  {local[worst].round(5)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
