"""
Checkpoint 5: does the local check-worthiness graph reproduce the logits its calibration was fitted on?

The verdict model got this check in Phase 07 and the detector shipped without it, which was an
oversight rather than a decision. The same argument applies with the same force: the vector-scaling
calibrator and the 0.35 threshold in `models/checkworthy/v1/calibration.json` were both fitted
against logits torch produced on Kaggle. If the exported graph drifts, nothing raises -- the filter
runs, the page renders, and a sentence is admitted or dropped by a boundary that was measured
against different numbers.

Four comparisons, in increasing order of what they protect:

  max |logit difference|   the numerical question. fp32 kernels differ between runtimes, so exact
                           equality is not the bar.
  argmax agreement         the three-class prediction itself.
  calibrated agreement     the scores the filter actually reads, after vector scaling.
  admission agreement      the operative one. Whether each sentence clears the frozen threshold is
                           the only thing the pipeline consumes, and it is the composition of
                           graph, calibrator and cut -- the whole path in one number.

Rebuilding the inputs is trivial here in a way it was not for the verdict model: there is no
evidence to select and no RNG to keep in step, just the sentence. That removes the failure mode
that made the verdict parity delicate, and leaves only the graph itself under test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.calibration.scaling import from_dict as calibrator_from_dict
from src.pipeline.detector import CALIBRATION_FILE, MODEL, CheckworthyDetector

DATA = Path("data/kaggle/checkworthy-v1")

LOGIT_TOLERANCE = 1e-3


def read_split(split: str) -> dict[str, str]:
    """Sentence text by id, which is the entire input to this model."""
    path = DATA / f"checkworthy_{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: r["text"] for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0,
                        help="fewer sentences, for a quick look. Takes the head of the file, "
                             "which is one debate -- not a sample of the split")
    parser.add_argument("--model", default=str(MODEL))
    args = parser.parse_args()

    root = Path(args.model)
    detector = CheckworthyDetector(root)
    frozen = json.loads((root / CALIBRATION_FILE).read_text(encoding="utf-8"))
    calibrator = calibrator_from_dict(frozen["calibrator"])
    thresholds = frozen["thresholds"]

    text = read_split(args.split)
    exported = [
        json.loads(line)
        for line in (root / f"predictions_{args.split}.jsonl").read_text(
            encoding="utf-8").splitlines() if line
    ]
    if args.limit:
        exported = exported[: args.limit]

    ids = [row["id"] for row in exported]
    missing = [i for i in ids if i not in text]
    if missing:
        raise SystemExit(
            f"{len(missing)} scored ids are not in the {args.split} split ({missing[:3]}). "
            "The dataset was rebuilt after the model was trained; re-run the training kernel "
            "rather than comparing against sentences it never saw."
        )
    sentences = [text[i] for i in ids]

    print(f"scoring {len(sentences):,} {args.split} sentences on {detector.threads} threads ...",
          flush=True)
    scored = []
    for start in range(0, len(sentences), args.batch):
        scored.extend(
            detector.score_batch(sentences[start:start + args.batch], batch=args.batch)
        )
        if (start // args.batch) % 20 == 0 and start:
            print(f"  {start:,}/{len(sentences):,}", flush=True)

    local = np.asarray([s.logits for s in scored], dtype=np.float64)
    kaggle = np.asarray([row["logits"] for row in exported], dtype=np.float64)

    difference = float(np.abs(local - kaggle).max())
    argmax_same = int((local.argmax(1) == kaggle.argmax(1)).sum())

    local_probabilities = calibrator.transform(local)
    kaggle_probabilities = calibrator.transform(kaggle)
    calibrated_close = int(
        (np.abs(local_probabilities - kaggle_probabilities).max(axis=1) < 1e-4).sum()
    )

    # The composition the pipeline actually consumes: graph, then calibrator, then the frozen cut.
    admissions_same = {}
    for name, cut in thresholds.items():
        if name not in ("factual", "check_worthy"):
            raise SystemExit(f"unknown binarization {name!r} in the calibration artifact")
        columns = (1, 2) if name == "factual" else (2,)
        local_score = local_probabilities[:, columns].sum(axis=1)
        kaggle_score = kaggle_probabilities[:, columns].sum(axis=1)
        admissions_same[name] = int(((local_score >= cut) == (kaggle_score >= cut)).sum())

    n = len(ids)
    print(f"\n{'check':<38} {'result':>16}   verdict")
    print(f"{'max |logit difference|':<38} {difference:>16.3e}   "
          f"{'ok' if difference < LOGIT_TOLERANCE else 'FAIL'}")
    print(f"{'argmax agreement':<38} {f'{argmax_same:,}/{n:,}':>16}   "
          f"{'ok' if argmax_same == n else 'FAIL'}")
    print(f"{'calibrated probability agreement':<38} {f'{calibrated_close:,}/{n:,}':>16}   "
          f"{'ok' if calibrated_close == n else 'FAIL'}")
    for name, same in admissions_same.items():
        print(f"{'admission agreement, ' + name:<38} {f'{same:,}/{n:,}':>16}   "
              f"{'ok' if same == n else 'FAIL'}")

    passed = (difference < LOGIT_TOLERANCE and argmax_same == n and calibrated_close == n
              and all(same == n for same in admissions_same.values()))
    print(f"\nCheckpoint 5: {'PASS -- the local detector is the model that was calibrated' if passed else 'FAIL'}")
    if not passed:
        worst = int(np.abs(local - kaggle).max(axis=1).argmax())
        print(f"  worst row: {ids[worst]}  {sentences[worst][:70]!r}")
        print(f"    kaggle {kaggle[worst].round(5)}")
        print(f"    local  {local[worst].round(5)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
