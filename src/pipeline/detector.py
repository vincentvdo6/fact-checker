"""
The learned check-worthiness detector, run locally through onnxruntime.

Phase 07's hand-written filter is a set of rules whose failures are legible; this is a DeBERTa
trained on 15,512 ClaimBuster debate sentences whose failures are not. That trade is only worth
making if the numbers say so on data neither model was fitted to, which is what the 120 hand
labelled State of the Union sentences are for.

**Three classes, and the middle one is the whole question.** ClaimBuster separates
factual-but-unimportant from check-worthy factual by *importance*; the Phase 07 rubric labelled by
verifiability alone. So the detector exposes both readings rather than picking one:

  `factual`      class 1 or 2 -- comparable to the hand labels
  `check_worthy` class 2 only -- ClaimBuster's own stricter task

Which one the demo should gate on is an empirical question, and `scripts/compare_checkworthy.py`
answers it instead of this module assuming it.

Same bounded thread pool as the verdict runtime, for the same reason: an unbounded onnxruntime
took this machine down twice under sustained all-core load.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.calibration.scaling import from_dict as calibrator_from_dict

MODEL = Path("models/checkworthy/v1")
CALIBRATION_FILE = "calibration.json"
ONNX_FILE = "onnx/model.onnx"
DEFAULT_THREADS = 4


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Row-wise softmax, shifted by the row maximum.

    The shift is not cosmetic. exp() of a raw logit overflows to inf around 710, and inf/inf is
    nan -- which propagates into every probability, compares False against any threshold, and so
    silently turns a confident sentence into one the filter drops. Subtracting the row max is
    algebraically identity and bounds the exponent at zero.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class Scored:
    """Calibration-free class probabilities, plus both binarizations of them."""

    probabilities: np.ndarray
    label: str
    logits: np.ndarray | None = None

    @property
    def factual(self) -> float:
        """P(the sentence asserts something checkable) -- the Phase 07 rubric's notion."""
        return float(self.probabilities[1] + self.probabilities[2])

    @property
    def check_worthy(self) -> float:
        """P(worth a fact-checker's time) -- ClaimBuster's stricter task."""
        return float(self.probabilities[2])


class CheckworthyDetector:
    """The trained detector behind one call. Loading is deferred; the graph is 738 MB."""

    def __init__(self, root: str | Path = MODEL, *, threads: int | None = None) -> None:
        self.root = Path(root)
        self.contract = json.loads((self.root / "contract.json").read_text(encoding="utf-8"))
        self.labels: tuple[str, ...] = tuple(self.contract["labels"])
        self.threads = threads if threads is not None else int(
            os.environ.get("VERDICT_THREADS", DEFAULT_THREADS)
        )
        self._session = None
        self._tokenizer = None

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort

            path = self.root / ONNX_FILE
            if not path.exists():
                raise SystemExit(
                    f"{path} is missing. Export it on Kaggle with the checkworthy-onnx kernel, "
                    "then place model.onnx here."
                )
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
        return self._session

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.root / "model_v1"))
        return self._tokenizer

    def score(self, sentence: str) -> Scored:
        return self.score_batch([sentence])[0]

    def score_batch(self, sentences: Sequence[str]) -> list[Scored]:
        if not sentences:
            return []
        batch = self.tokenizer(
            list(sentences),
            truncation=True,
            max_length=self.contract["max_length"],
            padding=True,
            return_tensors="np",
        )
        logits = self.session.run(
            None,
            {
                "input_ids": batch["input_ids"].astype(np.int64),
                "attention_mask": batch["attention_mask"].astype(np.int64),
            },
        )[0].astype(np.float64)
        return [
            Scored(probabilities=row, label=self.labels[int(row.argmax())], logits=raw)
            for row, raw in zip(softmax(logits), logits, strict=True)
        ]


@dataclass(frozen=True, slots=True)
class Decision:
    """
    Mirrors `src.pipeline.segment.Decision` so either filter can drive the pipeline.

    `reason` is where the two differ honestly. The rules name the clause that fired -- `no_anchor`,
    `imperative` -- and a reader can check that judgement. The detector has no such story: it has a
    number, so it reports the number. Inventing a rule-shaped reason for a learned score would be
    the worse kind of legibility, the kind that looks explanatory and explains nothing.
    """

    worthy: bool
    reason: str
    score: float = 0.0

    def __bool__(self) -> bool:
        return self.worthy


class DetectorFilter:
    """
    The learned detector as a drop-in for the hand-written filter.

    The threshold is read from the frozen calibration artifact, never chosen here. It was swept on
    ClaimBuster's held-out debates precisely so that the 120 SOTU labels stay unspent, and a
    default that quietly differed from the artifact would undo that.

    `binarization` picks which question is being asked. `factual` matches the Phase 07 rubric --
    assertable and lookupable -- and is the like-for-like setting; `check_worthy` is ClaimBuster's
    stricter notion of worth a fact-checker's time.
    """

    def __init__(
        self,
        binarization: str = "factual",
        *,
        root: str | Path = MODEL,
        threads: int | None = None,
    ) -> None:
        if binarization not in ("factual", "check_worthy"):
            raise ValueError(f"unknown binarization {binarization!r}")
        self.binarization = binarization
        self.detector = CheckworthyDetector(root, threads=threads)
        path = Path(root) / CALIBRATION_FILE
        if not path.exists():
            raise SystemExit(
                f"{path} is missing. Fit it with `python -m scripts.calibrate_checkworthy`; "
                "the threshold must come from the calibration split, not from a default here."
            )
        frozen = json.loads(path.read_text(encoding="utf-8"))
        self.calibrator = calibrator_from_dict(frozen["calibrator"])
        self.threshold = float(frozen["thresholds"][binarization])

    def decide_batch(self, sentences: Sequence[str]) -> list[Decision]:
        if not sentences:
            return []
        scored = self.detector.score_batch(list(sentences))
        # The calibrator is applied to the logits, which is what it was fitted on. Handing it the
        # softmax would calibrate a distribution rather than the scores that produced one.
        calibrated = self.calibrator.transform(
            np.asarray([s.logits for s in scored], dtype=np.float64)
        )
        decisions = []
        for row in calibrated:
            score = float(row[1] + row[2]) if self.binarization == "factual" else float(row[2])
            worthy = score >= self.threshold
            decisions.append(Decision(
                worthy=worthy,
                reason="check_worthy" if worthy else f"below_{self.binarization}_threshold",
                score=score,
            ))
        return decisions

    def decide(self, sentence: str) -> Decision:
        return self.decide_batch([sentence])[0]
