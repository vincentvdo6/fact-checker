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

MODEL = Path("models/checkworthy/v1")
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
            Scored(probabilities=row, label=self.labels[int(row.argmax())])
            for row in softmax(logits)
        ]
