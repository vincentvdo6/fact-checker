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
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.calibration.scaling import from_dict as calibrator_from_dict

MODEL = Path("models/checkworthy/v1")
CALIBRATION_FILE = "calibration.json"
ONNX_FILE = "onnx/model.onnx"
DEFAULT_THREADS = 4
# Sentences per forward pass. DeBERTa's disentangled attention materialises a Tile whose size grows
# with batch * sequence^2, and onnxruntime refuses any tensor over 4 GB -- so a caller that handed
# in a whole split got InvalidArgument rather than a slow answer. Chunking lives here so no caller
# has to know that, and the number is well under the limit at max_length 128.
DEFAULT_BATCH = 32

# A stated prior, not a fitted parameter, and the distinction matters.
#
# ClaimBuster holds 3 sentences of three words in 22,501 and none shorter, so the calibration split
# contains no evidence about short input at all -- every floor from 0 to 4 scores identically there.
# The 2016 State of the Union is 7.2% under four words ("Period.", "Just a guess.", "See?"). That
# is a coverage gap in the training distribution rather than a threshold to tune, and it is why the
# detector scored "I've done it." as a claim: it has never seen anything of the kind.
#
# Four words is the same floor the hand-written rules used, on the same reasoning -- a sentence
# that short cannot carry a subject, a predicate and something to look up. It is applied because
# the definition says so, not because a sweep chose it, and it is recorded that way.
MIN_WORDS = 4
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


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

    def score_batch(self, sentences: Sequence[str], *, batch: int = DEFAULT_BATCH) -> list[Scored]:
        """
        Score any number of sentences, chunked so the graph never sees an oversized tensor.

        Padding is per chunk, to the longest member of that chunk, which is also why chunking does
        not change any individual score: attention is masked either way.
        """
        # A bare str satisfies Sequence[str] and would be scored one character at a time -- 19
        # Decisions for one sentence, silently, since every downstream length check still agrees.
        if isinstance(sentences, str):
            raise TypeError("score_batch takes a sequence of sentences, not a single string")
        # Materialised once. Re-listing inside the loop cost 110x the __getitem__ calls on a lazy
        # Sequence, which is invisible on a list and quadratic on anything else.
        items = list(sentences)
        scored: list[Scored] = []
        for start in range(0, len(items), batch):
            scored.extend(self._score_chunk(items[start:start + batch]))
        return scored

    def _score_chunk(self, sentences: Sequence[str]) -> list[Scored]:
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
        min_words: int = MIN_WORDS,
    ) -> None:
        if binarization not in ("factual", "check_worthy"):
            raise ValueError(f"unknown binarization {binarization!r}")
        self.binarization = binarization
        self.min_words = min_words
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
        if isinstance(sentences, str):
            raise TypeError("decide_batch takes a sequence of sentences, not a single string")
        # len() rather than truthiness: a numpy array of sentences raises "truth value ambiguous".
        items = list(sentences)
        if len(items) == 0:
            return []
        scored = self.detector.score_batch(items)
        # The calibrator is applied to the logits, which is what it was fitted on. Handing it the
        # softmax would calibrate a distribution rather than the scores that produced one.
        calibrated = self.calibrator.transform(
            np.asarray([s.logits for s in scored], dtype=np.float64)
        )
        decisions = []
        for sentence, row in zip(items, calibrated, strict=True):
            score = float(row[1] + row[2]) if self.binarization == "factual" else float(row[2])
            # The floor is applied to the model's answer rather than instead of it, so the score is
            # still reported: a reader can see the detector was confident and was overruled.
            if len(_WORD.findall(sentence)) < self.min_words:
                decisions.append(Decision(False, "too_short", score))
                continue
            worthy = score >= self.threshold
            decisions.append(Decision(
                worthy=worthy,
                reason="check_worthy" if worthy else f"below_{self.binarization}_threshold",
                score=score,
            ))
        return decisions

    def decide(self, sentence: str) -> Decision:
        return self.decide_batch([sentence])[0]
