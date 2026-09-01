"""
Run the trained verdict model on this machine, without torch.

Phase 02 kept torch out of the project after ROCm proved a dead end on the RX 6800, and every
model since has been trained on Kaggle. That was the right call for training and it left the demo
with nowhere to run: the weights sit in `models/verdict/<variant>/model_v1/` and nothing here can
execute them. This is the runtime that closes that gap -- an ONNX graph exported once on Kaggle,
executed locally by onnxruntime, which is a 50 MB dependency rather than a deep-learning stack.

**The input is built by the shipped encoder, not reconstructed here.** `build_input` is the same
function the training notebook called, byte-identical across the Kaggle boundary since Phase 02.
Re-implementing the template locally would be the classic train/serve skew: the model would be
scored on text it was never trained on, nothing would raise, and every number would still compute.

**Faithfulness is a precondition, not a nicety.** The calibrator's temperature and three per-class
biases, the band thresholds, and the sufficiency gate were all fitted on logits this model
produced. If the ONNX graph drifts, the bands quietly stop meaning what they measured and a
sidebar that promises "strong: right about nine times in ten" becomes false. That is why
`scripts/check_onnx_parity.py` exists and why nothing downstream is built until it passes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.verdict.contract import CONTRACT_FILE, EncoderContract
from src.verdict.encode import build_input

MODELS = Path("models/verdict")
ONNX_FILE = "onnx/verdict.onnx"

Evidence = Sequence[Sequence]


@dataclass(frozen=True, slots=True)
class Scored:
    """Raw fp32 logits and the token count the model actually read."""

    logits: np.ndarray
    token_len: int


class VerdictRuntime:
    """
    The trained model, locally, behind one call.

    Loading is deferred to first use: the graph is 738 MB and a CLI that only prints help should
    not pay for it.
    """

    def __init__(self, variant: str = "retrieved", *, models: str | Path = MODELS) -> None:
        self.root = Path(models) / variant
        self.contract = EncoderContract.from_dict(
            json.loads((self.root / CONTRACT_FILE).read_text(encoding="utf-8"))
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
                    f"{path} is missing. Export it on Kaggle with notebooks/export_onnx.py, "
                    "then place verdict.onnx here."
                )
            # Single-threaded is deliberate: a transcript is scored claim by claim, and letting
            # onnxruntime spawn a pool per call costs more than it saves at batch size one.
            options = ort.SessionOptions()
            options.intra_op_num_threads = 0        # 0 lets ORT pick from the machine
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

    def measure(self, first: str, second: str) -> int:
        """Token length of a pair, for `pack` to budget against. Same call the notebook made."""
        return len(self.tokenizer(first, second)["input_ids"])

    def score(self, claim: str, evidence: Evidence) -> Scored:
        """Logits for one claim and the evidence it was given."""
        return self.score_batch([(claim, evidence)])[0]

    def score_batch(self, pairs: Sequence[tuple[str, Evidence]]) -> list[Scored]:
        """
        Logits for several claims at once.

        Padded to the longest member rather than to `max_length`: the graph was exported with a
        dynamic sequence axis precisely so short inputs cost what they are worth, and on a
        transcript most claims retrieve far less than 512 tokens of evidence.
        """
        if not pairs:
            return []
        built = [build_input(claim, evidence) for claim, evidence in pairs]
        batch = self.tokenizer(
            [first for first, _ in built],
            [second for _, second in built],
            truncation=True,
            max_length=self.contract.max_length,
            padding=True,
            return_tensors="np",
        )
        logits = self.session.run(
            None,
            {
                "input_ids": batch["input_ids"].astype(np.int64),
                "attention_mask": batch["attention_mask"].astype(np.int64),
            },
        )[0]
        lengths = batch["attention_mask"].sum(axis=1)
        return [
            Scored(logits=row.astype(np.float64), token_len=int(length))
            for row, length in zip(logits, lengths, strict=True)
        ]
