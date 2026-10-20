"""Let a large NLI model confirm supports the pair judge read just below its counting band.

On captions the pair judge missed plain paraphrases: "It was the third time in five months that
the economy lost jobs." reads as stating "It's also the third time in the last 5 months our
economy has lost jobs" at 0.83, under its 0.854 band, so it was shown and never counted. Lowering
the band admits mostly misreadings. A second reader trained on more NLI data (MNLI, FEVER, ANLI,
LingNLI, WANLI) is therefore asked only about those near-misses, and only in one direction:

- A pair the pair judge reads as stating the assertion with at least `SUPPORT_LOWER` confidence,
  but below its band, counts as stating it when the NLI model gives entailment of the text the
  judge saw (contradiction, for a negated assertion judged in its positive form) >= `SUPPORT_THRESHOLD`.
- Nothing else changes: the pair judge's own counts stand, and the NLI model never proposes a
  pair the pair judge read as context or vetoes a count. Both were tried on held-out claims and
  added wrong readings: a veto removed a right support and left a wrong contradiction standing,
  and proposals accepted altered claims ("cancelled permanently" for a suspension).
- Contradictions are never admitted this way: every wrong composed reading measured so far has
  been a contradiction.

So the stage can only add a support both readers agree on. The NLI probabilities are kept on the
judgment for inspection, and composition's rules apply unchanged. `direction_measurement.json`
beside the model records the held-out test of the whole system, and the card quotes it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.verdict.pair_judge import PairJudge

MODEL_DIR = Path("models/nli_confirm/deberta-v3-large-mfalw")
REPO, REVISION = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
# The tested files, by SHA-256; the installer refuses anything else.
FILES = {
    "onnx/model.onnx": "9f304954c0c3ed1eda620eecd8747f3f79f98e47d7593baf3160b0fef5e3bfbd",
    "tokenizer.json": "7aa118770f066a74530d161c7d0b994d0629cc0ff3a0df213f184192773f960a",
    "config.json": "7f9d420b616691c5b575bab8839719aefbddc656d8e1316519414b097ce43e3d",
    "tokenizer_config.json": "556a192d83e3edb45a09a6207095dd9295c3b2279062e5cc430feac4cb7a0817",
    "special_tokens_map.json": "fcfbeb4e25cd37fb5b8030406394253b245230fd3b552750092bab7d24d827da",
    "added_tokens.json": "cd0141b19e809eb963a8f8244fad72530233f729e1f6960d281e1fbb3537b2fb",
}
SUPPORT_LOWER, SUPPORT_THRESHOLD = 0.5, 0.95
MAX_LENGTH = 512
DEFAULT_THREADS = 4     # never every core: sustained all-core inference has hard-reset this machine


class NliRuntime:
    """The NLI model's ONNX graph and tokenizer, loaded on first use; long pairs are refused, not cut."""

    def __init__(self, model_dir: str | Path = MODEL_DIR, *, threads: int | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.threads = threads or int(os.environ.get("VERDICT_THREADS", DEFAULT_THREADS))
        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.labels = [config["id2label"][str(index)].lower() for index in range(len(config["id2label"]))]
        if sorted(self.labels) != ["contradiction", "entailment", "neutral"]:
            raise ValueError(f"Unexpected NLI labels: {self.labels}")
        self._session = self._tokenizer = None

    def _load(self) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = self.threads, 1
        self._session = ort.InferenceSession(str(self.model_dir / "onnx/model.onnx"), options,
                                             providers=["CPUExecutionProvider"])
        self._inputs = {item.name for item in self._session.get_inputs()}

    def __call__(self, pairs: list[tuple[str, str]]) -> list[dict[str, float] | None]:
        """Label probabilities per pair, or None for a pair longer than the model reads."""
        if not pairs:
            return []
        if self._session is None:
            self._load()
        lengths = [len(ids) for ids in self._tokenizer([p for p, _ in pairs], [h for _, h in pairs],
                                                        truncation=False)["input_ids"]]
        fitting = [index for index, length in enumerate(lengths) if length <= MAX_LENGTH]
        results: list[dict[str, float] | None] = [None] * len(pairs)
        for start in range(0, len(fitting), 8):
            chunk = fitting[start:start + 8]
            encoded = self._tokenizer([pairs[i][0] for i in chunk], [pairs[i][1] for i in chunk], padding=True,
                                      truncation=False, return_tensors="np")
            feed = {name: encoded[name].astype(np.int64) for name in encoded if name in self._inputs}
            logits = self._session.run(None, feed)[0]
            shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
            for index, row in zip(chunk, shifted / shifted.sum(axis=1, keepdims=True), strict=True):
                results[index] = {label: float(row[position]) for position, label in enumerate(self.labels)}
        return results


class SupportConfirmedJudge:
    """The pair judge, plus the near-miss supports an NLI model confirms; see the module docstring."""

    def __init__(self, primary: PairJudge, nli, *, model_dir: str | Path = MODEL_DIR) -> None:
        self.primary, self.nli = primary, nli
        self.measurement = primary.measurement
        record = Path(model_dir) / "direction_measurement.json"
        # The record must describe the system that produced the reading: this one, not the pair judge alone.
        self.direction_measurement = (json.loads(record.read_text(encoding="utf-8")) if record.is_file()
                                      else None)
        self.last_input_overflows: list[dict] = []

    def __call__(self, assertions: list[dict], units: list[dict]) -> list[dict]:
        rows = self.primary(assertions, units)
        self.last_input_overflows = list(getattr(self.primary, "last_input_overflows", []))
        by_unit = {unit["id"]: unit for unit in units}
        near = [row for row in rows if row.get("read_as") == "states" and row["relation"] == "bears_on"
                and row["confidence"] >= SUPPORT_LOWER]
        readings = self.nli([(PairJudge.premise(by_unit[row["unit_id"]]), row["judged_text"]) for row in near])
        for row, probabilities in zip(near, readings, strict=True):
            if probabilities is None:
                self.last_input_overflows.append({"assertion_id": row["assertion_id"], "unit_id": row["unit_id"],
                                                  "reason": "input_over_limit", "stage": "nli_confirm"})
                continue
            row["nli"] = probabilities
            # The judge's reading was of the text it saw; for a positive form, stating it denies the original.
            wanted = "entailment" if row["raw_label"] == "states" else "contradiction"
            if probabilities[wanted] >= SUPPORT_THRESHOLD:
                row["relation"] = "states"
        return rows
