"""Judge each (assertion, sentence) pair with the trained four-relation cross-encoder, locally.

The contract beside the graph names the label order, the maximum length and the template;
they are checked on load because a permuted label order turns `states` into `unrelated`
without an error anywhere. With a calibration file present, probabilities are the
calibrated ones and a counted relation -- `states` or `states_negation`, the two that move
a verdict -- is kept only when its confidence reaches a counting band fitted on exactly
those predictions; below that the sentence is shown as `bears_on` and never counted. The
raw label stays on the judgment for inspection. `bears_on` and `unrelated` are never
gated, since neither moves a verdict. The judge reads one sentence with its
own-paragraph definitions against one assertion and returns no prose.
Pairs exceeding the contract's token limit are omitted with a separate per-call
audit, not labelled unrelated. Neither the claim nor its source context is clipped.
Direct logits calls reject oversized inputs instead of producing partial readings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.calibration.bands import BAND_ORDER, Band, BandPolicy
from src.calibration.scaling import from_dict as calibrator_from_dict
from src.verdict.assertions import positive_form
from src.verdict.pair_judgment import RELATIONS
from src.verdict.quantities import hedged_form

MODEL_DIR = Path("models/pair_judge/v4")     # deberta-v3-base, two epochs; v2 (xsmall) stays installed for comparison
TEMPLATE_ID = "premise_hypothesis_v1"
DEFAULT_THREADS = 4
BATCH = 16
INVERTED = {"states": "states_negation", "states_negation": "states"}


class PairJudge:
    def __init__(self, *, model_dir: str | Path = MODEL_DIR, threads: int | None = None,
                 min_band: Band | None = Band.WEAK, probe_negation: bool = False, probe_hedges: bool = False) -> None:
        self.root = Path(model_dir)
        self.threads = threads if threads is not None else int(os.environ.get("VERDICT_THREADS", DEFAULT_THREADS))
        self.min_band = min_band
        self.probe_negation = probe_negation
        # A hedged figure in the assertion is read against the sentence's own fitting figure (quantities.py).
        self.probe_hedges = probe_hedges
        contract = json.loads((self.root / "contract.json").read_text(encoding="utf-8"))
        if tuple(contract["labels"]) != RELATIONS:
            raise ValueError(f"pair judge label order {contract['labels']} differs from {list(RELATIONS)}")
        if contract["template_id"] != TEMPLATE_ID:
            raise ValueError(f"pair judge template {contract['template_id']!r}, expected {TEMPLATE_ID!r}")
        self.max_length = int(contract["max_length"])
        self.calibrator = None
        self.bands = None
        calibration = self.root / "calibration.json"
        if calibration.exists():
            payload = json.loads(calibration.read_text(encoding="utf-8"))
            self.calibrator = calibrator_from_dict(payload["calibrator"])
            self.bands = BandPolicy.from_dict(payload.get("counting_bands", payload["bands"]))
        # What the judge did when it counted on labelled news pairs it never trained on, if anyone measured it
        # (`eval_pair_judge --record`); composition passes it to the page so a count is read with its error rate.
        measurement = self.root / "news_measurement.json"
        self.measurement = json.loads(measurement.read_text(encoding="utf-8")) if measurement.exists() else None
        # What the composed reading did with this judge on recorded clicks (`measure_directions --record`):
        # the record a viewer's page quotes, since a direction, not a sentence label, is what they see.
        direction = self.root / "direction_measurement.json"
        self.direction_measurement = json.loads(direction.read_text(encoding="utf-8")) if direction.exists() else None
        self._session = None
        self._tokenizer = None
        self.last_input_overflows: list[dict] = []

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(str(self.root / "onnx/model.onnx"), options,
                                                 providers=["CPUExecutionProvider"])
        return self._session

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.root / "model_v1"))
        return self._tokenizer

    def logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        lengths = self._token_lengths(pairs)
        if any(length > self.max_length for length in lengths):
            raise ValueError("A complete pair-judge input exceeds the token limit; no input was truncated.")
        rows = []
        for start in range(0, len(pairs), BATCH):
            chunk = pairs[start:start + BATCH]
            encoded = self.tokenizer([premise for premise, _ in chunk], [hypothesis for _, hypothesis in chunk],
                                     padding=True, truncation=False, return_tensors="np")
            rows.append(self.session.run(None, {"input_ids": encoded["input_ids"].astype(np.int64),
                                                "attention_mask": encoded["attention_mask"].astype(np.int64)})[0])
        return np.concatenate(rows).astype(np.float64) if rows else np.zeros((0, len(RELATIONS)))

    def _token_lengths(self, pairs: list[tuple[str, str]]) -> list[int]:
        """Count the actual paired template, including special tokens, without clipping either text."""
        if not pairs:
            return []
        encoded = self.tokenizer([premise for premise, _ in pairs], [hypothesis for _, hypothesis in pairs],
                                 padding=False, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def probabilities(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        logits = self.logits(pairs)
        if self.calibrator is not None:
            return self.calibrator.transform(logits)
        shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
        return shifted / shifted.sum(axis=1, keepdims=True)

    def _accepts(self, band: Band | None) -> bool:
        if self.bands is None:
            return True
        if band is None or self.min_band is None:
            return band is not None
        return BAND_ORDER.index(band) <= BAND_ORDER.index(self.min_band)

    @staticmethod
    def premise(unit: dict) -> str:
        return " ".join([unit["text"], *(item["text"] for item in unit.get("definitions", []))])

    def __call__(self, assertions: list[dict], units: list[dict]) -> list[dict]:
        self.last_input_overflows = []
        pairs, keys = [], []
        for assertion in assertions:
            positive = positive_form(assertion["text"]) if self.probe_negation and assertion["negated"] else None
            for unit in units:
                premise = self.premise(unit)
                judged, figures = positive or assertion["text"], None
                if self.probe_hedges and (hedged := hedged_form(judged, premise)):
                    judged, figures = hedged
                pairs.append((premise, judged))
                keys.append((assertion, unit, positive is not None, judged, figures))
        if not pairs:
            return []
        lengths = self._token_lengths(pairs)
        valid = [index for index, length in enumerate(lengths) if length <= self.max_length]
        probabilities = self.probabilities([pairs[index] for index in valid]) if valid else ()
        if len(probabilities) != len(valid):
            raise ValueError("Pair-judge predictions do not match the number of fitting pairs.")
        scored = iter(probabilities)
        judgments = []
        for (assertion, unit, inverted, judged, figures), length in zip(keys, lengths, strict=True):
            if length > self.max_length:
                self.last_input_overflows.append({"assertion_id": assertion["id"], "unit_id": unit["id"],
                                                  "reason": "input_over_limit", "input_tokens": length,
                                                  "max_input_tokens": self.max_length})
                continue
            row = next(scored)
            index = int(np.argmax(row))
            raw = RELATIONS[index]
            relation = INVERTED.get(raw, raw) if inverted else raw
            confidence = float(row[index])
            counted = relation in INVERTED
            band = self.bands.assign(confidence) if self.bands is not None and counted else None
            final = relation if not counted or self._accepts(band) else "bears_on"
            judgment = {"assertion_id": assertion["id"], "unit_id": unit["id"], "relation": final,
                        "span": unit["text"] if final != "unrelated" else "", "qualifiers": [],
                        "confidence": confidence, "status": "unverified", "raw_label": raw,
                        "read_as": relation,      # the judge's own reading before the counting band, inversion applied
                        "judged_text": judged, "band": band.value if band else None,
                        "probabilities": [float(value) for value in row]}
            if figures:
                judgment["figures_read"] = figures      # which hedged figure was read as which sentence figure
            judgments.append(judgment)
        return judgments
