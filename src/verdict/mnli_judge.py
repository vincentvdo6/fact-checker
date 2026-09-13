"""Judge each (assertion, sentence) pair with a small general-domain NLI cross-encoder.

Premise is the sentence with its own-paragraph definitions; hypothesis is the assertion,
or its mechanical positive form when the assertion is a denial. Entailment becomes
`states`, contradiction `states_negation`, neutral `unrelated`, and a denial's relations
are inverted back in code. The model is trained on SNLI and MNLI, not on this project's
data, so its softmax is uncalibrated here: the argmax is reported with its probability
and nothing about accuracy is promised until it is measured.

Topical bearing is a second question to the same model, asked the zero-shot way: does
the sentence entail "This text is about <the assertion's object>"? A neutral pair whose
sentence is about the assertion's subject is returned as `bears_on`, which composition
shows and never counts. The object is an exact span of the assertion, never a rewrite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.verdict.assertions import object_phrase, positive_form
from src.verdict.pair_judgment import nli_judgment

MODEL_DIR = Path("models/nli/nli-deberta-v3-xsmall")
LABELS = ("contradiction", "entailment", "neutral")
TO_FEVER = {"entailment": "supported", "contradiction": "contradicted", "neutral": "not_enough_evidence"}
INVERTED = {"supported": "contradicted", "contradicted": "supported"}
DEFAULT_THREADS = 4
BATCH = 16


class MnliJudge:
    def __init__(self, *, model_dir: str | Path = MODEL_DIR, threads: int | None = None,
                 bearing: bool = True) -> None:
        self.root = Path(model_dir)
        self.threads = threads if threads is not None else int(os.environ.get("VERDICT_THREADS", DEFAULT_THREADS))
        self.bearing = bearing
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        labels = tuple(config["id2label"][str(index)] for index in range(len(config["id2label"])))
        if labels != LABELS:
            raise ValueError(f"Unexpected NLI label order: {labels}")
        self._session = None
        self._tokenizer = None

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads       # bounded for the same reason VerdictRuntime is
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(str(self.root / "onnx/model.onnx"), options,
                                                 providers=["CPUExecutionProvider"])
        return self._session

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.root))
        return self._tokenizer

    def probabilities(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Softmax over (contradiction, entailment, neutral) for each (premise, hypothesis)."""
        rows = []
        for start in range(0, len(pairs), BATCH):
            chunk = pairs[start:start + BATCH]
            encoded = self.tokenizer([premise for premise, _ in chunk], [hypothesis for _, hypothesis in chunk],
                                     padding=True, truncation=True, max_length=512, return_tensors="np")
            logits = self.session.run(None, {"input_ids": encoded["input_ids"].astype(np.int64),
                                             "attention_mask": encoded["attention_mask"].astype(np.int64)})[0]
            shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
            rows.append(shifted / shifted.sum(axis=1, keepdims=True))
        return np.concatenate(rows) if rows else np.zeros((0, 3))

    @staticmethod
    def premise(unit: dict) -> str:
        return " ".join([unit["text"], *(item["text"] for item in unit.get("definitions", []))])

    def __call__(self, assertions: list[dict], units: list[dict]) -> list[dict]:
        pairs, topics, keys = [], [], []
        for assertion in assertions:
            probe = positive_form(assertion["text"]) if assertion["negated"] else None
            topic = object_phrase(assertion["text"])
            for unit in units:
                premise = self.premise(unit)
                pairs.append((premise, probe or assertion["text"]))
                topics.append((premise, f"This text is about {topic or assertion['text']}"))
                keys.append((assertion, unit, probe, topic))
        if not pairs:
            return []
        entail = self.probabilities(pairs)
        about = self.probabilities(topics) if self.bearing else np.zeros_like(entail)
        judgments = []
        for (assertion, unit, probe, topic), row, aboutness in zip(keys, entail, about, strict=True):
            label = TO_FEVER[LABELS[int(np.argmax(row))]]
            effective = INVERTED.get(label, label) if probe else label
            judgment = nli_judgment(assertion, unit, effective, float(row.max()))
            bearing = bool(self.bearing and aboutness[1] > max(aboutness[0], aboutness[2]))
            if judgment["relation"] == "unrelated" and bearing:
                judgment = judgment | {"relation": "bears_on", "span": unit["text"]}
            judgments.append(judgment | {"raw_label": label, "judged_text": probe or assertion["text"],
                                         "probabilities": [float(value) for value in row],
                                         "topic": topic or assertion["text"],
                                         "about": [float(value) for value in aboutness]})
        return judgments
