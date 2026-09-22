"""Lazy local inference for the exact full-precision ordering experiment.

The original contract, cloud graph manifest and lossless tokenizer serialization are
pinned independently. No artifact is downloaded or converted here. An oversized input
prevents the complete ordering pass before model loading; source text is never clipped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.verdict.context_order import ContextOrderingUnavailable

if TYPE_CHECKING:
    import onnxruntime as ort

CONTRACT_SHA256 = "e77254b204df5d4377c9b39a9ada6047370b950b42fab8f75818c7266adab939"
MANIFEST_SHA256 = "2399414d72030760d7c8475c08225da92567433a7a73565892e6b5161cca8e51"
TOKENIZER_SHA256 = "6238ff86c1fe4e152c931cc72bb99e01baf8dc38a31bcea0078a30c447ee2038"
BATCH = 4
MODEL_DIR = Path("models/context_ranker")


def verify_file(path: Path, expected: str) -> None:
    try:
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
    except FileNotFoundError as error:
        raise ContextOrderingUnavailable(f"Context ordering artifact is missing: {path.name}.") from error
    if actual != expected:
        raise ValueError(f"Context ordering artifact hash differs: {path.name}.")


def graph_files(root: Path) -> dict[str, str]:
    verify_file(root / "source-copy.json", MANIFEST_SHA256)
    manifest = json.loads((root / "source-copy.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Context ordering graph files must stay inside the model directory.")
        verify_file(path, expected)
    return manifest["files_sha256"]


class LocalContextRanker:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._contract = self._tokenizer = self._session = None

    def _prepare_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        from tokenizers import Tokenizer

        verify_file(self.root / "contract.json", CONTRACT_SHA256)
        verify_file(self.root / "tokenizer.json", TOKENIZER_SHA256)
        contract = json.loads((self.root / "contract.json").read_text(encoding="utf-8"))
        tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        for part in ("prefix", "suffix"):
            if tokenizer.encode(contract[part], add_special_tokens=False).ids != contract[part + "_ids"]:
                raise ValueError("Context ordering tokenizer changed the frozen prompt tokens.")
        self._contract, self._tokenizer = contract, tokenizer

    @property
    def session(self) -> ort.InferenceSession:
        if self._session is None:
            import onnxruntime as ort

            graph_files(self.root)
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            providers = ["CPUExecutionProvider"]
            if "DmlExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "DmlExecutionProvider")
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(str(self.root / "ranker.onnx"), options, providers=providers)
        return self._session

    @property
    def providers(self) -> list[str]:
        return self._session.get_providers() if self._session is not None else []

    def sequences(self, rows: list[dict]) -> list[list[int]]:
        self._prepare_tokenizer()
        contract = self._contract
        sequences = []
        for row in rows:
            document = "Target sentence: " + row["visible_sentence"] + (
                "\nAttached definitions: " + " ".join(row["visible_definitions"]) if row["visible_definitions"] else "")
            text = f"<Instruct>: {contract['instruction']}\n<Query>: {row['hypothesis']}\n<Document>: " + document
            tokens = contract["prefix_ids"] + self._tokenizer.encode(text, add_special_tokens=False).ids + contract["suffix_ids"]
            if len(tokens) > contract["max_length"]:
                raise ContextOrderingUnavailable("A complete context-ordering input exceeds the token limit; original order retained.")
            sequences.append(tokens)
        return sequences

    def batch_inputs(self, sequences: list[list[int]]) -> dict[str, np.ndarray]:
        length = max(map(len, sequences))
        ids = np.full((len(sequences), length), self._contract["pad_token_id"], dtype=np.int64)
        mask = np.zeros_like(ids)
        for index, tokens in enumerate(sequences):
            ids[index, -len(tokens):], mask[index, -len(tokens):] = tokens, 1
        return {"input_ids": ids, "attention_mask": mask}

    def __call__(self, rows: list[dict]) -> list[float]:
        if not rows:
            return []
        sequences = self.sequences(rows)
        scores = []
        for start in range(0, len(sequences), BATCH):
            chunk = sequences[start:start + BATCH]
            logits = np.asarray(self.session.run(None, self.batch_inputs(chunk))[0], dtype=np.float64)
            if logits.shape != (len(chunk), 2) or not np.isfinite(logits).all():
                raise ValueError("Context ordering requires two finite logits per pair, in no/yes order.")
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            scores.extend((probabilities[:, 1] / probabilities.sum(axis=1)).tolist())
        return scores
