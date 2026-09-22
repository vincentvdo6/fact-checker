"""No model inference: input, artifact and runtime selection contracts only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.verdict import context_ranker
from src.verdict.context_order import ContextOrderingUnavailable
from src.verdict.context_ranker import LocalContextRanker, verify_file


class Tokenizer:
    def __init__(self):
        self.seen = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.seen.append(text)
        return SimpleNamespace(ids=list(range(len(text))))


def prepared(tmp_path):
    scorer = LocalContextRanker(tmp_path)
    scorer._contract = {"instruction": "Frozen instruction.", "prefix_ids": [90], "suffix_ids": [91, 92],
                        "max_length": 1024, "pad_token_id": 99}
    scorer._tokenizer = Tokenizer()
    return scorer


def row(text="A measurement."):
    return {"hypothesis": "The rate did not increase.", "visible_sentence": text, "visible_definitions": []}


def test_constructor_and_empty_rows_are_lazy(tmp_path):
    scorer = LocalContextRanker(tmp_path / "not-present")
    assert scorer([]) == []
    assert scorer._contract is scorer._tokenizer is scorer._session is None


def test_exact_template_and_left_padding_keep_last_suffix(tmp_path):
    scorer = prepared(tmp_path)
    rows = [row(), row("Another measurement.") | {"visible_definitions": ["First definition.", "Second definition."]}]
    sequences = scorer.sequences(rows)
    assert scorer._tokenizer.seen == [
        "<Instruct>: Frozen instruction.\n<Query>: The rate did not increase.\n<Document>: Target sentence: A measurement.",
        "<Instruct>: Frozen instruction.\n<Query>: The rate did not increase.\n<Document>: Target sentence: Another measurement."
        "\nAttached definitions: First definition. Second definition."]
    inputs = scorer.batch_inputs(sequences)
    assert inputs["input_ids"].dtype == inputs["attention_mask"].dtype == np.int64
    padding = len(sequences[1]) - len(sequences[0])
    assert inputs["input_ids"][0].tolist() == [99] * padding + sequences[0]
    assert inputs["attention_mask"][0].tolist() == [0] * padding + [1] * len(sequences[0])
    assert inputs["attention_mask"][1].tolist() == [1] * len(sequences[1])
    assert inputs["input_ids"][:, -2:].tolist() == [[91, 92], [91, 92]]


def test_one_oversized_input_stops_entire_pass_before_inference(tmp_path):
    scorer = prepared(tmp_path)
    with pytest.raises(ContextOrderingUnavailable, match="complete.*token limit"):
        scorer([row()] * 4 + [row("x" * 1024)])
    assert scorer._session is None


def test_batch_four_and_float64_softmax_with_large_logits(tmp_path):
    scorer = prepared(tmp_path)
    calls = []

    def run(outputs, inputs):
        calls.append(inputs)
        return [np.tile(np.array([1000, 1001], dtype=np.float32), (len(inputs["input_ids"]), 1))]

    scorer._session = SimpleNamespace(run=run)
    actual = scorer([row()] * 5)
    assert [len(call["input_ids"]) for call in calls] == [4, 1]
    assert actual == [float(1 / (1 + np.exp(np.float64(-1))))] * 5
    assert all(type(score) is float for score in actual)


@pytest.mark.parametrize("values", [[[0, float("nan")]], [[0, float("inf")]], [[0, 1, 2]], []])
def test_invalid_logits_are_visible_errors(tmp_path, values):
    scorer = prepared(tmp_path)
    scorer._session = SimpleNamespace(run=lambda outputs, inputs: [values])
    with pytest.raises(ValueError, match="two finite logits"):
        scorer([row()])


def test_hashes_reject_missing_and_changed_artifacts(tmp_path):
    path = tmp_path / "artifact"
    with pytest.raises(ContextOrderingUnavailable, match="missing"):
        verify_file(path, "unused")
    path.write_bytes(b"frozen")
    digest = hashlib.sha256(b"frozen").hexdigest()
    verify_file(path, digest)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        verify_file(path, digest)


@pytest.mark.parametrize("part", ["prefix", "suffix"])
def test_prompt_tokens_must_match_the_contract(tmp_path, monkeypatch, part):
    import tokenizers

    contract = {"prefix": "start", "prefix_ids": [1], "suffix": "end", "suffix_ids": [2]}
    contract[part + "_ids"] = [99]
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    (tmp_path / "tokenizer.json").write_text("{}")
    for name, key in (("contract.json", "CONTRACT_SHA256"), ("tokenizer.json", "TOKENIZER_SHA256")):
        monkeypatch.setattr(context_ranker, key, hashlib.sha256((tmp_path / name).read_bytes()).hexdigest())
    tokenizer = SimpleNamespace(no_padding=lambda: None, no_truncation=lambda: None,
                                encode=lambda text, **kw: SimpleNamespace(ids=[1 if text == "start" else 2]))
    monkeypatch.setattr(tokenizers, "Tokenizer", SimpleNamespace(from_file=lambda path: tokenizer))
    scorer = LocalContextRanker(tmp_path)
    with pytest.raises(ValueError, match="frozen prompt tokens"):
        scorer.sequences([row()])
    assert scorer._session is None


def test_manifest_cannot_reference_a_file_outside_the_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (tmp_path / "outside").write_bytes(b"tensor")
    manifest = bundle / "source-copy.json"
    manifest.write_text(json.dumps({"files_sha256": {"../outside": hashlib.sha256(b"tensor").hexdigest()}}))
    monkeypatch.setattr(context_ranker, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="inside the model directory"):
        context_ranker.graph_files(bundle)


@pytest.mark.parametrize("dml", [False, True])
def test_provider_selection_and_pinned_graph_verification(tmp_path, monkeypatch, dml):
    import onnxruntime as ort

    graph = tmp_path / "ranker.onnx"
    graph.write_bytes(b"frozen graph")
    manifest = tmp_path / "source-copy.json"
    manifest.write_text(json.dumps({"files_sha256": {graph.name: hashlib.sha256(graph.read_bytes()).hexdigest()}}))
    monkeypatch.setattr(context_ranker, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["DmlExecutionProvider"] if dml else ["CPUExecutionProvider"])
    calls = []

    def create(path, options, providers):
        calls.append((path, options, providers))
        return "session"

    monkeypatch.setattr(ort, "InferenceSession", create)
    scorer = LocalContextRanker(tmp_path)
    assert scorer.session == scorer.session == "session" and len(calls) == 1
    path, options, providers = calls[0]
    assert Path(path) == graph
    assert options.intra_op_num_threads == 4 and options.inter_op_num_threads == 1
    assert providers == (["DmlExecutionProvider", "CPUExecutionProvider"] if dml else ["CPUExecutionProvider"])
    if dml:
        assert options.enable_mem_pattern is False and options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    graph.write_bytes(b"changed graph")
    with pytest.raises(ValueError, match="hash differs"):
        LocalContextRanker(tmp_path).session
    assert len(calls) == 1, "a changed graph must never reach the inference runtime"


@pytest.mark.slow
@pytest.mark.skipif(not (context_ranker.MODEL_DIR / "ranker.onnx").is_file(), reason="context ranker is not installed")
def test_installed_graph_matches_frozen_export_scores():
    scorer = LocalContextRanker(context_ranker.MODEL_DIR)
    rows = [{"hypothesis": "Riverton's water loss rate was about 25% in 2024.",
             "visible_sentence": sentence, "visible_definitions": definitions} for sentence, definitions in [
        ("Riverton's 2024 water loss rate was 24.8%.", []),
        ("Riverton's 2024 water loss rate was 8%, not the previously reported 25%.", []),
        ("Riverton's Water Institute was founded in 1998 by Maria Cole.",
         ["Water loss rate measures water entering the network that does not reach customers."])]]
    # FP32 cloud export fixtures: numerical parity, not a relevance accuracy measure.
    logits = np.array([[15.317933082580566, 20.269041061401367],
                       [14.870767593383789, 20.343503952026367],
                       [16.53447914123535, 20.107364654541016]])
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    assert scorer(rows) == pytest.approx(exps[:, 1] / exps.sum(axis=1), abs=.0005)
    assert scorer.providers and "CPUExecutionProvider" in scorer.providers
