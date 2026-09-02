"""
The local runtime.

Every calibration artifact in this project was fitted on logits torch produced on Kaggle, so the
one thing a local runtime must not do is score the model on text the model was never trained on.
That failure is silent -- the graph runs, the logits look plausible, every number computes -- so
the guards here are about the input path, not the arithmetic: the template comes from the shipped
encoder, the length limit comes from the contract, and padding a claim beside a longer one does
not move its verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.verdict.encode import build_input
from src.verdict.runtime import DEFAULT_THREADS, MODELS, ONNX_FILE, Scored, VerdictRuntime

RUNTIME = Path("src/verdict/runtime.py")

requires_onnx = pytest.mark.skipif(
    not (MODELS / "retrieved" / ONNX_FILE).exists(),
    reason="verdict.onnx absent; export it with notebooks/export_onnx.py",
)


# --- the input must come from the shipped encoder ----------------------------------------------

def test_the_runtime_builds_its_input_with_the_shipped_encoder():
    """
    Re-implementing the template locally is the classic train/serve skew: the model gets scored on
    text it never saw, nothing raises, and every number still computes. The Kaggle notebook called
    build_input, so this must too.
    """
    source = RUNTIME.read_text(encoding="utf-8")
    assert "from src.verdict.encode import build_input" in source
    assert "build_input(claim, evidence)" in source


def test_build_input_is_what_the_runtime_would_send():
    """The pair the tokenizer receives, checked directly rather than trusted."""
    first, second = build_input("a claim", [("Page", 0, "a sentence")])
    assert first == "a claim"
    assert "a sentence" in second


def test_an_empty_batch_scores_nothing_rather_than_erroring():
    """A transcript can filter every sentence out; that is an empty result, not a failure."""
    runtime = VerdictRuntime.__new__(VerdictRuntime)
    assert runtime.score_batch([]) == []


def test_a_missing_graph_names_the_remedy(tmp_path):
    """
    The export happens on Kaggle and the file is 738 MB, so a fresh clone will not have it. The
    error has to say what to run, not just that a path is absent.
    """
    (tmp_path / "retrieved").mkdir()
    (tmp_path / "retrieved" / "contract.json").write_text(
        json.dumps({
            "base_model": "microsoft/deberta-v3-base", "base_revision": "x",
            "tokenizer_sha256": "0" * 64, "labels": ["supported", "contradicted", "not_enough_evidence"],
            "max_length": 512, "template_id": "per_page_grouped_v1", "variant": "retrieved",
            "seed": 42, "torch_version": "2.10.0", "transformers_version": "5.0.0",
            "contract_version": 1,
        }), encoding="utf-8",
    )
    runtime = VerdictRuntime("retrieved", models=tmp_path)
    with pytest.raises(SystemExit, match="export_onnx"):
        _ = runtime.session


def test_the_contract_travels_with_the_runtime(tmp_path):
    """max_length and seed come from the artifact, never from a constant in this file."""
    source = RUNTIME.read_text(encoding="utf-8")
    assert "self.contract.max_length" in source
    assert "max_length=512" not in source


# --- the thread pool is bounded, and that is load-bearing -------------------------------------

def contract_at(root):
    """A minimal on-disk contract, so a runtime can be constructed without the 738 MB graph."""
    (root / "retrieved").mkdir(parents=True, exist_ok=True)
    (root / "retrieved" / "contract.json").write_text(
        json.dumps({
            "base_model": "microsoft/deberta-v3-base", "base_revision": "x",
            "tokenizer_sha256": "0" * 64,
            "labels": ["supported", "contradicted", "not_enough_evidence"],
            "max_length": 512, "template_id": "per_page_grouped_v1", "variant": "retrieved",
            "seed": 42, "torch_version": "2.10.0", "transformers_version": "5.0.0",
            "contract_version": 1,
        }), encoding="utf-8",
    )
    return root


def test_the_thread_pool_is_bounded_well_below_a_modern_core_count(tmp_path):
    """
    Left unbounded, onnxruntime takes every physical core and a 184M-parameter DeBERTa runs
    sustained all-core AVX matmuls. That hard-reset this machine twice while scoring the parity
    split -- no bugcheck logged either time, so a power or thermal cutout rather than a driver
    fault. The bound is not a nicety; it is why the check can finish.
    """
    assert 1 <= DEFAULT_THREADS <= 8
    assert VerdictRuntime("retrieved", models=contract_at(tmp_path)).threads == DEFAULT_THREADS


def test_the_thread_count_can_be_raised_for_hardware_that_carries_it(tmp_path, monkeypatch):
    monkeypatch.setenv("VERDICT_THREADS", "12")
    assert VerdictRuntime("retrieved", models=contract_at(tmp_path)).threads == 12


def test_an_explicit_thread_count_beats_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VERDICT_THREADS", "12")
    assert VerdictRuntime("retrieved", models=contract_at(tmp_path), threads=2).threads == 2


def test_the_session_is_built_with_that_bound_rather_than_ignoring_it():
    """
    Storing the number and then handing onnxruntime a zero would read as bounded and behave as
    unbounded -- the exact failure that took the machine down, back again and silent.
    """
    source = RUNTIME.read_text(encoding="utf-8")
    assert "options.intra_op_num_threads = self.threads" in source
    assert "intra_op_num_threads = 0" not in source


# --- against the real graph ---------------------------------------------------------------------

@requires_onnx
@pytest.mark.slow
def test_the_graph_scores_a_supported_claim_as_supported():
    """A sanity check that the export is a verdict model at all, not merely a valid ONNX file."""
    runtime = VerdictRuntime("retrieved")
    scored = runtime.score(
        "Barack Obama was born in Hawaii.",
        [("Barack Obama", 0, "Barack Obama was born in Honolulu, Hawaii, in 1961.")],
    )
    assert isinstance(scored, Scored)
    assert scored.logits.shape == (3,)
    assert int(np.argmax(scored.logits)) == 0, "supported is class 0 in the contract"


@requires_onnx
@pytest.mark.slow
def test_padding_does_not_change_a_claim_s_logits():
    """
    Batched claims pad to the longest member. If padding leaked into attention, a claim's verdict
    would depend on what happened to be batched beside it -- and on a transcript that is invisible.
    """
    runtime = VerdictRuntime("retrieved")
    claim = ("The Eiffel Tower is in Paris.", [("Eiffel Tower", 0, "The Eiffel Tower is in Paris.")])
    long = ("A longer claim about something else entirely.",
            [("Page", i, "Padding sentence with several words in it.") for i in range(12)])

    alone = runtime.score(*claim).logits
    batched = runtime.score_batch([claim, long])[0].logits
    assert alone == pytest.approx(batched, abs=1e-4)
