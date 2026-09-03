"""
The learned check-worthiness detector, run locally.

The verdict runtime's lesson applies unchanged here: the input path is where a silent failure
lives. Score the model on text shaped differently from its training data and nothing raises, the
probabilities look ordinary, and every number downstream still computes. So the guards are about
the contract travelling with the model and the two binarizations meaning what they say.

`factual` and `check_worthy` are not interchangeable and the difference is the open question of
this phase -- ClaimBuster's third class means *worth a fact-checker's time*, the Phase 07 rubric
means *assertable and lookupable*. A detector that collapsed them would answer that question by
accident.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.detector import DEFAULT_THREADS, MODEL, ONNX_FILE, CheckworthyDetector, Scored

DETECTOR = Path("src/pipeline/detector.py")

requires_onnx = pytest.mark.skipif(
    not (MODEL / ONNX_FILE).exists(),
    reason="checkworthy model.onnx absent; export it with the checkworthy-onnx kernel",
)


def contract_at(root: Path) -> Path:
    (root).mkdir(parents=True, exist_ok=True)
    (root / "contract.json").write_text(
        json.dumps({
            "base_model": "microsoft/deberta-v3-base", "base_revision": "x",
            "tokenizer_sha256": "0" * 64,
            "labels": ["non_factual", "unimportant_factual", "check_worthy"],
            "max_length": 128, "template_id": "sentence_v1", "variant": "checkworthy",
            "seed": 42, "torch_version": "2.10.0", "transformers_version": "5.0.0",
            "contract_version": 1,
        }), encoding="utf-8",
    )
    return root


# --- the two binarizations ------------------------------------------------------------------------

def probabilities(non_factual: float, unimportant: float, worthy: float) -> Scored:
    row = np.array([non_factual, unimportant, worthy])
    return Scored(probabilities=row, label="check_worthy")


def test_factual_is_both_factual_classes_and_check_worthy_is_only_the_third():
    """
    Asymmetric on purpose. If the two were computed from the same classes, this phase's open
    question -- which definition the demo should gate on -- would already be answered by a bug.
    """
    scored = probabilities(0.5, 0.2, 0.3)
    assert scored.factual == pytest.approx(0.5)
    assert scored.check_worthy == pytest.approx(0.3)
    assert scored.factual != scored.check_worthy


def test_a_sentence_that_is_purely_unimportant_factual_is_factual_but_not_check_worthy():
    """The class the two label definitions disagree about, in its pure form."""
    scored = probabilities(0.0, 1.0, 0.0)
    assert scored.factual == pytest.approx(1.0)
    assert scored.check_worthy == pytest.approx(0.0)


def test_check_worthy_never_exceeds_factual():
    """It is a subset by construction; the reverse would mean the classes were wired backwards."""
    for row in ((0.7, 0.2, 0.1), (0.1, 0.1, 0.8), (0.34, 0.33, 0.33)):
        scored = probabilities(*row)
        assert scored.check_worthy <= scored.factual + 1e-12


# --- the contract travels with the model ------------------------------------------------------

def test_the_label_space_comes_from_the_artifact(tmp_path):
    detector = CheckworthyDetector(contract_at(tmp_path / "v1"))
    assert detector.labels == ("non_factual", "unimportant_factual", "check_worthy")
    assert detector.contract["max_length"] == 128


def test_the_length_limit_is_read_rather_than_hard_coded():
    source = DETECTOR.read_text(encoding="utf-8")
    assert 'self.contract["max_length"]' in source
    assert "max_length=128" not in source


def test_a_missing_graph_names_the_kernel_that_produces_it(tmp_path):
    """The export happens on Kaggle and the file is 738 MB; a fresh clone will not have it."""
    detector = CheckworthyDetector(contract_at(tmp_path / "v1"))
    with pytest.raises(SystemExit, match="checkworthy-onnx"):
        _ = detector.session


def test_an_empty_batch_scores_nothing_rather_than_erroring():
    assert CheckworthyDetector.__new__(CheckworthyDetector).score_batch([]) == []


# --- the thread pool stays bounded, for the reason the verdict runtime learned ------------------

def test_the_thread_pool_is_bounded(tmp_path):
    """
    An unbounded onnxruntime hard-reset this machine twice while scoring the parity split. The
    same graph size and the same CPU apply here.
    """
    assert 1 <= DEFAULT_THREADS <= 8
    assert CheckworthyDetector(contract_at(tmp_path / "v1")).threads == DEFAULT_THREADS


def test_the_thread_count_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("VERDICT_THREADS", "9")
    assert CheckworthyDetector(contract_at(tmp_path / "v1")).threads == 9
    assert CheckworthyDetector(contract_at(tmp_path / "v1"), threads=2).threads == 2


def test_the_session_is_built_with_that_bound():
    source = DETECTOR.read_text(encoding="utf-8")
    assert "options.intra_op_num_threads = self.threads" in source
    assert "intra_op_num_threads = 0" not in source


# --- against the real graph -----------------------------------------------------------------------

@requires_onnx
@pytest.mark.slow
def test_probabilities_are_a_distribution():
    scored = CheckworthyDetector().score("The unemployment rate fell to 3.5 percent in 2019.")
    assert scored.probabilities.shape == (3,)
    assert scored.probabilities.sum() == pytest.approx(1.0)
    assert (scored.probabilities >= 0).all()


@requires_onnx
@pytest.mark.slow
def test_the_detector_separates_a_statistic_from_a_pleasantry():
    """
    A sanity check that this is a check-worthiness model at all. Not a measurement -- that is
    scripts/compare_checkworthy.py against held-out labels.
    """
    detector = CheckworthyDetector()
    claim, greeting = detector.score_batch([
        "We're consuming 50 percent of the world's cocaine.",
        "Thank you, and God bless the United States of America.",
    ])
    assert claim.factual > greeting.factual
    assert claim.label == "check_worthy"


@requires_onnx
@pytest.mark.slow
def test_padding_does_not_change_a_sentence_s_score():
    """
    Batched sentences pad to the longest member. If padding leaked into attention, a verdict would
    depend on what happened to be batched beside it, which on a transcript is invisible.
    """
    detector = CheckworthyDetector()
    short = "Wages rose in 2019."
    alone = detector.score(short).probabilities
    batched = detector.score_batch([short, " ".join(["A much longer sentence about policy."] * 8)])
    assert alone == pytest.approx(batched[0].probabilities, abs=1e-4)


# --- the softmax guard ------------------------------------------------------------------------

def test_softmax_survives_logits_large_enough_to_overflow():
    """
    exp() overflows to inf around 710 and inf/inf is nan. A nan probability compares False against
    every threshold, so an overflowing sentence is silently dropped by the filter rather than
    raising. The row-max shift is what prevents that, and only extreme logits can show it.
    """
    from src.pipeline.detector import softmax

    got = softmax(np.array([[1000.0, 999.0, 0.0]]))
    assert np.isfinite(got).all(), "no nan or inf reached the probabilities"
    assert got.sum() == pytest.approx(1.0)
    assert int(got.argmax()) == 0


def test_softmax_is_a_distribution_per_row():
    from src.pipeline.detector import softmax

    got = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert got.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert got[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3]), "equal logits are equal probabilities"
