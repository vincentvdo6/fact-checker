"""Keep diagnostic arms faithful to their inputs and model label order."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.evidence_probe import score_case, score_inputs
from src.verdict.encode import LABELS
from src.verdict.runtime import Scored


class Runtime:
    contract = SimpleNamespace(variant="retrieved", max_length=5, labels=LABELS)

    def measure(self, first: str, second: str) -> int:
        return len((first + " " + second).split())

    def score(self, claim: str, evidence: list) -> Scored:
        assert all(row[2] != "dropped" for row in evidence)
        return Scored(np.array([1.0, 2.0, 0.0]), 5)


class Calibrator:
    def transform(self, logits: np.ndarray) -> np.ndarray:
        np.testing.assert_array_equal(logits, [[1.0, 2.0, 0.0]])
        return np.array([[0.1, 0.2, 0.7]])


def test_packed_arms_keep_original_text_and_distinguish_calibration() -> None:
    evidence = [["title", 0, "kept"], ["title", 1, "dropped"]]
    case = {"original": "original exact text", "clarified": "different exact text",
            "retrieved": evidence, "manual": evidence}
    arms = score_case(case, Runtime(), Calibrator())
    assert len(arms) == 6
    for arm in arms:
        assert arm["claim"] == case[arm["wording"]]
        assert arm["raw_prediction"] == "contradicted"
        assert arm["calibrated_prediction"] == "not_enough_evidence"
        assert arm["evidence"] == ([] if arm["evidence_arm"] == "empty" else evidence[:1])
        assert arm["encoded_evidence"] == ("" if arm["evidence_arm"] == "empty" else "title: kept")
    assert case["retrieved"] == evidence


def test_overlong_claim_cannot_silently_truncate() -> None:
    case = {"original": "one two three four five six", "clarified": "short",
            "retrieved": [], "manual": []}
    with pytest.raises(ValueError, match="Claim exceeds"):
        score_case(case, Runtime(), Calibrator())


def test_separately_trained_baseline_never_receives_evidence() -> None:
    runtime = Runtime()
    runtime.contract = SimpleNamespace(variant="claim_only", max_length=5, labels=LABELS)
    arms = score_case({"original": "original", "clarified": "clarified"}, runtime, Calibrator())
    assert len(arms) == 2
    assert all(arm["evidence"] == [] and arm["variant"] == "claim_only" for arm in arms)


def test_named_controls_preserve_the_claim_evidence_cross_product() -> None:
    calls = []
    runtime = Runtime()

    def score(claim, evidence):
        calls.append((claim, evidence))
        return Scored(np.array([1.0, 2.0, 0.0]), 5)

    runtime.score = score
    claims = {"true": "rate fell", "false": "rate rose"}
    controls = {"empty": [], "fictional": [["test", 0, "reversed"]]}
    arms = score_inputs(claims, controls, runtime, Calibrator())
    assert calls == [(claim, evidence) for claim in claims.values() for evidence in controls.values()]
    assert [(arm["wording"], arm["evidence_arm"]) for arm in arms] == [
        (wording, evidence) for wording in claims for evidence in controls
    ]


@pytest.mark.parametrize("values", [[float("nan"), 1, 2], [1, 2]])
def test_invalid_logits_cannot_acquire_a_label(values) -> None:
    runtime = Runtime()
    runtime.score = lambda *args: Scored(np.array(values), 5)
    with pytest.raises(ValueError, match="logits"):
        score_inputs({"claim": "short"}, {"empty": []}, runtime, Calibrator())


@pytest.mark.parametrize("values", [[float("nan"), 0, 1], [-1, 1, 1], [0.2, 0.2, 0.2], [0.5, 0.5]])
def test_invalid_probabilities_cannot_acquire_a_label(values) -> None:
    calibrator = Calibrator()
    calibrator.transform = lambda _: np.array([values])
    with pytest.raises(ValueError, match="probabilities"):
        score_inputs({"claim": "short"}, {"empty": []}, Runtime(), calibrator)
