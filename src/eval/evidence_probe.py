"""Inspect selected failures without treating manual evidence as a retrieval or accuracy test.

Run with a frozen JSON manifest and output path. All arms retain the encoded inputs;
the empty-input ablation and separately trained claim-only baseline stay distinct.
No sufficiency gate or FEVER confidence-band promise applies to this diagnostic.
"""

from __future__ import annotations

import numpy as np

from src.verdict.encode import build_input, pack
from src.verdict.runtime import VerdictRuntime


def score_case(case: dict, runtime: VerdictRuntime, calibrator: object) -> list[dict]:
    """Cross wording with evidence; store the prefix actually visible to the model."""
    evidence_arms = {"empty": []}
    if runtime.contract.variant == "retrieved":
        evidence_arms.update(retrieved=case["retrieved"], manual=case["manual"])
    return score_inputs({name: case[name] for name in ("original", "clarified")}, evidence_arms, runtime, calibrator)


def score_inputs(claims: dict[str, str], evidence_arms: dict[str, list],
                 runtime: VerdictRuntime, calibrator: object) -> list[dict]:
    """Score explicit named inputs, preserving each claim and packed evidence arm."""
    results = []
    for wording, claim in claims.items():
        for arm, evidence in evidence_arms.items():
            used = evidence[:pack(claim, evidence, runtime.contract.max_length, runtime.measure)]
            first, second = build_input(claim, used)
            if runtime.measure(first, second) > runtime.contract.max_length:
                raise ValueError("Claim exceeds model budget")
            scored = runtime.score(claim, used)
            if scored.logits.shape != (len(runtime.contract.labels),) or not np.isfinite(scored.logits).all():
                raise ValueError("Invalid diagnostic logits")
            probabilities = calibrator.transform(scored.logits[None, :])[0]
            if (probabilities.shape != scored.logits.shape or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1)):
                raise ValueError("Invalid diagnostic probabilities")
            results.append({
                "variant": runtime.contract.variant, "wording": wording, "evidence_arm": arm,
                "claim": first, "encoded_evidence": second, "evidence": used,
                "evidence_offered": len(evidence), "evidence_used": len(used),
                "token_len": scored.token_len, "logits": scored.logits.tolist(),
                "raw_prediction": runtime.contract.labels[int(scored.logits.argmax())],
                "calibrated_prediction": runtime.contract.labels[int(probabilities.argmax())],
                "calibrated_scores": probabilities.tolist(),
            })
    return results
