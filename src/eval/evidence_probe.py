"""Inspect selected failures without treating manual evidence as a retrieval or accuracy test.

Run with a frozen JSON manifest and output path. All arms retain the encoded inputs;
the empty-input ablation and separately trained claim-only baseline stay distinct.
No sufficiency gate or FEVER confidence-band promise applies to this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from src.calibration.scaling import from_dict
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    cutoff = date.fromisoformat(manifest["as_of"])
    for case in manifest["cases"]:
        for source in case["sources"]:
            if date.fromisoformat(source["published"]) > cutoff:
                raise ValueError(f"Future evidence: {source['url']}")
    output = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest": manifest, "models": {}, "results": [],
    }
    for variant in ("retrieved", "claim_only"):
        runtime = VerdictRuntime(variant)
        if not (runtime.root / "onnx/verdict.onnx").exists():
            output["models"][variant] = {"status": "unavailable", "reason": "Local ONNX export missing"}
            continue
        calibration_bytes = (runtime.root / "calibration.json").read_bytes()
        calibrator = from_dict(json.loads(calibration_bytes)["calibrator"])
        with (runtime.root / "onnx/verdict.onnx").open("rb") as model_file:
            model_hash = hashlib.file_digest(model_file, "sha256").hexdigest()
        output["models"][variant] = {
            "onnx_sha256": model_hash,
            "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
            "contract": json.loads((runtime.root / "contract.json").read_bytes()),
        }
        for case in manifest["cases"]:
            output["results"].append({"id": case["id"], "arms": score_case(case, runtime, calibrator)})
            print(f"Scored {variant}: {case['id']}", flush=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
