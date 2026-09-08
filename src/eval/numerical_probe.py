"""Frozen numerical minimal pairs diagnose evidence use without estimating live accuracy.

Synthetic evidence is an explicitly fictional sensitivity control. Predictions are
from the existing verdict head; no sufficiency gate or confidence-band promise applies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from src.calibration.scaling import from_dict
from src.eval.evidence_probe import score_inputs
from src.verdict.encode import MAX_SENTENCE_WORDS
from src.verdict.runtime import VerdictRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frozen = args.manifest.read_bytes()
    manifest = json.loads(frozen)
    if any(len(row[2].split()) > MAX_SENTENCE_WORDS
           for evidence in manifest["evidence_arms"].values() for row in evidence):
        raise ValueError("Diagnostic evidence would be clipped")
    for source in manifest["sources"]:
        if date.fromisoformat(source["published"]) > date.fromisoformat(manifest["as_of"]):
            raise ValueError("Source published after speech")
    runtime = VerdictRuntime("retrieved")
    calibration = (runtime.root / "calibration.json").read_bytes()
    calibrator = from_dict(json.loads(calibration)["calibrator"])
    with (runtime.root / "onnx/verdict.onnx").open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    results = []
    for pair in manifest["pairs"]:
        arms = score_inputs(pair["claims"], manifest["evidence_arms"], runtime, calibrator)
        for arm in arms:
            if arm["evidence_used"] != arm["evidence_offered"]:
                raise ValueError("Diagnostic evidence was dropped during packing")
            arm["expected_evidence_relation"] = pair["expected"][arm["evidence_arm"]][arm["wording"]]
        results.append({"id": pair["id"], "arms": arms})
        print(f"Scored {pair['id']}", flush=True)
    output = {
        "manifest_sha256": hashlib.sha256(frozen).hexdigest(), "manifest": manifest,
        "model_sha256": model_hash, "calibration_sha256": hashlib.sha256(calibration).hexdigest(),
        "contract": json.loads((runtime.root / "contract.json").read_bytes()),
        "baseline": "Empty evidence is a retrieved-model ablation, not the separately trained claim-only model.",
        "results": results,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
