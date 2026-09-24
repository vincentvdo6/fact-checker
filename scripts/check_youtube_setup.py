"""Find missing local prerequisites before a viewer waits for their first check.

Read-only and network-free: no model loading, downloads, registration or calibration.
A passing inventory does not certify graph integrity, browser connectivity or accuracy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR_LABELS = ["states", "states_negation", "bears_on", "unrelated"]
DETECTOR_LABELS = ["non_factual", "unimportant_factual", "check_worthy"]
MODEL_FILES = ("contract.json", "model_v1/tokenizer.json", "model_v1/tokenizer_config.json", "onnx/model.onnx")


def inspect_setup(root: Path, environment: Mapping[str, str]) -> dict:
    """Inspect the selected configuration without mutating it or importing model runtimes."""
    checks: list[dict] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    def files(name: str, directory: Path, names: tuple[str, ...], severity: str = "error") -> bool:
        missing = [str(directory / item) for item in names
                   if not (directory / item).is_file() or (directory / item).stat().st_size == 0]
        add(name, severity if missing else "ok", "Missing or empty: " + ", ".join(missing) if missing
            else f"Required files present: {directory}")
        return not missing

    def model(name: str, directory: Path, labels: list[str], template: str, *, optional: bool = False) -> None:
        activated = all((directory / item).is_file() for item in ("contract.json", "onnx/model.onnx"))
        severity = "warning" if optional and not activated else "error"
        if not files(name, directory, MODEL_FILES, severity):
            return
        try:
            contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))
            if contract["labels"] != labels or contract["template_id"] != template:
                raise ValueError("label order or input template differs from the runtime")
            if type(contract["max_length"]) is not int or contract["max_length"] < 32:
                raise ValueError("invalid input length")
            for item in ("model_v1/tokenizer.json", "model_v1/tokenizer_config.json"):
                if not isinstance(json.loads((directory / item).read_text(encoding="utf-8")), dict):
                    raise ValueError(f"{item} must contain an object")
        except (OSError, ValueError, KeyError, TypeError) as error:
            add(name + " contract", "error", str(error))
        else:
            add(name + " contract", "ok", "Labels, template, length and tokenizer JSON checked; graph not loaded.")
        calibration(name, directory, len(labels), detector=template == "sentence_v1")

    def calibration(name: str, directory: Path, classes: int, *, detector: bool) -> None:
        if not files(name + " calibration", directory, ("calibration.json",)):
            return
        try:
            import numpy as np

            from src.calibration.bands import BandPolicy
            from src.calibration.scaling import from_dict

            value = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
            calibrator = from_dict(value["calibrator"])
            with np.errstate(all="raise"):
                probabilities = calibrator.transform(np.zeros((1, classes)))
            if probabilities.shape != (1, classes) or not np.isfinite(probabilities).all():
                raise ValueError("calibrator output does not match the model classes")
            if detector:
                threshold = value["thresholds"]["factual"]
                if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
                    raise ValueError("factual threshold must be in [0, 1]")
            else:
                BandPolicy.from_dict(value.get("counting_bands", value["bands"]))
        except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, ArithmeticError) as error:
            add(name + " calibration schema", "error", str(error))

    switches = {}
    for name, default, allowed in (
        ("WEB_SEARCH", "news", ("news", "off")), ("DECOMPOSED", "on", ("on", "off")),
        ("CONTEXT_REVIEW", "on", ("on", "off")), ("CONTEXT_ORDERING", "on", ("on", "off")),
        ("CONCEPTS", "on", ("on", "off")),
    ):
        full = "FACT_CHECKER_" + name
        switches[name] = environment.get(full, default).strip()
        add(full, "ok" if switches[name] in allowed else "error",
            switches[name] if switches[name] in allowed else f"Must be one of {', '.join(allowed)}.")
    try:
        threads = int(environment.get("VERDICT_THREADS", "4"))
        if threads < 1:
            raise ValueError
        add("inference threads", "warning" if threads > 4 else "ok", f"{threads}; recommended maximum is 4.")
    except ValueError:
        add("inference threads", "error", "VERDICT_THREADS must be a positive integer; 0 uses every core.")

    detector = root / "models/checkworthy/v1"
    model("claim detector", detector, DETECTOR_LABELS, "sentence_v1")
    if switches["DECOMPOSED"] == "on":
        configured = environment.get("FACT_CHECKER_PAIR_JUDGE", "").strip() or "models/pair_judge/v4"
        primary = root / configured
        model("sentence judge", primary, PAIR_LABELS, "premise_hypothesis_v1")
        if switches["CONTEXT_REVIEW"] == "on":
            model("context reviewer", root / "models/pair_judge/v5", PAIR_LABELS,
                  "premise_hypothesis_v1", optional=True)
        if switches["CONTEXT_ORDERING"] == "on":
            directory = root / "models/context_ranker"
            names = ("contract.json", "source-copy.json", "tokenizer.json", "ranker.onnx")
            activated = all((directory / name).is_file() for name in names)
            if files("context ordering", directory, names, "error" if activated else "warning"):
                try:
                    manifest = json.loads((directory / "source-copy.json").read_text(encoding="utf-8"))
                    names = manifest["files_sha256"]
                    if not isinstance(names, dict) or not names:
                        raise ValueError("empty graph manifest")
                    for name in names:
                        if not (directory / name).resolve().is_relative_to(directory.resolve()):
                            raise ValueError("graph file outside the model directory")
                    files("context ordering graph data", directory, tuple(names))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    add("context ordering manifest", "error", str(error))
    else:
        add("sentence-level reading", "warning", "Disabled: checks will not include a sentence-level reading.")
    if switches["CONCEPTS"] == "on" or switches["WEB_SEARCH"] == "off":
        files("Wikipedia corpus", root, ("data/fever/wiki.sqlite3",),
              "error" if switches["WEB_SEARCH"] == "off" else "warning")
    if switches["WEB_SEARCH"] == "off":
        files("offline retrieval index", root / "data/fever/index",
              ("indices.npy", "tf.npy", "indptr.npy", "doclen.npy", "meta.json"))
        files("offline verdict", root / "models/verdict/retrieved",
              ("contract.json", "calibration.json", "sufficiency.json", "onnx/verdict.onnx",
               "model_v1/tokenizer.json", "model_v1/tokenizer_config.json"))
    return {"inventory_ok": not any(row["status"] == "error" for row in checks), "checks": checks,
            "scope": "Local configuration and file inventory only; no inference, graph integrity, browser or network test."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable diagnostic report")
    args = parser.parse_args()
    report = inspect_setup(ROOT, os.environ)
    missing = [name for name in ("numpy", "scipy", "transformers", "tokenizers", "onnxruntime", "sentencepiece")
               if importlib.util.find_spec(name) is None]
    report["checks"].append({"name": "Python runtime", "status": "error" if missing or sys.version_info < (3, 12) else "ok",
                             "detail": f"{sys.executable}; Python 3.12+ required; missing packages: {', '.join(missing) or 'none'}"})
    report["inventory_ok"] = not any(row["status"] == "error" for row in report["checks"])
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for row in report["checks"]:
            print(f"{row['status'].upper()}: {row['name']}: {row['detail']}")
        print(report["scope"])
    return 0 if report["inventory_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
