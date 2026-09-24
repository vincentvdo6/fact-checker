"""Installation diagnostics must distinguish degraded features from unusable checks."""

from __future__ import annotations

import json

import pytest

from scripts.check_youtube_setup import DETECTOR_LABELS, MODEL_FILES, PAIR_LABELS, inspect_setup


def bundle(root, path, labels, template):
    directory = root / path
    for name in MODEL_FILES:
        file = directory / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("{}")
    (directory / "contract.json").write_text(json.dumps({"labels": labels, "template_id": template, "max_length": 192}))
    bands = {"thresholds": {}, "within": {}, "cumulative": {}, "support": {}, "targets": {}, "min_support": 30,
             "fitted_on": 1}
    (directory / "calibration.json").write_text(json.dumps({"thresholds": {"factual": 0.5},
        "calibrator": {"name": "uncalibrated"}, "bands": bands}))
    return directory


@pytest.fixture
def installed(tmp_path):
    bundle(tmp_path, "models/checkworthy/v1", DETECTOR_LABELS, "sentence_v1")
    bundle(tmp_path, "models/pair_judge/v4", PAIR_LABELS, "premise_hypothesis_v1")
    return tmp_path


def test_news_inventory_does_not_require_old_offline_index(installed):
    before = sorted(str(path) for path in installed.rglob("*"))
    report = inspect_setup(installed, {})
    assert report["inventory_ok"]
    assert not any(row["name"] == "offline retrieval index" for row in report["checks"])
    assert any(row["name"] == "context reviewer" and row["status"] == "warning" for row in report["checks"])
    assert before == sorted(str(path) for path in installed.rglob("*"))


@pytest.mark.parametrize("name,value", [("WEB_SEARCH", "brave"), ("DECOMPOSED", "maybe"),
                                       ("CONTEXT_REVIEW", "true"), ("CONTEXT_ORDERING", "false"), ("CONCEPTS", "yes")])
def test_invalid_switch_fails_before_click(installed, name, value):
    assert not inspect_setup(installed, {"FACT_CHECKER_" + name: value})["inventory_ok"]


@pytest.mark.parametrize("threads", ["0", "-1", "many"])
def test_unbounded_or_invalid_threads_fail(installed, threads):
    assert not inspect_setup(installed, {"VERDICT_THREADS": threads})["inventory_ok"]


def test_offline_mode_requires_corpus_index_and_verdict(installed):
    report = inspect_setup(installed, {"FACT_CHECKER_WEB_SEARCH": "off", "FACT_CHECKER_CONCEPTS": "off"})
    assert not report["inventory_ok"]
    errors = {row["name"] for row in report["checks"] if row["status"] == "error"}
    assert {"Wikipedia corpus", "offline retrieval index", "offline verdict"} <= errors


def test_custom_judge_path_and_wrong_label_order(installed):
    custom = bundle(installed, "custom", list(reversed(PAIR_LABELS)), "premise_hypothesis_v1")
    report = inspect_setup(installed, {"FACT_CHECKER_PAIR_JUDGE": str(custom)})
    assert not report["inventory_ok"]
    assert any(row["name"] == "sentence judge contract" and row["status"] == "error" for row in report["checks"])


@pytest.mark.parametrize("file,content", [("contract.json", "null"), ("model_v1/tokenizer.json", "[]"),
                                         ("model_v1/tokenizer_config.json", "bad JSON"), ("onnx/model.onnx", "")])
def test_broken_required_artifacts_fail(installed, file, content):
    (installed / "models/pair_judge/v4" / file).write_text(content)
    assert not inspect_setup(installed, {})["inventory_ok"]


def test_activated_reviewer_missing_tokenizer_is_error(installed):
    reviewer = bundle(installed, "models/pair_judge/v5", PAIR_LABELS, "premise_hypothesis_v1")
    (reviewer / "model_v1/tokenizer.json").unlink()
    assert not inspect_setup(installed, {})["inventory_ok"]
    assert inspect_setup(installed, {"FACT_CHECKER_CONTEXT_REVIEW": "off"})["inventory_ok"]


def test_ranker_requires_external_data_and_rejects_manifest_escape(installed):
    directory = installed / "models/context_ranker"
    directory.mkdir()
    for name in ("contract.json", "source-copy.json", "tokenizer.json", "ranker.onnx"):
        (directory / name).write_text("{}")
    manifest = directory / "source-copy.json"
    manifest.write_text(json.dumps({"files_sha256": {"weights.data": "hash"}}))
    assert not inspect_setup(installed, {})["inventory_ok"]
    (directory / "weights.data").write_bytes(b"fixture")
    assert inspect_setup(installed, {})["inventory_ok"]
    outside = directory.parent / "v4/model.onnx"
    outside.parent.mkdir()
    outside.write_bytes(b"fixture")
    manifest.write_text(json.dumps({"files_sha256": {"../v4/model.onnx": "hash"}}))
    assert not inspect_setup(installed, {})["inventory_ok"]


def test_empty_activated_ranker_is_error(installed):
    directory = installed / "models/context_ranker"
    directory.mkdir()
    for name in ("contract.json", "source-copy.json", "tokenizer.json", "ranker.onnx"):
        (directory / name).write_text("")
    assert not inspect_setup(installed, {})["inventory_ok"]


@pytest.mark.parametrize("threshold", [None, True, -0.1, 1.1, float("nan")])
def test_missing_or_invalid_detector_threshold_fails(installed, threshold):
    path = installed / "models/checkworthy/v1/calibration.json"
    path.write_text(json.dumps({"thresholds": {"factual": threshold}, "calibrator": {"name": "uncalibrated"}}))
    assert not inspect_setup(installed, {})["inventory_ok"]


@pytest.mark.parametrize("content", ["bad JSON", "{}", '{"calibrator": {}}',
                                     '{"calibrator": {"name": "uncalibrated"}, "bands": {}}'])
def test_malformed_judge_calibration_fails(installed, content):
    (installed / "models/pair_judge/v4/calibration.json").write_text(content)
    assert not inspect_setup(installed, {})["inventory_ok"]


def test_reviewer_requires_calibration_when_activated(installed):
    reviewer = bundle(installed, "models/pair_judge/v5", PAIR_LABELS, "premise_hypothesis_v1")
    (reviewer / "calibration.json").unlink()
    assert not inspect_setup(installed, {})["inventory_ok"]


@pytest.mark.parametrize("calibrator", [{}, {"name": "unknown"}, {"name": "vector_scaling", "temperature": 1,
                                                                   "bias": [0, 0, 0]}])
def test_judge_calibrator_must_load_with_its_class_count(installed, calibrator):
    path = installed / "models/pair_judge/v4/calibration.json"
    value = json.loads(path.read_text())
    value["calibrator"] = calibrator
    path.write_text(json.dumps(value))
    assert not inspect_setup(installed, {})["inventory_ok"]


@pytest.mark.parametrize("field,value", [("template_id", "wrong"), ("max_length", True), ("max_length", 0)])
def test_judge_contract_rejects_wrong_template_and_length(installed, field, value):
    path = installed / "models/pair_judge/v4/contract.json"
    contract = json.loads(path.read_text())
    contract[field] = value
    path.write_text(json.dumps(contract))
    assert not inspect_setup(installed, {})["inventory_ok"]


def test_cli_returns_nonzero_json_for_a_missing_install(tmp_path, monkeypatch, capsys):
    from scripts import check_youtube_setup

    monkeypatch.setattr(check_youtube_setup, "ROOT", tmp_path)
    monkeypatch.setattr(check_youtube_setup.sys, "argv", ["check_youtube_setup", "--json"])
    assert check_youtube_setup.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["inventory_ok"]
    assert any(row["name"] == "claim detector" and row["status"] == "error" for row in report["checks"])
