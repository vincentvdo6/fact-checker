"""Installing an artifact: every reason to refuse, and refusing without writing anything."""

from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from scripts.install_artifacts import REQUIRED, check
from src.verdict.contract import CONTRACT_FILE
from src.verdict.encode import LABELS, TEMPLATE_ID

TOKENIZER_FILES = {
    "model_v1/tokenizer_config.json": b'{"model_max_length": 512}',
    "model_v1/special_tokens_map.json": b"{}",
    "model_v1/spm.model": b"sentencepiece bytes",
}


def tokenizer_sha() -> str:
    digest = hashlib.sha256()
    for name in sorted(TOKENIZER_FILES):
        digest.update(TOKENIZER_FILES[name])
    return digest.hexdigest()


def contract(**overrides) -> dict:
    return {
        "base_model": "microsoft/deberta-v3-base",
        "base_revision": "abc123",
        "tokenizer_sha256": tokenizer_sha(),
        "labels": list(LABELS),
        "max_length": 512,
        "template_id": TEMPLATE_ID,
        "variant": "retrieved",
        "seed": 42,
        "torch_version": "2.4.0",
        "transformers_version": "4.44.2",
        "contract_version": 1,
        **overrides,
    }


def archive(tmp_path, *, contract_overrides=None, metrics_overrides=None, drop=()):
    path = tmp_path / "artifacts_retrieved_v1.zip"
    metrics = {"steps": 8000, "planned_steps": 8000, "accuracy": {"test": 0.7}, "smoke": False}
    metrics.update(metrics_overrides or {})
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in TOKENIZER_FILES.items():
            zf.writestr(name, data)
        zf.writestr("model_v1/model.safetensors", b"weights")
        zf.writestr(CONTRACT_FILE, json.dumps(contract(**(contract_overrides or {}))))
        zf.writestr("metrics.json", json.dumps(metrics))
        for name in ("predictions_calibration.jsonl", "predictions_test.jsonl"):
            if name not in drop:
                zf.writestr(name, "")
    return path


def test_a_well_formed_archive_passes(tmp_path):
    with zipfile.ZipFile(archive(tmp_path)) as zf:
        parsed, metrics = check(zf)
    assert parsed.variant == "retrieved"
    assert metrics["steps"] == 8000


def test_every_required_file_is_demanded(tmp_path):
    with zipfile.ZipFile(archive(tmp_path, drop=("predictions_test.jsonl",))) as zf:
        with pytest.raises(SystemExit, match="missing"):
            check(zf)


def test_a_permuted_label_order_is_refused(tmp_path):
    """
    The worst drift there is: every metric still computes, against the wrong classes, so nothing
    about the output looks wrong.
    """
    permuted = [LABELS[1], LABELS[0], LABELS[2]]
    with zipfile.ZipFile(archive(tmp_path, contract_overrides={"labels": permuted})) as zf:
        with pytest.raises(SystemExit, match="label order mismatch"):
            check(zf)


def test_a_changed_template_is_refused(tmp_path):
    with zipfile.ZipFile(archive(tmp_path, contract_overrides={"template_id": "per_sentence_v1"})) as zf:
        with pytest.raises(SystemExit, match="template mismatch"):
            check(zf)


def test_a_tokenizer_that_does_not_match_its_own_files_is_refused(tmp_path):
    """The tokenizer travels in the zip, so its identity is checkable rather than asserted."""
    with zipfile.ZipFile(archive(tmp_path, contract_overrides={"tokenizer_sha256": "0" * 64})) as zf:
        with pytest.raises(SystemExit, match="tokenizer mismatch"):
            check(zf)


def test_a_smoke_run_is_refused(tmp_path):
    """A smoke artifact trains on 512 rows; installing one would report a number nobody meant."""
    with zipfile.ZipFile(archive(tmp_path, metrics_overrides={"smoke": True})) as zf:
        with pytest.raises(SystemExit, match="smoke run"):
            check(zf)


def test_a_contract_missing_a_field_is_refused(tmp_path):
    payload = contract()
    del payload["max_length"]
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in TOKENIZER_FILES.items():
            zf.writestr(name, data)
        zf.writestr("model_v1/model.safetensors", b"weights")
        zf.writestr(CONTRACT_FILE, json.dumps(payload))
        zf.writestr("metrics.json", json.dumps({"steps": 1, "planned_steps": 1, "smoke": False}))
        for name in ("predictions_calibration.jsonl", "predictions_test.jsonl"):
            zf.writestr(name, "")
    with zipfile.ZipFile(path) as zf:
        with pytest.raises(ValueError, match="missing"):
            check(zf)


@pytest.mark.parametrize("overrides", [
    {"labels": [LABELS[1], LABELS[0], LABELS[2]]},
    {"template_id": "other"},
    {"tokenizer_sha256": "0" * 64},
])
def test_a_refusal_writes_nothing(tmp_path, overrides):
    """
    A half-installed artifact is worse than a rejected one: the contract would describe one model
    and the weights beside it another.
    """
    dest = tmp_path / "models"
    dest.mkdir()
    with zipfile.ZipFile(archive(tmp_path, contract_overrides=overrides)) as zf:
        with pytest.raises(SystemExit):
            check(zf)
    assert list(dest.iterdir()) == []


def test_the_required_set_names_what_later_phases_read():
    """Predictions are the whole local interface; Phase 03 fits its temperature on those logits."""
    assert "predictions_calibration.jsonl" in REQUIRED
    assert "predictions_test.jsonl" in REQUIRED
    assert CONTRACT_FILE in REQUIRED
