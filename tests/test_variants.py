"""
Installing and reporting a second model beside the control rather than over it.

Phase 05 trains the same `retrieved` encoder variant on regrounded data. Its whole value is the
comparison against the Phase 02 model, so the two failures that matter are installing over the
control -- which destroys the baseline irreversibly -- and reporting from a hard-coded variant
list, which would omit the new model from every table with nothing raising to say so.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.install_artifacts import check
from src.verdict.contract import CONTRACT_FILE, KNOWN_VARIANTS, installed_variants

SCRIPTS = ("eval_verdict", "eval_calibration", "calibrate")


def make_variant(root: Path, name: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / CONTRACT_FILE).write_text("{}", encoding="utf-8")


def test_known_variants_come_first_then_anything_else(tmp_path):
    for name in ("gold", "zebra", "retrieved", "retrieved_grounded", "claim_only"):
        make_variant(tmp_path, name)
    assert installed_variants(tmp_path) == [
        "retrieved", "claim_only", "gold", "retrieved_grounded", "zebra",
    ]


def test_a_new_variant_is_discovered_without_editing_a_constant():
    """The property the refactor exists for: nothing has to be edited for the comparison to appear."""
    assert "retrieved_grounded" not in KNOWN_VARIANTS


def test_a_directory_without_a_contract_is_not_a_variant(tmp_path):
    """runs/, caches and stray directories must not be reported as models."""
    make_variant(tmp_path, "retrieved")
    (tmp_path / "not_a_model").mkdir()
    assert installed_variants(tmp_path) == ["retrieved"]


def test_a_missing_models_directory_is_empty_not_an_error(tmp_path):
    assert installed_variants(tmp_path / "absent") == []


def test_the_reporting_scripts_do_not_hard_code_a_variant_list():
    """
    A tuple here would leave the Phase 05 model out of every report. The omission is silent: the
    table simply has one fewer row, and every number in it is still correct.
    """
    for name in SCRIPTS:
        source = Path(f"scripts/{name}.py").read_text(encoding="utf-8")
        assert 'VARIANTS = ("retrieved"' not in source, name
        assert "installed_variants(" in source, name


def test_the_installer_can_place_a_model_under_another_name():
    """
    Without --as, a second `retrieved` run overwrites the control it is measured against, and the
    baseline cannot be recovered without re-downloading a 668 MB artifact.
    """
    source = Path("scripts/install_artifacts.py").read_text(encoding="utf-8")
    assert 'args.install_as or contract.variant' in source
    assert '"--as", dest="install_as"' in source


def test_the_contract_still_declares_the_encoder_variant(tmp_path):
    """
    --as changes where the model lands, never what it claims to be. The contract governs how inputs
    are built, and rewriting it to match a directory name would make the encoder check meaningless.
    """
    from tests.test_install_artifacts import TOKENIZER_FILES, contract

    path = tmp_path / "artifacts_retrieved_v1.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in TOKENIZER_FILES.items():
            zf.writestr(name, data)
        zf.writestr("model_v1/model.safetensors", b"weights")
        zf.writestr(CONTRACT_FILE, json.dumps(contract()))
        zf.writestr("metrics.json", json.dumps({"steps": 1, "planned_steps": 1, "smoke": False}))
        for name in ("predictions_calibration.jsonl", "predictions_test.jsonl"):
            zf.writestr(name, "")

    with zipfile.ZipFile(path) as zf:
        parsed, _ = check(zf)
    assert parsed.variant == "retrieved"
