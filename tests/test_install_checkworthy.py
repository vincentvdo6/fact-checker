"""
The install gate for the check-worthiness artifacts.

Every check here exists because the failure it catches produces numbers rather than an exception.
A permuted label order is the sharpest: `factual` is read off classes 1 and 2 and `check_worthy`
off class 2, both by index, so swapping two entries in the contract gates the demo on a different
quantity while every probability still sums to one and every metric still returns a float.

The refusal has to happen before anything is copied. A half-installed detector -- new contract,
old graph, or the reverse -- is worse than none, because the mismatch is invisible from either
file alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_checkworthy_dataset import LABELS
from scripts.install_checkworthy import MAX_LENGTH, TEMPLATE_ID, validate


def contract(**overrides) -> dict:
    base = {
        "base_model": "microsoft/deberta-v3-base",
        "labels": list(LABELS),
        "max_length": MAX_LENGTH,
        "template_id": TEMPLATE_ID,
        "variant": "checkworthy",
    }
    return base | overrides


def test_the_shipped_contract_passes():
    validate(contract())


def test_a_permuted_label_order_is_refused():
    """
    The worst failure available: every metric still computes, against the wrong classes. Swapping
    the two factual classes would make `check_worthy` mean "unimportant" throughout the demo.
    """
    swapped = [LABELS[0], LABELS[2], LABELS[1]]
    with pytest.raises(SystemExit, match="label order"):
        validate(contract(labels=swapped))


def test_a_missing_class_is_refused():
    with pytest.raises(SystemExit, match="label order"):
        validate(contract(labels=list(LABELS[:2])))


def test_a_different_sequence_budget_is_refused():
    """A model trained at one length and read at another sees different text on long sentences."""
    with pytest.raises(SystemExit, match="max_length"):
        validate(contract(max_length=512))


def test_a_different_template_is_refused():
    with pytest.raises(SystemExit, match="template"):
        validate(contract(template_id="per_page_grouped_v1"))


def test_the_verdict_model_cannot_be_installed_here_by_accident():
    """
    Both exports are 738 MB DeBERTa graphs written by the same notebook under the same filename.
    Size and shape cannot tell them apart; the contract can.
    """
    with pytest.raises(SystemExit, match="not the check-worthiness head"):
        validate(contract(variant="retrieved"))


def test_the_expected_values_match_what_the_notebook_declares():
    """
    These constants are a second copy of the training config. If the notebook changes and this
    does not, the gate would reject the very artifact it is meant to admit.
    """
    template = Path("notebooks/checkworthy_notebook.py").read_text(encoding="utf-8")
    assert f'template_id="{TEMPLATE_ID}"' in template
    assert f"max_length={MAX_LENGTH}," in template


def test_the_installed_artifact_still_satisfies_the_gate():
    """The model actually on disk, checked against the same rules a fresh install would face."""
    path = Path("models/checkworthy/v1/contract.json")
    if not path.exists():
        pytest.skip("detector not installed")
    validate(json.loads(path.read_text(encoding="utf-8")))


def test_the_graph_is_renamed_away_from_the_verdict_name():
    """
    The shared export notebook writes `verdict.onnx` whatever it was given. Leaving that name in
    a check-worthiness directory would imply the wrong model to anyone reading the tree.
    """
    from scripts.install_checkworthy import EXPORT_NAME, ONNX_NAME

    assert EXPORT_NAME == "verdict.onnx"
    assert ONNX_NAME == "onnx/model.onnx"
    assert "verdict" not in ONNX_NAME


def test_nothing_is_written_before_the_contract_is_validated():
    """
    Ordering is the property, and it is checked in the source because the alternative is letting a
    test write a half-installed model somewhere to prove it.
    """
    source = Path("scripts/install_checkworthy.py").read_text(encoding="utf-8")
    assert source.index("validate(contract)") < source.index("mkdir(parents=True")
    assert source.index("validate(contract)") < source.index("shutil.copy")
