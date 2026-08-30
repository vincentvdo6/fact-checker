"""The declarations an artifact must match before its numbers are trusted."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.verdict.contract import CONTRACT_VERSION, EncoderContract
from src.verdict.encode import LABELS, TEMPLATE_ID


def contract(**overrides) -> EncoderContract:
    base = dict(
        base_model="microsoft/deberta-v3-base",
        base_revision="deadbeef",
        tokenizer_sha256="a" * 64,
        labels=LABELS,
        max_length=512,
        template_id=TEMPLATE_ID,
        variant="retrieved",
        seed=42,
        torch_version="2.4.0",
        transformers_version="4.44.0",
        contract_version=CONTRACT_VERSION,
    )
    return EncoderContract(**{**base, **overrides})


def test_round_trips_through_json(tmp_path):
    original = contract()
    path = tmp_path / "contract.json"
    original.save(path)
    assert EncoderContract.load(path) == original


def test_labels_survive_as_an_ordered_tuple():
    """JSON has no tuples; a list would compare unequal and reorder silently."""
    restored = EncoderContract.from_dict(json.loads(json.dumps(contract().to_dict())))
    assert restored.labels == LABELS
    assert isinstance(restored.labels, tuple)


def test_a_missing_field_raises_rather_than_defaulting():
    """A default here would paper over exactly the drift the contract exists to catch."""
    payload = contract().to_dict()
    del payload["max_length"]
    with pytest.raises(ValueError, match=r"missing \['max_length'\]"):
        EncoderContract.from_dict(payload)


def test_an_unknown_field_raises():
    payload = contract().to_dict() | {"batch_size": 16}
    with pytest.raises(ValueError, match="unknown fields"):
        EncoderContract.from_dict(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("labels", ("contradicted", "supported", "not_enough_evidence")),
        ("template_id", "per_sentence_title_v1"),
        ("max_length", 384),
        ("base_model", "roberta-base"),
        ("tokenizer_sha256", "b" * 64),
        ("contract_version", CONTRACT_VERSION + 1),
    ],
)
def test_each_input_defining_field_is_checked(field, value):
    """One case per field, because each is a distinct way to get plausible wrong numbers."""
    with pytest.raises(ValueError, match=f"mismatch on {field}"):
        contract().assert_compatible(contract(**{field: value}))


def test_a_permuted_label_order_is_caught():
    """
    The worst drift: every metric still computes, just against the wrong classes. Nothing about
    the output looks wrong.
    """
    permuted = contract(labels=(LABELS[1], LABELS[0], LABELS[2]))
    with pytest.raises(ValueError, match="mismatch on labels"):
        contract().assert_compatible(permuted)


@pytest.mark.parametrize("field,value", [("variant", "gold"), ("seed", 7)])
def test_variant_and_seed_do_not_break_compatibility(field, value):
    """Three variants ship from one contract by design, and two seeds are still comparable."""
    contract().assert_compatible(contract(**{field: value}))


def test_an_identical_contract_is_compatible():
    contract().assert_compatible(contract())


def test_replace_keeps_the_contract_usable():
    assert replace(contract(), seed=1).seed == 1
