"""Loader behaviour against the released FEVER files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.fever import LABELS, NOT_ENOUGH_INFO, claim_key, load_claims

TRAIN = "data/fever/train.jsonl"

requires_fever = pytest.mark.skipif(
    not Path(TRAIN).exists(), reason="FEVER data absent; run python -m scripts.fetch_fever"
)


def test_claim_key_normalizes_case_and_whitespace():
    assert claim_key("  The   Sky Is  Blue.\n") == "the sky is blue."


def test_unexpected_label_is_rejected(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": 1, "label": "MAYBE", "claim": "x", "evidence": []}))
    with pytest.raises(ValueError, match="unexpected label"):
        load_claims(bad)


def test_pages_are_deduplicated_and_nulls_dropped(tmp_path):
    row = {
        "id": 1,
        "label": "SUPPORTS",
        "claim": "x",
        "evidence": [[[0, 0, "Page_A", 1], [0, 0, "Page_A", 2]], [[0, 0, None, None]]],
    }
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(row))
    assert load_claims(path)[0].pages == ("Page_A",)


@pytest.mark.slow
@requires_fever
def test_real_train_loads_with_known_shape():
    claims = load_claims(TRAIN)
    assert len(claims) == 145_449
    assert {c.label for c in claims} == set(LABELS)
    # No NOT ENOUGH INFO row carries evidence pages; retrieval must not assume they do.
    assert not any(c.pages for c in claims if c.label == NOT_ENOUGH_INFO)
