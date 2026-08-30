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


def one(tmp_path, evidence, label="SUPPORTS"):
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps({"id": 1, "label": label, "claim": "x", "evidence": evidence}))
    return load_claims(path)[0]


def test_pages_are_deduplicated_and_nulls_dropped(tmp_path):
    evidence = [[[0, 0, "Page_A", 1], [0, 0, "Page_A", 2]], [[0, 0, None, None]]]
    assert one(tmp_path, evidence).pages == ("Page_A",)


def test_a_group_is_kept_whole(tmp_path):
    """Both sentences verify the claim jointly, so they are one group, not two."""
    evidence = [[[0, 0, "Page_A", 1], [0, 0, "Page_B", 3]]]
    assert one(tmp_path, evidence).groups == (frozenset({("Page_A", 1), ("Page_B", 3)}),)


def test_duplicate_annotator_groups_collapse(tmp_path):
    evidence = [[[0, 0, "Page_A", 1]], [[1, 1, "Page_A", 1]], [[2, 2, "Page_B", 0]]]
    groups = one(tmp_path, evidence).groups
    assert groups == (frozenset({("Page_A", 1)}), frozenset({("Page_B", 0)}))


def test_a_group_with_a_null_page_is_dropped_whole(tmp_path):
    """
    Not emptied. An empty frozenset is a subset of everything, so a retained one would make
    every recall metric downstream unconditionally perfect.
    """
    evidence = [[[0, 0, "Page_A", 1], [0, 0, None, None]], [[1, 1, "Page_B", 2]]]
    assert one(tmp_path, evidence).groups == (frozenset({("Page_B", 2)}),)


def test_not_enough_info_has_no_groups(tmp_path):
    claim = one(tmp_path, [[[0, 0, None, None]]], label=NOT_ENOUGH_INFO)
    assert claim.groups == ()
    assert claim.pages == ()


@pytest.mark.slow
@requires_fever
def test_no_verifiable_claim_lacks_groups():
    """A verifiable claim with no evidence would be scored against nothing."""
    for claim in load_claims(TRAIN):
        assert bool(claim.groups) == (claim.label != NOT_ENOUGH_INFO)


@pytest.mark.slow
@requires_fever
def test_real_train_loads_with_known_shape():
    claims = load_claims(TRAIN)
    assert len(claims) == 145_449
    assert {c.label for c in claims} == set(LABELS)
    # No NOT ENOUGH INFO row carries evidence pages; retrieval must not assume they do.
    assert not any(c.pages for c in claims if c.label == NOT_ENOUGH_INFO)
