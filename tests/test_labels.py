"""Verdict space and dataset mappings."""

from __future__ import annotations

import pytest

from src.data.fever import LABELS as FEVER_NATIVE
from src.verdict.labels import AVERITEC, FEVER, LABEL_SPACES, Verdict


def test_fever_maps_every_native_label():
    assert {FEVER.to_verdict(label) for label in FEVER_NATIVE} == set(FEVER.verdicts)


def test_fever_has_no_mixed_verdict():
    assert Verdict.MIXED not in FEVER.verdicts
    assert FEVER.size == 3


def test_averitec_carries_all_four():
    assert set(AVERITEC.verdicts) == set(Verdict)
    assert AVERITEC.size == 4


def test_unmapped_label_raises():
    with pytest.raises(ValueError, match="unmapped label"):
        FEVER.to_verdict("MOSTLY TRUE")


def test_class_indices_are_stable_and_dense():
    for space in LABEL_SPACES.values():
        assert sorted(space.index(v) for v in space.verdicts) == list(range(space.size))

