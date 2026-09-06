"""Validate the transcript wire contract before accepting speech."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.pipeline.transcript import TranscriptUpdate


def update(id="a", start=0, **changes):
    return replace(TranscriptUpdate(id, 0, "Jobs increased.", start, start + 1, True, "speaker"), **changes)


@pytest.mark.parametrize("changes", [
    {"id": ""}, {"id": "x" * 129}, {"revision": -1}, {"revision": True},
    {"text": "x" * 4001}, {"text": None}, {"text": " "}, {"final": "true"},
    {"speaker": None}, {"start": float("nan")}, {"end": float("inf")},
    {"start": True}, {"start": -1}, {"end": -1}, {"start": 2, "end": 1},
])
def test_invalid_updates_are_rejected(changes):
    with pytest.raises(ValueError):
        update(**changes)


@pytest.mark.parametrize("value", [[], {}, {"unexpected": 1}])
def test_json_contract_refuses_wrong_shapes(value):
    with pytest.raises(ValueError):
        TranscriptUpdate.from_dict(value)
