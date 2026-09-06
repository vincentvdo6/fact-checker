"""Reject invalid Unicode before transcript state or logs can change."""

from __future__ import annotations

import pytest

from src.pipeline.transcript import TranscriptUpdate


@pytest.mark.parametrize("field", ["id", "text", "speaker"])
def test_wire_strings_are_utf8_encodable(field):
    data = dict(id="a", revision=0, text="Jobs grew.", start=0, end=1, final=True, speaker="Speaker")
    data[field] = "\ud800"
    with pytest.raises(ValueError):
        TranscriptUpdate.from_dict(data)
