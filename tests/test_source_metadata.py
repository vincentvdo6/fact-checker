"""A source publication date must not silently become the date described by speech."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from src.pipeline.context import SpeechMetadata, build_query
from src.pipeline.transcript import ClaimContext, TranscriptUpdate


def test_publication_date_is_retained_without_supplying_speech_time():
    metadata = SpeechMetadata(source_published_at="2025-07-20")
    assert asdict(metadata)["source_published_at"] == "2025-07-20"
    assert metadata.spoken_at == ""
    claim = ClaimContext(TranscriptUpdate("a", 0, "Jobs increased.", 0, 1, True), ())
    assert build_query(claim, metadata, mode="context").query == "Jobs increased."
    assert SpeechMetadata(spoken_at="2020-01-01", source_published_at="2025-07-20").spoken_at == "2020-01-01"


@pytest.mark.parametrize("value", [None, True, 2025, "x" * 2049, "2025-02-30", "yesterday", "\ud800"])
def test_invalid_source_publication_metadata_is_rejected(value):
    with pytest.raises((ValueError, UnicodeError)):
        SpeechMetadata(source_published_at=value)


def test_missing_publication_date_stays_unknown():
    assert SpeechMetadata().source_published_at == ""
