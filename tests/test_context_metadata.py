"""Metadata must be safe to include in the UTF-8 event stream."""

from __future__ import annotations

import pytest

from src.pipeline.context import SpeechMetadata


def test_metadata_must_be_encodable():
    with pytest.raises(ValueError):
        SpeechMetadata(source="\ud800")
