"""Preserve distinct explicit legislation names."""
from __future__ import annotations

import pytest

from src.pipeline.references import law_mentions


@pytest.mark.parametrize("text", ["The Act changed.", "This Law changed.", "affordable care act"])
def test_generic_or_uncased_words_are_not_named_laws(text: str) -> None:
    assert law_mentions(text) == []


def test_title_years_and_spans():
    text = "The Civil Rights Act of 1964 and the Civil Rights Act of 1991."
    rows = law_mentions(text)
    assert [r[0] for r in rows] == ["Civil Rights Act of 1964", "Civil Rights Act of 1991"]
    assert all(text[start:end] == title for title, start, end in rows)
