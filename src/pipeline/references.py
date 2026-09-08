"""Conservative legislation-title hints, not pronoun resolution or factual evidence.

Only explicit capitalized names ending in Act, Bill or Law qualify. Multiple names
remain ambiguous; casing lost by ASR is deliberately left unresolved.
"""

from __future__ import annotations

import re

REFERENCE_SEGMENTS = 8
LAW_REFERENCE = re.compile(r"\b(?:(?:it|this|that) became law|(?:this|that|the) (?:law|act|bill))\b", re.I)
LAW_NAME = re.compile(r"\b(?:[A-Z][\w'-]*\s+){1,7}(?:Act|Bill|Law)(?:\s+of\s+\d{4})?\b")


def law_mentions(text: str) -> list[tuple[str, int, int]]:
    """Return exact source spans, excluding a leading article from the title."""
    mentions = []
    for match in LAW_NAME.finditer(text):
        start, end = match.span()
        for article in ("The ", "This ", "That "):
            if text[start:end].startswith(article):
                start += len(article)
                break
        if " " in text[start:end]:
            mentions.append((text[start:end], start, end))
    return mentions
