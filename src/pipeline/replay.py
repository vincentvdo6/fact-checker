"""Synthetic arrival times exercise streaming without pretending a text file contains audio timing."""

from __future__ import annotations

import math
from collections.abc import Iterator

from src.pipeline.segment import segment
from src.pipeline.transcript import TranscriptUpdate


def replay_updates(text: str, *, words_per_minute: float = 150, speaker: str = "") -> Iterator[TranscriptUpdate]:
    if not math.isfinite(words_per_minute) or words_per_minute <= 0:
        raise ValueError("words_per_minute must be finite and positive")
    elapsed = 0.0
    for sentence in segment(text):
        duration = len(sentence.text.split()) * 60 / words_per_minute
        yield TranscriptUpdate(str(sentence.index), 0, sentence.text, elapsed, elapsed + duration, True, speaker)
        elapsed += duration
