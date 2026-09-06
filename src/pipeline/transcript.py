"""Versioned ASR segments with bounded, causal context.

The producer owns sentence boundaries and media timestamps. Interim hypotheses are stored but
never checked. A correction uses the same id and a higher revision; callers can invalidate a
verdict by its claim revision and the exact context revisions it consumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    id: str
    revision: int
    text: str
    start: float
    end: float
    final: bool
    speaker: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 128:
            raise ValueError("id must contain 1-128 characters")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a nonnegative integer")
        if not isinstance(self.text, str) or len(self.text) > 4000:
            raise ValueError("text must be a string of at most 4000 characters")
        if type(self.final) is not bool or (self.final and not self.text.strip()):
            raise ValueError("final must be boolean and final text must not be empty")
        if not isinstance(self.speaker, str) or len(self.speaker) > 128:
            raise ValueError("speaker must be a string of at most 128 characters")
        for value in (self.id, self.text, self.speaker):
            value.encode("utf-8")
        for value in (self.start, self.end):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("timestamps must be finite nonnegative seconds")
        if self.end < self.start:
            raise ValueError("end must not precede start")

    @classmethod
    def from_dict(cls, value: dict) -> TranscriptUpdate:
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"invalid transcript fields: {error}") from error

