"""Versioned ASR segments with bounded, causal context.

The producer owns sentence boundaries and media timestamps. Interim hypotheses are stored but
never checked. A correction uses the same id and a higher revision; callers can invalidate a
verdict by its claim revision and the exact context revisions it consumed.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

from src.pipeline.references import LAW_REFERENCE, REFERENCE_SEGMENTS


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


@dataclass(frozen=True, slots=True)
class ClaimContext:
    claim: TranscriptUpdate
    preceding: tuple[TranscriptUpdate, ...]
    reference_context: tuple[TranscriptUpdate, ...] = ()

    @property
    def dependencies(self) -> tuple[tuple[str, int], ...]:
        prior = {s.id: s for s in (*self.reference_context, *self.preceding)}
        ordered = sorted(prior.values(), key=lambda s: (s.end, s.start, s.id))
        return tuple((s.id, s.revision) for s in (*ordered, self.claim))


class TranscriptWindow:
    """Retain a bounded revision window; reject late new segments instead of using future text."""

    def __init__(self, *, capacity: int = 64, context_size: int = 2, use_references: bool = False) -> None:
        if type(capacity) is not int or type(context_size) is not int or not 0 <= context_size < capacity:
            raise ValueError("require 0 <= context_size < capacity, both integers")
        self.capacity = capacity
        self.context_size = context_size
        if type(use_references) is not bool:
            raise ValueError("use_references must be boolean")
        self.reference_size = min(REFERENCE_SEGMENTS, capacity - 1) if use_references else 0
        self.segments: OrderedDict[str, TranscriptUpdate] = OrderedDict()
        self._retired_through = -1.0

    def accept(self, update: TranscriptUpdate) -> ClaimContext | None:
        old = self.segments.get(update.id)
        if old is not None:
            if update.revision < old.revision or update == old:
                return None
            if update.revision == old.revision:
                raise ValueError("conflicting content for the same revision")
            if update.start != old.start:
                raise ValueError("a segment's start timestamp is its stable anchor")
        else:
            latest = max((s.start for s in self.segments.values()), default=-1.0)
            if update.start <= self._retired_through or update.start < latest:
                raise ValueError("new segments must arrive in media order within the revision window")
        self.segments[update.id] = update
        while len(self.segments) > self.capacity:
            _, retired = self.segments.popitem(last=False)
            self._retired_through = max(self._retired_through, retired.start)
        return self.snapshot(update.id) if update.final else None

    def snapshot(self, segment_id: str) -> ClaimContext:
        claim = self.segments[segment_id]
        preceding = [
            s for s in self.segments.values()
            if s.id != claim.id and s.final and s.end <= claim.start
            and s.start < claim.start and s.speaker == claim.speaker
        ]
        preceding.sort(key=lambda s: (s.end, s.start, s.id))
        chosen = preceding[-self.context_size:] if self.context_size else []
        reference_context = preceding[-self.reference_size:] if (
            self.reference_size and LAW_REFERENCE.search(claim.text)
        ) else []
        return ClaimContext(claim, tuple(chosen), tuple(reference_context))

    def current(self, context: ClaimContext) -> bool:
        if self.segments.get(context.claim.id) != context.claim:
            return False
        prior = (*context.preceding, *context.reference_context)
        if any(self.segments[s.id] != s for s in prior if s.id in self.segments):
            return False
        # Retired context cannot be revised; its immutable snapshot remains valid. Forgetting it
        # would requeue old claims with less context every time the history window advances.
        fresh = self.snapshot(context.claim.id)
        for field, size in (("preceding", self.context_size), ("reference_context", self.reference_size)):
            saved = getattr(context, field)
            candidates = [s for s in saved if s.id not in self.segments]
            candidates.extend(getattr(fresh, field))
            candidates.sort(key=lambda s: (s.end, s.start, s.id))
            chosen = candidates[-size:] if size else []
            if tuple(chosen) != saved:
                return False
        return True
