"""Make retrieval context explicit without silently rewriting the asserted claim.

Query expansion is experimental. It adds supplied geography/date and, for referential language,
nearby finalized speech. These words are retrieval hints, never evidence or resolved facts. The
optional law hint carries the exact earlier transcript span and refuses competing titles. The
verdict encoder still receives the original assertion; FEVER calibration does not validate this
new retrieval distribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from src.pipeline.references import LAW_REFERENCE, law_mentions
from src.pipeline.topic_query import topic_query
from src.pipeline.transcript import ClaimContext

REFERENTIAL = re.compile(r"\b(it|its|this|that|these|those|we|our|ours|they|their|he|she|his|her)\b", re.I)
CONTEXT_WORDS = 48


@dataclass(frozen=True, slots=True)
class SpeechMetadata:
    source: str = ""
    country: str = ""
    spoken_at: str = ""
    source_published_at: str = ""

    def __post_init__(self) -> None:
        for value in (self.source, self.country, self.spoken_at, self.source_published_at):
            if not isinstance(value, str) or len(value) > 2048:
                raise ValueError("speech metadata must be strings of at most 2048 characters")
            value.encode("utf-8")
        if self.spoken_at:
            date.fromisoformat(self.spoken_at)
        if self.source_published_at:
            date.fromisoformat(self.source_published_at)


@dataclass(frozen=True, slots=True)
class ReferenceHint:
    text: str
    segment_id: str
    revision: int
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    claim: str
    query: str
    mode: str
    context_ids: tuple[str, ...]
    references: tuple[ReferenceHint, ...] = ()
    context_text: tuple[str, ...] = ()


def _law_hint(context: ClaimContext) -> tuple[ReferenceHint, ...]:
    if not LAW_REFERENCE.search(context.claim.text) or law_mentions(context.claim.text):
        return ()
    candidates = {}
    for segment in context.reference_context:
        for text, start, end in law_mentions(segment.text):
            candidates[text.casefold()] = ReferenceHint(text, segment.id, segment.revision, start, end)
    return tuple(candidates.values()) if len(candidates) == 1 else ()


def build_query(context: ClaimContext, metadata: SpeechMetadata, *, mode: str = "claim") -> RetrievalQuery:
    if mode not in ("claim", "context", "topic"):
        raise ValueError("query mode must be claim, context or topic")
    claim = context.claim.text
    if mode == "claim":
        return RetrievalQuery(claim, claim, mode, ())
    if mode == "topic":
        query, ids, text = topic_query(context)
        return RetrievalQuery(claim, query, mode, ids, context_text=text)
    hints = [metadata.country, metadata.spoken_at[:4]]
    preceding = context.preceding if REFERENTIAL.search(claim) else ()
    words = " ".join(s.text for s in preceding).split()
    hints.append(" ".join(words[-CONTEXT_WORDS:]))
    references = _law_hint(context)
    hints.extend(reference.text for reference in references)
    query = " ".join([claim, *(h for h in hints if h)])
    ids = tuple(dict.fromkeys([*(s.id for s in preceding), *(r.segment_id for r in references)]))
    return RetrievalQuery(claim, query, mode, ids, references)
