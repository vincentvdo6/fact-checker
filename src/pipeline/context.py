"""Make retrieval context explicit without silently rewriting the asserted claim.

Query expansion is experimental. It adds supplied geography/date and, for referential language,
nearby finalized speech. These words are retrieval hints, never evidence or resolved facts. The
verdict encoder still receives the original assertion; FEVER calibration does not validate this
new retrieval distribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from src.pipeline.transcript import ClaimContext

REFERENTIAL = re.compile(r"\b(it|its|this|that|these|those|we|our|ours|they|their|he|she|his|her)\b", re.I)
CONTEXT_WORDS = 48


@dataclass(frozen=True, slots=True)
class SpeechMetadata:
    source: str = ""
    country: str = ""
    spoken_at: str = ""

    def __post_init__(self) -> None:
        for value in (self.source, self.country, self.spoken_at):
            if not isinstance(value, str) or len(value) > 2048:
                raise ValueError("speech metadata must be strings of at most 2048 characters")
            value.encode("utf-8")
        if self.spoken_at:
            date.fromisoformat(self.spoken_at)


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    claim: str
    query: str
    mode: str
    context_ids: tuple[str, ...]


def build_query(context: ClaimContext, metadata: SpeechMetadata, *, mode: str = "claim") -> RetrievalQuery:
    if mode not in ("claim", "context"):
        raise ValueError("query mode must be claim or context")
    claim = context.claim.text
    if mode == "claim":
        return RetrievalQuery(claim, claim, mode, ())
    hints = [metadata.country, metadata.spoken_at[:4]]
    preceding = context.preceding if REFERENTIAL.search(claim) else ()
    words = " ".join(s.text for s in preceding).split()
    hints.append(" ".join(words[-CONTEXT_WORDS:]))
    query = " ".join([claim, *(h for h in hints if h)])
    return RetrievalQuery(claim, query, mode, tuple(s.id for s in preceding))
