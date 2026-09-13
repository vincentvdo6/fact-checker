"""Bound local query selection to literal caption phrases instead of generated words.

Content-word runs supply topic-neutral search candidates. They are neither resolved
entities nor evidence. A selector may choose their IDs, but cannot alter their text,
numbers or provenance. Quantitative runs are omitted whole, never stripped of units.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from src.pipeline.research_queries import NUMBER_WORDS
from src.pipeline.topic_query import content_terms
from src.pipeline.transcript import ClaimContext

CONTEXT_WORDS = 192
MAX_CANDIDATES = 32
MAX_CHOICES = 2
_EDGE_PUNCTUATION = ',.!?;:"“”()[]'


def _phrases(text: str, source_id: str, origin: str, offset: int = 0) -> list[dict]:
    candidates = []
    run: list[tuple[int, int]] = []

    def finish() -> None:
        if 2 <= len(run) <= 8 and run[0][0] >= offset:
            start, end = run[0][0], run[-1][1]
            literal = text[start:end]
            if not any(character.isnumeric() for character in literal) and not NUMBER_WORDS.intersection(content_terms(literal)):
                identity = f"{origin}:{source_id}:{start}:{end}:{literal}"
                candidates.append({"id": "phrase-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                                   "text": literal, "origin": origin, "source_id": source_id,
                                   "start": start - offset, "end": end - offset})
        run.clear()

    for match in re.finditer(r"\S+", text):
        raw = match.group()
        if raw.startswith(tuple(_EDGE_PUNCTUATION)):
            finish()
        clean = raw.strip(_EDGE_PUNCTUATION)
        if content_terms(clean):
            start = match.start() + len(raw) - len(raw.lstrip(_EDGE_PUNCTUATION))
            run.append((start, start + len(clean)))
        else:
            finish()
        if raw.endswith(tuple(_EDGE_PUNCTUATION)):
            finish()
    finish()
    return candidates


def concept_packet(context: ClaimContext) -> dict:
    """Keep complete phrase spans within the last 192 words of finalized earlier speech."""
    earlier = sorted((item for item in context.preceding if item.final
                      and item.start <= context.claim.start and item.end <= context.claim.end),
                     key=lambda item: (item.end, item.start))
    records = []
    candidates = []
    remaining = CONTEXT_WORDS
    for item in reversed(earlier):
        words = list(re.finditer(r"\S+", item.text))
        if not words:
            continue
        kept = min(len(words), remaining)
        offset = words[-kept].start()
        records.append({"id": item.id, "text": item.text[offset:], "source_offset": offset})
        candidates.append(_phrases(item.text, item.id, "context", offset))
        remaining -= kept
        if not remaining:
            break
    records.reverse()
    candidates.reverse()
    return {"claim": context.claim.text, "context": records,
            "claim_candidates": _phrases(context.claim.text, context.claim.id, "claim")[:MAX_CANDIDATES],
            "context_candidates": [candidate for group in candidates for candidate in group][-MAX_CANDIDATES:]}


def concept_selection_schema(packet: dict) -> dict:
    """Constrain model output to known context IDs, including an empty candidate set."""
    identities = [item["id"] for item in packet["context_candidates"]]
    return {"type": "object", "properties": {
        "selected_context_ids": {"type": "array", "items": {"type": "string", **({"enum": identities} if identities else {})},
                                 "maxItems": min(MAX_CHOICES, len(identities)), "uniqueItems": True}},
        "required": ["selected_context_ids"], "additionalProperties": False}


def select_concepts(packet: dict, selection: dict) -> list[dict]:
    """Resolve validated IDs to the original spans; reject generated queries or duplicates."""
    if not isinstance(selection, dict) or set(selection) != {"selected_context_ids"}:
        raise ValueError("Caption selection must contain only selected_context_ids")
    identities = selection["selected_context_ids"]
    if (not isinstance(identities, list) or len(identities) > MAX_CHOICES
            or any(not isinstance(identity, str) for identity in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError("Caption selection must contain at most two distinct IDs")
    available = {candidate["id"]: candidate for candidate in packet["context_candidates"]}
    if any(identity not in available for identity in identities):
        raise ValueError("Caption selection contains an unknown context ID")
    return [dict(available[identity]) for identity in identities]


class CorpusSpecificity:
    """Inverse document frequency from the local Wikipedia index's term table, opened on first use.

    A topic-neutral measure of how specific a phrase is; the corpus is the one retrieval already
    ships, so no new artifact is needed. Unknown terms return None rather than a guessed value,
    and so does every term when the store is absent or was never built: `wiki.connect` would
    create an empty file, so the path is checked first and nothing is created on a machine
    without the corpus.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        from src.retrieval.wiki import DB_PATH

        self.path = Path(path) if path is not None else DB_PATH
        self._conn = None
        self._pages = 0
        self._broken = False

    @property
    def available(self) -> bool:
        return self.path.is_file() and not self._broken

    def __call__(self, term: str) -> float | None:
        if not self.available:
            return None
        try:
            if self._conn is None:
                from src.retrieval import wiki

                self._conn = wiki.connect(self.path)
                self._pages = wiki.page_count(self._conn)
            row = self._conn.execute("SELECT df FROM terms WHERE term = ?", (term,)).fetchone()
        except sqlite3.Error:
            self._broken = True        # a store without the index tables: no specificity, no crash
            return None
        return None if row is None else math.log((self._pages + 1) / (row[0] + 1))


def select_context_phrases(packet: dict, idf: Callable[[str], float | None]) -> dict:
    """Choose up to two context phrases by corpus specificity, with no model and no generated words.

    A capitalized token the corpus has never seen is treated as a caption mis-transcription and
    the phrase is skipped: "Lwood Institute" would search for nothing. Phrases with no known
    content term are skipped too. Ties keep caption order. The result is a selection in the
    shape `select_concepts` validates, so the IDs and text stay exactly the candidates'.
    """
    ranked = []
    for position, candidate in enumerate(packet["context_candidates"]):
        words = candidate["text"].split()
        if any(word[:1].isupper() and idf(word.lower()) is None for word in words):
            continue
        scores = [value for value in (idf(term) for term in content_terms(candidate["text"])) if value is not None]
        if scores:
            ranked.append((-sum(scores) / len(scores), position, candidate["id"]))
    return {"selected_context_ids": [identity for _, _, identity in sorted(ranked)[:MAX_CHOICES]]}
