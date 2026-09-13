"""Reject lexical distractors in live search before they reach the verdict model.

Evidence must match a claim content term and a multiword topic phrase in its body,
not just its title. Phrases come from the assertion and explicitly selected context.
This conservative lexical screen cannot establish semantic relevance, scope or truth.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.pipeline.topic_query import FUNCTION_WORDS, content_terms
from src.retrieval.text import tokenize

CLAUSE_BOUNDARY = re.compile(r"[.!?,;:]")


def _phrases(text: str) -> set[tuple[str, ...]]:
    phrases: set[tuple[str, ...]] = set()
    for clause in CLAUSE_BOUNDARY.split(text):
        run: list[str] = []
        for term in [*tokenize(clause), "the"]:
            if term in FUNCTION_WORDS:
                size = min(3, len(run))
                if size >= 2:
                    phrases.update(tuple(run[i:i + size]) for i in range(len(run) - size + 1))
                run = []
            else:
                run.append(term)
    return phrases


def topic_evidence(
    claim: str, query: str, evidence: Sequence[tuple[str, int, str]], scores: Sequence[float],
    *, context: Sequence[str] = (),
) -> tuple[list[tuple[str, int, str]], list[float]]:
    """Keep aligned evidence/scores in their original rank order, without filling rejected slots."""
    claim_terms, query_terms = set(content_terms(claim)), set(content_terms(query))
    phrases = _phrases(claim)
    for text in context:
        phrases.update(_phrases(text))
    kept, values = [], []
    for row, score in zip(evidence, scores, strict=True):
        body = set(content_terms(row[2]))
        if not body.intersection(claim_terms) or len(body.intersection(query_terms)) < min(2, len(query_terms)):
            continue
        if phrases:
            clauses = [tokenize(clause) for clause in CLAUSE_BOUNDARY.split(row[2])]
            if not any(tuple(tokens[i:i + len(phrase)]) == phrase
                       for tokens in clauses for phrase in phrases
                       for i in range(len(tokens) - len(phrase) + 1)):
                continue
        kept.append(row)
        values.append(score)
    return kept, values
