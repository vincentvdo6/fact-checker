"""Broaden discovery with literal claim words, without rewriting the asserted claim.

The fallback drops conversational filler and preceding context, never invents synonyms
or publishers. It is a search hint, not a resolved interpretation of the assertion.
"""

from __future__ import annotations

import re

from src.pipeline.topic_query import content_terms

_SCOPE_WORDS = {"before", "after", "during", "until", "since", "between", "from", "to", "this",
                "more", "less", "least", "most", "than", "under", "over", "up", "at", "about", "around", "not", "no"}
NUMBER_WORDS = frozenset("""
    zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen
    fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy
    eighty ninety hundred thousand million billion trillion half quarter dozen
""".split())


def fallback_query(claim: str, country: str, cutoff: str) -> str:
    """Keep literal numbers, punctuation and supplied scope in a compact search hint."""
    terms = content_terms(claim)
    if len(terms) < 2:
        return ""
    tokens = claim.split()
    quantitative = any(character.isdigit() for character in claim) or bool(NUMBER_WORDS.intersection(terms))
    words = tokens if quantitative else [
        word for word in tokens if content_terms(word)
        or word.lower().strip(".,!?;:") in _SCOPE_WORDS
        or not re.fullmatch(r"[A-Za-z'’]+[.,!?;:]?", word)]
    query = " ".join(filter(None, (" ".join(words), country, cutoff[:4])))
    return query if len(query) <= 500 else ""
