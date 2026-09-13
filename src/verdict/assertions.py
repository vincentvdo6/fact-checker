"""Split a claim into the exact assertions a verdict must account for, in code.

A corrective contrast ("We don't have X. We have Y.") is two assertions and one
claim: neither half may be judged as if it were the whole. The split reuses the
caption grammar so the assertion texts are exact spans of the original claim,
never paraphrases. Anything the grammar does not recognize stays one assertion.
"""

from __future__ import annotations

import re

from src.pipeline.caption_claims import contrast_parts

_NEGATED = re.compile(r"\b(?:not|no|n't|never|nobody|nothing)\b|n['’]t\b", re.I)
_CLAUSE = re.compile(
    r"(?P<subject>we|they|you|i|he|she|it|this|that|there)\s+"
    r"(?P<verb>do not have|does not have|don't have|doesn't have|have no|has no|"
    r"are not|is not|am not|aren't|isn't|are no|is no)\s+"
    r"(?P<object>[^.!?]+?)(?P<end>[.!?]?)", re.I,
)
_ANY_CLAUSE = re.compile(
    r"(?P<subject>we|they|you|i|he|she|it|this|that|there)\s+"
    r"(?P<verb>do not have|does not have|don't have|doesn't have|have no|has no|"
    r"are not|is not|am not|aren't|isn't|are no|is no|have|has|are|is|am)\s+"
    r"(?P<object>[^.!?]+?)(?P<end>[.!?]?)", re.I,
)
# Copula and auxiliary forms whose negation is a mechanical insertion, for general subjects.
_COPULA = re.compile(r"^(?P<head>.+?\s)(?P<verb>is|are|was|were|has|have|had|can|could|will|would|does|do|did)\s+"
                     r"(?P<rest>(?!not\b|no\b|never\b)[^.!?]+?)(?P<end>[.!?]?)$", re.I)
_POSITIVE = {"do not have": "have", "don't have": "have", "have no": "have", "does not have": "has",
             "doesn't have": "has", "has no": "has", "are not": "are", "aren't": "are", "are no": "are",
             "is not": "is", "isn't": "is", "is no": "is", "am not": "am"}


def claim_assertions(claim: str) -> list[dict]:
    """Exact assertion spans with a stable ID and whether each is a denial."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("A claim must be nonempty text.")
    parts = contrast_parts(claim) or (claim.strip(),)
    for part in parts:
        if part not in claim:
            raise ValueError("Assertions must be exact spans of the claim.")
    return [{"id": f"assertion-{index + 1}", "text": part, "negated": bool(_NEGATED.search(part)),
             "contrast": len(parts) > 1} for index, part in enumerate(parts)]


def positive_form(text: str) -> str | None:
    """The same clause without its negation, for a judge that reads negation words as a tell.

    Only the narrow contrast grammar is inverted, so the result is a mechanical edit of the
    original with a known inverse: a relation found for the positive form is flipped back
    by the caller. Anything the grammar does not match returns None and is judged as written.
    """
    match = _CLAUSE.fullmatch(text.strip().replace("’", "'"))
    if not match:
        return None
    verb = _POSITIVE[match["verb"].casefold()]
    return f"{match['subject']} {verb} {match['object']}{match['end']}"


def object_phrase(text: str) -> str | None:
    """The clause's object as an exact span, for asking what a sentence is about."""
    match = _ANY_CLAUSE.fullmatch(text.strip().replace("’", "'"))
    return match["object"].strip() if match else None


def negate_form(text: str) -> str | None:
    """The same clause with one negation inserted after its first copula or auxiliary.

    Used only to build training pairs: a sentence that states H contradicts this form of H,
    and a sentence that merely bears on H bears on it too. A clause that already carries a
    negation, or has no copula or auxiliary to hang one on, returns None.
    """
    clean = text.strip().replace("’", "'")
    if _NEGATED.search(clean):
        return None
    match = _COPULA.match(clean)
    if not match or not match["head"].strip() or not match["rest"].strip():
        return None
    return f"{match['head']}{match['verb']} not {match['rest']}{match['end']}"
