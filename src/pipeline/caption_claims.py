"""Keep adjacent corrective contrasts together without rewriting caption text.

Caption punctuation is not a reliable claim boundary. A short negated assertion followed
by the same subject/predicate and a different qualifier of the same noun is one contrast.
This deliberately narrow grammar leaves unrelated sentences and questions separate.
"""

from __future__ import annotations

import re

from src.pipeline.segment import Sentence, segment

_CLAUSE = re.compile(
    r"(?P<subject>we|they|you|i|he|she|it|this|that|there)\s+"
    r"(?P<verb>do not have|does not have|don't have|doesn't have|have no|has no|"
    r"are not|is not|am not|aren't|isn't|are no|is no|have|has|are|is|am)\s+"
    r"(?P<object>[^.!?]+)\.?", re.I,
)


def _contrast(first: str, second: str) -> bool:
    left, right = (_CLAUSE.fullmatch(text.replace("’", "'")) for text in (first, second))
    if not left or not right or max(len(first.split()), len(second.split())) > 25:
        return False
    if left["subject"].casefold() != right["subject"].casefold():
        return False
    negative, positive = left["verb"].casefold(), right["verb"].casefold()
    if not any(word in negative for word in ("not", "n't", " no")):
        return False
    predicate = "have" if "have" in negative or "has" in negative else "be"
    if positive not in ({"have", "has"} if predicate == "have" else {"is", "are", "am"}):
        return False
    objects = [re.findall(r"[a-z]+", match["object"].casefold()) for match in (left, right)]
    return bool(objects[0] and objects[1] and objects[0] != objects[1] and objects[0][-1] == objects[1][-1])


def contrast_parts(text: str) -> tuple[str, ...]:
    """Expose exact assertion spans for research without splitting the displayed claim."""
    sentences = segment(text)
    if len(sentences) == 2:
        parts = (sentences[0].text, sentences[1].text)
        if _contrast(*parts):
            return parts
    if len(sentences) == 1:
        for match in re.finditer(r"[,;]\s+(?:but\s+)?", text, flags=re.I):
            if _contrast(text[:match.start()], text[match.end():]):
                return text[:match.start()], text[match.end():]
    return ()


def corrective_contrast(text: str) -> bool:
    """Recognize paired assertions whether captions used a period, comma or semicolon."""
    return bool(contrast_parts(text))


def caption_claims(text: str) -> list[Sentence]:
    """Return source spans, merging only immediately adjacent corrective pairs."""
    sentences = segment(text)
    grouped: list[Sentence] = []
    index = 0
    while index < len(sentences):
        first = last = sentences[index]
        if index + 1 < len(sentences) and _contrast(first.text, sentences[index + 1].text):
            index += 1
            last = sentences[index]
        grouped.append(Sentence(len(grouped), text[first.start:last.end], first.start, last.end))
        index += 1
    return grouped
