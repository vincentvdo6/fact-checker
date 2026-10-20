"""Keep adjacent corrective contrasts together without rewriting caption text.

Caption punctuation is not a reliable claim boundary. A short negated assertion followed
by the same subject/predicate and a different qualifier of the same noun is one contrast.
This deliberately narrow grammar leaves unrelated sentences and questions separate.

YouTube captions mark a change of speaker with ">>". That is a boundary punctuation may not
show, and the marker is never part of what anyone said, so no span includes it.

A caption sentence can also be a fragment whose subject was said earlier -- "Last month it
was 32,000" -- and searching its words alone finds any page with that number. Such a
sentence names nothing to check and is not researched.
"""

from __future__ import annotations

import re

from src.pipeline.segment import Sentence, segment
from src.pipeline.topic_query import content_terms

_SPEAKER_CHANGE = re.compile(r"\s*>>+\s*")
# Words that say when or how many, never what: a sentence made only of these and pronouns is a fragment.
_NOT_A_SUBJECT = frozenset("""
    last next past previous recent recently current currently ago now today yesterday tomorrow tonight
    day days week weeks month months year years decade decades quarter quarters century period time times
    annual annually monthly weekly daily morning evening night
    january february march april may june july august september october november december
    jan feb mar apr jun jul aug sep sept oct nov dec
    one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty sixty seventy
    eighty ninety hundred hundreds thousand thousands million millions billion billions trillion percent
    first second third fourth fifth sixth seventh eighth ninth tenth
""".split())


def names_a_subject(text: str) -> bool:
    """Whether a sentence names anything besides pronouns, numbers and times, so it can be researched."""
    return any(term.isalpha() and term not in _NOT_A_SUBJECT for term in content_terms(text))


_THIRD_PERSON = re.compile(r"^\W*(?:(?:and|but|so|then|well|look|now|because|uh|um)\W+)*"
                           r"(?:he|she|they|his|her|their|him|them)\b", re.I)
# A capitalised word in mid-sentence, after a lowercase word or a comma: not a sentence or quotation opener.
_MID_NAME = re.compile(r"(?:[a-z0-9,;]\s+)(?!I\b|I['’])[A-Z]")


def names_whom(text: str) -> bool:
    """Whether a sentence says whom it is about, or could: false for "They don't have equity in
    their home", whose subject was named earlier and whose words alone find any page about anyone.
    A third-person pronoun subject with no capitalised name after it names no one. Captions with
    no capitals at all cannot show a name, so they are left alone."""
    if not _THIRD_PERSON.match(text) or text == text.lower():
        return True
    return bool(_MID_NAME.search(text))


# "He said ...", "I told them ...": the speaker recounting what someone said. Quotations are never
# counted as evidence, so a source can only appear to settle such a claim by misreading. Only a
# pronoun subject: "Amazon announced 14,000 job cuts" names a company, and its announcement is the event.
_REPORTED_SPEECH = re.compile(
    r"^\W*(?:(?:and|but|so|then|well|look|now|because)\W+)*(?:he|she|they|i|we|you)\s+"
    r"(?:said|says|told|tells|claimed|claims|stated|argued|insisted|wrote|tweeted)\b", re.I)


def reports_speech(text: str) -> bool:
    """Whether a sentence's main assertion is that someone, named by a pronoun, said something."""
    return bool(_REPORTED_SPEECH.match(text))


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
    """Return source spans, merging only immediately adjacent corrective pairs by one speaker."""
    sentences: list[Sentence] = []
    turns: list[int] = []
    cursor = 0
    for turn, match in enumerate([*_SPEAKER_CHANGE.finditer(text), None]):
        end = match.start() if match else len(text)
        for sentence in segment(text[cursor:end]):
            sentences.append(Sentence(len(sentences), sentence.text, sentence.start + cursor, sentence.end + cursor))
            turns.append(turn)
        if match:
            cursor = match.end()
    grouped: list[Sentence] = []
    index = 0
    while index < len(sentences):
        first = last = sentences[index]
        if (index + 1 < len(sentences) and turns[index] == turns[index + 1]
                and _contrast(first.text, sentences[index + 1].text)):
            index += 1
            last = sentences[index]
        grouped.append(Sentence(len(grouped), text[first.start:last.end], first.start, last.end))
        index += 1
    return grouped
