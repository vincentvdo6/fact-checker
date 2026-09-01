"""
Find the sentences in a transcript that are worth checking.

A transcript is mostly not check-worthy. Greetings, questions, exhortations, opinions and
pleasantries all pass through a fact checker untouched, and running the verdict model on them
would not be merely wasteful -- it would put a calibrated confidence beside a sentence that has no
truth value, which is the same category error the project's abstention machinery exists to avoid.

**The filter is a heuristic, deliberately.** A learned check-worthiness detector is a Phase 08
question. Here a heuristic wins on a property that matters more than accuracy at this stage: when
it is wrong, the reason is inspectable. Every rejection carries the rule that fired, so a claim
missing from the sidebar can be traced to `no_anchor` or `opinion` rather than to a score. That
also makes the filter measurable against hand labels -- see `scripts/measure_checkworthiness.py`,
because a filter that silently decides what the demo shows must not go unmeasured.

**What "check-worthy" means here:** the sentence asserts something about the world that could be
looked up and found true or false. Three conditions, each a separate rejection reason:

  it is an assertion   not a question, not an imperative, not hedged opinion
  it has a finite verb something is predicated, rather than a fragment or a title
  it has an anchor     a name, a number or a date -- something retrieval can search for

The last is the one that departs from a linguistic definition of a claim, and it earns its place
empirically: FEVER claims are entity-centred, the retrieval stack is BM25 over Wikipedia titles,
and a sentence with no proper noun and no number has nothing for either to grip. "That was a
terrible decision" is an assertion with a finite verb and it is not checkable by this system.

**Tense coverage is a closed-class argument, not a word list.** Regular past tense is `-ed`;
irregular past is a genuinely closed class in English, so it is enumerated. Present-tense
assertions almost always route through a copula or auxiliary, which is also closed. What this
misses is bare present-tense lexical verbs ("The bill raises taxes"), and that gap is real,
reported by the measurement script rather than papered over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Segmentation. Splitting on [.!?] alone breaks on abbreviations and initials, and each break
# invents a sentence that was never said -- which then gets scored as a claim.
ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sen rep gov pres sec gen adm capt lt sgt rev hon st
    jr sr inc ltd corp co dept univ assn bros
    jan feb mar apr jun jul aug sept sep oct nov dec
    vs etc al ca approx est fig no vol pp ed eds cf
    u.s u.k u.n e.g i.e a.m p.m
    """.split()
)

# The trailing \s+ is load-bearing: it is why decimals need no guard at all, since the period in
# "3.5" is followed by a digit and never matches. A guard for them rejected every sentence
# ending in a year instead -- "Wages rose in 2019." merged into the sentence after it.
_BOUNDARY = re.compile(r"([.!?]+[\"')\]]*)(\s+)")
_INITIAL = re.compile(r"\b[A-Z]\.$")           # "John F. Kennedy"
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")

# Check-worthiness. Each set is closed by construction; see the module docstring.
AUXILIARIES = frozenset(
    """
    is are was were am be been being has have had does do did
    will would can could may might shall
    isn't aren't wasn't weren't hasn't haven't hadn't doesn't don't didn't
    won't wouldn't can't cannot couldn't
    """.split()
)
IRREGULAR_PAST = frozenset(
    """
    went made took gave saw came got said told put cut held won lost sent spent
    built brought bought paid ran rose fell grew left kept met began became
    chose drove wrote spoke broke stood found led sold taught fought sat
    """.split()
)
QUESTION_OPENERS = frozenset("who what when where why how which whose whom".split())
IMPERATIVE_OPENERS = frozenset(
    """
    look listen tell think consider imagine remember forget ask let stop wait
    vote join give take make see watch
    """.split()
)
# Hedges and deontics. A sentence about what *should* happen has no truth value to retrieve
# against, and one hedged to "probably" is not asserting the thing the model would score.
OPINION_MARKERS = frozenset(
    """
    should shouldn't must mustn't ought needs need believe believes think thinks feel feels
    probably maybe perhaps arguably seems seem apparently allegedly supposedly
    """.split()
)
SUBJECTIVE_PHRASES = ("i think", "i believe", "i feel", "in my opinion", "in my view", "if you ask me")

MONTHS = frozenset(
    "january february march april may june july august september october november december".split()
)
_NUMBER = re.compile(r"\d")
MIN_WORDS = 4


@dataclass(frozen=True, slots=True)
class Sentence:
    """One sentence, with the span it occupies in the transcript."""

    index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether a sentence is worth checking, and which rule decided."""

    worthy: bool
    reason: str

    def __bool__(self) -> bool:
        return self.worthy


def segment(transcript: str) -> list[Sentence]:
    """
    Split a transcript into sentences, keeping each one's span.

    Spans travel with the sentence so the sidebar can point at the transcript rather than at a
    copy of it -- a verdict a reader cannot locate in the source is a verdict they cannot audit.
    """
    sentences: list[Sentence] = []
    start = 0
    for match in _BOUNDARY.finditer(transcript):
        end = match.end(1)
        if not _splits_here(transcript[start:end]):
            continue
        _append(sentences, transcript, start, end)
        start = match.end(2)
    _append(sentences, transcript, start, len(transcript))
    return sentences


def _append(sentences: list[Sentence], transcript: str, start: int, end: int) -> None:
    """Record one sentence, with whitespace trimmed off both the text and its span."""
    raw = transcript[start:end]
    text = raw.strip()
    if not text:
        return
    offset = start + (len(raw) - len(raw.lstrip()))
    sentences.append(Sentence(len(sentences), text, offset, offset + len(text)))


def _splits_here(candidate: str) -> bool:
    """False when the terminator belongs to an abbreviation or an initial rather than a sentence."""
    if candidate.endswith(("!", "?", '."', ".'", ".)", '.”', ".’")):
        return True
    if _INITIAL.search(candidate):
        return False
    words = _WORD.findall(candidate)
    if not words:
        return True
    trailing = candidate.rstrip(".").split()[-1] if candidate.rstrip(".").split() else ""
    return trailing.lower().strip(".,;:\"'()") not in ABBREVIATIONS


def check_worthy(sentence: str) -> Decision:
    """
    Is this sentence an assertion about the world that retrieval could look up?

    Order matters: the cheapest and most certain rejections come first, so the reason a reader
    sees is the most specific one that applies.
    """
    text = sentence.strip()
    lowered = text.lower()
    words = [w.lower() for w in _WORD.findall(text)]

    if len(words) < MIN_WORDS:
        return Decision(False, "too_short")
    if text.endswith("?") or words[0] in QUESTION_OPENERS or words[0] in AUXILIARIES:
        # Auxiliary-initial is subject-auxiliary inversion: "Did the bill pass?" -- a question even
        # when the transcript lost its question mark, which ASR routinely does.
        return Decision(False, "question")
    if words[0] in IMPERATIVE_OPENERS:
        return Decision(False, "imperative")
    if any(phrase in lowered for phrase in SUBJECTIVE_PHRASES) or OPINION_MARKERS & set(words):
        return Decision(False, "opinion")
    if not _has_finite_verb(words):
        return Decision(False, "no_assertion")
    if not _has_anchor(text, words):
        return Decision(False, "no_anchor")
    return Decision(True, "check_worthy")


def _has_finite_verb(words: list[str]) -> bool:
    return any(
        word in AUXILIARIES or word in IRREGULAR_PAST or (len(word) > 4 and word.endswith("ed"))
        for word in words
    )


def _has_anchor(text: str, words: list[str]) -> bool:
    """A number, a month, or a proper noun -- something BM25 can match a title against."""
    if _NUMBER.search(text):
        return True
    if MONTHS & set(words):
        return True
    tokens = _WORD.findall(text)
    return any(token[0].isupper() for token in tokens[1:])


def claims(transcript: str) -> list[tuple[Sentence, Decision]]:
    """Every sentence with its decision -- rejections included, so the demo can show its work."""
    return [(s, check_worthy(s.text)) for s in segment(transcript)]
