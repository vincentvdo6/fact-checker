"""Content-focused lexical queries for conversational claims, without altering assertions.

The FEVER tokenizer and index remain unchanged. Removing closed-class words and repeated
query terms prevents contractions and conversational framing from dominating title matches.
Only preceding speech sharing a claim term supplies context; it remains a search hint.
This is an uncalibrated live retrieval policy, not an evidence-sufficiency test.
"""

from __future__ import annotations

from src.pipeline.transcript import ClaimContext
from src.retrieval.text import tokenize

FUNCTION_WORDS = frozenset("""
    a an the and or but so nor for yet as at by from in into of on onto to with without
    about above after again against all also am any are around be because been before
    being below between both can could did do does doing down during each either else
    few further had has have having he her here hers herself him himself his how i if
    is it its itself just let like me more most much my myself neither no not now off
    once only other our ours ourselves out over own same shall she should some such
    than that their theirs them themselves then there these they this those through
    too under until up us very was we were what when where which while who whom why
    will would you your yours yourself yourselves yes okay ok well
    don doesn didn isn aren wasn weren hasn haven hadn won wouldn shouldn couldn
    mustn needn t s re ve ll d m
""".split())
CONTEXT_WORDS = 96


def content_terms(text: str) -> list[str]:
    """Keep index-compatible terms, with each query term contributing once."""
    return list(dict.fromkeys(term for term in tokenize(text) if term not in FUNCTION_WORDS))


def topic_query(context: ClaimContext) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Use bounded earlier speech with lexical topic overlap; never future speech or a title."""
    claim_terms = content_terms(context.claim.text)
    anchors = {term for term in claim_terms if term.isalpha()}
    preceding = [item for item in context.preceding if item.final and item.start <= context.claim.start
                 and item.end <= context.claim.end
                 and anchors.intersection(content_terms(item.text))]
    selected = []
    remaining = CONTEXT_WORDS
    for item in reversed(preceding):
        words = item.text.split()[-remaining:]
        selected.append((item.id, " ".join(words)))
        remaining -= len(words)
        if remaining <= 0:
            break
    selected.reverse()
    terms = list(dict.fromkeys([*claim_terms, *content_terms(" ".join(text for _, text in selected))]))
    return (" ".join(terms) or context.claim.text,
            tuple(key for key, _ in selected), tuple(text for _, text in selected))
