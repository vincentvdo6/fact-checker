"""Keep literal article blocks and nearby qualifications available for source reading.

Lexical matches are discovery anchors, never entailment. Reading windows may include
definitions that use different wording from the caption. They remain unverified
source context and cannot establish an assertion just by being near an anchor.
"""

from __future__ import annotations

import re

from src.pipeline.topic_query import content_terms
from src.retrieval.source_dates import publication_notices
from src.retrieval.visible_text import visible_text


def _blocks(markdown: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", markdown[:160_000])
    if len(markdown) > 160_000:
        blocks.pop()
    return [block.strip() for block in blocks]


def _notes(blocks: list[str]) -> tuple[set[int], bool]:
    positions: set[int] = set()
    in_notes = False
    for index, text in enumerate(blocks):
        if text.startswith("# "):
            in_notes = text[2:].strip().casefold() in {"notes", "footnotes", "endnotes"}
        if in_notes:
            positions.add(index)
    return positions, in_notes


def _prose(text: str, minimum: int = 80) -> str:
    if not minimum <= len(text) <= 1400 or text.startswith(("|", "![", "#", "](")):
        return ""
    if re.match(r"^(?:Table|Chart|Figure)\s+[A-Z0-9][.:]", text):
        return ""
    visible, linked_length = visible_text(text)
    if re.search(r"(?:\[\s*(?:\.{3}|…)\s*\]|…|\.{3})(?:\s*(?:Read more|Continue reading))?[.!]?$", visible, re.I):
        return ""
    if linked_length > len(visible) // 2:
        return ""
    if re.search(r"Highcharts|Hover over chart|Choose another chart|Go to selected chart", text, re.I):
        return ""
    return visible


def source_excerpts(markdown: str, query: str, *, context_query: str = "") -> list[str]:
    """Return the strongest intact lexical match, preserving the existing candidate policy."""
    terms = set(content_terms(query))
    context_terms = set(content_terms(context_query)) - terms
    candidates = []
    blocks = _blocks(markdown)
    notes, _ = _notes(blocks)
    for position, text in enumerate(blocks):
        if position in notes:
            continue
        passage_terms = set(content_terms(_prose(text)))
        overlap = terms.intersection(passage_terms)
        if terms and len(overlap) >= min(2, len(terms)):
            candidates.append((len(overlap), len(context_terms.intersection(passage_terms)),
                               len(overlap) / len(passage_terms), position, text))
    selected = sorted(candidates, key=lambda item: (-item[0], -item[1], -item[2], item[3]))[:1]
    return [text for _, _, _, _, text in selected]


def reading_passages(markdown: str, query: str) -> list[str]:
    """Keep up to nine complete paragraphs around matching prose or section headings.

    Heading text may locate a definition but is never itself quoted as evidence.
    reading_window separately carries short qualifications and section context.
    Measurements and qualifications can precede or follow an anchor, so its window
    includes four intact blocks in each direction without favoring numeric text.
    """
    terms = set(content_terms(query))
    if not terms:
        return []
    blocks = _blocks(markdown)
    notes, _ = _notes(blocks)
    anchors = []
    for position, text in enumerate(blocks):
        if position in notes:
            continue
        if text.startswith("# "):
            visible, linked_length = visible_text(text)
            if linked_length > len(visible) // 2:
                continue
        else:
            visible = _prose(text)
        overlap = terms.intersection(content_terms(visible))
        if len(overlap) >= min(2, len(terms)):
            anchors.append((-len(overlap), position))
    chosen: set[int] = set()
    for _, position in sorted(anchors):
        for nearby in range(max(0, position - 4), min(position + 5, len(blocks))):
            if nearby not in notes and _prose(blocks[nearby]):
                chosen.add(nearby)
            if len(chosen) == 9:
                return [blocks[index] for index in sorted(chosen)]
    return [blocks[index] for index in sorted(chosen)]


def reading_window(markdown: str, query: str) -> dict:
    """Pair citable paragraphs with literal headings and short adjacent qualifications.

    All preceding headings travel as section context because the HTML reader does not
    preserve their nesting. Complete notes sections also travel as context, since a
    distant footnote can qualify a measurement. Context is not independently citable.
    If it cannot fit intact, return a reason so callers cannot use unqualified excerpts.
    """
    passages = reading_passages(markdown, query)
    blocks = _blocks(markdown)
    positions = [index for index, text in enumerate(blocks) if text in passages]
    if not positions:
        return {"passages": [], "context": []}
    if len(markdown) > 2_000_000:
        return {"passages": [], "context": [], "unavailable_reason": "Source context exceeds the validation limit."}
    if len(markdown) > 160_000:
        readable_end = max((separator.start() for separator in re.finditer(r"\n\s*\n", markdown[:160_000])), default=0)
        for heading in re.finditer(r"(?im)^# (?:notes|footnotes|endnotes)[ \t\r]*$", markdown):
            if heading.end() > readable_end:
                return {"passages": [], "context": [], "unavailable_reason": "Source notes extend beyond the reading limit."}
    context_positions = {index for index, text in enumerate(blocks[:max(positions) + 1]) if text.startswith("# ")}
    notices = publication_notices(markdown)
    context_positions.update(index for index, text in enumerate(blocks) if text in notices)
    notes, in_notes = _notes(blocks)
    context_positions.update(notes)
    if in_notes and len(markdown) > 160_000:
        return {"passages": [], "context": [], "unavailable_reason": "Source notes extend beyond the reading limit."}
    for position in positions:
        for direction in (-1, 1):
            adjacent = position + direction
            while 0 <= adjacent < len(blocks):
                text = blocks[adjacent]
                if text.startswith("# ") or (len(text) < 80 and _prose(text, minimum=1)):
                    context_positions.add(adjacent)
                    adjacent += direction
                else:
                    break
    context = [blocks[index] for index in sorted(context_positions) if blocks[index] not in passages]
    if sum(map(len, context)) > 8000:
        return {"passages": [], "context": [], "unavailable_reason": "Complete source context exceeds the reading limit."}
    return {"passages": passages, "context": context}
