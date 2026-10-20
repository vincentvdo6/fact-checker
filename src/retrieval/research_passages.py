"""Keep literal article blocks and nearby qualifications available for source reading.

Lexical matches are discovery anchors, never entailment. Reading windows may include
definitions that use different wording from the caption. They remain unverified
source context and cannot establish an assertion just by being near an anchor.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

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


def _bounded_positions(blocks: list[str], notes: set[int], anchors: Sequence[int],
                       selected: Sequence[int] = ()) -> list[int]:
    chosen = set(selected)
    for position in [*selected, *anchors]:
        if len(chosen) >= 9:
            return sorted(chosen)
        for nearby in range(max(0, position - 4), min(position + 5, len(blocks))):
            if nearby not in notes and _prose(blocks[nearby]):
                chosen.add(nearby)
            if len(chosen) >= 9:
                return sorted(chosen)
    return sorted(chosen)


def _passage_positions(blocks: list[str], query: str, selected_excerpts: Sequence[str]) -> list[int]:
    terms = set(content_terms(query))
    notes, _ = _notes(blocks)
    selected = []
    for excerpt in selected_excerpts:
        position = next((index for index, block in enumerate(blocks)
                         if block == excerpt and index not in notes and _prose(block)), None)
        if position is None:
            raise ValueError("A selected excerpt must be an exact citable source block.")
        if position not in selected:
            selected.append(position)
    if len(selected) > 9:
        raise ValueError("More selected excerpts than the nine-paragraph reading limit.")
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
        if terms and len(overlap) >= min(2, len(terms)):
            anchors.append((-len(overlap), position))

    ordered = [position for _, position in sorted(anchors)]
    original = _bounded_positions(blocks, notes, ordered)
    if not selected or set(selected).issubset(original):
        return original

    # Re-anchor only when the original window omits a selected excerpt.
    selected_texts = {blocks[position] for position in selected}
    remaining = [position for position in ordered if blocks[position] not in selected_texts or position in selected]
    return _bounded_positions(blocks, notes, remaining, selected)


def reading_passages(markdown: str, query: str, *, selected_excerpts: Sequence[str] = ()) -> list[str]:
    """Keep up to nine complete paragraphs around matching prose or section headings.

    Heading text may locate a definition but is never itself quoted as evidence.
    reading_window separately carries short qualifications and section context.
    Measurements and qualifications can precede or follow an anchor, so its window
    includes four intact blocks in each direction without favoring numeric text.
    Selected excerpts must be exact citable blocks; re-anchor only if one is missing.
    """
    blocks = _blocks(markdown)
    return [blocks[index] for index in _passage_positions(blocks, query, selected_excerpts)]


def reading_window(markdown: str, query: str, *, selected_excerpts: Sequence[str] = ()) -> dict:
    """Pair citable paragraphs with literal headings and short adjacent qualifications.

    All preceding headings travel as section context because the HTML reader does not
    preserve their nesting. Complete notes sections also travel as context, since a
    distant footnote can qualify a measurement. Context is not independently citable.
    If it cannot fit intact, return a reason so callers cannot use unqualified excerpts.
    """
    blocks = _blocks(markdown)
    positions = _passage_positions(blocks, query, selected_excerpts)
    passages = [blocks[index] for index in positions]
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
