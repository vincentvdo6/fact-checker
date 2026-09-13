"""Preserve literal inline references while keeping destinations out of prose relevance."""

from __future__ import annotations

import re


def _closing(text: str, start: int, opening: str, closing: str) -> int | None:
    depth, end = 1, start
    while end < len(text) and depth:
        if text[end] == "\\" and end + 1 < len(text):
            end += 2
            continue
        if text[end] == opening:
            depth += 1
        elif text[end] == closing:
            depth -= 1
        end += 1
    return end if not depth else None


def _links(text: str) -> list[dict] | None:
    links, cursor = [], 0
    for match in re.finditer(r"\[([^\]]*)\]\(", text):
        if match.start() < cursor:
            continue
        end = _closing(text, match.end(), "(", ")")
        if end is None:
            return None
        links.append({"label": match[1], "destination": text[match.end():end - 1],
                      "start": match.start(), "end": end})
        cursor = end
    return links


def _escaped(text: str, position: int) -> bool:
    backslashes = 0
    while position > 0 and text[position - 1] == "\\":
        backslashes += 1
        position -= 1
    return bool(backslashes % 2)


def inline_links(text: str) -> list[dict]:
    """Return literal label/destination and end-exclusive markup spans for text references.

    Images, nested labels and escaped opening brackets are omitted. Label escapes stay
    literal. Unmatched label openers are skipped; an unclosed destination rejects the
    whole input. Destinations retain exact escapes and require caller transport validation.
    """
    links, cursor = [], 0
    while (start := text.find("[", cursor)) >= 0:
        cursor = start + 1
        if _escaped(text, start):
            continue
        label_end = _closing(text, start + 1, "[", "]")
        if label_end is None:
            continue
        cursor = label_end
        if label_end == len(text) or text[label_end] != "(":
            continue
        end = _closing(text, label_end + 1, "(", ")")
        if end is None:
            return []
        cursor = end
        label = text[start + 1:label_end - 1]
        if "[" in label or (start > 0 and text[start - 1] == "!" and not _escaped(text, start - 1)):
            continue
        links.append({"label": label, "destination": text[label_end + 1:end - 1], "start": start, "end": end})
    return links


def visible_text(text: str) -> tuple[str, int]:
    """Keep labels and balanced parentheses; malformed destinations cannot leak terms."""
    links = _links(text)
    if links is None:
        return "", 0
    parts, linked_length, cursor = [], 0, 0
    for link in links:
        parts.extend((text[cursor:link["start"]], link["label"]))
        linked_length += len(link["label"])
        cursor = link["end"]
    parts.append(text[cursor:])
    return re.sub(r"https?://\S+", "", "".join(parts)), linked_length
