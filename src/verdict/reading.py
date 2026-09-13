"""Give every source sentence a stable identity before anything reads it for a claim.

The decomposed chain judges sentences, so each one needs an ID that survives from the
source-only role reading through the gate, the pair judgments and the rendered answer.
Offsets refer to the untouched paragraph text: a unit is a span of the original, never
a rewritten sentence. Role labels attached to these IDs are model readings of the
source alone; they carry no claim and prove nothing about it.
"""

from __future__ import annotations

import re

from src.pipeline.segment import segment

_LINE = re.compile(r"[^\n]+")

ROLES = ("reported_observation", "definition", "hypothetical", "forecast",
         "attributed_opinion", "instruction", "unknown")


def reading_packet(packet: dict) -> dict:
    """Passages with sentence units for every source excerpt; no claim or captions travel.

    `excerpts` were found by researching the claim's own assertions. `context_excerpts` were
    found by caption-concept research, which explains the claim's terms and, by the retrieval
    rules, never resolves the assertions themselves; each passage records which it was.
    """
    sources, passages = [], []
    for source in packet["sources"]:
        sources.append({key: source[key] for key in ("id", "url", "published_at", "publication_basis",
                                                      "temporal_status", "reading_context") if key in source})
        excerpts = [(text, "assertion") for text in source["excerpts"]]
        excerpts += [(text, "context") for text in source.get("context_excerpts", [])]
        for index, (text, origin) in enumerate(excerpts):
            identity = f"{source['id']}:p{index + 1}"
            # A heading kept with its paragraph sits on its own line; a line break ends a sentence.
            spans = [(line.start() + sentence.start, line.start() + sentence.end)
                     for line in _LINE.finditer(text) for sentence in segment(line.group())]
            units = [{"id": f"{identity}:u{number}", "text": text[start:end], "start": start, "end": end}
                     for number, (start, end) in enumerate(spans, 1)]
            if any(text[unit["start"]:unit["end"]] != unit["text"] for unit in units):
                raise ValueError("Sentence spans must be exact substrings of the paragraph.")
            passages.append({"id": identity, "source_id": source["id"], "text": text, "origin": origin, "units": units})
    identities = [unit["id"] for passage in passages for unit in passage["units"]]
    if len(identities) != len(set(identities)):
        raise ValueError("Sentence identities must be unique.")
    return {"sources": sources, "passages": passages}


def validate_roles(reading: dict, annotations: object) -> list[dict]:
    """One valid role set per supplied sentence, in source order; nothing invented or missing."""
    identities = [unit["id"] for passage in reading["passages"] for unit in passage["units"]]
    if not isinstance(annotations, list):
        raise ValueError("Role annotations must be a list.")
    found: dict[str, list[str]] = {}
    for row in annotations:
        if not isinstance(row, dict) or not isinstance(row.get("unit_id"), str):
            raise ValueError("Role annotations must name a sentence.")
        identity, roles = row["unit_id"], row.get("roles")
        if identity not in identities or identity in found:
            raise ValueError("Role annotations omitted, repeated or invented a source sentence.")
        if (not isinstance(roles, list) or not roles or len(set(roles)) != len(roles)
                or any(not isinstance(role, str) or role not in ROLES for role in roles)
                or ("unknown" in roles and len(roles) != 1)):
            raise ValueError("Role annotations contain invalid or conflicting labels.")
        found[identity] = list(roles)
    if set(found) != set(identities):
        raise ValueError("Role annotations must cover every source sentence.")
    return [{"unit_id": identity, "roles": found[identity]} for identity in identities]
