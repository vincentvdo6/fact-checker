"""Decide in code which source sentences may ever support or contradict a claim.

Every local model tried so far promoted attributed opinion, forecasts and worked
examples to proof once they sat beside a claim. Prompt instructions did not stop it.
This gate removes those sentences before any claim-facing judgment sees them: a
sentence is a verdict candidate only when its source-only role reading calls it a
reported observation and nothing else that a reader would have to discount. Everything
else stays visible as context and is never cited as evidence. Role labels remain
unverified model judgments; the gate composes them, it does not certify them.

Definitions in the same paragraph travel with an observation so the judgment stage
reads a measurement together with what it measures. They cannot be candidates alone.
"""

from __future__ import annotations

from src.verdict.reading import ROLES

# A sentence carrying any of these is read as context however it is also labelled.
CONTEXT_ROLES = ("attributed_opinion", "forecast", "hypothetical", "instruction", "caption")

REASONS = {
    "attributed_opinion": "attributed opinion",
    "forecast": "forecast or expectation",
    "hypothetical": "hypothetical or illustration",
    "instruction": "instruction or navigation text",
    "caption": "image caption or credit",
    "definition": "definition without a reported observation",
    "unknown": "role not established",
}


def _annotations(annotations: object) -> dict[str, list[str]]:
    if not isinstance(annotations, list):
        raise ValueError("Role annotations must be a list.")
    found: dict[str, list[str]] = {}
    for row in annotations:
        if not isinstance(row, dict) or not isinstance(row.get("unit_id"), str) or row["unit_id"] in found:
            raise ValueError("Role annotations must name each sentence once.")
        roles = row.get("roles")
        if (not isinstance(roles, list) or not roles or len(set(roles)) != len(roles)
                or any(not isinstance(role, str) or role not in ROLES for role in roles)):
            raise ValueError("Role annotations contain invalid labels.")
        found[row["unit_id"]] = list(roles)
    return found


def _reason(roles: list[str]) -> str:
    for role in CONTEXT_ROLES:
        if role in roles:
            return REASONS[role]
    if "unknown" in roles:
        return REASONS["unknown"]
    if "reported_observation" not in roles:
        return REASONS["definition"]
    return ""


def gate_units(reading: dict, annotations: object) -> dict:
    """Return every sentence with its verdict eligibility and the reason it was withheld.

    `reading` is a source_reading_packet; every sentence must be annotated and no
    annotation may name a sentence that is not there. Eligible units carry the
    definition sentences of their own paragraph as read-only context.
    """
    roles = _annotations(annotations)
    identities = [unit["id"] for passage in reading["passages"] for unit in passage["units"]]
    if len(identities) != len(set(identities)):
        raise ValueError("The reading packet contains duplicate sentence identities.")
    if set(roles) != set(identities):
        raise ValueError("Role annotations must cover exactly the supplied sentences.")
    units, withheld = [], {}
    for passage in reading["passages"]:
        definitions = [unit for unit in passage["units"] if "definition" in roles[unit["id"]]
                       and not any(role in CONTEXT_ROLES for role in roles[unit["id"]])]
        for unit in passage["units"]:
            reason = _reason(roles[unit["id"]])
            row = {"id": unit["id"], "passage_id": passage["id"], "source_id": passage["source_id"],
                   "origin": passage.get("origin", "assertion"), "text": unit["text"],
                   "roles": list(roles[unit["id"]]), "eligible": not reason}
            if reason:
                row["withheld"] = reason
                withheld[reason] = withheld.get(reason, 0) + 1
            else:
                row["definitions"] = [{"id": item["id"], "text": item["text"]}
                                      for item in definitions if item["id"] != unit["id"]]
            units.append(row)
    return {"units": units, "eligible_ids": [row["id"] for row in units if row["eligible"]],
            "withheld": withheld, "rule": "reported_observation without opinion, forecast, hypothetical or instruction"}
