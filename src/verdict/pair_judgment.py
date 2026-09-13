"""Ask one narrow question per (assertion, sentence) pair and accept only verbatim answers.

Whole-claim synthesis over dozens of paragraphs let every model tried rewrite
definitions, change a statistic's population and misattribute sources in the prose
where citation checks could not see it. Here the judge sees one assertion, one
eligible observation sentence and that sentence's own-paragraph definitions, and
returns a relation label plus exact substrings of the sentence. There is no prose
field to mutate anything in. `states` and `states_negation` are the only relations
that can move a verdict; `bears_on` marks a relevant measurement or finding that
stops short of the assertion and is only ever shown, never counted.

Qualifiers are exact substrings that narrow what the sentence is about (a place, a
population, a period). Composition uses them only to weaken a relation, never to
strengthen one, so over-extraction errs toward abstention. A judgment is a model
reading bound to one sentence; validation proves it quotes the sentence, not that it
read it correctly.
"""

from __future__ import annotations

RELATIONS = ("states", "states_negation", "bears_on", "unrelated")
MAX_QUALIFIERS = 4

INSTRUCTIONS = """\
You are given one assertion and one sentence from a source, with any definitions from
the sentence's own paragraph. Read the sentence literally. Decide which one relation
holds and return a JSON object with exactly these fields:

relation: one of
  "states"          the sentence itself reports that the assertion is the case.
  "states_negation" the sentence itself reports that the assertion is not the case.
  "bears_on"        the sentence reports a measurement, event or finding about the
                    same subject that neither states nor denies the assertion.
  "unrelated"       the sentence is about something else.
span: the exact words of the sentence that carry the relation, copied character for
  character; an empty string when the relation is "unrelated".
qualifiers: exact substrings of the sentence naming a place, population, period or
  other limit on what the sentence reports; an empty list if there are none.

Rules. A sentence that reports what someone thinks, expects, argues or predicts does
not state a fact. Reporting that something was not announced, not measured or not
tested is not a statement that it did not happen. A definition or an example does
not report a measurement. A finding about a narrower place or group than the
assertion is "states" or "states_negation" only with that place or group listed as
a qualifier. Do not use the assertion to fill in what the sentence leaves unsaid."""


def pair_request(assertion: dict, unit: dict) -> dict:
    """Exactly what the judge may read: the assertion, the sentence and its own definitions."""
    definitions = [item["text"] for item in unit.get("definitions", [])]
    return {"assertion": assertion["text"], "sentence": unit["text"], "definitions": definitions}


def pair_schema(unit: dict) -> dict:
    """Bound every text field by the sentence it must quote."""
    length = len(unit["text"])
    return {"type": "object", "additionalProperties": False,
            "properties": {"relation": {"type": "string", "enum": list(RELATIONS)},
                           "span": {"type": "string", "maxLength": length},
                           "qualifiers": {"type": "array", "maxItems": MAX_QUALIFIERS,
                                          "items": {"type": "string", "minLength": 1, "maxLength": length}}},
            "required": ["relation", "span", "qualifiers"]}


def validate_pair_judgment(assertion: dict, unit: dict, response: object) -> dict:
    """Accept a relation only when its span and qualifiers are exact substrings of the sentence."""
    if not isinstance(response, dict) or set(response) != {"relation", "span", "qualifiers"}:
        raise ValueError("A pair judgment must contain exactly relation, span and qualifiers.")
    relation, span, qualifiers = response["relation"], response["span"], response["qualifiers"]
    if relation not in RELATIONS:
        raise ValueError("A pair judgment names an unknown relation.")
    text = unit["text"]
    if not isinstance(span, str):
        raise ValueError("A pair judgment span must be text.")
    if relation == "unrelated":
        if span:
            raise ValueError("An unrelated judgment must not quote a span.")
    elif not span.strip() or span not in text:
        raise ValueError("A pair judgment span must be an exact substring of the sentence.")
    if (not isinstance(qualifiers, list) or len(qualifiers) > MAX_QUALIFIERS
            or any(not isinstance(item, str) or not item.strip() or item not in text for item in qualifiers)):
        raise ValueError("Pair judgment qualifiers must be exact substrings of the sentence.")
    return {"assertion_id": assertion["id"], "unit_id": unit["id"], "relation": relation, "span": span,
            "qualifiers": list(dict.fromkeys(qualifiers)), "status": "unverified"}


NLI_RELATIONS = {"supported": "states", "contradicted": "states_negation", "not_enough_evidence": "unrelated"}


def nli_judgment(assertion: dict, unit: dict, label: str, confidence: float) -> dict:
    """Map a three-way NLI verdict onto the relation space; it cannot express bears_on or qualifiers."""
    if label not in NLI_RELATIONS:
        raise ValueError("Unknown NLI label.")
    relation = NLI_RELATIONS[label]
    return {"assertion_id": assertion["id"], "unit_id": unit["id"], "relation": relation,
            "span": unit["text"] if relation != "unrelated" else "", "qualifiers": [],
            "confidence": float(confidence), "status": "unverified"}
