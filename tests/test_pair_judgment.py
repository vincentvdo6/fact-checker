"""A pair judgment may only quote the one sentence it was asked about."""

from __future__ import annotations

import pytest

from src.verdict.pair_judgment import (
    INSTRUCTIONS,
    RELATIONS,
    nli_judgment,
    pair_request,
    pair_schema,
    validate_pair_judgment,
)

ASSERTION = {"id": "assertion-1", "text": "We don't have a labor shortage."}
UNIT = {"id": "s1:p1:u1", "passage_id": "s1:p1", "source_id": "s1",
        "text": "Wisconsin's labor shortage is a major barrier to growing the state's economy, a new report finds.",
        "roles": ["reported_observation"], "eligible": True,
        "definitions": [{"id": "s1:p1:u2", "text": "A shortage means fewer workers than openings."}]}


def test_request_carries_only_assertion_sentence_and_own_definitions():
    request = pair_request(ASSERTION, UNIT)
    assert request == {"assertion": ASSERTION["text"], "sentence": UNIT["text"],
                       "definitions": ["A shortage means fewer workers than openings."]}
    assert "s1" not in str(request) and "url" not in request
    schema = pair_schema(UNIT)
    assert schema["properties"]["relation"]["enum"] == list(RELATIONS)
    assert schema["properties"]["span"]["maxLength"] == len(UNIT["text"])
    assert schema["properties"]["qualifiers"]["items"]["maxLength"] == len(UNIT["text"])
    flat = " ".join(INSTRUCTIONS.split())
    assert "does not state a fact" in flat and "not measured" in flat


def test_spans_and_qualifiers_must_be_exact_substrings():
    good = validate_pair_judgment(ASSERTION, UNIT, {"relation": "states_negation",
                                                    "span": "Wisconsin's labor shortage is a major barrier",
                                                    "qualifiers": ["Wisconsin", "Wisconsin"]})
    assert good == {"assertion_id": "assertion-1", "unit_id": "s1:p1:u1", "relation": "states_negation",
                    "span": "Wisconsin's labor shortage is a major barrier", "qualifiers": ["Wisconsin"],
                    "status": "unverified"}
    for bad in [
        {"relation": "states", "span": "Wisconsin has a labor shortage", "qualifiers": []},
        {"relation": "states", "span": "", "qualifiers": []},
        {"relation": "states", "span": "  ", "qualifiers": []},
        {"relation": "unrelated", "span": "Wisconsin", "qualifiers": []},
        {"relation": "bears_on", "span": "Wisconsin", "qualifiers": ["the state of Wisconsin"]},
        {"relation": "bears_on", "span": "Wisconsin", "qualifiers": ["Wisconsin", "labor", "state", "report", "new"]},
        {"relation": "proves", "span": "Wisconsin", "qualifiers": []},
        {"relation": "states", "span": "Wisconsin", "qualifiers": [], "note": "extra"},
        {"relation": "states", "span": "Wisconsin"},
    ]:
        with pytest.raises(ValueError):
            validate_pair_judgment(ASSERTION, UNIT, bad)
    assert validate_pair_judgment(ASSERTION, UNIT, {"relation": "unrelated", "span": "", "qualifiers": []})["span"] == ""


def test_nli_adapter_cannot_express_bears_on_or_qualifiers():
    row = nli_judgment(ASSERTION, UNIT, "contradicted", 0.71)
    assert row["relation"] == "states_negation" and row["span"] == UNIT["text"] and row["qualifiers"] == []
    assert row["confidence"] == 0.71 and row["status"] == "unverified"
    assert nli_judgment(ASSERTION, UNIT, "not_enough_evidence", 0.5)["relation"] == "unrelated"
    assert nli_judgment(ASSERTION, UNIT, "not_enough_evidence", 0.5)["span"] == ""
    with pytest.raises(ValueError):
        nli_judgment(ASSERTION, UNIT, "mixed", 0.5)
