"""Only reported observations may become evidence; everything a reader must discount is context."""

from __future__ import annotations

import pytest

from src.verdict.eligibility import CONTEXT_ROLES, gate_units
from src.verdict.reading import reading_packet, validate_roles

PARAGRAPH = ("The true rate was 24.3% in April. It is defined as the share of the labor force without a living wage. "
             "In our view this shows a shortage of workers. Analysts expect a rise next week. "
             "For example, a ratio of 0.39 means 39 workers per 100 openings. Click here for data.")


def reading():
    return reading_packet({"sources": [
        {"id": "s1", "url": "https://example.org/a", "published_at": "2025-05-01",
         "temporal_status": "published_by_cutoff", "excerpts": [PARAGRAPH, "Employment fell 2%. Definitions vary."]}]})


def roles(**overrides):
    base = {"s1:p1:u1": ["reported_observation"], "s1:p1:u2": ["definition"],
            "s1:p1:u3": ["reported_observation", "attributed_opinion"], "s1:p1:u4": ["forecast"],
            "s1:p1:u5": ["hypothetical", "definition"], "s1:p1:u6": ["instruction"],
            "s1:p2:u1": ["reported_observation", "definition"], "s1:p2:u2": ["definition"]}
    base.update(overrides)
    return [{"unit_id": key, "roles": value} for key, value in base.items()]


def test_a_caption_role_withholds_with_its_own_reason():
    gate = gate_units(reading(), roles(**{"s1:p1:u1": ["reported_observation", "caption"]}))
    first = gate["units"][0]
    assert not first["eligible"] and first["withheld"] == "image caption or credit"
    assert gate["withheld"]["image caption or credit"] == 1 and "caption" in CONTEXT_ROLES


def test_context_roles_withhold_even_when_paired_with_an_observation():
    gate = gate_units(reading(), validate_roles(reading(), roles()))
    by_id = {unit["id"]: unit for unit in gate["units"]}
    assert gate["eligible_ids"] == ["s1:p1:u1", "s1:p2:u1"]
    assert by_id["s1:p1:u3"]["withheld"] == "attributed opinion" and not by_id["s1:p1:u3"]["eligible"]
    assert by_id["s1:p1:u4"]["withheld"] == "forecast or expectation"
    assert by_id["s1:p1:u5"]["withheld"] == "hypothetical or illustration"
    assert by_id["s1:p1:u6"]["withheld"] == "instruction or navigation text"
    assert by_id["s1:p1:u2"]["withheld"] == "definition without a reported observation"
    assert gate["withheld"] == {"attributed opinion": 1, "forecast or expectation": 1,
                                "hypothetical or illustration": 1, "instruction or navigation text": 1,
                                "definition without a reported observation": 2}
    assert set(CONTEXT_ROLES) == {"attributed_opinion", "forecast", "hypothetical", "instruction", "caption"}


def test_unknown_role_is_withheld_not_guessed():
    gate = gate_units(reading(), validate_roles(reading(), roles(**{"s1:p1:u1": ["unknown"]})))
    assert "s1:p1:u1" not in gate["eligible_ids"]
    assert gate["withheld"]["role not established"] == 1


def test_same_paragraph_definitions_travel_without_self_context_roles_or_other_paragraphs():
    gate = gate_units(reading(), validate_roles(reading(), roles()))
    by_id = {unit["id"]: unit for unit in gate["units"]}
    attached = [item["id"] for item in by_id["s1:p1:u1"]["definitions"]]
    assert attached == ["s1:p1:u2"], "the hypothetical example is a definition with a context role"
    assert "labor force" in by_id["s1:p1:u1"]["definitions"][0]["text"]
    assert [item["id"] for item in by_id["s1:p2:u1"]["definitions"]] == ["s1:p2:u2"]
    assert "definitions" not in by_id["s1:p1:u2"]


def test_annotations_must_match_the_reading_exactly():
    rows = validate_roles(reading(), roles())
    with pytest.raises(ValueError):
        gate_units(reading(), rows[:-1])
    with pytest.raises(ValueError):
        gate_units(reading(), rows + [{"unit_id": "s1:p3:u1", "roles": ["definition"]}])
    with pytest.raises(ValueError):
        gate_units(reading(), rows[:-1] + [{"unit_id": rows[-1]["unit_id"], "roles": ["evidence"]}])
    with pytest.raises(ValueError):
        gate_units(reading(), rows[:-1] + [rows[0]])


def test_gate_rows_carry_the_passage_origin():
    packet = reading()
    packet["passages"][1]["origin"] = "context"
    gate = gate_units(packet, validate_roles(packet, roles()))
    by_id = {unit["id"]: unit for unit in gate["units"]}
    assert by_id["s1:p1:u1"]["origin"] == "assertion" and by_id["s1:p2:u1"]["origin"] == "context"
    assert gate["eligible_ids"] == ["s1:p1:u1", "s1:p2:u1"], "origin is not an eligibility rule"
