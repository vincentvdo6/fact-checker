"""Sentence identities must be exact spans of the source, and role sets must cover them exactly."""

from __future__ import annotations

import pytest

from src.verdict.reading import ROLES, reading_packet, validate_roles

TEXT = "The rate was 24.3% in April. It is defined as the share of the labor force without a living wage."


def packet():
    return {"claim": "We have a good job shortage.", "claim_context": "caption words",
            "sources": [{"id": "s1", "url": "https://example.org/a", "published_at": "2025-05-01",
                         "temporal_status": "published_by_cutoff", "excerpts": [TEXT, "Second paragraph."]}]}


def test_units_are_exact_spans_with_stable_ids_and_no_claim_text():
    reading = reading_packet(packet())
    assert set(reading) == {"sources", "passages"}
    assert "caption words" not in str(reading) and "good job shortage" not in str(reading)
    first = reading["passages"][0]
    assert first["id"] == "s1:p1" and [unit["id"] for unit in first["units"]] == ["s1:p1:u1", "s1:p1:u2"]
    for unit in first["units"]:
        assert TEXT[unit["start"]:unit["end"]] == unit["text"]
    assert "24.3%" in first["units"][0]["text"] and reading["passages"][1]["id"] == "s1:p2"
    assert reading["sources"][0]["temporal_status"] == "published_by_cutoff"


def test_roles_must_cover_every_sentence_exactly_once_with_known_labels():
    reading = reading_packet(packet())
    rows = [{"unit_id": "s1:p2:u1", "roles": ["definition"]}, {"unit_id": "s1:p1:u2", "roles": ["definition"]},
            {"unit_id": "s1:p1:u1", "roles": ["reported_observation"]}]
    checked = validate_roles(reading, rows)
    assert [row["unit_id"] for row in checked] == ["s1:p1:u1", "s1:p1:u2", "s1:p2:u1"]
    with pytest.raises(ValueError):
        validate_roles(reading, rows[:2])
    with pytest.raises(ValueError):
        validate_roles(reading, rows + [{"unit_id": "s1:p9:u1", "roles": ["definition"]}])
    with pytest.raises(ValueError):
        validate_roles(reading, rows + [rows[0]])
    with pytest.raises(ValueError):
        validate_roles(reading, rows[:2] + [{"unit_id": "s1:p1:u1", "roles": ["proof"]}])
    with pytest.raises(ValueError):
        validate_roles(reading, rows[:2] + [{"unit_id": "s1:p1:u1", "roles": ["unknown", "definition"]}])
    with pytest.raises(ValueError):
        validate_roles(reading, rows[:2] + [{"unit_id": "s1:p1:u1", "roles": []}])
    assert "reported_observation" in ROLES and "unknown" in ROLES


def test_context_excerpts_follow_assertion_excerpts_and_carry_their_origin():
    source = packet()["sources"][0] | {"context_excerpts": ["Living wage pages explain the term.", TEXT]}
    reading = reading_packet({"sources": [source]})
    assert [(row["id"], row["origin"]) for row in reading["passages"]] == [
        ("s1:p1", "assertion"), ("s1:p2", "assertion"), ("s1:p3", "context"), ("s1:p4", "context")]
    assert reading["passages"][2]["units"][0]["id"] == "s1:p3:u1"
    assert all(row["origin"] == "assertion" for row in reading_packet(packet())["passages"])


def test_a_line_break_ends_a_sentence_so_a_heading_is_its_own_unit():
    glued = "About LISEP\nThe institute was created in 2019. It does research.\n\nSecond line."
    source = packet()["sources"][0] | {"excerpts": [glued]}
    passage = reading_packet({"sources": [source]})["passages"][0]
    assert [unit["text"] for unit in passage["units"]] == [
        "About LISEP", "The institute was created in 2019.", "It does research.", "Second line."]
    assert [unit["id"] for unit in passage["units"]] == ["s1:p1:u1", "s1:p1:u2", "s1:p1:u3", "s1:p1:u4"]
    assert all(glued[unit["start"]:unit["end"]] == unit["text"] for unit in passage["units"])


def test_archived_annotations_project_onto_split_units_by_span_not_identity():
    from scripts.replay_decomposed import project_annotations

    glued = "About TRU\nThe paper was issued in 2020. It is cited widely."
    packet = {"sources": [{"id": "s1", "url": "https://example.org/a", "excerpts": [glued]}]}
    archived = [{"unit_id": "s1:p1:u1", "roles": ["definition"]}, {"unit_id": "s1:p1:u2", "roles": ["attributed_opinion"]}]
    assert project_annotations(packet, archived) == [
        {"unit_id": "s1:p1:u1", "roles": ["definition"]}, {"unit_id": "s1:p1:u2", "roles": ["definition"]},
        {"unit_id": "s1:p1:u3", "roles": ["attributed_opinion"]}]
