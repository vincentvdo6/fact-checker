"""Lexical evidence screening must preserve score alignment and reject distractors."""

from __future__ import annotations

import pytest

from src.pipeline.topic_evidence import topic_evidence


def test_topic_filter_drops_song_titles_and_empty_text_without_reordering_scores():
    evidence = [
        ("We_Don't_Have_to_Dance", 0, "We Do n't Have to Dance is a song by Andy Black."),
        ("Labor_market", 0, "."),
        ("Good_Job!", 0, "Good Job !"),
        ("Artist", 0, "His work protests the shortage of green space, labor rights and unemployment."),
        ("Secondary_labor_market", 1, "This labor market consists of low-pay part-time work."),
        ("Unemployment", 2, "The job market includes unemployment and part-time work."),
    ]
    rows, scores = topic_evidence("We don't have a labor shortage. We have a good job shortage.",
                                 "labor shortage good job unemployment part time work", evidence, [6, 5, 4, 3, 2, 1],
                                 context=["The labor market. Part-time work."])
    assert rows == evidence[-2:] and scores == [2, 1]
    assert topic_evidence("We don't have it.", "We don't have it.", evidence, [6, 5, 4, 3, 2, 1]) == ([], [])
    with pytest.raises(ValueError):
        topic_evidence("labor shortage", "labor shortage", evidence, [1])


def test_direct_claim_evidence_does_not_need_incidental_context_words():
    row = ("Labor", 0, "A labor shortage means insufficient workers.")
    assert topic_evidence("We don't have a labor shortage.", "labor shortage wages unemployment", [row], [1],
                          context=["Labor market wages and unemployment."]) == ([row], [1])


def test_topic_phrases_end_at_sentence_boundaries():
    row = ("Labor", 0, "The labor market consists of employees and employers.")
    assert topic_evidence("We don't have a labor shortage.", "labor shortage market people disagree", [row], [1],
                          context=["The labor market. People disagree."]) == ([row], [1])


@pytest.mark.parametrize("boundary", [". ", "! ", "? ", "; ", ": ", ", "])
def test_evidence_cannot_assemble_a_topic_phrase_across_clauses(boundary):
    unrelated = ("Unrelated", 0, "The party was called Labor" + boundary + "Shortage is an unrelated song title.")
    relevant = ("Employment", 1, "A labor shortage means insufficient workers.")
    assert topic_evidence("We don't have a labor shortage.", "labor shortage",
                          [unrelated, relevant], [5.0, 2.0]) == ([relevant], [2.0])


def test_topic_phrase_can_appear_inside_a_longer_evidence_clause():
    row = ("Employment", 0, "Persistent severe labor shortages increased regional hiring costs.")
    assert topic_evidence("Labor shortages increased.", "labor shortages increased", [row], [2.0]) == ([row], [2.0])
