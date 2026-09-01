"""
Known-answer tests for segmentation and check-worthiness.

This filter decides what the demo shows. A bug here does not raise, it just quietly drops claims
or admits sentences that have no truth value -- so the tests are written as cases with an obvious
right answer, and the reason each rejection gives is asserted alongside the rejection itself. A
filter that rejects the right sentence for the wrong reason is one refactor away from rejecting
the wrong sentence.
"""

from __future__ import annotations

import pytest

from src.pipeline.segment import Sentence, check_worthy, claims, segment

# --- segmentation --------------------------------------------------------------------------

def test_a_plain_transcript_splits_on_terminators():
    got = segment("Unemployment fell. Wages rose. Inflation held steady.")
    assert [s.text for s in got] == ["Unemployment fell.", "Wages rose.", "Inflation held steady."]
    assert [s.index for s in got] == [0, 1, 2]


@pytest.mark.parametrize(
    "transcript",
    [
        "Sen. Smith voted against the bill in 2019.",
        "John F. Kennedy was elected in 1960.",
        "The rate fell to 3.5 percent last year.",
        "The U.S. economy added 400,000 jobs in June.",
        "Growth was 2.1 percent vs. 1.8 percent the year before.",
    ],
)
def test_a_terminator_inside_an_abbreviation_or_number_is_not_a_boundary(transcript):
    """
    Every false split invents a sentence nobody said, and the fragment then gets scored as a
    claim. Abbreviations, initials and decimals are where that happens in political speech.
    """
    assert len(segment(transcript)) == 1


def test_a_span_points_back_at_the_transcript():
    """
    A verdict a reader cannot locate in the source is a verdict they cannot audit, so the span has
    to be the text exactly -- no stripping on either side of the comparison. The transcript is
    padded with irregular whitespace because that is the only thing that makes an untrimmed offset
    visible; comparing stripped strings passes whether the span is right or not.
    """
    transcript = "   Wages rose in 2019.    Unemployment fell to 3.5 percent.  "
    got = segment(transcript)
    assert [s.text for s in got] == ["Wages rose in 2019.", "Unemployment fell to 3.5 percent."]
    for sentence in got:
        assert transcript[sentence.start:sentence.end] == sentence.text


def test_a_transcript_with_no_final_terminator_keeps_its_last_sentence():
    """ASR output routinely ends mid-punctuation; dropping the tail loses a real claim."""
    got = segment("Wages rose in 2019. Unemployment fell to 3.5 percent")
    assert len(got) == 2
    assert got[1].text == "Unemployment fell to 3.5 percent"


def test_an_empty_transcript_yields_no_sentences():
    assert segment("") == []
    assert segment("   \n  ") == []


# --- check-worthiness: the four cases the plan named ------------------------------------------

def test_a_dated_numeric_assertion_is_kept():
    decision = check_worthy("The unemployment rate fell to 3.5 percent in June 2019.")
    assert decision.worthy
    assert decision.reason == "check_worthy"


@pytest.mark.parametrize(
    ("sentence", "reason"),
    [
        ("Why did the unemployment rate fall in 2019?", "question"),
        ("Did the Senate pass the 2019 budget bill?", "question"),
        ("Did the Senate pass the 2019 budget bill", "question"),   # ASR loses the mark
        ("Look at what happened to wages in 2019.", "imperative"),
        ("Vote for the candidate who created 400,000 jobs.", "imperative"),
        ("I think the 2019 budget was a disaster.", "opinion"),
        ("Congress should have passed the 2019 bill.", "opinion"),
        ("The rate probably fell in 2019.", "opinion"),
        ("Good evening, everyone.", "too_short"),
        ("The greatest economy in American history.", "no_assertion"),
        ("That was a terrible decision.", "no_anchor"),
    ],
)
def test_a_sentence_with_no_checkable_content_is_rejected_for_the_right_reason(sentence, reason):
    decision = check_worthy(sentence)
    assert not decision.worthy
    assert decision.reason == reason


def test_an_assertion_with_a_name_but_no_number_is_still_checkable():
    """FEVER is entity-centred; a proper noun is enough for BM25 to grip."""
    assert check_worthy("Barack Obama was born in Honolulu, Hawaii.").worthy


def test_a_decision_is_truthy_so_it_reads_as_the_answer_it_is():
    assert bool(check_worthy("Barack Obama was born in Honolulu, Hawaii."))
    assert not bool(check_worthy("That was a terrible decision."))


# --- the two together --------------------------------------------------------------------------

def test_claims_returns_every_sentence_including_the_rejected_ones():
    """
    The rejections are the demo's honesty: a reader can see what was skipped and why, rather than
    a page that silently shows four claims out of forty.
    """
    got = claims(
        "Good evening. The unemployment rate fell to 3.5 percent in 2019. "
        "Why does that matter? I think it was the best year on record."
    )
    assert len(got) == 4
    assert [d.reason for _, d in got] == ["too_short", "check_worthy", "question", "opinion"]
    assert [s.index for s, _ in got] == [0, 1, 2, 3]
    assert isinstance(got[0][0], Sentence)
