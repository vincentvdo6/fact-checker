"""
The transcript run's bookkeeping, which is where this demo's honesty actually lives.

None of this touches the model. What it guards is the accounting around it, and every failure
available here is silent and flattering: a page that lists only verdicts looks confident rather
than selective, a coverage figure divided by the wrong denominator looks better than the policy
is, and a suppressed prediction dropped from the record makes abstention unfalsifiable.

The distinction that must survive to the page is NOT ENOUGH EVIDENCE against declining. The first
is a verdict about the world; the second is the system declining to answer. Both read as "no
verdict" to a hurried reader, so the data keeps them in different fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_transcript import coverage_of, to_row
from src.calibration.bands import Band
from src.pipeline.segment import Decision, Sentence
from src.pipeline.verify import Judgement, Outcome
from src.verdict.labels import Verdict


def sentence(index: int = 0, text: str = "Wages rose in 2019.") -> Sentence:
    return Sentence(index=index, text=text, start=0, end=len(text))


def judgement(outcome: Outcome, verdict: Verdict | None, band: Band | None = None) -> Judgement:
    return Judgement(
        claim="Wages rose in 2019.", outcome=outcome, verdict=verdict,
        predicted=Verdict.SUPPORTED, band=band, confidence=0.8, sufficiency=0.6,
        probabilities=(0.8, 0.1, 0.1), evidence=(("Page", 0, "some text"),),
    )


# --- every sentence is accounted for ------------------------------------------------------------

def test_a_sentence_the_filter_rejected_still_gets_a_row():
    """
    The rejections are the demo's honesty. Dropping them would leave the abstention rate without
    a denominator and let a page show four verdicts out of forty with nothing to compare against.
    """
    row = to_row(sentence(), Decision(False, "no_anchor"), None)
    assert row["check_worthy"] is False
    assert row["filter_reason"] == "no_anchor"
    assert "outcome" not in row


def test_a_check_worthy_sentence_left_unverified_is_marked_rather_than_silently_bare():
    """Reachable only under --limit, and it must not be mistaken for a filter rejection."""
    row = to_row(sentence(), Decision(True, "check_worthy"), None)
    assert row["outcome"] == "not_verified"


def test_the_claim_text_and_its_span_both_travel_with_the_row():
    """
    The page renders this text, so an empty or truncated field is a blank claim beside a verdict.
    The span is what lets a reader find it back in the transcript and audit the call.
    """
    row = to_row(sentence(index=7), Decision(True, "check_worthy"), None)
    assert row["text"] == "Wages rose in 2019."
    assert (row["index"], row["start"], row["end"]) == (7, 0, len("Wages rose in 2019."))


# --- the three ways to have no verdict stay apart -----------------------------------------------

def test_a_confident_no_evidence_finding_is_recorded_as_a_verdict():
    """
    NOT ENOUGH EVIDENCE is a claim about the world -- we looked and found nothing that settles it.
    It is an answer, and collapsing it into abstention would erase the distinction the whole
    project is arguing for.
    """
    row = to_row(sentence(), Decision(True, "check_worthy"),
                 judgement(Outcome.ANSWERED, Verdict.NOT_ENOUGH_EVIDENCE, Band.STRONG))
    assert row["outcome"] == "answered"
    assert row["verdict"] == "not_enough_evidence"


@pytest.mark.parametrize(
    "outcome",
    [Outcome.DECLINED_LOW_CONFIDENCE, Outcome.DECLINED_INSUFFICIENT_EVIDENCE,
     Outcome.DECLINED_BOTH],
)
def test_a_declined_claim_carries_no_verdict_but_keeps_what_was_suppressed(outcome):
    """
    Hiding the withheld prediction would make the abstention unfalsifiable: there would be no way
    to ask whether the system declined the claims it was wrong about.
    """
    row = to_row(sentence(), Decision(True, "check_worthy"), judgement(outcome, None))
    assert row["verdict"] is None
    assert row["predicted"] == "supported"
    assert row["outcome"].startswith("declined")


def test_the_reason_for_declining_is_kept_rather_than_flattened_to_one_word():
    """Low confidence and inadequate evidence are different diagnoses and stay distinguishable."""
    low = to_row(sentence(), Decision(True, "check_worthy"),
                 judgement(Outcome.DECLINED_LOW_CONFIDENCE, None))
    thin = to_row(sentence(), Decision(True, "check_worthy"),
                  judgement(Outcome.DECLINED_INSUFFICIENT_EVIDENCE, None))
    assert low["outcome"] != thin["outcome"]


# --- coverage has the right denominator ---------------------------------------------------------

def test_coverage_is_measured_among_claims_that_reached_the_model():
    """
    The gate governs claims it actually saw. Dividing by every sentence would fold the
    check-worthiness filter into a number that is supposed to describe the abstention policy --
    and would fall as the filter got more permissive, which is backwards.
    """
    rows = [
        {"outcome": "answered"}, {"outcome": "answered"}, {"outcome": "answered"},
        {"outcome": "declined_low_confidence"},
        {"check_worthy": False, "filter_reason": "question"},
        {"check_worthy": False, "filter_reason": "opinion"},
    ]
    answered, declined, coverage = coverage_of(rows)
    assert (len(answered), len(declined)) == (3, 1)
    assert coverage == pytest.approx(0.75), "6 sentences, but only 4 reached the model"


def test_every_kind_of_declining_counts_against_coverage():
    rows = [{"outcome": "answered"},
            {"outcome": "declined_low_confidence"},
            {"outcome": "declined_insufficient_evidence"},
            {"outcome": "declined_both"}]
    _, declined, coverage = coverage_of(rows)
    assert len(declined) == 3
    assert coverage == pytest.approx(0.25)


def test_a_transcript_where_nothing_reached_the_model_does_not_divide_by_zero():
    answered, declined, coverage = coverage_of([{"check_worthy": False, "filter_reason": "question"}])
    assert (answered, declined, coverage) == ([], [], 0.0)


def test_an_unverified_row_is_not_counted_as_either():
    """--limit leaves check-worthy claims unscored; they are neither answered nor declined."""
    _, declined, coverage = coverage_of([{"outcome": "answered"}, {"outcome": "not_verified"}])
    assert declined == []
    assert coverage == pytest.approx(1.0)


# --- which filter selects the claims --------------------------------------------------------

def test_the_learned_detector_is_the_default_and_the_rules_stay_reachable():
    """
    The default is the measured winner -- F1 0.7619 against 0.3103 on labels neither system saw.
    The rules stay available because they are the only filter whose rejections name a clause a
    reader can argue with, which is the better instrument when the question is why.
    """
    source = Path("scripts/verify_transcript.py").read_text(encoding="utf-8")
    assert 'default="detector", choices=("detector", "rules")' in source
    assert 'if args.filter == "rules":' in source


def test_the_run_records_which_filter_produced_its_claims():
    """
    Two filters admit different sentences, so a verdicts.json without this is not interpretable:
    the same transcript yields different pages and nothing says why.
    """
    source = Path("scripts/verify_transcript.py").read_text(encoding="utf-8")
    assert '"filter": args.filter,' in source
    assert '"binarization": args.binarization if args.filter == "detector" else None,' in source


def test_a_learned_decision_carries_its_score_and_a_rule_decision_does_not():
    """
    The score is the honest asymmetry. A rule rejection means a named clause fired; a detector
    rejection means a number fell below a cut, and the page should be able to tell them apart.
    """
    from src.pipeline.detector import Decision as LearnedDecision

    learned = to_row(sentence(), LearnedDecision(False, "below_factual_threshold", 0.12), None)
    ruled = to_row(sentence(), Decision(False, "no_anchor"), None)
    assert learned["filter_score"] == pytest.approx(0.12)
    assert "filter_score" not in ruled
