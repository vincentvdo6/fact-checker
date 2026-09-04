"""
Checkpoint 4: the page has to tell the truth, and the checks have to be able to fail.

A rendered page is the one artifact anyone outside this machine will ever see, so every way it
could mislead is worth a test. The failures are all of the flattering kind -- they make the system
look more decisive than it is:

  an abstention rendered as a missing row      the page looks confident instead of selective
  a band accuracy shown without provenance     a FEVER number reads as a promise about this speech
  the domain warning quietly dropped           the whole framing goes with it
  NEI rendered the same as declining           the distinction the project argues for disappears

The `CHECKS` table is Checkpoint 4 itself, and a checkpoint that cannot fail is worse than none,
so these drive it against pages that are deliberately wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.render_sidebar import CHECKS, render

BANDS = {
    "strong": {"promised": 0.9, "measured": 0.9121495327102803, "n": 535},
    "moderate": {"promised": 0.75, "measured": 0.8005780346820809, "n": 346},
    "weak": {"promised": 0.6, "measured": 0.5984251968503937, "n": 635},
}


def row(**overrides) -> dict:
    base = {
        "index": 0, "text": "Unemployment fell to 5 percent.", "start": 0, "end": 31,
        "check_worthy": True, "filter_reason": "check_worthy",
        "outcome": "answered", "verdict": "supported", "predicted": "supported",
        "band": "strong", "confidence": 0.91, "sufficiency": 0.72,
        "evidence": [["Unemployment", 0, "The rate fell in 2019."]],
    }
    return base | overrides


def payload(*rows: dict) -> dict:
    answered = [r for r in rows if r.get("outcome") == "answered"]
    declined = [r for r in rows if str(r.get("outcome", "")).startswith("declined")]
    return {
        "transcript": "sotu-2016", "variant": "retrieved",
        "source": "https://www.govinfo.gov/content/pkg/DCPD-201600012/html/DCPD-201600012.htm",
        "sentences": len(rows), "check_worthy": len(answered) + len(declined),
        "verified": len(answered) + len(declined),
        "answered": len(answered), "declined": len(declined),
        "coverage": len(answered) / max(len(answered) + len(declined), 1),
        "calibration": {
            "fitted_on": "FEVER calibration split (13,332 claims)",
            "measured_on": "FEVER test split (2,000 claims)",
            "bands": BANDS,
            "out_of_domain": "These promises were measured on FEVER, ...",
        },
        "rows": list(rows),
    }


# --- what the page must contain ------------------------------------------------------------------

def test_an_abstention_is_rendered_as_its_own_state_not_omitted():
    data = payload(row(outcome="declined_low_confidence", verdict=None, band=None))
    page = render(data)
    assert 'class="tag declined"' in page
    assert "No verdict" in page
    assert "Unemployment fell to 5 percent." in page, "the declined claim is still shown"


def test_the_reason_for_declining_reaches_the_reader():
    page = render(payload(row(outcome="declined_insufficient_evidence", verdict=None, band=None)))
    assert "retrieval did not return the kind of evidence" in page


def test_a_withheld_prediction_is_shown_rather_than_hidden():
    """A reader can see what the gate suppressed, which is what makes the abstention auditable."""
    page = render(payload(row(outcome="declined_both", verdict=None, band=None,
                              predicted="contradicted")))
    assert "would have said" in page
    assert "Contradicted" in page


def test_a_band_accuracy_never_appears_without_the_split_it_was_measured_on():
    """
    "Right about nine times in ten" is a fact about FEVER's test split. Beside an out-of-domain
    political claim, shown bare, it is a false promise.
    """
    page = render(payload(row()))
    assert "91.2%" in page
    where = page.index("91.2%")
    context = page[max(0, where - 260):where + 260]
    assert "FEVER" in context and "test split" in context


def test_the_page_says_the_claim_is_outside_the_measured_distribution():
    page = render(payload(row()))
    assert "outside the distribution" in page


def test_a_missed_band_promise_is_shown_as_missed_rather_than_quietly_dropped():
    """The weak band came in at 0.5984 against a 0.60 promise. Reported, never retuned."""
    page = render(payload(row()))
    assert "MISSED" in page
    assert "59.8%" in page


def test_no_evidence_found_and_declining_are_labelled_differently():
    """
    Per src/verdict/labels.py these are different claims -- one about the world, one about the
    model. Both read as "no verdict" to a hurried reader, which is exactly why the words differ.
    """
    page = render(payload(
        row(verdict="not_enough_evidence", band="moderate"),
        row(index=1, outcome="declined_low_confidence", verdict=None, band=None),
    ))
    assert "No evidence found" in page
    assert "No verdict" in page


def test_the_skipped_sentences_are_counted_by_reason():
    page = render(payload(
        row(),
        row(index=1, check_worthy=False, filter_reason="question", outcome=None),
        row(index=2, check_worthy=False, filter_reason="question", outcome=None),
        row(index=3, check_worthy=False, filter_reason="no_anchor", outcome=None),
    ))
    assert "a question, not an assertion" in page
    assert "no name, number or date" in page


def test_claim_text_is_escaped_rather_than_interpolated_as_markup():
    """Transcript text is data. A sentence containing markup must not become markup."""
    page = render(payload(row(text='He said <script>alert("x")</script> loudly.')))
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


# --- Checkpoint 4 itself must be able to fail ----------------------------------------------------

def test_every_check_passes_on_a_page_that_is_telling_the_truth():
    data = payload(row(), row(index=1, outcome="declined_both", verdict=None, band=None))
    page = render(data)
    assert all(predicate(page, data) for predicate in CHECKS.values())


@pytest.mark.parametrize(
    ("name", "damage"),
    [
        ("abstentions are visible as their own state", 'class="tag declined"'),
        ("every band promise carries where it was measured", "FEVER's test split this band was right"),
        ("the out-of-domain warning is present", "outside the distribution"),
        ("the skipped sentences are accounted for", "Sentences the filter skipped"),
    ],
)
def test_each_check_fails_when_the_thing_it_guards_is_removed(name, damage):
    """
    A checkpoint that cannot fail is worse than no checkpoint, because it gets mistaken for
    evidence. Each property is deleted from the rendered page and the matching check must notice.
    """
    data = payload(row(), row(index=1, outcome="declined_both", verdict=None, band=None))
    page = render(data)
    assert CHECKS[name](page, data), "precondition: the check passes on the real page"
    assert not CHECKS[name](page.replace(damage, ""), data)


# --- the page names the filter that chose its claims --------------------------------------------

def test_the_page_says_which_filter_selected_the_claims():
    """
    Two filters admit different sentences from the same transcript. A page that does not name the
    one that ran is not interpretable: a reader comparing two renderings sees different claims and
    no reason for the difference.
    """
    page = render(payload(row()) | {"filter": "detector", "binarization": "factual"})
    assert "detector trained on ClaimBuster" in page
    assert "factual" in page


def test_a_rules_run_says_so_rather_than_claiming_a_model():
    page = render(payload(row()) | {"filter": "rules", "binarization": None})
    assert "hand-written rules" in page
    assert "ClaimBuster" not in page


def test_a_learned_rejection_is_described_as_a_score_not_as_a_rule():
    """
    The detector has no clause to point at. Rendering "below threshold" as though a named rule
    fired would be legibility that explains nothing, so the page shows the distribution instead.
    """
    data = payload(
        row(),
        row(index=1, check_worthy=False, filter_reason="below_factual_threshold",
            outcome=None, filter_score=0.08),
        row(index=2, check_worthy=False, filter_reason="below_factual_threshold",
            outcome=None, filter_score=0.12),
    ) | {"filter": "detector", "binarization": "factual", "filter_threshold": 0.35}
    page = render(data)
    assert "scored rather than ruled on" in page
    assert "median check-worthiness" in page
    assert "0.35" in page, "the threshold it was measured against"
    assert "fixed on held-out debates before this transcript was seen" in page


def test_a_rules_run_shows_no_score_distribution():
    """There is nothing to show: a rule rejection is a clause, and the clause is already listed."""
    data = payload(
        row(),
        row(index=1, check_worthy=False, filter_reason="no_anchor", outcome=None),
    ) | {"filter": "rules", "binarization": None}
    page = render(data)
    assert "scored rather than ruled on" not in page
    assert "no name, number or date" in page


def test_the_score_distribution_excludes_sentences_the_floor_overruled():
    """
    Two rejection mechanisms, and only one of them consulted the score. The length floor drops a
    sentence whatever the model said -- one it dropped scored 0.875 -- so folding those into the
    median both understates it and calls "scored" something that was ruled on. The page reports
    them separately or it is describing the wrong thing.
    """
    data = payload(
        row(),
        row(index=1, check_worthy=False, filter_reason="below_factual_threshold",
            outcome=None, filter_score=0.10),
        row(index=2, check_worthy=False, filter_reason="below_factual_threshold",
            outcome=None, filter_score=0.10),
        row(index=3, check_worthy=False, filter_reason="too_short",
            outcome=None, filter_score=0.90),
    ) | {"filter": "detector", "binarization": "factual", "filter_threshold": 0.35}
    page = render(data)

    assert "2 of these were scored rather than ruled on" in page
    assert "median check-worthiness of 0.10" in page, "0.90 was ruled on, not scored"
    assert "A further 1 were ruled out before the score was consulted" in page



# --- why the verdicts are what they are -----------------------------------------------------

MEASURED = {"claims": 40, "relevant": 14, "relevance_rate": 0.35,
            "auc": {"relevance_here": 0.5797, "relevance_here_ci": [0.395, 0.755]}}


def test_the_page_says_retrieval_mostly_found_nothing_usable():
    """
    54 of 60 answers come back "no evidence found". A reader seeing only that concludes the verdict
    model is weak; the measured cause is that retrieval returned something bearing on the claim for
    barely a third of them. Reporting the symptom without the cause is the page misleading by
    omission rather than by statement.
    """
    page = render(payload(row()), MEASURED)
    assert "could bear on the claim" in page
    assert "35%" in page
    assert "40" in page and "14" in page, "the sample is quoted as a sample"


def test_the_inconclusive_auc_is_not_put_on_the_page():
    """
    The same artifact holds an AUC of 0.5797 whose interval spans chance and the FEVER number. It
    settles nothing, and a number on a page stops being read as inconclusive no matter what the
    caption says. The relevance rate is a direct count and belongs there; the AUC does not.
    """
    page = render(payload(row()), MEASURED)
    assert "0.5797" not in page
    assert "0.58" not in page
    assert "AUC" not in page


def test_a_run_without_the_measurement_simply_omits_it():
    """The claim needs labels. Without them the page says nothing rather than guessing a rate."""
    page = render(payload(row()), None)
    assert "could bear on the claim" not in page


def test_checkpoint_four_notices_if_the_retrieval_note_goes_missing():
    data = payload(row(), row(index=1, outcome="declined_both", verdict=None, band=None))
    data["retrieval"] = MEASURED
    page = render(data)
    check = CHECKS["the retrieval quality behind the verdicts is stated"]
    assert check(page, data)
    assert not check(page.replace("could bear on the claim", ""), data)


def test_the_retrieval_check_reads_the_payload_not_the_filesystem():
    """
    A check whose answer depends on which files happen to sit on the machine passes or fails for
    reasons that have nothing to do with the page. It has to be decidable from what render was
    handed, which is also what lets it be driven from a fixture.
    """
    source = Path("scripts/render_sidebar.py").read_text(encoding="utf-8")
    assert 'not data.get("retrieval")' in source
    assert "RETRIEVAL.exists()" not in source

    without = payload(row())
    assert CHECKS["the retrieval quality behind the verdicts is stated"](render(without), without)
