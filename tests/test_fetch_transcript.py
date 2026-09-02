"""
Turning a govinfo compilation entry into speech.

Everything this strips is something nobody said. Audience reaction, speaker labels, the editorial
note and the subject index are the compilation's apparatus, and each one that survives becomes a
"sentence" the segmenter hands to the check-worthiness filter and, if it passes, to the verdict
model -- which then attaches a calibrated confidence to text that was never asserted.

The index is the case that actually happened: 514 words of semicolon-separated subject headings
arriving as a single sentence, longer than any real one in the speech by an order of magnitude.
"""

from __future__ import annotations

import pytest

from scripts.fetch_transcript import to_text

HEADER = (
    "<P>Administration of Barack Obama, 2016</P>"
    "<P>Address Before a Joint Session of the Congress on the State of the Union</P>"
    "<P>January 12, 2016</P>"
)


def document(*paragraphs: str) -> str:
    return HEADER + "".join(f"<P>{p}</P>" for p in paragraphs)


def test_the_three_header_paragraphs_are_provenance_not_speech():
    text, header = to_text(document("Unemployment fell to 5 percent."))
    assert text == "Unemployment fell to 5 percent."
    assert header == [
        "Administration of Barack Obama, 2016",
        "Address Before a Joint Session of the Congress on the State of the Union",
        "January 12, 2016",
    ]


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("Wages rose. [Laughter] And so did hiring.", "Wages rose. And so did hiring."),
        ("Our economy is strong. [Applause]", "Our economy is strong."),
        ("The President. Wages rose in 2019.", "Wages rose in 2019."),
        ("We <I>did</I> that.", "We did that."),
        ("Growth was 2.1 percent&mdash;the best yet.", "Growth was 2.1 percent—the best yet."),
        ("He said &quot;yes&quot; to it.", 'He said "yes" to it.'),
    ],
)
def test_editorial_insertions_and_markup_do_not_reach_the_transcript(raw, want):
    text, _ = to_text(document(raw))
    assert text == want


@pytest.mark.parametrize(
    "marker",
    [
        "Names: Biden, Joseph R., Jr.; Francis, Pope; Ryan, Paul D.",
        "Subjects: Africa : Agricultural production; Budget, Federal : Deficit",
        "DCPD Number: DCPD201600012.",
        "NOTE: The President spoke at 9:10 p.m. in the House Chamber.",
        "In his remarks, he referred to Speaker of the House Paul D. Ryan.",
    ],
)
def test_the_trailing_apparatus_is_cut_rather_than_segmented(marker):
    """
    Each of these is the compilation talking, not the speaker. The subject index in particular
    arrives as one 514-word pseudo-sentence that the filter would then have to rule on.
    """
    text, _ = to_text(document("Unemployment fell to 5 percent.", marker))
    assert text == "Unemployment fell to 5 percent."


def test_everything_after_the_first_apparatus_marker_is_dropped_too():
    """The index runs to several paragraphs; stopping at the first is what makes the cut clean."""
    text, _ = to_text(document(
        "Unemployment fell to 5 percent.",
        "Names: Ryan, Paul D.",
        "Subjects: Budget, Federal : Deficit and national debt.",
        "DCPD Number: DCPD201600012.",
    ))
    assert text == "Unemployment fell to 5 percent."


def test_speech_that_merely_mentions_a_marker_word_is_not_cut():
    """
    The markers anchor at the start of a paragraph. A sentence that happens to contain "Names:"
    mid-line is speech, and truncating the transcript there would silently lose the rest.
    """
    body = "The report listed Names: three of them, and we acted on it."
    text, _ = to_text(document(body, "Unemployment fell to 5 percent."))
    assert text.endswith("Unemployment fell to 5 percent.")
    assert body in text


def test_paragraphs_are_separated_so_sentences_cannot_run_together():
    text, _ = to_text(document("Wages rose", "Unemployment fell"))
    assert text == "Wages rose\n\nUnemployment fell"


def test_an_empty_paragraph_does_not_become_a_sentence():
    text, _ = to_text(document("Wages rose in 2019.", "   ", "Unemployment fell."))
    assert text == "Wages rose in 2019.\n\nUnemployment fell."
