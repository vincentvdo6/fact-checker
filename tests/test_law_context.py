"""Earlier-law hints must survive revisions without guessing or borrowing future speech."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.pipeline.context import SpeechMetadata, build_query
from src.pipeline.references import law_mentions
from src.pipeline.transcript import TranscriptUpdate, TranscriptWindow


def segment(index: int, text: str = "Health costs slowed.", **changes: object) -> TranscriptUpdate:
    return replace(TranscriptUpdate(str(index), 0, text, index * 2, index * 2 + 1, True, "speaker"), **changes)


def setup(**changes: object) -> tuple[TranscriptWindow, TranscriptUpdate]:
    window = TranscriptWindow(use_references=True)
    law = segment(0, "That is what the Affordable Care Act is all about.", **changes)
    window.accept(law)
    for i in range(1, 4):
        window.accept(segment(i))
    window.accept(segment(4, "Our businesses added jobs since it became law."))
    return window, law


def query(window: TranscriptWindow):
    return build_query(window.snapshot("4"), SpeechMetadata(), mode="context")


def test_law_four_sentences_back_has_exact_provenance_and_keeps_claim() -> None:
    window, law = setup()
    result = query(window)
    reference, = result.references
    assert result.claim == window.segments["4"].text
    assert result.query.endswith("Affordable Care Act")
    assert result.context_ids == ("2", "3", "0")
    assert (reference.segment_id, reference.revision) == (law.id, law.revision)
    assert law.text[reference.char_start:reference.char_end] == reference.text == "Affordable Care Act"
    assert (law.id, law.revision) in window.snapshot("4").dependencies
    plain = build_query(window.snapshot("4"), SpeechMetadata())
    assert plain.query == plain.claim and plain.references == ()


@pytest.mark.parametrize("changes", [{"speaker": "someone else"}, {"final": False}, {"end": 10}])
def test_ineligible_speech_cannot_supply_a_law(changes: dict) -> None:
    window, _ = setup(**changes)
    assert query(window).references == ()


def test_ambiguous_names_and_explicit_current_name_do_not_get_a_hint() -> None:
    window, _ = setup()
    window.accept(segment(2, "The Clean Air Act also changed.", revision=1))
    assert query(window).references == ()
    window, _ = setup()
    window.accept(segment(4, "The Voting Rights Act changed; jobs grew since it became law.", revision=1))
    assert query(window).references == ()


def test_revisions_retract_old_hints_and_new_candidates_invalidate_context() -> None:
    window, law = setup()
    old = window.snapshot("4")
    window.accept(replace(law, revision=1, text="This is about insurance."))
    assert not window.current(old)
    assert query(window).references == ()
    no_hint = window.snapshot("4")
    window.accept(segment(1, "The Clean Air Act changed.", revision=1))
    assert not window.current(no_hint)
    assert query(window).references[0].segment_id == "1"


def test_finalized_earlier_source_invalidates_reference_history() -> None:
    window, law = setup(final=False)
    old = window.snapshot("4")
    window.accept(replace(law, final=True, revision=1))
    assert not window.current(old)
    assert query(window).references[0].segment_id == "0"


def test_lookback_is_bounded_and_future_names_are_excluded() -> None:
    window = TranscriptWindow(use_references=True)
    window.accept(segment(0, "The Affordable Care Act changed."))
    for i in range(1, 9):
        window.accept(segment(i))
    window.accept(segment(9, "Jobs increased since it became law."))
    window.accept(segment(10, "The Clean Air Act changed."))
    context = window.snapshot("9")
    assert len(context.reference_context) == 8
    assert build_query(context, SpeechMetadata(), mode="context").references == ()


def test_retiring_reference_history_preserves_an_unchanged_snapshot() -> None:
    window = TranscriptWindow(capacity=6, use_references=True)
    for i in range(5):
        text = "The Affordable Care Act changed." if i == 0 else "Health costs slowed."
        window.accept(segment(i, text))
    context = window.accept(segment(5, "Jobs increased since it became law."))
    window.accept(segment(6))
    assert window.current(context)
    assert build_query(context, SpeechMetadata(), mode="context").references[0].segment_id == "0"


@pytest.mark.parametrize("text", ["The Act changed.", "This Law changed.", "affordable care act"])
def test_generic_or_uncased_words_are_not_named_laws(text: str) -> None:
    assert law_mentions(text) == []


def test_years_distinguish_laws_instead_of_collapsing_ambiguity() -> None:
    window, law = setup()
    window.accept(replace(law, revision=1, text="The Civil Rights Act of 1964 and the Civil Rights Act of 1991."))
    assert query(window).references == ()
    assert [name for name, _, _ in law_mentions(window.segments["0"].text)] == [
        "Civil Rights Act of 1964", "Civil Rights Act of 1991",
    ]


def test_default_window_does_not_acquire_reference_dependencies() -> None:
    window = TranscriptWindow(context_size=0)
    window.accept(segment(0, "The Affordable Care Act changed."))
    context = window.accept(segment(1, "Jobs increased since it became law."))
    assert context.reference_context == () and context.dependencies == (("1", 0),)
    window.accept(segment(0, "The Clean Air Act changed.", revision=1))
    assert window.current(context)


def test_reference_option_requires_boolean() -> None:
    with pytest.raises(ValueError, match="boolean"):
        TranscriptWindow(use_references="yes")
