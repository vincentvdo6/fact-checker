"""Reference candidates remain causal and revision-aware."""
from __future__ import annotations

from dataclasses import replace

import pytest

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


def test_earlier_candidates_and_finalization():
    window, law = setup(final=False)
    old = window.snapshot("4")
    window.accept(replace(law, final=True, revision=1))
    fresh = window.snapshot("4")
    assert not window.current(old)
    assert fresh.reference_context[0].id == "0"
    assert ("0", 1) in fresh.dependencies
    window.accept(segment(5, "The Clean Air Act changed."))
    assert all(s.id != "5" for s in window.snapshot("4").reference_context)

def test_retired_reference_stays_current():
    window = TranscriptWindow(capacity=3, use_references=True)
    for i in range(3):
        window.accept(segment(i, "Jobs grew since it became law."))
    old = window.snapshot("2")
    window.accept(segment(3))
    assert window.current(old)

@pytest.mark.parametrize("changes", [{"final": False}, {"speaker": "other"}, {"end": 10}])
def test_ineligible_history_is_excluded(changes):
    window, _ = setup(**changes)
    assert all(s.id != "0" for s in window.snapshot("4").reference_context)

def test_history_cap():
    window = TranscriptWindow(use_references=True)
    for i in range(11):
        window.accept(segment(i, "Jobs grew since it became law."))
    assert [s.id for s in window.snapshot("10").reference_context] == [str(i) for i in range(2,10)]
