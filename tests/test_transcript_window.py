"""Revisions and media order must prevent stale claims and future context from being judged."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.pipeline.transcript import TranscriptUpdate, TranscriptWindow


def update(id="a", start=0, **changes):
    return replace(TranscriptUpdate(id, 0, "Jobs increased.", start, start + 1, True, "speaker"), **changes)


@pytest.mark.parametrize("changes", [
    {"id": ""}, {"id": "x" * 129}, {"revision": -1}, {"revision": True},
    {"text": "x" * 4001}, {"text": None}, {"text": " "}, {"final": "true"},
    {"speaker": None}, {"start": float("nan")}, {"end": float("inf")},
    {"start": True}, {"start": -1}, {"end": -1}, {"start": 2, "end": 1},
])
def test_invalid_updates_are_rejected(changes):
    with pytest.raises(ValueError):
        update(**changes)


def test_interim_duplicate_and_older_revisions_do_not_emit_claims():
    window = TranscriptWindow()
    interim = update(final=False, text="")
    assert window.accept(interim) is None
    final = update(revision=1)
    assert window.accept(final).claim == final
    assert window.accept(final) is None
    assert window.accept(interim) is None


def test_conflicting_revision_and_moved_anchor_are_rejected():
    window = TranscriptWindow()
    window.accept(update())
    with pytest.raises(ValueError, match="same revision"):
        window.accept(update(text="Jobs decreased."))
    with pytest.raises(ValueError, match="stable anchor"):
        window.accept(update(start=1, revision=1))


def test_context_is_final_prior_nonoverlapping_and_same_speaker():
    window = TranscriptWindow()
    window.accept(update("first", 0))
    window.accept(update("other", 1, speaker="opponent"))
    window.accept(update("interim", 2, final=False))
    window.accept(update("overlap", 3, end=6))
    context = window.accept(update("claim", 4))
    window.accept(update("future", 7))
    assert [s.id for s in context.preceding] == ["first"]
    assert window.snapshot("claim") == context


def test_corrections_invalidate_claim_and_context_dependencies():
    window = TranscriptWindow()
    window.accept(update())
    context = window.accept(update("b", 2))
    assert window.current(context)
    window.accept(update(revision=1, text="Jobs fell."))
    assert not window.current(context)
    fresh = window.snapshot("b")
    assert fresh.dependencies == (("a", 1), ("b", 0))
    window.accept(update("b", 2, revision=1, final=False))
    assert not window.current(fresh)


def test_newly_final_context_also_invalidates_an_existing_snapshot():
    window = TranscriptWindow()
    window.accept(update(final=False))
    context = window.accept(update("b", 2))
    window.accept(update(revision=1))
    assert not window.current(context)


def test_evicted_claim_is_stale_instead_of_crashing():
    window = TranscriptWindow(capacity=2, context_size=1)
    context = window.accept(update())
    window.accept(update("b", 2))
    window.accept(update("c", 4))
    assert not window.current(context)


def test_zero_duration_peer_is_not_prior_context():
    window = TranscriptWindow()
    window.accept(update("peer", 0, end=0))
    assert window.accept(update("claim", 0)).preceding == ()


def test_history_is_bounded_and_evicted_ids_cannot_reenter():
    window = TranscriptWindow(capacity=3, context_size=1)
    for i in range(6):
        context = window.accept(update(str(i), i * 2))
    assert list(window.segments) == ["3", "4", "5"]
    assert [s.id for s in context.preceding] == ["4"]
    with pytest.raises(ValueError, match="media order"):
        window.accept(update("0", 0, revision=2))
    with pytest.raises(ValueError, match="media order"):
        window.accept(update("late", 9))


@pytest.mark.parametrize("capacity,size", [(0, 0), (2, 2), (3, -1), (True, 0), (3, 1.5)])
def test_invalid_window_sizes_are_rejected(capacity, size):
    with pytest.raises(ValueError):
        TranscriptWindow(capacity=capacity, context_size=size)


def test_context_can_be_disabled():
    window = TranscriptWindow(context_size=0)
    window.accept(update())
    assert window.accept(update("b", 2)).preceding == ()


@pytest.mark.parametrize("value", [[], {}, {"unexpected": 1}])
def test_json_contract_refuses_wrong_shapes(value):
    with pytest.raises(ValueError):
        TranscriptUpdate.from_dict(value)
