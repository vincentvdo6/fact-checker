"""Retiring immutable history must not trigger a context-stripped recheck."""

from __future__ import annotations

from src.pipeline.transcript import TranscriptUpdate, TranscriptWindow


def test_retired_context_remains_valid_until_the_claim_retires():
    window = TranscriptWindow(capacity=3, context_size=1)
    snapshots = []
    for i in range(4):
        snapshots.append(window.accept(TranscriptUpdate(str(i), 0, "Jobs grew.", i * 2, i * 2 + 1, True)))
    assert window.current(snapshots[1])
    assert not window.current(snapshots[0])
