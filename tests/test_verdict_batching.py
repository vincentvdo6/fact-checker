"""A caller's queue must not become a single attention tensor or reorder its answers."""

from __future__ import annotations

import numpy as np
import pytest

from src.verdict.runtime import DEFAULT_BATCH, Scored, VerdictRuntime


def test_large_batch_is_bounded_and_preserves_every_result(monkeypatch):
    runtime = VerdictRuntime.__new__(VerdictRuntime)
    sizes = []

    def score_chunk(pairs):
        sizes.append(len(pairs))
        return [Scored(np.array([int(claim), 0, 0]), int(claim) + 1) for claim, _ in pairs]

    monkeypatch.setattr(runtime, "_score_chunk", score_chunk)
    pairs = [(str(i), []) for i in range(DEFAULT_BATCH * 2 + 1)]
    scored = runtime.score_batch(pairs)
    assert sizes == [DEFAULT_BATCH, DEFAULT_BATCH, 1]
    assert [s.logits[0] for s in scored] == list(range(len(pairs)))
    assert [s.token_len for s in scored] == list(range(1, len(pairs) + 1))


def test_empty_batch_never_opens_the_graph(monkeypatch):
    runtime = VerdictRuntime.__new__(VerdictRuntime)
    monkeypatch.setattr(runtime, "_score_chunk", lambda _: pytest.fail("graph called"))
    assert runtime.score_batch([]) == []


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "4"])
def test_invalid_batch_sizes_fail_before_scoring(size):
    runtime = VerdictRuntime.__new__(VerdictRuntime)
    with pytest.raises(ValueError, match="positive integer"):
        runtime.score_batch([], batch_size=size)


def test_explicit_batch_size_controls_all_chunks(monkeypatch):
    runtime = VerdictRuntime.__new__(VerdictRuntime)
    sizes = []
    monkeypatch.setattr(runtime, "_score_chunk", lambda pairs: sizes.append(len(pairs)) or [])
    runtime.score_batch([("a", [])] * 5, batch_size=2)
    assert sizes == [2, 2, 1]
