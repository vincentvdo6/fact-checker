"""Text replay timing must remain explicit, ordered and reproducible."""

from __future__ import annotations

import pytest

from src.pipeline.replay import replay_updates


def test_replay_preserves_text_and_assigns_monotonic_synthetic_times():
    updates = list(replay_updates("Jobs grew. Real wages rose.", words_per_minute=60, speaker="Speaker"))
    assert [(u.id, u.text, u.start, u.end) for u in updates] == [
        ("0", "Jobs grew.", 0, 2), ("1", "Real wages rose.", 2, 5),
    ]
    assert all(u.final and u.speaker == "Speaker" for u in updates)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_replay_refuses_invalid_speaking_rates(rate):
    with pytest.raises(ValueError):
        list(replay_updates("Jobs grew.", words_per_minute=rate))
