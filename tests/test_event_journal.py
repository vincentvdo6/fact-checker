"""Durable events precede publication and cannot overwrite sessions."""
from __future__ import annotations

import io

import pytest

from src.pipeline.journal import EventJournal


def test_journal_refuses_to_overwrite_an_existing_session(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        EventJournal(path, io.StringIO(), {})
    assert path.read_text(encoding="utf-8") == "original"


def test_event_is_fsynced_before_it_is_published(tmp_path, monkeypatch):
    order = []

    class Output(io.StringIO):
        def write(self, value):
            order.append("published")
            return super().write(value)

    monkeypatch.setattr("src.pipeline.journal.os.fsync", lambda _: order.append("durable"))
    journal = EventJournal(tmp_path / "events.jsonl", Output(), {})
    journal.write({"type": "test"})
    journal.close()
    assert order == ["durable", "published", "durable", "published"]


def test_journal_refuses_nonfinite_numbers_before_writing(tmp_path):
    path, output = tmp_path / "events.jsonl", io.StringIO()
    journal = EventJournal(path, output, {})
    try:
        with pytest.raises(ValueError):
            journal.write({"type": "result", "latency_ms": float("nan")})
    finally:
        journal.close()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_failures_are_counted_and_sequence_is_owned(tmp_path):
    import json
    output = io.StringIO()
    journal = EventJournal(tmp_path / "events.jsonl", output, {})
    for kind in ["failed", "rejected", "result"]:
        journal.write({"type": kind, "sequence": 99})
    journal.close()
    assert journal.failures == 2
    assert [json.loads(line)["sequence"] for line in output.getvalue().splitlines()] == [0,1,2,3]
