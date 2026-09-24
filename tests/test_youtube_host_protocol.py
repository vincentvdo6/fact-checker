"""Only completed native responses produce distinct, valid click records."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts import youtube_host


def requests(*messages: dict) -> io.BytesIO:
    stream = io.BytesIO()
    for message in messages:
        youtube_host.write_message(stream, message)
    stream.seek(0)
    return stream


def test_same_second_checks_keep_both_records_and_never_use_untrusted_video_as_a_path(tmp_path, monkeypatch):
    instant = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(youtube_host, "datetime", SimpleNamespace(now=lambda _: instant))
    first = youtube_host.record_click(tmp_path, {"video_id": "abcdefghijk", "marker": 1}, {"video_id": "abcdefghijk"})
    second = youtube_host.record_click(tmp_path, {"video_id": "abcdefghijk", "marker": 2}, {"video_id": "abcdefghijk"})
    assert first != second
    assert first.name == "click-2026-09-24T120000Z-abcdefghijk.json"
    assert second.name == "click-2026-09-24T120000Z-abcdefghijk-1.json"
    assert [json.loads(path.read_text(encoding="utf-8"))["payload"]["marker"] for path in (first, second)] == [1, 2]
    unsafe = youtube_host.record_click(tmp_path, {"video_id": "abcdefghijk"}, {"video_id": "../outside"})
    assert unsafe.parent == tmp_path and unsafe.name.endswith("-unknown.json")


def test_invalid_utf8_record_is_rejected_before_creating_a_file(tmp_path):
    with pytest.raises(UnicodeEncodeError):
        youtube_host.record_click(tmp_path, {"unused": "\ud800"}, {"video_id": "abcdefghijk"})
    assert not list(tmp_path.iterdir())


def test_unencodable_result_does_not_leave_a_success_record(tmp_path, monkeypatch):
    original = youtube_host.record_click
    attempts = []

    def record(*args):
        attempts.append(args)
        return original(*args)

    monkeypatch.setattr(youtube_host, "record_click", record)
    checker = SimpleNamespace(check=lambda payload: {"value": float("nan")} if payload["bad"] else {"ok": True},
                              close=lambda: None)
    incoming = requests({"id": "bad", "payload": {"bad": True}}, {"id": "good", "payload": {"bad": False}})
    outgoing = io.BytesIO()
    youtube_host.serve(incoming, outgoing, checker, tmp_path)
    outgoing.seek(0)
    assert "error" in youtube_host.read_message(outgoing)
    assert youtube_host.read_message(outgoing) == {"id": "good", "result": {"ok": True}}
    assert len(attempts) == 1 and attempts[0][1] == {"bad": False}
    assert len(list(tmp_path.glob("click-*.json"))) == 1


def test_recording_failure_does_not_change_response_or_next_request(tmp_path, monkeypatch, capsys):
    def cannot_record(*args):
        raise ValueError("recording value invalid")

    monkeypatch.setattr(youtube_host, "record_click", cannot_record)
    checker = SimpleNamespace(check=lambda payload: {"ok": payload["number"]}, close=lambda: None)
    incoming = requests({"id": "one", "payload": {"number": 1}},
                        {"id": "two", "payload": {"number": 2}})
    outgoing = io.BytesIO()
    youtube_host.serve(incoming, outgoing, checker, tmp_path)
    outgoing.seek(0)
    assert youtube_host.read_message(outgoing) == {"id": "one", "result": {"ok": 1}}
    assert youtube_host.read_message(outgoing) == {"id": "two", "result": {"ok": 2}}
    assert capsys.readouterr().err.count("click not recorded") == 2


def test_broken_output_never_records_a_response_the_browser_could_not_receive(tmp_path):
    class Broken(io.BytesIO):
        def write(self, frame):
            raise OSError("native pipe closed")

    checker = SimpleNamespace(check=lambda payload: {"ok": True}, close=lambda: None)
    with pytest.raises(OSError, match="native pipe closed"):
        youtube_host.serve(requests({"id": "one", "payload": {}}), Broken(), checker, tmp_path)
    assert not list(tmp_path.iterdir())
