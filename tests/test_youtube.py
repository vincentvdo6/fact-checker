"""Caption time boundaries and native framing must preserve the click's provenance."""

from __future__ import annotations

import io
import json
import socket
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.youtube_host import MAX_MESSAGE, read_message, serve, write_message
from src.pipeline.youtube import YouTubeChecker, caption_excerpt


def payload(**fields):
    return {"video_id": "abcdefghijk", "title": "A speech", "time": 20, "language": "en",
            "source": "track", "captions": [{"start": 10, "end": 15, "text": "The law passed in 2010."}], **fields}


def test_web_search_can_be_disabled_without_topic_specific_network_routes(monkeypatch):
    """An explicit local check must never contact a search provider."""
    monkeypatch.setenv("FACT_CHECKER_WEB_SEARCH", "off")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "must-not-enable-client-search")
    def unexpected(*args, **kwargs):
        pytest.fail("Local check attempted web research or a network connection")

    monkeypatch.setattr(socket, "socket", unexpected)
    monkeypatch.setattr("src.pipeline.live_processor.DetectorFilter", lambda: SimpleNamespace(
        decide=lambda _: SimpleNamespace(worthy=True, reason="rule", score=0.4)))
    monkeypatch.setattr("src.pipeline.live_processor.Verifier", lambda: SimpleNamespace(
        retrieve=lambda _: ([], []), judge=unexpected, close=lambda: None))
    checker = YouTubeChecker()
    try:
        result = checker.check(payload(captions=[{
            "start": 10, "end": 15,
            "text": "We don't have a labor shortage. We have a good job shortage.",
        }]))
        assert checker.processor.web is None
        assert len(result["rows"]) == 1
        claim = result["rows"][0]["result"]
        assert claim["outcome"] == "declined_no_relevant_evidence"
        assert claim["verdict"] is None and "research" not in claim
        assert "web research is off" in result["scope"]
    finally:
        checker.close()


def test_caption_selection_excludes_future_speech_and_stale_context():
    data = payload(time=60, captions=[
        {"start": 5, "end": 10, "text": "Old claim."},
        {"start": 50, "end": 55, "text": "The law passed in 2010."},
        {"start": 56, "end": 100, "text": "Not yet heard."},
        {"start": 61, "end": 63, "text": "Future claim."},
    ])
    excerpt = caption_excerpt(data)
    assert excerpt.text == "The law passed in 2010."
    assert (excerpt.start, excerpt.end, excerpt.clicked_at) == (50, 55, 60)
    with pytest.raises(ValueError, match="No recent"):
        caption_excerpt(payload(time=30))


@pytest.mark.parametrize("claim", ["Solar panels produce electricity.", "The bridge opened in 1932."])
def test_default_news_search_runs_without_keys_or_a_backend_and_assigns_no_web_verdict(monkeypatch, claim):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("FACT_CHECKER_WEB_SEARCH", raising=False)
    monkeypatch.delenv("FACT_CHECKER_SEARCH_URL", raising=False)
    calls = []
    monkeypatch.setattr("src.pipeline.live_processor.DetectorFilter", lambda: SimpleNamespace(
        decide=lambda _: SimpleNamespace(worthy=True, reason="rule", score=0.4)))
    monkeypatch.setattr("src.pipeline.live_processor.Verifier", lambda: pytest.fail("News search loaded offline artifacts"))
    monkeypatch.setattr("src.pipeline.youtube.GoogleNewsSources", lambda: SimpleNamespace(
        always_search=True, begin_check=lambda: calls.append("begin"), search=lambda query: calls.append(query) or [],
        scrape=lambda _: pytest.fail("Empty search must not fetch a page")))
    checker = YouTubeChecker()
    try:
        result = checker.check(payload(captions=[{"start": 10, "end": 15, "text": claim}]))
        assert calls == ["begin", claim]
        assert "experimental news research enabled" in result["scope"]
        row = result["rows"][0]["result"]
        assert row["verdict"] is None and row["outcome"] == "declined_web_review"
        assert row["research"]["sources"] == []
    finally:
        checker.close()


def test_rolling_caption_dedup_keeps_nonoverlapping_repetitions():
    excerpt = caption_excerpt(payload(time=8, captions=[
        {"start": 0, "end": 3, "text": "The law"},
        {"start": 2, "end": 5, "text": "The law passed."},
        {"start": 5, "end": 8, "text": "The law passed."},
    ]))
    assert excerpt.text == "The law passed. The law passed."


def test_caption_window_keeps_the_complete_cue_crossing_its_start():
    excerpt = caption_excerpt(payload(time=60, captions=[
        {"start": 20, "end": 30, "text": "Stale sentence."},
        {"start": 29, "end": 35, "text": "Among the surveyed households,"},
        {"start": 35, "end": 55, "text": "24% had solar panels."},
        {"start": 55, "end": 61, "text": "Not finished yet."},
        {"start": 61, "end": 65, "text": "Future sentence."},
    ]))
    assert excerpt.text == "Among the surveyed households, 24% had solar panels."
    assert (excerpt.start, excerpt.end) == (29, 55)


def test_later_click_keeps_population_definition_from_saved_captions():
    from src.pipeline.caption_claims import caption_claims

    fixture = json.loads(Path("tests/youtube_labor_case.json").read_text(encoding="utf-8"))
    fixture["time"] = 3184.5
    fixture["captions"].append({"start": 3173, "end": 3184, "text": "Later speech."})
    excerpt = caption_excerpt(fixture)
    expected = fixture["captions"][2]["text"].removeprefix("Okay. ") + " " + fixture["captions"][3]["text"].split(". So")[0] + "."
    claims = [sentence.text for sentence in caption_claims(excerpt.text)]
    assert expected in claims
    assert "We don't have a labor shortage. We have a good job shortage." in claims
    assert excerpt.start == 3153


def test_nested_caption_ends_do_not_make_a_recent_excerpt_stale():
    excerpt = caption_excerpt(payload(time=20, captions=[
        {"start": 0, "end": 20, "text": "The law passed."},
        {"start": 1, "end": 2, "text": "The law passed."},
    ]))
    assert excerpt.text == "The law passed."
    assert excerpt.end == excerpt.clicked_at == 20


def test_nested_caption_ends_preserve_the_rolling_overlap_window():
    excerpt = caption_excerpt(payload(time=10, captions=[
        {"start": 0, "end": 10, "text": "The law passed."},
        {"start": 1, "end": 2, "text": "The law passed."},
        {"start": 3, "end": 10, "text": "The law passed. Wages rose."},
    ]))
    assert excerpt.text == "The law passed. Wages rose."


@pytest.mark.parametrize("fields", [
    {"time": True}, {"time": float("nan")}, {"time": -1}, {"language": "es"},
    {"video_id": "other"}, {"source": "invented"}, {"captions": []},
    {"captions": [{"start": 1, "end": 0, "text": "Wrong order"}]},
    {"captions": [{"start": 1, "end": 130, "text": "Too long"}]},
])
def test_invalid_caption_requests_are_rejected(fields):
    with pytest.raises(ValueError):
        caption_excerpt(payload(**fields))


def test_clicks_are_independent_and_abstention_stays_abstention():
    contexts = []

    class Processor:
        def process(self, context):
            contexts.append(context)
            return {"status": "verified", "outcome": "declined_both", "predicted": "supported", "verdict": None}

        def close(self):
            pass

    app = YouTubeChecker(lambda _: Processor())
    try:
        result = app.check(payload(captions=[{"start": 10, "end": 15, "text": "First claim. Second claim. Third claim. Fourth claim."}]))
        assert result["omitted_claims"] == 1 and len(result["rows"]) == 3
        assert [row["text"] for row in result["rows"]] == ["Second claim.", "Third claim.", "Fourth claim."]
        assert result["rows"][0]["result"]["verdict"] is None
        assert [context.claim.text for context in contexts] == ["Fourth claim.", "Third claim.", "Second claim."],             "the clicked claim is researched first"
        other = app.check(payload(video_id="0123456789_"))
        assert other["video_id"] == "0123456789_"
        assert contexts[0].preceding[0].text == "First claim."
        assert all(prior.text != context.claim.text for context in contexts for prior in context.preceding)
        assert not contexts[-1].preceding
        assert all(context.claim.end <= result["clicked_at"] for context in contexts)
    finally:
        app.close()


def test_source_metadata_changes_on_each_click_without_reusing_the_previous_video():
    from src.pipeline.context import SpeechMetadata

    received = []

    class Processor:
        def begin_check(self, metadata):
            received.append(metadata)

        def process(self, context):
            return {"status": "skipped"}

        def close(self):
            pass

    checker = YouTubeChecker(lambda metadata: Processor())
    first = checker.check(payload(source_published_at="2025-07-20"))
    second = checker.check(payload(video_id="0123456789_"))
    assert first["source_published_at"] == "2025-07-20" and second["source_published_at"] == ""
    assert received == [SpeechMetadata(source="https://www.youtube.com/watch?v=abcdefghijk", source_published_at="2025-07-20"),
                        SpeechMetadata(source="https://www.youtube.com/watch?v=0123456789_")]


@pytest.mark.parametrize("published", [None, "2025-02-30", "20250720", "yesterday"])
def test_invalid_publication_metadata_is_rejected(published):
    with pytest.raises(ValueError):
        caption_excerpt(payload(source_published_at=published))


def test_labor_clip_keeps_one_contrast_and_earlier_context_without_future_captions():
    from src.pipeline.context import SpeechMetadata, build_query

    data = json.loads(Path(__file__).with_name("youtube_labor_case.json").read_text(encoding="utf-8"))
    data["captions"].append({"start": 3173, "end": 3180, "text": "Future labor commentary."})
    contexts = []

    class Processor:
        def process(self, context):
            contexts.append(context)
            return {"status": "skipped"}

        def close(self):
            pass

    app = YouTubeChecker(lambda _: Processor())
    result = app.check(data)
    claim = contexts[0]         # the clicked claim is processed first
    assert claim.claim.text == "We don't have a labor shortage. We have a good job shortage."
    assert sum("shortage" in row["text"] for row in result["rows"]) == 1
    assert "Future" not in result["excerpt"]
    query = build_query(claim, SpeechMetadata(), mode="topic")
    assert {"labor", "shortage", "job", "unemployment", "part", "time"} <= set(query.query.split())
    assert all("shortage" not in item.text and "Future" not in item.text for item in claim.preceding)
    assert (result["start"], result["end"], result["clicked_at"]) == (3145, 3173, 3173)
    app.close()


@pytest.mark.slow
def test_labor_clip_with_installed_models_keeps_contrast_and_rejects_song_distractors():
    data = json.loads(Path(__file__).with_name("youtube_labor_case.json").read_text(encoding="utf-8"))
    from src.pipeline.live_processor import LiveProcessor

    app = YouTubeChecker(lambda metadata: LiveProcessor(metadata, query_mode="topic"))
    try:
        result = app.check(data)
        rows = [row for row in result["rows"] if "shortage" in row["text"]]
        assert len(rows) == 1
        checked = rows[0]["result"]
        assert checked["outcome"] == "declined_compound_claim"
        assert checked["verdict"] is None and "confidence" not in checked
        assert checked["evidence"]
        assert checked["evidence"][0][0] == "Secondary_labor_market"
        assert all(not title.startswith(("We_Don't", "Good_Job", "Good_job", "Gagosh", "Labor_market_of_Japan"))
                   for title, _, _ in checked["evidence"])
    finally:
        app.close()


def test_unicode_framing_handles_partial_reads_and_eof():
    class Partial(io.BytesIO):
        def read(self, length=-1):
            return super().read(min(length, 2))

    output = io.BytesIO()
    value = {"id": "é\n", "payload": "世"}
    write_message(output, value)
    stream = Partial(output.getvalue())
    assert read_message(stream) == value
    assert read_message(stream) is None


@pytest.mark.parametrize("raw", [b"\x01", struct.pack("=I", MAX_MESSAGE + 1), struct.pack("=I", 20) + b"{}"])
def test_invalid_native_frames_stop_before_processing(raw):
    with pytest.raises(ValueError):
        read_message(io.BytesIO(raw))


def test_missing_model_returns_correlated_error_and_next_request_is_read():
    class Checker:
        closed = False

        def check(self, value):
            if not value:
                raise SystemExit("missing model export")
            return {"verified": True}

        def close(self):
            self.closed = True

    incoming, outgoing = io.BytesIO(), io.BytesIO()
    write_message(incoming, {"id": "first", "payload": {}})
    write_message(incoming, {"id": "second", "payload": {"valid": True}})
    incoming.seek(0)
    checker = Checker()
    serve(incoming, outgoing, checker)
    outgoing.seek(0)
    assert read_message(outgoing) == {"id": "first", "error": "missing model export"}
    assert read_message(outgoing) == {"id": "second", "result": {"verified": True}}
    assert checker.closed


@pytest.mark.parametrize("invalid", ["x" * MAX_MESSAGE, float("nan"), object(), "\ud800"],
                         ids=["oversized", "nonfinite", "unsupported_type", "invalid_unicode"])
def test_unserializable_result_returns_correlated_error_without_losing_next_request(invalid):
    checker = SimpleNamespace(
        check=lambda value: {"value": invalid} if value["bad"] else {"ok": True},
        close=lambda: None,
    )
    incoming, outgoing = io.BytesIO(), io.BytesIO()
    write_message(incoming, {"id": "bad", "payload": {"bad": True}})
    write_message(incoming, {"id": "good", "payload": {"bad": False}})
    incoming.seek(0)
    serve(incoming, outgoing, checker)
    outgoing.seek(0)
    first = read_message(outgoing)
    assert first["id"] == "bad" and first.get("error") and "result" not in first
    assert read_message(outgoing) == {"id": "good", "result": {"ok": True}}
    assert read_message(outgoing) is None


def test_native_output_limit_counts_encoded_bytes_and_writes_no_partial_frame():
    output = io.BytesIO()
    with pytest.raises(ValueError, match="output limit"):
        write_message(output, {"text": "世" * (MAX_MESSAGE // 2)})
    assert output.getvalue() == b""


def test_broken_native_transport_stops_without_retrying_or_processing_next_request():
    calls = []

    class BrokenOutput(io.BytesIO):
        def write(self, value):
            calls.append("write")
            raise OSError("pipe closed")

    checker = SimpleNamespace(check=lambda value: calls.append("check") or {},
                              close=lambda: calls.append("close"))
    incoming = io.BytesIO()
    write_message(incoming, {"id": "first"})
    write_message(incoming, {"id": "second"})
    incoming.seek(0)
    with pytest.raises(OSError, match="pipe closed"):
        serve(incoming, BrokenOutput(), checker)
    assert calls == ["check", "write", "close"]


def test_context_review_factory_uses_installed_model_and_allows_opt_out(monkeypatch, tmp_path):
    from src.pipeline import youtube as module
    from src.pipeline.context import SpeechMetadata
    from src.verdict.context_review import ContextReviewedJudge

    primary, reviewer = tmp_path / "primary", tmp_path / "reviewer"
    primary.mkdir()
    (reviewer / "onnx").mkdir(parents=True)
    (primary / "contract.json").write_text("{}")
    (reviewer / "contract.json").write_text("{}")
    graph = reviewer / "onnx/model.onnx"
    graph.write_bytes(b"fixture")
    monkeypatch.setattr("src.verdict.pair_judge.MODEL_DIR", primary)
    monkeypatch.setattr("src.verdict.context_review.MODEL_DIR", reviewer)
    monkeypatch.setattr("src.verdict.pair_judge.PairJudge", lambda **kw: SimpleNamespace(
        root=kw["model_dir"], measurement=None, direction_measurement=None))
    monkeypatch.setattr(module, "LiveProcessor", lambda metadata, **kw: SimpleNamespace(**kw))
    monkeypatch.setenv("FACT_CHECKER_CONCEPTS", "off")
    monkeypatch.setenv("FACT_CHECKER_WEB_SEARCH", "off")
    monkeypatch.setenv("FACT_CHECKER_DECOMPOSED", "on")
    monkeypatch.setenv("FACT_CHECKER_CONTEXT_ORDERING", "off")
    monkeypatch.delenv("FACT_CHECKER_PAIR_JUDGE", raising=False)
    monkeypatch.delenv("FACT_CHECKER_CONTEXT_REVIEW", raising=False)
    processor = module.youtube_processor(SpeechMetadata())
    assert isinstance(processor.judge, ContextReviewedJudge)
    assert processor.judge.primary.root == primary and processor.judge.reviewer.root == reviewer
    monkeypatch.setenv("FACT_CHECKER_CONTEXT_REVIEW", "off")
    assert module.youtube_processor(SpeechMetadata()).judge.root == primary
    monkeypatch.setenv("FACT_CHECKER_CONTEXT_REVIEW", "on")
    graph.unlink()
    processor = module.youtube_processor(SpeechMetadata())
    assert processor.judge.root == primary
    assert processor.notes == ["Context review is off: the second local judge is not installed."]


def test_context_ordering_factory_is_lazy_and_allows_opt_out(monkeypatch, tmp_path):
    from src.pipeline import youtube as module
    from src.pipeline.context import SpeechMetadata
    from src.verdict.context_order import ContextOrderedJudge

    primary, ranker = tmp_path / "primary", tmp_path / "ranker"
    primary.mkdir()
    ranker.mkdir()
    (primary / "contract.json").write_text("{}")
    for name in ("contract.json", "source-copy.json", "tokenizer.json", "ranker.onnx"):
        (ranker / name).write_text("not loaded at construction")
    monkeypatch.setattr("src.verdict.pair_judge.MODEL_DIR", primary)
    monkeypatch.setattr("src.verdict.context_ranker.MODEL_DIR", ranker)
    monkeypatch.setattr("src.verdict.pair_judge.PairJudge", lambda **kw: SimpleNamespace(root=kw["model_dir"]))
    monkeypatch.setattr(module, "LiveProcessor", lambda metadata, **kw: SimpleNamespace(**kw))
    for key in ("CONCEPTS", "CONTEXT_REVIEW", "WEB_SEARCH"):
        monkeypatch.setenv("FACT_CHECKER_" + key, "off")
    monkeypatch.setenv("FACT_CHECKER_DECOMPOSED", "on")
    monkeypatch.delenv("FACT_CHECKER_PAIR_JUDGE", raising=False)
    monkeypatch.delenv("FACT_CHECKER_CONTEXT_ORDERING", raising=False)
    processor = module.youtube_processor(SpeechMetadata())
    assert isinstance(processor.judge, ContextOrderedJudge)
    assert processor.judge.primary.root == primary and processor.judge.scorer._session is None
    monkeypatch.setenv("FACT_CHECKER_CONTEXT_ORDERING", "off")
    assert module.youtube_processor(SpeechMetadata()).judge.root == primary
    monkeypatch.setenv("FACT_CHECKER_CONTEXT_ORDERING", "on")
    (ranker / "ranker.onnx").unlink()
    processor = module.youtube_processor(SpeechMetadata())
    assert processor.judge.root == primary
    assert processor.notes == ["Context ordering is off: the local ranker is not installed."]
    monkeypatch.setenv("FACT_CHECKER_CONTEXT_ORDERING", "sometimes")
    with pytest.raises(ValueError, match="FACT_CHECKER_CONTEXT_ORDERING"):
        module.youtube_processor(SpeechMetadata())


def test_processor_factory_defaults_both_stages_on_and_reports_missing_artifacts(monkeypatch, tmp_path):
    from src.pipeline import youtube as module
    from src.pipeline.context import SpeechMetadata

    built = []
    monkeypatch.setattr(module, "LiveProcessor", lambda metadata, **kwargs: built.append(kwargs) or "processor")
    monkeypatch.setattr(module, "WebResearch", lambda sources: "web")
    monkeypatch.setattr("src.verdict.pair_judge.MODEL_DIR", tmp_path / "absent")
    monkeypatch.setattr("src.retrieval.wiki.DB_PATH", tmp_path / "absent.sqlite3")
    monkeypatch.delenv("FACT_CHECKER_CONCEPTS", raising=False)
    monkeypatch.delenv("FACT_CHECKER_DECOMPOSED", raising=False)
    assert module.youtube_processor(SpeechMetadata()) == "processor"
    assert built[-1]["judge"] is None and built[-1]["specificity"] is None
    assert built[-1]["notes"] == ["Caption-concept research is off: the local corpus store is not installed.",
                                  "Sentence-level reading is off: the pair judge is not installed."]
    monkeypatch.setenv("FACT_CHECKER_CONCEPTS", "off")
    monkeypatch.setenv("FACT_CHECKER_DECOMPOSED", "off")
    module.youtube_processor(SpeechMetadata())
    assert built[-1]["notes"] == [] and built[-1]["judge"] is None and built[-1]["specificity"] is None
    monkeypatch.setenv("FACT_CHECKER_DECOMPOSED", "sometimes")
    with pytest.raises(ValueError, match="FACT_CHECKER_DECOMPOSED"):
        module.youtube_processor(SpeechMetadata())


def test_skipped_sentences_do_not_use_up_the_three_claim_slots():
    class Processor:
        def process(self, context):
            if len(context.claim.text.split()) < 5:
                return {"status": "skipped"}
            return {"status": "verified", "outcome": "declined_web_review", "verdict": None,
                    "research": {"sources": [], "assertions": [], "gaps": [], "errors": []}}

        def close(self):
            pass

    app = YouTubeChecker(lambda _: Processor())
    text = "Rates rose in April by two points. That is around 25% of the population. No. Yes, we do. Okay? We have a good job shortage."
    try:
        result = app.check(payload(captions=[{"start": 10, "end": 15, "text": text}]))
    finally:
        app.close()
    assert [row["text"] for row in result["rows"]] == [
        "Rates rose in April by two points.", "That is around 25% of the population.", "No. Yes, we do.", "Okay?",
        "We have a good job shortage."], "skipped sentences stay visible in caption order"
    assert sum(row["result"]["status"] != "skipped" for row in result["rows"]) == 3
    assert result["omitted_claims"] == 0
    longer = app.check(payload(captions=[{"start": 10, "end": 15, "text": "Earlier claim about rates here. " + text}]))
    assert longer["omitted_claims"] == 1 and longer["rows"][0]["text"] == "Rates rose in April by two points."


def test_viewer_declared_country_reaches_the_metadata_and_the_result_but_is_never_inferred():
    seen = []

    class Processor:
        def __init__(self, metadata):
            seen.append(metadata)

        def process(self, context):
            return {"status": "skipped"}

        def close(self):
            pass

    app = YouTubeChecker(Processor)
    try:
        result = app.check(payload(claim_country="  United   States "))
        assert seen[-1].country == "United States" and result["claim_country"] == "United States"
        plain = YouTubeChecker(Processor)
        assert plain.check(payload())["claim_country"] == "" and seen[-1].country == ""
    finally:
        app.close()
    assert caption_excerpt(payload(claim_country="United Kingdom")).claim_country == "United Kingdom"
    for bad in ("<script>", "U", "x" * 41, 7, "United States; drop"):
        with pytest.raises(ValueError, match="claim country"):
            caption_excerpt(payload(claim_country=bad))


def test_viewer_declared_speech_date_bounds_the_search_and_is_never_later_than_the_upload():
    seen = []

    class Processor:
        def __init__(self, metadata):
            seen.append(metadata)

        def process(self, context):
            return {"status": "skipped"}

        def close(self):
            pass

    app = YouTubeChecker(Processor)
    try:
        result = app.check(payload(source_published_at="2025-07-20", spoken_at="2025-07-15"))
        assert seen[-1].spoken_at == "2025-07-15" and result["spoken_at"] == "2025-07-15"
        assert YouTubeChecker(Processor).check(payload())["spoken_at"] == "" and seen[-1].spoken_at == ""
    finally:
        app.close()
    for bad in ("July 15, 2025", "2025-13-01", 20250715):
        with pytest.raises(ValueError, match="speech date"):
            caption_excerpt(payload(spoken_at=bad))
    with pytest.raises(ValueError, match="later than the video"):
        caption_excerpt(payload(source_published_at="2025-07-20", spoken_at="2025-07-21"))


def test_clicks_are_recorded_only_when_a_directory_is_given_and_a_failed_write_never_fails_the_check(tmp_path, capsys):
    checker = SimpleNamespace(check=lambda value: {"video_id": value["video_id"], "rows": []}, close=lambda: None)
    incoming, outgoing = io.BytesIO(), io.BytesIO()
    write_message(incoming, {"id": "one", "payload": {"video_id": "abcdefghijk", "captions": [{"text": "x"}]}})
    incoming.seek(0)
    serve(incoming, outgoing, checker)
    assert not list(tmp_path.iterdir()), "nothing is written without a directory"
    incoming.seek(0)
    serve(incoming, io.BytesIO(), checker, tmp_path / "clicks")
    files = list((tmp_path / "clicks").glob("click-*-abcdefghijk.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert record["payload"]["captions"] == [{"text": "x"}] and record["result"]["video_id"] == "abcdefghijk"
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")
    incoming.seek(0)
    outgoing = io.BytesIO()
    serve(incoming, outgoing, checker, blocked)
    outgoing.seek(0)
    assert read_message(outgoing)["result"]["video_id"] == "abcdefghijk", "the check still answers"
    assert "click not recorded" in capsys.readouterr().err


def test_the_host_launcher_bakes_in_the_record_directory_only_when_asked():
    from pathlib import PureWindowsPath

    from scripts.install_youtube_extension import launcher_text

    workspace = PureWindowsPath(r"C:\work\repo")
    executable = workspace / ".venv/Scripts/python.exe"
    plain = launcher_text(workspace, executable, None)
    assert 'set "FACT_CHECKER_RECORD_CLICKS="' in plain and plain.endswith('-m scripts.youtube_host\n')
    recording = launcher_text(workspace, executable, workspace / "runs/clicks")
    lines = recording.splitlines()
    assert 'set "FACT_CHECKER_RECORD_CLICKS=C:\\work\\repo\\runs\\clicks"' in lines
    assert lines.index('set "FACT_CHECKER_RECORD_CLICKS=C:\\work\\repo\\runs\\clicks"') < lines.index('cd /d "C:\\work\\repo" || exit /b 1')
    workspace = PureWindowsPath(r"C:\100%\repo")
    percent = launcher_text(workspace, workspace / "python.exe", workspace / "runs/clicks")
    assert "C:\\100%%\\repo\\runs\\clicks" in percent, "percent signs are escaped for batch"
