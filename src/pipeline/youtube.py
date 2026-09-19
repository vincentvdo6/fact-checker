"""Check a bounded caption excerpt, tied to one YouTube video and click timestamp.

Captions are the asserted speech, never evidence. No video date is inferred from its
upload date. Each click is independent so seeking and navigation cannot reuse context.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from src.pipeline.caption_claims import caption_claims
from src.pipeline.context import SpeechMetadata
from src.pipeline.live_processor import LiveProcessor
from src.pipeline.transcript import ClaimContext, TranscriptUpdate
from src.pipeline.web_research import WebResearch
from src.retrieval.google_news_sources import GoogleNewsSources


class Processor(Protocol):
    def process(self, context: ClaimContext) -> dict: ...
    def close(self) -> None: ...


def seconds(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 604800:
        raise ValueError("Caption timestamps must be finite nonnegative seconds.")
    return float(value)


@dataclass(frozen=True)
class CaptionExcerpt:
    video_id: str
    title: str
    clicked_at: float
    start: float
    end: float
    text: str
    language: str
    source: str
    source_published_at: str = ""
    claim_country: str = ""     # declared by the viewer for this video; never inferred from sources or the news edition
    spoken_at: str = ""         # declared by the viewer; the search boundary when given, never later than the upload


COUNTRY = re.compile(r"[A-Za-z][A-Za-z .'-]{1,39}")


def caption_excerpt(payload: dict) -> CaptionExcerpt:
    if not isinstance(payload, dict):
        raise ValueError("Expected caption request fields.")
    video_id, language = payload.get("video_id"), payload.get("language", "")
    if not isinstance(video_id, str) or not re.fullmatch(r"[\w-]{11}", video_id, re.ASCII):
        raise ValueError("Open a YouTube video before checking a claim.")
    if not isinstance(language, str) or not re.fullmatch(r"en(?:-[A-Za-z]{2,8})?", language):
        raise ValueError("The checker currently needs English captions. Select English or Auto-translate → English in YouTube.")
    clicked_at = seconds(payload.get("time"))
    title, source = payload.get("title", ""), payload.get("source", "")
    published = payload.get("source_published_at", "")
    if not isinstance(published, str) or (published and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", published)):
        raise ValueError("Invalid video publication date.")
    if published:
        date.fromisoformat(published)
    if not isinstance(title, str) or len(title) > 500 or source not in ("track", "text-track", "transcript", "visible"):
        raise ValueError("Invalid caption source metadata.")
    country = payload.get("claim_country", "")
    if not isinstance(country, str) or (country and not COUNTRY.fullmatch(country.strip())):
        raise ValueError("The claim country must be a short place name, or left unset.")
    country = " ".join(country.split())
    spoken_at = payload.get("spoken_at", "")
    if not isinstance(spoken_at, str) or (spoken_at and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", spoken_at)):
        raise ValueError("The speech date must be a calendar day (YYYY-MM-DD), or left unset.")
    if spoken_at:
        try:
            date.fromisoformat(spoken_at)
        except ValueError:
            raise ValueError("The speech date must be a calendar day (YYYY-MM-DD), or left unset.") from None
        if published and spoken_at > published:
            raise ValueError("The speech date cannot be later than the video's publication date.")
    captions = payload.get("captions")
    if not isinstance(captions, list) or not 1 <= len(captions) <= 500:
        raise ValueError("No captions were available around this playback position.")
    selected = []
    for cue in captions:
        if not isinstance(cue, dict):
            raise ValueError("Invalid caption cue.")
        start, end = seconds(cue.get("start")), seconds(cue.get("end"))
        text = cue.get("text")
        if end < start or end - start > 120 or not isinstance(text, str) or len(text) > 4000:
            raise ValueError("Invalid caption text or duration.")
        # A cue crossing the left edge may contain the subject of the next cue's claim.
        if max(0, clicked_at - 30) < end <= clicked_at and text.strip():
            selected.append((start, end, " ".join(text.split())))
    selected.sort(key=lambda cue: (cue[0], cue[1]))
    latest_end = max((cue[1] for cue in selected), default=-1.0)
    if not selected or latest_end < clicked_at - 8:
        raise ValueError("No recent captions at this position. Play a captioned section and try again.")
    words: list[str] = []
    previous_end = -1.0
    for start, end, text in selected:
        incoming = text.split()
        overlap = 0
        if start < previous_end:
            for size in range(min(len(words), len(incoming)), 0, -1):
                if words[-size:] == incoming[:size]:
                    overlap = size
                    break
        words.extend(incoming[overlap:])
        previous_end = max(previous_end, end)
    text = " ".join(words)
    if not text or len(text) > 4000:
        raise ValueError("This caption excerpt is too long to check safely. Try a shorter section.")
    return CaptionExcerpt(video_id, title, clicked_at, selected[0][0], latest_end, text, language, source, published, country,
                          spoken_at)


def _switch(name: str, default: str = "on") -> bool:
    value = os.environ.get(name, default).strip()
    if value not in {"on", "off"}:
        raise ValueError(f"{name} must be on or off.")
    return value == "on"


def youtube_processor(metadata: SpeechMetadata) -> LiveProcessor:
    """Try news discovery locally; no search account or hosted service is required.

    The sentence-level reading and caption-concept research are on unless switched off. Each
    needs a local artifact -- the pair judge under `models/pair_judge`, the FEVER wiki store for
    phrase specificity -- and a missing one is reported on every result rather than raised, so a
    click on a machine without it still returns the research it could do.
    """
    mode = os.environ.get("FACT_CHECKER_WEB_SEARCH", "news").strip()
    if mode not in {"news", "off"}:
        raise ValueError("FACT_CHECKER_WEB_SEARCH must be news or off.")
    web = WebResearch(GoogleNewsSources()) if mode == "news" else None
    notes: list[str] = []
    specificity = None
    if _switch("FACT_CHECKER_CONCEPTS"):
        from src.pipeline.caption_concepts import CorpusSpecificity

        specificity = CorpusSpecificity()
        if not specificity.available:
            specificity = None
            notes.append("Caption-concept research is off: the local corpus store is not installed.")
    judge = None
    if _switch("FACT_CHECKER_DECOMPOSED"):
        from src.calibration.bands import Band
        from src.verdict.pair_judge import MODEL_DIR, PairJudge

        model_dir = Path(os.environ.get("FACT_CHECKER_PAIR_JUDGE", "").strip() or MODEL_DIR)
        if (model_dir / "contract.json").is_file():
            judge = PairJudge(model_dir=model_dir, min_band=Band.STRONG, probe_negation=True, probe_hedges=True)
        else:
            notes.append("Sentence-level reading is off: the pair judge is not installed.")
    return LiveProcessor(metadata, query_mode="topic", web=web, judge=judge, specificity=specificity, notes=notes)


MAX_CLAIMS = 3      # check-worthy sentences checked per click, counting back from the playback position


class YouTubeChecker:
    def __init__(self, factory: Callable[[SpeechMetadata], Processor] | None = None) -> None:
        self.factory = factory or youtube_processor
        self.processor: Processor | None = None

    def check(self, payload: dict) -> dict:
        excerpt = caption_excerpt(payload)
        metadata = SpeechMetadata(source=f"https://www.youtube.com/watch?v={excerpt.video_id}",
                                  country=excerpt.claim_country, spoken_at=excerpt.spoken_at,
                                  source_published_at=excerpt.source_published_at)
        if self.processor is None:
            self.processor = self.factory(metadata)
        begin_check = getattr(self.processor, "begin_check", None)
        if begin_check is not None:
            begin_check(metadata)
        sentences = caption_claims(excerpt.text)
        rows = []
        # Most recent first, so the clicked claim gets the shared web budget before anything earlier,
        # and the three slots go to check-worthy sentences: a "No." or an "Okay?" the detector skips
        # is shown as skipped but does not use one up.
        for sentence in reversed(sentences):
            if sum(row["result"].get("status") != "skipped" for row in rows) >= MAX_CLAIMS:
                break
            # Cue timings bound the excerpt; sentence-level word timings are unavailable.
            claim = TranscriptUpdate(str(sentence.index), 0, sentence.text, excerpt.start, excerpt.end, True, "captions")
            preceding = tuple(
                TranscriptUpdate(str(prior.index), 0, prior.text, excerpt.start, excerpt.end, True, "captions")
                for prior in sentences[:sentence.index]
            )
            result = self.processor.process(ClaimContext(claim, preceding))
            rows.append({"text": sentence.text, "result": result})
        rows.reverse()
        web_scope = ("experimental news research enabled; source applicability unverified"
                     if getattr(self.processor, "web", None) is not None
                     else "web research is off")
        return {"video_id": excerpt.video_id, "title": excerpt.title, "clicked_at": excerpt.clicked_at,
                "start": excerpt.start, "end": excerpt.end, "excerpt": excerpt.text, "source": excerpt.source,
                "language": excerpt.language, "rows": rows, "omitted_claims": len(sentences) - len(rows),
                "source_published_at": excerpt.source_published_at, "claim_country": excerpt.claim_country,
                "spoken_at": excerpt.spoken_at,
                "scope": f"English captions; local Wikipedia June 2017; {web_scope}; YouTube accuracy unmeasured."}

    def close(self) -> None:
        if self.processor is not None:
            self.processor.close()
