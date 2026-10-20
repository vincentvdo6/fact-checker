"""Run one extension-equivalent click from an exported YouTube transcript.

The browser transcript exporter writes timestamped lines such as ``[1:37] text``.
This helper converts the preceding caption window to the payload accepted by
``YouTubeChecker`` and stores it with the same ``record_click`` function used by
the native messaging host.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from scripts.youtube_host import record_click
from src.pipeline.youtube import YouTubeChecker

LINE = re.compile(r"^\[(?:(\d+):)?(\d+):(\d+)]\s+(.*)$")


def parse_time(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise argparse.ArgumentTypeError("time must be MM:SS or HH:MM:SS")


def transcript_cues(path: Path) -> list[dict]:
    cues: list[dict] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = LINE.match(line)
        if not match:
            continue
        hours, minutes, seconds, text = match.groups()
        start = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
        cues.append({"start": float(start), "text": " ".join(text.split())})
    if not cues:
        raise ValueError(f"no timestamped transcript lines in {path}")
    for index, cue in enumerate(cues):
        following = cues[index + 1]["start"] if index + 1 < len(cues) else cue["start"] + 3
        cue["end"] = max(cue["start"], following - 0.01)
    return cues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--at", type=parse_time, required=True, help="MM:SS or HH:MM:SS, just after the claim")
    parser.add_argument("--country", required=True)
    parser.add_argument("--published", default="")
    parser.add_argument("--spoken", default="")
    parser.add_argument("--out", type=Path, default=Path("runs/clicks"))
    args = parser.parse_args()

    clicked_at = args.at
    cues = [cue for cue in transcript_cues(args.transcript)
            if clicked_at - 30 <= cue["start"] and cue["end"] <= clicked_at]
    payload = {
        "video_id": args.video_id,
        "title": args.title,
        "time": clicked_at,
        "language": "en",
        "source": "transcript",
        "source_published_at": args.published,
        "claim_country": args.country,
        "spoken_at": args.spoken,
        "captions": cues,
    }
    checker = YouTubeChecker()
    try:
        result = checker.check(payload)
    finally:
        checker.close()
    path = record_click(args.out, payload, result)
    print(path)
    for row in result["rows"]:
        print(row["result"].get("status"), row["text"])


if __name__ == "__main__":
    main()
