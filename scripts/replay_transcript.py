"""Emit saved speech as paced final ASR segments, with explicitly simulated timestamps.

Pipe stdout into scripts.verify_stream. --speed 1 simulates the requested speaking rate;
--speed 0 emits a burst for an overload test. Neither mode measures real ASR latency.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from src.pipeline.replay import replay_updates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--words-per-minute", type=float, default=150)
    parser.add_argument("--speed", type=float, default=1, help="0 for burst; otherwise playback multiplier")
    parser.add_argument("--speaker", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed < 0 or args.start_index < 0 or args.limit < 0:
        parser.error("speed, start-index and limit must be finite and nonnegative")
    sys.stdout.reconfigure(encoding="utf-8")
    updates = replay_updates(args.transcript.read_text(encoding="utf-8"),
                             words_per_minute=args.words_per_minute, speaker=args.speaker)
    origin = time.monotonic()
    media_origin: float | None = None
    sent = 0
    for update in updates:
        if int(update.id) < args.start_index:
            continue
        if media_origin is None:
            media_origin = update.start
        if args.speed:
            deadline = origin + (update.end - media_origin) / args.speed
            while (remaining := deadline - time.monotonic()) > 0:
                time.sleep(min(remaining, 0.5))
        print(json.dumps(asdict(update), ensure_ascii=False), flush=True)
        sent += 1
        if args.limit and sent >= args.limit:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
