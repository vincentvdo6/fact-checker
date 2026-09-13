"""Chrome/Edge native messaging host. Stdout is reserved for framed JSON messages.

Set FACT_CHECKER_RECORD_CLICKS to a directory to keep every check -- the caption payload the
extension sent and the full result -- as one JSON file per click. Off by default: the panel's
promise that no transcript is saved holds unless the viewer chooses this. Recorded clicks are
what `scripts/harvest_clicks.py` turns into label files, so the judge can be measured on the
claims a viewer actually checked rather than on FEVER.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from src.pipeline.youtube import YouTubeChecker

MAX_MESSAGE = 1024 * 1024


def record_click(directory: Path, payload: dict, result: dict) -> Path:
    """One file per click, named by time and video, holding exactly what was sent and returned."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    video = str(result.get("video_id") or payload.get("video_id") or "unknown")[:11]
    path = directory / f"click-{stamp}-{video}.json"
    path.write_text(json.dumps({"recorded_at": stamp, "payload": payload, "result": result}, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    return path


def read_exact(stream: BinaryIO, length: int) -> bytes:
    parts = bytearray()
    while len(parts) < length:
        part = stream.read(length - len(parts))
        if not part:
            raise ValueError("Incomplete native message.")
        parts.extend(part)
    return bytes(parts)


def read_message(stream: BinaryIO) -> dict | None:
    first = stream.read(1)
    if not first:
        return None
    length, = struct.unpack("=I", first + read_exact(stream, 3))
    if not 0 < length <= MAX_MESSAGE:
        raise ValueError("Native message exceeds the input limit.")
    value = json.loads(read_exact(stream, length))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def encode_message(value: dict) -> bytes:
    """Validate the entire response before writing any part of its native frame."""
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_MESSAGE:
        raise ValueError("Native response exceeds the output limit.")
    return struct.pack("=I", len(encoded)) + encoded


def write_message(stream: BinaryIO, value: dict) -> None:
    stream.write(encode_message(value))
    stream.flush()


def serve(incoming: BinaryIO, outgoing: BinaryIO, checker: YouTubeChecker, record_dir: Path | None = None) -> None:
    try:
        while (message := read_message(incoming)) is not None:
            request_id = message.get("id")
            if not isinstance(request_id, str) or len(request_id) > 100:
                raise ValueError("Invalid request id.")
            try:
                payload = message.get("payload", {})
                result = checker.check(payload)
                if record_dir is not None:
                    try:
                        record_click(record_dir, payload, result)
                    except OSError as error:      # a full or read-only disk must not fail the check itself
                        print(f"click not recorded: {error}", file=sys.stderr)
                response = {"id": request_id, "result": result}
            except (Exception, SystemExit) as error:
                response = {"id": request_id, "error": str(error)}
            try:
                frame = encode_message(response)
            except (TypeError, ValueError):
                frame = encode_message({"id": request_id, "error": "The check response could not be encoded within the native message limit."})
            outgoing.write(frame)
            outgoing.flush()
    finally:
        checker.close()


def main() -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    outgoing = sys.stdout.buffer
    # Third-party diagnostics must not corrupt the browser's message stream.
    sys.stdout = sys.stderr
    record = os.environ.get("FACT_CHECKER_RECORD_CLICKS", "").strip()
    serve(sys.stdin.buffer, outgoing, YouTubeChecker(), Path(record) if record else None)


if __name__ == "__main__":
    main()
