"""Flush each event to durable storage before publishing it to a live consumer.

Each session owns a new file. Reusing a path fails instead of mixing independent revision
histories. A crash can leave a partial last line; complete earlier lines remain inspectable.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import TextIO


class EventJournal:
    def __init__(self, path: Path, output: TextIO, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("x", encoding="utf-8", newline="\n")
        self._output = output
        self._lock = threading.Lock()
        self._sequence = 0
        self.failures = 0
        try:
            self.write({"type": "session", "schema": 1, "metadata": metadata})
        except BaseException:
            self._handle.close()
            raise

    def write(self, event: dict) -> None:
        with self._lock:
            line = json.dumps(event | {"sequence": self._sequence}, ensure_ascii=False, allow_nan=False) + "\n"
            self._handle.write(line)
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._output.write(line)
            self._output.flush()
            self._sequence += 1
            if event["type"] in ("failed", "rejected"):
                self.failures += 1

    def close(self) -> None:
        self._handle.close()
