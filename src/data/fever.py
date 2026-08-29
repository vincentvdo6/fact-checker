"""
FEVER claim loading.

Records are read as released; the only derived field is `key`, a lowercased and
whitespace-collapsed form of the claim text. That key exists because claims repeat --
6.8% of train rows and 2.0% of dev rows -- and some repeats carry conflicting labels
(753 in train, 55 in dev), so identical text must be kept together and must never
straddle an evaluation split.

Evidence page names are retained for later retrieval scoring, but are empty for
every NOT ENOUGH INFO row (0 of 35,639 in train), so nothing downstream may assume
a claim has pages.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
NOT_ENOUGH_INFO = "NOT ENOUGH INFO"
LABELS: tuple[str, ...] = (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Claim:
    id: int
    label: str
    text: str
    key: str                    # grouping key; see module docstring
    pages: tuple[str, ...]      # gold evidence pages, empty for NOT ENOUGH INFO


def claim_key(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


def _pages(evidence: list) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for group in evidence:
        for item in group:
            page = item[2]
            if page is not None:
                seen[page] = None
    return tuple(seen)


def load_claims(path: str | Path) -> list[Claim]:
    claims = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = row["label"]
            if label not in LABELS:
                raise ValueError(f"unexpected label {label!r} in {path}")
            claims.append(
                Claim(
                    id=row["id"],
                    label=label,
                    text=row["claim"],
                    key=claim_key(row["claim"]),
                    pages=_pages(row["evidence"]),
                )
            )
    return claims
