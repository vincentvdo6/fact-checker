"""
FEVER claim loading.

Records are read as released; the only derived field is `key`, a lowercased and
whitespace-collapsed form of the claim text. That key exists because claims repeat --
6.8% of train rows and 2.0% of dev rows -- and some repeats carry conflicting labels
(753 in train, 55 in dev), so identical text must be kept together and must never
straddle an evaluation split.

Evidence is retained for retrieval scoring as `groups` -- see _groups for why the
shape matters -- with `pages` derived from it. Both are empty for every NOT ENOUGH INFO
row (0 of 35,639 in train), so nothing downstream may assume a claim has evidence.
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
    groups: tuple[frozenset[tuple[str, int]], ...]   # gold evidence; see _groups
    pages: tuple[str, ...]      # pages named by groups, empty for NOT ENOUGH INFO


def claim_key(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


def _groups(evidence: list) -> tuple[frozenset[tuple[str, int]], ...]:
    """
    Gold evidence as deduplicated sets of (page, sentence_index) references.

    A claim is verified by any ONE group in full: conjunctive within a group, disjunctive
    between them. Annotators overlap heavily -- 21.5% of dev groups and 22.0% of train
    groups are exact duplicates -- so they are deduplicated here rather than double-counted
    by everything downstream.

    A group holding a null page is dropped whole rather than emptied. Every NOT ENOUGH INFO
    row is exactly one such group, and an empty frozenset is a subset of everything, so
    keeping one would make any recall metric unconditionally perfect.
    """
    seen: dict[frozenset[tuple[str, int]], None] = {}
    for group in evidence:
        refs: list[tuple[str, int]] = []
        for item in group:
            page, sentence = item[2], item[3]
            if page is None or sentence is None:
                refs = []
                break
            refs.append((page, sentence))
        if refs:
            seen[frozenset(refs)] = None
    return tuple(seen)


def _pages(groups: tuple[frozenset[tuple[str, int]], ...]) -> tuple[str, ...]:
    # Sorted within each group because frozenset iteration order is not stable across runs.
    seen: dict[str, None] = {}
    for group in groups:
        for page, _ in sorted(group):
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
            groups = _groups(row["evidence"])
            claims.append(
                Claim(
                    id=row["id"],
                    label=label,
                    text=row["claim"],
                    key=claim_key(row["claim"]),
                    groups=groups,
                    pages=_pages(groups),
                )
            )
    return claims
