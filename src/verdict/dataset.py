"""
Assembling and validating the rows the training notebooks read.

The expensive failure in this phase is a bad dataset discovered on a GPU. Every check here runs
before a byte is written, and scripts/check_verdict_dataset.py runs them again against the
written files, because what must be correct is what the notebook will actually read, not what the
builder believed it wrote.

Rows carry sentence text inline rather than references into the corpus. Deduplicating would save
roughly 40% of a file that gzips to about 130 MB, and would buy a join that Kaggle would be the
first place to get wrong.

Titles are composed on both sides. Gold arrives from FEVER as released, which is NFD for 170 of
the 14,533 distinct titles, while retrieval holds the store's composed form -- so one page can
appear under two visually identical titles, defeating the gold/fill dedup in encode and costing
an evidence slot. encode.py cannot import title_key, because Kaggle has no src package, so
normalising is this module's job and its precondition.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from src.data.fever import NOT_ENOUGH_INFO, Claim
from src.retrieval import wiki
from src.retrieval.text import title_key
from src.verdict.encode import FEVER_TO_LABEL


@dataclass(frozen=True, slots=True)
class Row:
    id: int
    label: str                       # the project verdict, not FEVER's native string
    claim: str
    evidence: tuple[tuple[str, int, str], ...]
    gold: tuple[tuple[str, int, str], ...]
    gold_resolved: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "claim": self.claim,
            "evidence": [list(e) for e in self.evidence],
            "gold": [list(g) for g in self.gold],
            "gold_resolved": self.gold_resolved,
        }


class SentenceStore:
    """Sentence text by (composed title, index), cached per page and never by list position."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._pages: dict[str, dict[int, str]] = {}

    def page(self, title: str) -> dict[int, str]:
        # Composed for the cache key, not for the lookup -- wiki.doc_id normalises on its own.
        # Without it an NFD and an NFC spelling of one page would occupy two cache entries.
        key = title_key(title)
        if key not in self._pages:
            doc_id = wiki.doc_id(self._conn, key)
            # The store drops empty sentences, so index i is not position i. Dict, never list.
            self._pages[key] = dict(wiki.sentences(self._conn, doc_id)) if doc_id is not None else {}
        return self._pages[key]

    def text(self, title: str, index: int) -> str | None:
        return self.page(title).get(index)

    def resolve(self, refs: Iterable[tuple[str, int]]) -> list[tuple[str, int, str]]:
        out: list[tuple[str, int, str]] = []
        for title, index in refs:
            text = self.text(title, index)
            if text:
                out.append((title_key(title), index, text))
        return out


def smallest_gold(claim: Claim, store: SentenceStore) -> tuple[list[tuple[str, int, str]], bool]:
    """
    The smallest fully resolvable gold group, and whether one was found.

    Smallest because it is the cheapest to reserve against the token budget -- 88.2% are a single
    sentence. A group with an unresolvable sentence falls through to the next; if none resolve,
    the row is flagged and the oracle notebook skips it, since training an oracle on retrieved
    evidence would quietly make it something else.
    """
    for group in sorted(claim.groups, key=len):
        resolved = store.resolve(sorted(group))
        if len(resolved) == len(group):
            return resolved, True
    return [], False


def build_rows(claims: Iterable[Claim], retrieved: dict[int, list], store: SentenceStore) -> Iterator[Row]:
    for claim in claims:
        refs = retrieved.get(claim.id)
        if refs is None:
            raise ValueError(f"claim {claim.id} has no retrieved evidence")
        gold, resolved = ([], True) if claim.label == NOT_ENOUGH_INFO else smallest_gold(claim, store)
        yield Row(
            id=claim.id,
            label=FEVER_TO_LABEL[claim.label],
            claim=claim.text,
            evidence=tuple(store.resolve(tuple(ref) for ref in refs)),
            gold=tuple(gold),
            gold_resolved=resolved,
        )


def validate(rows: list[Row], claims: dict[int, Claim], *, split: str) -> None:
    """
    Everything that must hold before a byte is written. Raises on the first failure, naming it.

    These are the mistakes that do not announce themselves: a duplicated claim inflates a score,
    an empty evidence list trains on nothing, and a gold ref outside a claim's own groups makes
    the oracle an oracle for some other claim.
    """
    seen: set[int] = set()
    for row in rows:
        if row.id in seen:
            raise ValueError(f"{split}: claim {row.id} appears twice")
        seen.add(row.id)

        claim = claims.get(row.id)
        if claim is None:
            raise ValueError(f"{split}: claim {row.id} is not in the split")
        if row.label != FEVER_TO_LABEL[claim.label]:
            raise ValueError(f"{split}: claim {row.id} label {row.label} does not match {claim.label}")
        if not row.evidence:
            raise ValueError(f"{split}: claim {row.id} has no evidence")

        if row.gold:
            allowed = {(title_key(t), i) for group in claim.groups for t, i in group}
            stray = {(t, i) for t, i, _ in row.gold} - allowed
            if stray:
                raise ValueError(f"{split}: claim {row.id} gold {sorted(stray)} is outside its own groups")
        elif row.gold_resolved and claim.label != NOT_ENOUGH_INFO:
            raise ValueError(f"{split}: claim {row.id} is verifiable but carries no gold")


def assert_splits_disjoint(by_split: dict[str, list[Row]], claims: dict[str, dict[int, Claim]]) -> None:
    """
    No claim id and no claim key may cross splits.

    Re-derived here rather than trusted from drop_leaked upstream: this is the last point before
    the data leaves for a GPU, and a leak found later invalidates every number trained on it.
    """
    seen_ids: dict[int, str] = {}
    seen_keys: dict[str, str] = {}
    for split, rows in by_split.items():
        for row in rows:
            if row.id in seen_ids:
                raise ValueError(f"claim {row.id} is in both {seen_ids[row.id]} and {split}")
            seen_ids[row.id] = split
            key = claims[split][row.id].key
            if key in seen_keys and seen_keys[key] != split:
                raise ValueError(f"claim key {key} spans {seen_keys[key]} and {split}")
            seen_keys[key] = split


def write_rows(rows: Iterable[Row], path: str | Path) -> str:
    """Write gzipped jsonl, returning the sha256 of the file as written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict()) + "\n")
    return sha256(path)


def read_rows(path: str | Path) -> Iterator[Row]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            yield Row(
                id=payload["id"],
                label=payload["label"],
                claim=payload["claim"],
                evidence=tuple(tuple(e) for e in payload["evidence"]),
                gold=tuple(tuple(g) for g in payload["gold"]),
                gold_resolved=payload["gold_resolved"],
            )


def sha256(path: str | Path) -> str:
    """Whole-file digest. Capping it would collide across files that share a long prefix."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
