"""
MultiNLI loading.

Rows are read as released. `gold_label` is the annotators' majority vote and is "-" when there
was none; those rows are dropped here, since a pair with no agreed relation cannot supervise
one. `promptID` groups every hypothesis written for one premise -- entailment, neutral and
contradiction alike -- so it is the unit that must never straddle an evaluation split: the
premise text would otherwise be seen in training and again at test time under another label.
Both IDs are unique only within one release file -- the matched and mismatched dev sets restart
numbering -- so they are namespaced by the file they came from; and 6,988 train pairIDs are
duplicated within that file, so the line number is appended to make the pair identity unique.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LABELS: tuple[str, ...] = ("entailment", "neutral", "contradiction")


@dataclass(frozen=True, slots=True)
class NliPair:
    pair_id: str
    prompt_id: str        # the premise's identity; the leakage key
    genre: str
    label: str
    premise: str
    hypothesis: str


def load_pairs(path: str | Path) -> list[NliPair]:
    pairs: list[NliPair] = []
    tag = Path(path).stem.replace("multinli_1.0_", "")
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row["gold_label"] not in LABELS:
                continue
            premise, hypothesis = row["sentence1"].strip(), row["sentence2"].strip()
            if not premise or not hypothesis:
                continue
            pairs.append(NliPair(pair_id=f"{tag}:{row['pairID']}:{number}", prompt_id=f"{tag}:{row['promptID']}", genre=row["genre"],
                                 label=row["gold_label"], premise=premise, hypothesis=hypothesis))
    return pairs
