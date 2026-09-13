"""Build (sentence, assertion) pairs labelled with the four relations the pair judge must learn.

The judge answers one narrow question: does this one sentence state the assertion, state
its negation, bear on its subject without settling it, or concern something else. No
public dataset labels exactly that, but two nearly do, and they fail in complementary
ways, which is why both are used:

  FEVER  gold sentence of a SUPPORTS / REFUTES claim -> states / states_negation, taken only
         from single-sentence groups, since a sentence that establishes a claim jointly with
         another does not establish it alone. Retrieved-but-not-gold sentences -> bears_on:
         BM25 found them for the claim's entities, and annotation did not cite them.
         Sentences retrieved for another claim about other pages -> unrelated.
  MNLI   entailment / contradiction / neutral over ordinary prose, with negation on both
         sides; neutral is the one public source of "same topic, no entailment", which is
         the bears_on relation. Premises crossed with hypotheses from another genre ->
         unrelated.

Two known label noises are accepted and recorded rather than hidden. FEVER annotation is not
exhaustive, so a retrieved sentence labelled bears_on can occasionally entail the claim.
And FEVER's REFUTES claims carry negation words ten times as often as SUPPORTS claims, the
tell Phase 02 measured; MNLI dilutes it and `negated_hypothesis` lets held-out splits report
the subgroup. Grouping keys keep every pair built from one claim, or one MNLI premise, on a
single side of the split.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.data.fever import NOT_ENOUGH_INFO, Claim
from src.data.mnli import NliPair
from src.retrieval.text import title_key
from src.verdict.assertions import negate_form
from src.verdict.pair_judgment import RELATIONS

FEVER_TO_RELATION = {"SUPPORTS": "states", "REFUTES": "states_negation"}
MNLI_TO_RELATION = {"entailment": "states", "contradiction": "states_negation", "neutral": "bears_on"}

_NEGATED = re.compile(r"\b(?:not|no|never|none|nobody|nothing|neither|nor|cannot)\b|n['’]t\b", re.I)


@dataclass(frozen=True, slots=True)
class Pair:
    id: str
    source: str                 # "fever" or "mnli"
    relation: str
    premise: str                # the sentence
    hypothesis: str             # the assertion
    group: str                  # leakage key: claim key, or MNLI promptID
    negated_hypothesis: bool
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "relation": self.relation, "premise": self.premise,
                "hypothesis": self.hypothesis, "group": self.group, "negated_hypothesis": self.negated_hypothesis,
                "provenance": dict(self.provenance)}


def negated(text: str) -> bool:
    return bool(_NEGATED.search(text))


class SentenceLookup:
    """The two calls the builder needs from a sentence store, so tests can supply a dict."""

    def __init__(self, store) -> None:
        self._store = store

    def text(self, title: str, index: int) -> str | None:
        return self._store.text(title, index)


def _resolved(refs: Iterable[tuple[str, int]], lookup: SentenceLookup) -> list[tuple[str, int, str]]:
    out = []
    for title, index in refs:
        text = lookup.text(title, index)
        if text:
            out.append((title_key(title), int(index), text))
    return out


def fever_pairs(claims: list[Claim], retrieved: dict[int, list], lookup: SentenceLookup, *, rng: random.Random,
                gold_per_claim: int = 2, bears_per_claim: int = 1, unrelated_per_claim: int = 1) -> list[Pair]:
    """Pairs from one split's claims; the donor pool for unrelated sentences is the same split."""
    donors: list[tuple[int, frozenset[str], tuple[str, int, str]]] = []
    prepared = []
    for claim in claims:
        refs = retrieved.get(claim.id)
        if refs is None:
            raise ValueError(f"claim {claim.id} has no retrieved evidence")
        found = _resolved((tuple(ref) for ref in refs), lookup)
        pages = frozenset(title_key(title) for title in claim.pages) | frozenset(title for title, _, _ in found)
        prepared.append((claim, found, pages))
        donors.extend((claim.id, pages, row) for row in found)

    pairs: list[Pair] = []
    emitted: set[tuple[str, str]] = set()      # repeated claims share a key; their pairs must not repeat
    for claim, found, pages in prepared:
        negated_claim = negated(claim.text)

        def add(kind: str, relation: str, row: tuple[str, int, str], extra: dict) -> None:
            title, index, text = row
            if text == claim.text or (text, claim.text) in emitted:
                return
            emitted.add((text, claim.text))
            pairs.append(Pair(id=f"fever:{claim.id}:{kind}:{title}:{index}", source="fever", relation=relation,
                              premise=text, hypothesis=claim.text, group=claim.key, negated_hypothesis=negated_claim,
                              provenance={"claim_id": claim.id, "label": claim.label, "title": title,
                                          "index": index, **extra}))

        gold_refs: set[tuple[str, int]] = set()
        if claim.label != NOT_ENOUGH_INFO:
            all_gold = {(title_key(title), index) for group in claim.groups for title, index in group}
            singles = sorted({(title_key(title), index) for group in claim.groups if len(group) == 1
                              for title, index in group})
            for row in _resolved(singles, lookup)[:gold_per_claim]:
                add("gold", FEVER_TO_RELATION[claim.label], row, {"kind": "gold"})
            gold_refs = all_gold
        candidates = [row for row in found if (row[0], row[1]) not in gold_refs]
        for row in rng.sample(candidates, min(bears_per_claim, len(candidates))):
            add("retrieved", "bears_on", row, {"kind": "retrieved_not_gold"})
        for _ in range(unrelated_per_claim):
            for _attempt in range(20):
                donor_id, donor_pages, row = donors[rng.randrange(len(donors))]
                if donor_id != claim.id and not (donor_pages & pages) and row[0] not in pages:
                    add("unrelated", "unrelated", row, {"kind": "other_claim_retrieval", "donor_claim_id": donor_id})
                    break
    return pairs


def mnli_pairs(rows: list[NliPair], *, rng: random.Random, unrelated_every: int = 3) -> list[Pair]:
    """Native relations, plus cross-genre premise swaps as unrelated."""
    pairs, emitted = [], set()
    for row in rows:
        if row.premise == row.hypothesis or (row.premise, row.hypothesis) in emitted:
            continue
        emitted.add((row.premise, row.hypothesis))
        pairs.append(Pair(id=f"mnli:{row.pair_id}", source="mnli", relation=MNLI_TO_RELATION[row.label],
                          premise=row.premise, hypothesis=row.hypothesis, group=f"mnli:{row.prompt_id}",
                          negated_hypothesis=negated(row.hypothesis),
                          provenance={"pair_id": row.pair_id, "genre": row.genre, "label": row.label}))
    by_genre: dict[str, list[NliPair]] = {}
    for row in rows:
        by_genre.setdefault(row.genre, []).append(row)
    genres = sorted(by_genre)
    if len(genres) > 1 and unrelated_every > 0:
        for position, row in enumerate(rows):
            if position % unrelated_every:
                continue
            other = genres[(genres.index(row.genre) + 1 + rng.randrange(len(genres) - 1)) % len(genres)]
            donor = by_genre[other][rng.randrange(len(by_genre[other]))]
            if (donor.premise, row.hypothesis) in emitted:
                continue
            emitted.add((donor.premise, row.hypothesis))
            pairs.append(Pair(id=f"mnli:{row.pair_id}:unrelated:{donor.pair_id}", source="mnli", relation="unrelated",
                              premise=donor.premise, hypothesis=row.hypothesis, group=f"mnli:{row.prompt_id}",
                              negated_hypothesis=negated(row.hypothesis),
                              provenance={"pair_id": row.pair_id, "genre": row.genre, "donor_pair_id": donor.pair_id,
                                          "donor_genre": donor.genre, "label": "cross_genre"}))
    return pairs


def negation_augment(pairs: list[Pair], *, rng: random.Random, rate: float,
                     exclude: set[tuple[str, str]] | None = None) -> list[Pair]:
    """Decorrelate negation from the label by negating hypotheses under every label that permits it.

    A sentence that states H contradicts the negated H; one that bears on H, or is unrelated to
    it, stands in the same relation to the negated H. `states_negation` pairs are left alone:
    un-negating a refuted claim does not make its gold sentence support it. Training only, and
    `exclude` must hold every pair of every split: a negated claim can already exist as its own
    FEVER claim, on the same gold sentence, on the held-out side.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must lie in [0, 1]")
    flipped = {"states": "states_negation", "bears_on": "bears_on", "unrelated": "unrelated"}
    existing = {(pair.premise, pair.hypothesis) for pair in pairs} | set(exclude or ())
    added = []
    for pair in pairs:
        if pair.relation not in flipped or pair.negated_hypothesis or rng.random() >= rate:
            continue
        hypothesis = negate_form(pair.hypothesis)
        if hypothesis is None or (pair.premise, hypothesis) in existing:
            continue
        existing.add((pair.premise, hypothesis))
        added.append(Pair(id=f"{pair.id}:neg", source=pair.source, relation=flipped[pair.relation],
                          premise=pair.premise, hypothesis=hypothesis, group=pair.group, negated_hypothesis=True,
                          provenance=dict(pair.provenance) | {"kind": "negation_augment", "from": pair.id}))
    return added


def validate(by_split: dict[str, list[Pair]]) -> dict:
    """Refuse duplicates, empty text, unknown relations and any group on two sides of a split."""
    seen_pairs: dict[tuple[str, str], str] = {}
    seen_groups: dict[str, str] = {}
    seen_ids: set[str] = set()
    counts: dict[str, dict[str, int]] = {}
    for split, pairs in by_split.items():
        histogram: dict[str, int] = {}
        for pair in pairs:
            if pair.relation not in RELATIONS:
                raise ValueError(f"{pair.id}: unknown relation {pair.relation!r}")
            if not pair.premise.strip() or not pair.hypothesis.strip() or pair.premise == pair.hypothesis:
                raise ValueError(f"{pair.id}: empty or identical premise and hypothesis")
            if pair.id in seen_ids:
                raise ValueError(f"{pair.id}: duplicate pair id")
            seen_ids.add(pair.id)
            key = (pair.premise, pair.hypothesis)
            if key in seen_pairs:
                raise ValueError(f"{pair.id}: duplicate of {seen_pairs[key]}")
            seen_pairs[key] = pair.id
            if seen_groups.setdefault(pair.group, split) != split:
                raise ValueError(f"group {pair.group!r} appears in {seen_groups[pair.group]} and {split}")
            label = f"{pair.source}:{pair.relation}"
            histogram[label] = histogram.get(label, 0) + 1
        counts[split] = dict(sorted(histogram.items()))
    return counts
