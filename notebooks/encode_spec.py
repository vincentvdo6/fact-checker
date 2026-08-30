"""
Build the model's input string from a claim and its evidence.

This module is the train/serve contract. A tabular model can pin a feature list; a transformer
has no such list -- the thing that must not drift is the *function that assembles the text*. So
this file is copied byte-for-byte into the Kaggle dataset as `encode_spec.py`, the notebook
imports that copy, and a test asserts the two are identical. Training then runs the serving code
rather than a reimplementation of it.

That copy is why nothing here may import from `src` -- there is no `src` on Kaggle. LABELS is
therefore written out literally, and a test asserts it equals the label space in
src/verdict/labels.py, which stays the single source of truth.

Evidence is packed dynamically rather than at a fixed k. Retrieved sentences average about 20
words, so 25 of them overrun a 512-token context; a fixed k would then have to be set for the
worst case and would waste room on every claim shorter than it. Packing to the budget gives each
claim as much evidence as it can hold, and the count actually used is exported per prediction so
the achievable ceiling stays a measurement rather than an assumption.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

TEMPLATE_ID = "per_page_grouped_v1"

# Mirrors src/verdict/labels.py in class-index order. Written literally because Kaggle has no
# src package; tests/test_encode.py asserts the two agree.
LABELS: tuple[str, ...] = ("supported", "contradicted", "not_enough_evidence")

FEVER_TO_LABEL = {
    "SUPPORTS": "supported",
    "REFUTES": "contradicted",
    "NOT ENOUGH INFO": "not_enough_evidence",
}

# A sentence longer than this is truncated before packing. The 99th percentile of retrieved
# sentences is 56 words, so this bites on well under 1% while capping the pathological ones that
# would otherwise consume a whole budget on their own.
MAX_SENTENCE_WORDS = 64

Evidence = Sequence[Sequence]              # rows of (title, sentence_index, text)
Measure = Callable[[str, str], int]        # a tokenizer's length, or a word count in tests


def _clip(text: str) -> str:
    words = text.split()
    return text if len(words) <= MAX_SENTENCE_WORDS else " ".join(words[:MAX_SENTENCE_WORDS])


def readable_title(title: str) -> str:
    """Underscores to spaces and bracket escapes decoded, so the model reads prose."""
    out = title.replace("_", " ")
    for escaped, plain in (("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("-COLON-", ":")):
        out = out.replace(escaped, plain)
    return out


def render(evidence: Evidence) -> str:
    """
    Evidence as text, consecutive sentences from one page grouped under a single title.

    Grouping is not cosmetic. Repeating the title on every sentence costs roughly 50-70 tokens
    per example, which is two or three more sentences of evidence for free.
    """
    parts: list[str] = []
    current: str | None = None
    for title, _, text in evidence:
        if title != current:
            parts.append(f"{readable_title(title)}:")
            current = title
        parts.append(_clip(text))
    return " ".join(parts)


def build_input(claim: str, evidence: Evidence) -> tuple[str, str]:
    """The (first, second) segment pair a cross-encoder tokenizes. Empty evidence is legal."""
    return claim, render(evidence)


def shuffle_pages(rows: Sequence[Sequence], rng: random.Random) -> list:
    """
    Permute the title runs, so gold stops always arriving first.

    Shuffling individual rows would break the consecutive-title runs that render() collapses,
    adding one title repetition per break -- measured at +12 tokens on a three-page fixture,
    enough to push a packed set over its budget where the tokenizer truncates in silence.

    This permutes *runs*, not pages. A page occupying two non-adjacent runs -- ordinary on the
    gold path, where the fill arrives in retrieval order -- may end up with its runs adjacent and
    merged. So the token count never increases, but it is not invariant either: it can fall.
    pack()'s measurement is therefore an upper bound after a shuffle, never an overrun.
    """
    groups: list[list] = []
    for row in rows:
        if groups and groups[-1][0][0] == row[0]:
            groups[-1].append(row)
        else:
            groups.append([row])
    rng.shuffle(groups)
    return [row for group in groups for row in group]


def pack(claim: str, evidence: Evidence, budget: int, measure: Measure) -> int:
    """
    How many leading evidence rows fit alongside the claim, given a budget and a length function.

    The claim is measured, not assumed away: it occupies the same sequence as the evidence, so
    packing against an empty first segment overruns by the length of the claim on every example.

    `measure` takes the rendered pair and returns its token count, so this module stays free of
    any tokenizer dependency -- the notebook passes a real one, tests pass a word counter.

    Binary search rather than a linear scan. Packing is monotone in prefix length because adding
    a row either extends the current title run or opens a new one, and neither shortens the
    render -- so this costs about five measurements instead of one per sentence.
    """
    if not evidence or measure(*build_input(claim, [])) > budget:
        return 0
    low, high = 0, len(evidence)
    while low < high:
        middle = (low + high + 1) // 2
        if measure(*build_input(claim, evidence[:middle])) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def select_evidence(
    row: dict,
    variant: str,
    budget: int,
    measure: Measure,
    rng: random.Random | None = None,
) -> list:
    """
    The rows of evidence this variant shows the model.

    claim_only sees none: the baseline exists to detect FEVER's claim-only artifacts, so any
    evidence at all would defeat it.

    gold reserves the gold group first and fills the rest from retrieval. Without that fill the
    oracle would be trained on one or two sentences for verifiable claims and fifteen retrieved
    ones for NOT ENOUGH INFO, and would learn to read the sentence count instead of the text --
    scoring near-perfectly while measuring nothing. shuffle_pages then removes the position tell
    of gold always arriving first, at no cost in tokens.
    """
    if variant == "claim_only":
        return []

    claim = row["claim"]
    retrieved = list(row["evidence"])
    if variant == "retrieved":
        return retrieved[: pack(claim, retrieved, budget, measure)]

    if variant != "gold":
        raise ValueError(f"unknown variant {variant!r}")

    gold = [tuple(ref) for ref in row.get("gold") or []]
    if not gold:
        # NOT ENOUGH INFO has no gold by construction, so it keeps the retrieved condition.
        return retrieved[: pack(claim, retrieved, budget, measure)]

    # Titles are compared as given. Gold arrives from FEVER as released, which is NFD for 170 of
    # the 14,533 distinct titles, while retrieval holds the store's composed form -- so the
    # dataset builder must NFC both sides before a row reaches here, or the same page appears
    # twice under two visually identical titles and costs an evidence slot.
    reserved = {(title, index) for title, index, _ in gold}
    fill = [ref for ref in retrieved if (ref[0], ref[1]) not in reserved]
    candidates = list(gold) + fill
    # The floor keeps gold whole even when it alone overruns: without it a verifiable claim gets
    # zero evidence while NOT ENOUGH INFO keeps a full retrieved set, which is the sentence-count
    # artifact this function exists to prevent. The tokenizer then truncates the tail, so the
    # exported count overstates what the model read -- 1 of 13,332 dev claims, and worth the trade.
    budgeted = max(len(gold), pack(claim, candidates, budget, measure))
    # Never show more rows than the retrieved condition could. Gold sentences retrieval missed
    # make `candidates` longer than `retrieved`, so on a row where they all fit the total lands
    # one above the retrieved cap -- and because only verifiable claims carry gold, that single
    # extra row is a perfect tell for verifiability wherever it fires. Measured at 3 of 256 test
    # rows, all verifiable. The gold floor still wins when the group alone is larger.
    chosen = candidates[: min(budgeted, max(len(gold), len(retrieved)))]
    return shuffle_pages(chosen, rng) if rng is not None else chosen
