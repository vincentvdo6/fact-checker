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

from collections.abc import Sequence

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

Evidence = Sequence[Sequence]   # rows of (title, sentence_index, text)


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


def shuffle_pages(rows: list, rng) -> list:
    """
    Permute whole pages, keeping each page's sentences together and in order.

    Shuffling individual rows would break the consecutive-title runs that render() collapses,
    adding one title repetition per break -- measured at +12 tokens on a three-page fixture, which
    is enough to push a set packed to its budget over it, where the tokenizer truncates silently.
    Permuting groups leaves every page contributing exactly one title, so the token count is
    unchanged while gold no longer always arrives first.
    """
    groups: list[list] = []
    for row in rows:
        if groups and groups[-1][0][0] == row[0]:
            groups[-1].append(row)
        else:
            groups.append([row])
    rng.shuffle(groups)
    return [row for group in groups for row in group]


def pack(claim: str, evidence: Evidence, budget: int, measure) -> int:
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


def select_evidence(row: dict, variant: str, budget: int, measure, rng=None) -> list:
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

    reserved = {(title, index) for title, index, _ in gold}
    fill = [ref for ref in retrieved if (ref[0], ref[1]) not in reserved]
    candidates = list(gold) + fill
    chosen = candidates[: max(len(gold), pack(claim, candidates, budget, measure))]
    return shuffle_pages(chosen, rng) if rng is not None else chosen
