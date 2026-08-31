"""
The relabel rule: what it moves, what it leaves alone, and what it must never be pointed at.

Phase 02 traced a 38-point NEI gap to verifiable rows whose evidence cannot justify their label.
This rule removes them, and every way it can go wrong is silent: relabel too much and the model
learns to abstain on claims it could have answered; relabel too little and the contradiction that
caused the gap survives; relabel the evaluation splits and every number since Phase 02 becomes
incomparable while still computing.
"""

from __future__ import annotations

import pytest

from src.data.fever import NOT_ENOUGH_INFO, Claim
from src.verdict.dataset import Row, reground, validate

SUPPORTED, CONTRADICTED, NEI = "supported", "contradicted", "not_enough_evidence"


def claim(claim_id: int, label: str, groups) -> Claim:
    return Claim(
        id=claim_id,
        label=label,
        text="a claim",
        key="a claim",
        groups=tuple(frozenset(g) for g in groups),
        pages=tuple(sorted({title for group in groups for title, _ in group})),
    )


def row(claim_id: int, label: str, refs, gold=()) -> Row:
    return Row(
        id=claim_id,
        label=label,
        claim="a claim",
        evidence=tuple((t, i, "text") for t, i in refs),
        gold=tuple((t, i, "text") for t, i in gold),
        gold_resolved=bool(gold),
    )


def test_a_row_whose_gold_was_never_read_becomes_nei():
    """
    Gold sits at rank 3; the encoder packed 2 sentences. The label says SUPPORTED and the input
    contains nothing that supports it -- exactly the contradiction Phase 02 measured.
    """
    refs = [("A", 0), ("A", 1), ("A", 2), ("A", 3)]
    rows = [row(1, SUPPORTED, refs, gold=[("A", 3)])]
    claims = {1: claim(1, "SUPPORTS", [[("A", 3)]])}

    out, stats = reground(rows, claims, {1: 2})
    assert out[0].label == NEI
    assert out[0].gold == ()
    assert stats["moved"] == 1


def test_the_same_row_is_left_alone_when_the_budget_reaches_its_gold():
    """The only difference is how much the encoder packed, and that decides everything."""
    refs = [("A", 0), ("A", 1), ("A", 2), ("A", 3)]
    rows = [row(1, SUPPORTED, refs, gold=[("A", 3)])]
    claims = {1: claim(1, "SUPPORTS", [[("A", 3)]])}

    out, stats = reground(rows, claims, {1: 4})
    assert out[0].label == SUPPORTED
    assert out[0].gold == (("A", 3, "text"),)
    assert stats["moved"] == 0


def test_a_larger_group_counts_even_when_the_stored_smallest_one_does_not():
    """
    Row.gold keeps only the cheapest group to reserve. A two-sentence group can be fully present
    while the one-sentence group is not, and judging on the stored gold alone would relabel a row
    whose evidence was there the whole time.
    """
    refs = [("A", 0), ("A", 1)]
    rows = [row(1, SUPPORTED, refs, gold=[("Z", 9)])]      # stored smallest group, not retrieved
    claims = {1: claim(1, "SUPPORTS", [[("Z", 9)], [("A", 0), ("A", 1)]])}

    out, stats = reground(rows, claims, {1: 2})
    assert out[0].label == SUPPORTED, "the two-sentence group was fully read"
    assert stats["moved"] == 0


def test_a_partially_read_group_does_not_count():
    """Groups are conjunctive: half the evidence for a claim is not evidence for it."""
    refs = [("A", 0), ("B", 5)]
    rows = [row(1, CONTRADICTED, refs, gold=[("A", 0)])]
    claims = {1: claim(1, "REFUTES", [[("A", 0), ("A", 7)]])}

    out, _ = reground(rows, claims, {1: 2})
    assert out[0].label == NEI


def test_rows_that_were_already_nei_are_untouched():
    """NEI has no gold by construction; relabelling it would be a no-op that inflates the count."""
    rows = [row(1, NEI, [("A", 0)])]
    claims = {1: claim(1, NOT_ENOUGH_INFO, [])}

    out, stats = reground(rows, claims, {1: 0})
    assert out[0] is rows[0]
    assert stats["moved"] == 0


def test_relabelled_rows_keep_their_evidence_and_their_claim():
    """Only the label and the gold change: the model must see the same input, judged honestly."""
    refs = [("A", 0), ("A", 1)]
    rows = [row(1, SUPPORTED, refs, gold=[("Z", 9)])]
    claims = {1: claim(1, "SUPPORTS", [[("Z", 9)]])}

    out, _ = reground(rows, claims, {1: 2})
    assert out[0].evidence == rows[0].evidence
    assert out[0].claim == rows[0].claim
    assert out[0].id == rows[0].id


def test_the_stats_describe_what_moved():
    refs = [("A", 0)]
    rows = [
        row(1, SUPPORTED, refs, gold=[("Z", 9)]),        # ungrounded -> moves
        row(2, CONTRADICTED, refs, gold=[("A", 0)]),     # grounded   -> stays
        row(3, NEI, refs),                               # already NEI
    ]
    claims = {
        1: claim(1, "SUPPORTS", [[("Z", 9)]]),
        2: claim(2, "REFUTES", [[("A", 0)]]),
        3: claim(3, NOT_ENOUGH_INFO, []),
    }

    out, stats = reground(rows, claims, {1: 1, 2: 1, 3: 1})
    assert stats["moved"] == 1
    assert stats["verifiable_before"] == 2
    assert stats["share_of_verifiable"] == pytest.approx(0.5)
    assert stats["prior_before"][SUPPORTED] == pytest.approx(1 / 3)
    assert stats["prior_after"][NEI] == pytest.approx(2 / 3)
    assert [r.label for r in out] == [NEI, CONTRADICTED, NEI]


def test_a_missing_budget_is_refused_rather_than_assumed():
    """
    Defaulting to the stored 25 would silently mark every unbudgeted row grounded, which is the
    direction that quietly does nothing and reports success.
    """
    rows = [row(1, SUPPORTED, [("A", 0)], gold=[("A", 0)])]
    claims = {1: claim(1, "SUPPORTS", [[("A", 0)]])}
    with pytest.raises(ValueError, match="no packing budget"):
        reground(rows, claims, {})


def test_a_missing_claim_is_refused():
    rows = [row(1, SUPPORTED, [("A", 0)], gold=[("A", 0)])]
    with pytest.raises(ValueError, match="missing from the claim map"):
        reground(rows, {}, {1: 1})


# --- validate must permit exactly one departure from FEVER's labels ----------------------------

def rows_and_claims(row_label: str, claim_label: str):
    rows = [row(1, row_label, [("A", 0)], gold=() if row_label == NEI else [("A", 0)])]
    return rows, {1: claim(1, claim_label, [] if claim_label == NOT_ENOUGH_INFO else [[("A", 0)]])}


def test_validate_rejects_a_relabelled_row_unless_told_regrounding_happened():
    """The guard stays on by default, so a stray relabel in an ordinary build still raises."""
    rows, claims = rows_and_claims(NEI, "SUPPORTS")
    with pytest.raises(ValueError, match="does not match"):
        validate(rows, claims, split="train")


def test_validate_accepts_a_verifiable_claim_moved_to_nei_when_regrounded():
    rows, claims = rows_and_claims(NEI, "SUPPORTS")
    validate(rows, claims, split="train", regrounded=True)


def test_validate_still_rejects_a_swapped_verdict_when_regrounded():
    """
    Regrounding permits one direction only. A SUPPORTED row on a REFUTES claim is a bug whether or
    not the split was regrounded, and it is the kind that trains a model on inverted supervision.
    """
    rows, claims = rows_and_claims(SUPPORTED, "REFUTES")
    with pytest.raises(ValueError, match="does not match"):
        validate(rows, claims, split="train", regrounded=True)


def test_validate_still_rejects_a_claim_moved_out_of_nei_when_regrounded():
    """The reverse direction invents evidence for a claim FEVER says has none."""
    rows, claims = rows_and_claims(SUPPORTED, NOT_ENOUGH_INFO)
    with pytest.raises(ValueError, match="does not match"):
        validate(rows, claims, split="train", regrounded=True)


def test_the_builder_regrounds_only_the_training_splits():
    """
    Relabelling calibration or test would grade the model against our own relabelling, and every
    number since Phase 02 would become incomparable while still computing.
    """
    import inspect

    from scripts import build_verdict_dataset

    source = inspect.getsource(build_verdict_dataset.main)
    assert 'split in ("train", "trainval")' in source
    assert "args.reground" in source


def test_the_checker_takes_regrounding_from_the_manifest_not_a_default():
    """
    The permission must be per-split and derived from what the build recorded. Hard-coding it True
    would let a build that relabelled rows *without* recording it pass validation -- and no
    existing dataset would catch that, because a dataset with no relabels passes either way.
    """
    from pathlib import Path

    source = Path("scripts/check_verdict_dataset.py").read_text(encoding="utf-8")
    assert "regrounded=split in regrounded" in source
    assert "regrounded=True" not in source
    assert 'manifest.get("regrounded")' in source
