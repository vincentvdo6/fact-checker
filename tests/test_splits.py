"""Split construction: determinism, disjointness, and label balance."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from src.data.fever import LABELS, Claim, claim_key, load_claims
from src.data.splits import HOLDOUT_PER_LABEL, drop_leaked, holdout, split_averitec, split_dev

DEV = "data/fever/shared_task_dev.jsonl"
TRAIN = "data/fever/train.jsonl"

requires_fever = pytest.mark.skipif(
    not (Path(DEV).exists() and Path(TRAIN).exists()),
    reason="FEVER data absent; run python -m scripts.fetch_fever",
)


def make(text: str, label: str, id_: int = 0) -> Claim:
    return Claim(id=id_, label=label, text=text, key=claim_key(text), groups=(), pages=())


def synthetic(per_label: int = 60) -> list[Claim]:
    # Ids are sequential, not hashed: hashed ids collide and Python randomizes string
    # hashing per process, which made this helper produce a different fixture per run.
    pairs = itertools.product(LABELS, range(per_label))
    return [make(f"{label} claim number {i}", label, id_=n) for n, (label, i) in enumerate(pairs)]


def test_calibration_and_test_partition_the_input():
    claims = synthetic()
    calibration, test = split_dev(claims)
    assert len(calibration) + len(test) == len(claims)
    assert {c.id for c in calibration} & {c.id for c in test} == set()


def test_no_claim_text_appears_on_both_sides():
    claims = synthetic()
    calibration, test = split_dev(claims)
    assert {c.key for c in calibration} & {c.key for c in test} == set()


def test_repeated_claim_text_stays_together():
    claims = synthetic() + [make("SUPPORTS claim number 0", "SUPPORTS", id_=99_999)]
    calibration, test = split_dev(claims)
    side = calibration if any(c.id == 99_999 for c in calibration) else test
    assert sum(1 for c in side if c.key == claim_key("SUPPORTS claim number 0")) == 2


def test_a_key_with_two_labels_is_cut_by_only_one_of_them():
    """
    Every key belongs to exactly one label's candidate pool, so each label contributes
    round(per_label * fraction) keys and no more. Drop the guard and a key that already
    has a home re-enters a later label's pool, inflating that pool's cut.

    The conflicts have to span both later labels. Confined to one, the hash-ordered
    prefixes come out the same size either way and the guard looks like a no-op.
    """
    per_label, fraction = 40, 0.75
    claims = synthetic(per_label)
    claims += [make(f"SUPPORTS claim number {i}", "REFUTES", id_=90_000 + i) for i in range(20)]
    claims += [make(f"REFUTES claim number {i}", "NOT ENOUGH INFO", id_=95_000 + i) for i in range(20)]

    calibration, _ = split_dev(claims, calibration_fraction=fraction)
    assert len({c.key for c in calibration}) == len(LABELS) * round(per_label * fraction)


def test_split_is_deterministic():
    claims = synthetic()
    first, _ = split_dev(claims)
    second, _ = split_dev(list(reversed(claims)))
    assert {c.key for c in first} == {c.key for c in second}


def test_salt_changes_the_split():
    claims = synthetic()
    default, _ = split_dev(claims)
    other, _ = split_dev(claims, salt="different")
    assert {c.key for c in default} != {c.key for c in other}


def test_label_balance_is_preserved():
    claims = synthetic(per_label=100)
    calibration, test = split_dev(claims)
    for label in LABELS:
        assert sum(1 for c in calibration if c.label == label) == 50
        assert sum(1 for c in test if c.label == label) == 50


def test_fraction_is_respected():
    claims = synthetic(per_label=100)
    calibration, test = split_dev(claims, calibration_fraction=0.25)
    assert len(calibration) == 75
    assert len(test) == 225


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        split_dev(synthetic(), calibration_fraction=fraction)


def test_drop_leaked_matches_on_normalized_text():
    """Casing and spacing differ, so this only passes if the comparison uses `key`."""
    train = [make("Shared   Claim", "SUPPORTS", 1), make("unique claim", "REFUTES", 2)]
    held_out = [make("shared claim", "NOT ENOUGH INFO", 3)]
    assert [c.id for c in drop_leaked(train, held_out)] == [2]


@pytest.mark.slow
@requires_fever
def test_real_dev_split_is_balanced_and_disjoint():
    calibration, test = split_dev(load_claims(DEV))
    assert len(calibration) + len(test) == 19_998
    assert {c.key for c in calibration} & {c.key for c in test} == set()
    for label in LABELS:
        share = sum(1 for c in calibration if c.label == label) / 6_666
        assert 0.45 < share < 0.55


@pytest.mark.slow
@requires_fever
def test_real_train_has_no_evaluation_leakage():
    calibration, test = split_dev(load_claims(DEV))
    cleaned = drop_leaked(load_claims(TRAIN), calibration, test)
    blocked = {c.key for c in calibration} | {c.key for c in test}
    assert not any(c.key in blocked for c in cleaned)


def test_holdout_is_balanced_by_label():
    """Train runs 55/20/25 but both dev halves are 33/33/33; selection must match evaluation."""
    claims = synthetic(per_label=100)
    remaining, held = holdout(claims, per_label=20)
    for label in LABELS:
        assert sum(1 for c in held if c.label == label) == 20
        assert sum(1 for c in remaining if c.label == label) == 80


def test_holdout_partitions_the_input():
    claims = synthetic(per_label=100)
    remaining, held = holdout(claims, per_label=20)
    assert len(remaining) + len(held) == len(claims)
    assert {c.id for c in remaining} & {c.id for c in held} == set()


def test_no_claim_key_spans_the_holdout_boundary():
    """A key selected on and trained on at once would make the selection meaningless."""
    claims = synthetic(per_label=60)
    remaining, held = holdout(claims, per_label=10)
    assert {c.key for c in remaining} & {c.key for c in held} == set()


def test_holdout_takes_every_row_of_a_repeated_key():
    claims = synthetic(per_label=30) + [make("SUPPORTS claim number 0", "SUPPORTS", id_=70_000)]
    _, held = holdout(claims, per_label=30)
    key = claim_key("SUPPORTS claim number 0")
    assert sum(1 for c in held if c.key == key) == 2


def test_holdout_is_deterministic_and_order_independent():
    claims = synthetic(per_label=40)
    first = {c.key for c in holdout(claims, per_label=10)[1]}
    second = {c.key for c in holdout(list(reversed(claims)), per_label=10)[1]}
    assert first == second


def test_holdout_uses_a_different_cut_than_the_dev_split():
    """Sharing a salt would correlate the two splits for no reason."""
    claims = synthetic(per_label=40)
    from src.data.splits import DEFAULT_SALT

    assert {c.key for c in holdout(claims, per_label=10)[1]} != {
        c.key for c in holdout(claims, per_label=10, salt=DEFAULT_SALT)[1]
    }


def test_holdout_refuses_when_a_label_is_too_small():
    with pytest.raises(ValueError, match="need 50"):
        holdout(synthetic(per_label=40), per_label=50)


@pytest.mark.slow
@requires_fever
def test_real_holdout_leaves_train_usable():
    calibration, test = split_dev(load_claims(DEV))
    train = drop_leaked(load_claims(TRAIN), calibration, test)
    remaining, held = holdout(train)

    assert len(remaining) + len(held) == len(train)
    assert {c.key for c in remaining} & {c.key for c in held} == set()
    # Balanced by key; rows run slightly above 3 * per_label because keys repeat.
    assert 3 * HOLDOUT_PER_LABEL <= len(held) < 3 * HOLDOUT_PER_LABEL * 1.2
    for label in LABELS:
        assert sum(1 for c in held if c.label == label) >= HOLDOUT_PER_LABEL


# --- AVeriTeC: three ways, from a pool rather than a released dev ------------------------------

class FakeAveritec:
    """Minimal stand-in: split_averitec only reads .label and .key."""

    def __init__(self, key: str, label: str) -> None:
        self.key = key
        self.label = label


def averitec_claims(per_label: dict[str, int]) -> list[FakeAveritec]:
    return [
        FakeAveritec(f"{label}-{i}", label)
        for label, count in per_label.items()
        for i in range(count)
    ]


def test_averitec_splits_are_disjoint_and_exhaustive():
    claims = averitec_claims({"supported": 400, "contradicted": 800, "mixed": 90, "not_enough_evidence": 130})
    train, calibration, test = split_averitec(claims)

    assert len(train) + len(calibration) + len(test) == len(claims)
    keys = [{c.key for c in part} for part in (train, calibration, test)]
    assert not keys[0] & keys[1]
    assert not keys[0] & keys[2]
    assert not keys[1] & keys[2]


def test_averitec_splits_are_stratified_by_verdict():
    """
    MIXED is under 6% of AVeriTeC. An unstratified cut can leave calibration with a handful of
    them, and a per-class bias fitted on a handful is noise wearing a number.
    """
    per_label = {"supported": 400, "contradicted": 800, "mixed": 90, "not_enough_evidence": 130}
    claims = averitec_claims(per_label)
    _, calibration, test = split_averitec(claims, calibration_fraction=0.14, test_fraction=0.14)

    # Exact counts, not a tolerance band. A tolerance wide enough to be safe is also wide enough
    # for an unstratified split to pass, because random assignment lands each label near its
    # population share anyway -- which is how a vacuous version of this test survived until
    # mutation testing removed the stratification and nothing failed.
    for label, total in per_label.items():
        expected = round(total * 0.14)
        assert sum(1 for c in calibration if c.label == label) == expected, label
        assert sum(1 for c in test if c.label == label) == expected, label


def test_averitec_assignment_is_stable_under_reordering():
    """Hash of the key, not a shuffle: reproducible with no stored index."""
    claims = averitec_claims({"supported": 200, "contradicted": 300, "mixed": 60, "not_enough_evidence": 80})
    first = split_averitec(claims)
    second = split_averitec(list(reversed(claims)))
    for a, b in zip(first, second, strict=True):
        assert {c.key for c in a} == {c.key for c in b}


def test_a_repeated_claim_lands_on_one_side():
    """Two rows sharing a key must not straddle the boundary, or evaluation sees training text."""
    claims = averitec_claims({"supported": 100, "contradicted": 200, "mixed": 40, "not_enough_evidence": 60})
    claims += [FakeAveritec("supported-0", "supported"), FakeAveritec("contradicted-0", "contradicted")]
    parts = split_averitec(claims)
    for key in ("supported-0", "contradicted-0"):
        holders = [i for i, part in enumerate(parts) if any(c.key == key for c in part)]
        assert len(holders) == 1, f"{key} appears in {len(holders)} splits"


def test_averitec_fractions_that_leave_no_train_are_refused():
    claims = averitec_claims({"supported": 50, "contradicted": 50, "mixed": 20, "not_enough_evidence": 20})
    with pytest.raises(ValueError, match="leave some train"):
        split_averitec(claims, calibration_fraction=0.6, test_fraction=0.5)
