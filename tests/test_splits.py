"""Split construction: determinism, disjointness, and label balance."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from src.data.fever import LABELS, Claim, claim_key, load_claims
from src.data.splits import drop_leaked, split_dev

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
