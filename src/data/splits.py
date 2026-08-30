"""
Train / calibration / test construction.

FEVER ships a large imbalanced train set and a label-balanced dev set. Dev is cut in
half to form calibration and test, so both come from the same balanced distribution.
Fitting a temperature on one and reporting ECE on the other is the whole point: fit
on anything that overlaps test and the reported calibration error is roughly halved.

Assignment is by hash of the claim key rather than a shuffle. That is stable under
input reordering, needs no stored index to reproduce, and places every copy of a
repeated claim on the same side automatically. The cut is made within each label so
dev's balance survives into both halves.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from src.data.fever import LABELS, Claim

DEFAULT_SALT = "fever-calibration-v1"
HOLDOUT_SALT = "fever-trainval-v1"

# Keys per label in the model-selection holdout; three labels, so about 5,000 claims.
HOLDOUT_PER_LABEL = 1667


def _rank(key: str, salt: str) -> int:
    digest = hashlib.sha1(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def split_dev(
    claims: list[Claim],
    *,
    salt: str = DEFAULT_SALT,
    calibration_fraction: float = 0.5,
) -> tuple[list[Claim], list[Claim]]:
    """Cut dev into (calibration, test) by claim key, stratified by label."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")

    keys_by_label: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        keys_by_label[claim.label].add(claim.key)

    # A key carrying two labels sits in both pools, so without this it gets a second
    # chance at the cut and calibration drifts above the requested fraction. It can never
    # land on both sides -- the partition below is by one key set -- so the cost is a
    # skewed fraction, not leakage. First label to claim it wins, its other rows follow.
    calibration_keys: set[str] = set()
    assigned: set[str] = set()
    for label in LABELS:
        candidates = sorted(keys_by_label[label] - assigned, key=lambda k: _rank(k, salt))
        cut = round(len(candidates) * calibration_fraction)
        calibration_keys.update(candidates[:cut])
        assigned.update(candidates)

    calibration = [c for c in claims if c.key in calibration_keys]
    test = [c for c in claims if c.key not in calibration_keys]
    return calibration, test


def drop_leaked(train: list[Claim], *held_out: list[Claim]) -> list[Claim]:
    """Remove train rows whose claim text also appears in an evaluation split."""
    blocked = {claim.key for split in held_out for claim in split}
    return [claim for claim in train if claim.key not in blocked]


def holdout(
    claims: list[Claim],
    *,
    per_label: int = HOLDOUT_PER_LABEL,
    salt: str = HOLDOUT_SALT,
) -> tuple[list[Claim], list[Claim]]:
    """
    Carve a model-selection split out of train. Returns (remaining, held_out).

    Checkpoints have to be selected on something, and selecting on the calibration split would
    mean the model had seen it -- which is the one thing this project cannot afford, because
    calibration is the contribution rather than a side effect. So selection happens here, on
    claims train can spare, and calibration stays untouched by training in every sense.

    Balanced by label, not proportional. Train runs 55/20/25 while both dev halves are 33/33/33,
    so a checkpoint chosen on train's prior is chosen for a distribution it is never evaluated
    on. Train holds 29,775 REFUTES claims, so balance costs nothing worth having.

    `per_label` counts keys, not rows: a key appearing twice takes both its rows, because a
    claim that straddled the boundary would be selected on and trained on at once. The held-out
    row count therefore runs slightly above `3 * per_label`.
    """
    keys_by_label: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        keys_by_label[claim.label].add(claim.key)

    chosen: set[str] = set()
    assigned: set[str] = set()
    for label in LABELS:
        # Same guard as split_dev: a key with two labels is cut once, by the first to reach it.
        candidates = sorted(keys_by_label[label] - assigned, key=lambda k: _rank(k, salt))
        if len(candidates) < per_label:
            raise ValueError(f"{label} has {len(candidates)} keys, need {per_label}")
        chosen.update(candidates[:per_label])
        assigned.update(candidates)

    held = [claim for claim in claims if claim.key in chosen]
    remaining = [claim for claim in claims if claim.key not in chosen]
    return remaining, held
