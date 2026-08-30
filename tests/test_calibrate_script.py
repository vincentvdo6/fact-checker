"""
The fitting script: label alignment, a loadable artifact, and the leak that would be invisible.

Nothing here checks a number. It checks that the one script permitted to choose something chooses
it on the right split and writes something the reporting side can actually load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate import CALIBRATION_FILE, CANDIDATES, read_predictions
from src.calibration.bands import BandPolicy
from src.calibration.scaling import from_dict
from src.verdict.encode import LABELS

SCRIPT = Path("scripts/calibrate.py")


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_labels_are_read_into_the_model_class_order():
    """
    The prediction file stores a label *name*; the logits are in class-index order. Mapping by
    anything other than LABELS would silently score every claim against the wrong class, and
    every metric downstream would still compute.
    """
    rows = [
        {"id": 1, "label": LABELS[2], "logits": [0.1, 0.2, 0.7]},
        {"id": 2, "label": LABELS[0], "logits": [0.9, 0.05, 0.05]},
    ]
    path = Path("_tmp_predictions.jsonl")
    try:
        write_predictions(path, rows)
        logits, labels, ids = read_predictions(path)
    finally:
        path.unlink(missing_ok=True)

    assert labels.tolist() == [2, 0]
    assert ids == [1, 2]
    assert logits.shape == (2, 3)


def test_every_candidate_round_trips_through_the_saved_payload():
    """calibrate.py writes; eval_calibration.py reads. A calibrator that cannot survive that trip
    would only fail after the fitting run had already been done."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(300, 3))
    labels = rng.integers(0, 3, size=300)
    for name, cls in CANDIDATES.items():
        fitted = cls().fit(logits, labels)
        restored = from_dict(json.loads(json.dumps(fitted.to_dict())))
        assert restored.transform(logits) == pytest.approx(fitted.transform(logits)), name


def test_the_band_policy_round_trips_through_the_saved_payload():
    rng = np.random.default_rng(1)
    probabilities = rng.uniform(0.34, 1.0, size=500)
    correct = rng.random(500) < probabilities
    from src.calibration.bands import fit_bands

    policy = fit_bands(probabilities, correct)
    restored = BandPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
    assert restored.thresholds == policy.thresholds
    assert restored.assign(0.99) == policy.assign(0.99)


def test_the_fitting_script_never_opens_the_test_split():
    """
    The leak this project cannot afford, guarded structurally rather than by discipline. Choosing
    a calibrator or a band threshold against test makes every promise circular, and the failure is
    silent -- the numbers just improve. If this script ever needs test data, that is a design
    change that should have to delete this test on purpose.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "predictions_test" not in source
    assert "predictions_calibration" in source


def test_uncalibrated_is_a_real_candidate():
    """
    If no transform beats the raw softmax out-of-fold, shipping none is the honest outcome.
    Dropping it from the candidate set would force a calibrator to win by default.
    """
    assert "uncalibrated" in CANDIDATES
    assert CALIBRATION_FILE.endswith(".json")
