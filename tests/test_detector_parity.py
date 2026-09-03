"""
Checkpoint 5's guards: the detector's parity check has to be able to fail.

The verdict model got this check in Phase 07 and the detector shipped without it, which was an
oversight rather than a decision. The exposure is identical -- the vector-scaling calibrator and
the 0.35 threshold were both fitted against logits torch produced on Kaggle, so a drifting export
moves the boundary deciding which sentences reach the verdict model, and nothing raises.

What makes this check stronger than the verdict one is its last comparison. Admission agreement is
the composition of graph, calibrator and frozen cut -- the only thing the pipeline consumes -- so
it covers the whole path in one number rather than three that each cover a stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_detector_parity import LOGIT_TOLERANCE, read_split

SCRIPT = Path("scripts/check_detector_parity.py")


def test_the_tolerance_is_tight_enough_to_mean_something():
    """
    fp32 kernels differ between runtimes, so exact equality is not the bar -- but a tolerance loose
    enough to admit any export is not a checkpoint. The observed drift is 9.775e-06.
    """
    assert LOGIT_TOLERANCE == 1e-3


def test_the_check_compares_admissions_and_not_only_logits():
    """
    Logits agreeing to 1e-3 says nothing about whether a sentence lands on the same side of 0.35.
    The admission comparison is the only one covering graph, calibrator and threshold together.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "admission agreement" in source
    assert "(local_score >= cut) == (kaggle_score >= cut)" in source


def test_both_binarizations_are_checked():
    """
    factual and check_worthy have separate frozen thresholds, so agreement on one says nothing
    about the other -- and the pipeline can be configured to either.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "for name, cut in thresholds.items():" in source
    assert "columns = (1, 2) if name == \"factual\" else (2,)" in source


def test_the_calibrator_is_applied_to_both_sides_of_the_comparison():
    """
    Calibrating only the local logits would compare a calibrated score against a raw one and fail
    for a reason that has nothing to do with the export.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "local_probabilities = calibrator.transform(local)" in source
    assert "kaggle_probabilities = calibrator.transform(kaggle)" in source


def test_a_dataset_rebuilt_after_training_is_refused_rather_than_scored():
    """
    Sentence ids are the join. If the dataset were rebuilt, the ids would silently address
    different sentences and the parity number would be meaningless rather than wrong.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "raise SystemExit(" in source
    assert "are not in the" in source


def test_a_failure_returns_non_zero_and_names_the_worst_row():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "return 1" in source
    assert "worst row" in source


def test_the_split_is_read_as_text_by_id(tmp_path, monkeypatch):
    """The entire input to this model is the sentence, which is what makes the check simple."""
    import scripts.check_detector_parity as module

    data = tmp_path
    (data / "checkworthy_test.jsonl").write_text(
        "\n".join(json.dumps({"id": f"cb-{i}", "text": f"sentence {i}", "label": "non_factual"})
                  for i in range(3)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DATA", data)
    assert read_split("test") == {"cb-0": "sentence 0", "cb-1": "sentence 1", "cb-2": "sentence 2"}


@pytest.mark.parametrize("split", ["test", "calibration"])
def test_either_split_can_be_checked(split):
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'choices=("calibration", "test")' in source
    assert f'"{split}"' in source or split in source
