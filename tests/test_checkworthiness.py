"""
Checkpoint 3's arithmetic, and the guards that keep the measurement meaningful.

The number this produces is bad -- precision 0.2571, recall 0.3913 on the 2016 State of the Union
-- and the point of these tests is that it stays honest rather than that it improves. Three ways
this measurement could quietly stop meaning anything:

  the labels drift out of step with the segmentation, so indices score different sentences
  precision and recall get swapped, which reads plausibly and inverts the conclusion
  the rubric acquires the heuristic's own rules, making the score 1.0 by construction

The last is the one worth stating loudest. The labels are deliberately written against a general
definition of a checkable claim, *not* against the filter's rules -- so the anchor requirement,
which is a limitation of a BM25-over-Wikipedia stack rather than part of what makes a claim worth
checking, shows up as lost recall. That gap is the finding, and a rubric that encoded the rules
would erase it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.measure_checkworthiness import counts, missed_by_reason

LABELS = Path("labels/checkworthy-sotu-2016.json")
SCRIPT = Path("scripts/measure_checkworthiness.py")


# --- the confusion matrix ----------------------------------------------------------------------

def test_the_four_cells_are_counted_where_they_belong():
    """
    Asymmetric on purpose: 3 true positives, 5 false positives, 2 false negatives, 1 true
    negative. Equal counts would pass whether or not the cells were swapped.
    """
    predicted = [True] * 3 + [True] * 5 + [False] * 2 + [False] * 1
    actual = [True] * 3 + [False] * 5 + [True] * 2 + [False] * 1
    assert counts(predicted, actual) == {"tp": 3, "fp": 5, "fn": 2, "tn": 1}


def test_precision_and_recall_cannot_be_swapped_unnoticed():
    """
    With 3 tp, 5 fp and 2 fn, precision is 3/8 and recall 3/5 -- different numbers, so a swap
    changes both. This is the arithmetic the script performs, checked against those values.
    """
    c = counts([True] * 3 + [True] * 5 + [False] * 2, [True] * 3 + [False] * 5 + [True] * 2)
    precision = c["tp"] / (c["tp"] + c["fp"])
    recall = c["tp"] / (c["tp"] + c["fn"])
    assert precision == pytest.approx(0.375)
    assert recall == pytest.approx(0.6)


def test_a_filter_that_keeps_nothing_has_no_recall_rather_than_full_precision():
    c = counts([False] * 4, [True, True, False, False])
    assert c == {"tp": 0, "fp": 0, "fn": 2, "tn": 2}


# --- the labels ---------------------------------------------------------------------------------

def test_the_labels_carry_a_rubric_a_second_annotator_could_follow():
    truth = json.loads(LABELS.read_text(encoding="utf-8"))
    for field in ("rubric", "sample", "source", "labeller", "limitation"):
        assert truth[field].strip(), f"{field} is empty"
    assert len(truth["rubric"]) > 120, "a one-line rubric is not reproducible"


def test_the_stated_limitation_names_the_shared_author_problem():
    """
    One person wrote the heuristic and the labels. That bias has a direction, and a reader of the
    number has to be told rather than left to infer it.
    """
    truth = json.loads(LABELS.read_text(encoding="utf-8"))
    limitation = truth["limitation"].lower()
    assert "author" in limitation
    assert "annotator" in limitation


def test_the_rubric_does_not_encode_the_heuristic_s_own_rules():
    """Labelling by the rules being scored makes precision and recall 1.0 by construction."""
    rubric = json.loads(LABELS.read_text(encoding="utf-8"))["rubric"].lower()
    for rule in ("no_anchor", "no_assertion", "check_worthy ="):
        assert rule not in rubric.replace("check_worthy = the sentence asserts", "")


def test_the_sample_is_described_precisely_enough_to_reproduce():
    truth = json.loads(LABELS.read_text(encoding="utf-8"))
    assert "[::3]" in truth["sample"]
    assert len(truth["labels"]) == 120


def test_every_label_is_a_boolean_and_every_key_an_index():
    truth = json.loads(LABELS.read_text(encoding="utf-8"))
    for key, value in truth["labels"].items():
        assert key.isdigit(), f"{key!r} is not a sentence index"
        assert isinstance(value, bool), f"label {key} is {type(value).__name__}, not bool"


def test_the_labels_are_not_all_one_class():
    """A degenerate label set would make recall or precision trivially defined."""
    values = list(json.loads(LABELS.read_text(encoding="utf-8"))["labels"].values())
    assert 0 < sum(values) < len(values)


# --- the script refuses to score against the wrong sentences ------------------------------------

def test_labels_that_no_longer_match_the_segmentation_are_a_hard_error():
    """
    If the transcript or the segmenter changes, index 51 stops being the sentence that was
    labelled. Scoring on anyway would produce a number that looks fine and means nothing, so the
    script raises instead.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "raise SystemExit(" in source
    assert "set(labelled) - set(sentences)" in source


def test_the_report_names_the_rule_that_dropped_each_missed_claim():
    """
    Recall alone says the filter is losing claims; the per-reason table says which rule to fix.
    On this transcript that table is the finding -- half the misses are the anchor requirement.
    Counted here rather than checked for in the source, so an empty table cannot pass.
    """
    from src.pipeline.segment import Decision

    labelled = {0: True, 1: True, 2: True, 3: False, 4: True}
    decisions = {
        0: Decision(False, "no_anchor"),
        1: Decision(False, "no_anchor"),
        2: Decision(False, "imperative"),
        3: Decision(False, "no_anchor"),      # not a real claim; must not be counted
        4: Decision(True, "check_worthy"),    # kept; must not be counted
    }
    assert missed_by_reason(labelled, decisions) == Counter({"no_anchor": 2, "imperative": 1})


def test_the_limitation_is_printed_with_the_number_not_buried_in_a_file():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'truth[\'limitation\']' in source or 'truth["limitation"]' in source
