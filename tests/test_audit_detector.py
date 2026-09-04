"""
The subgroup audit, and the discipline that keeps a small cell from becoming a finding.

An aggregate F1 says a filter is good; it does not say whether it is good evenly. For a filter
deciding which claims get fact-checked at all, uneven is the interesting failure -- a detector that
admitted one party's sentences more readily would tilt everything downstream, and the aggregate
number would not move.

Two ways this audit could mislead, both guarded here. A cell of eight sentences produces a rate
that looks like a result, so small cells are reported and excluded from comparison. And a group
whose sentences genuinely contain more claims *should* be admitted more often -- so the base rate
is printed beside the error rate and neither is shown alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.audit_detector import MIN_CELL, group_rows, length_band, report
from src.eval.intervals import wilson

SCRIPT = Path("scripts/audit_detector.py")


def test_length_bands_separate_the_known_coverage_gap():
    """
    Under four words is where ClaimBuster has almost no training data -- 3 sentences in 22,501 --
    so it needs its own band rather than being folded into the shortest ordinary one.
    """
    assert length_band("Period.") == "under 4 words"
    assert length_band("I have done it now.") == "4-9 words"
    assert length_band(" ".join(["word"] * 15)) == "10-19 words"
    assert length_band(" ".join(["word"] * 40)) == "20+ words"


def test_the_band_boundaries_do_not_overlap_or_leave_a_gap():
    seen = [length_band(" ".join(["word"] * n)) for n in range(1, 40)]
    assert seen[0] == "under 4 words" and seen[2] == "under 4 words"
    assert seen[3] == "4-9 words", "four words is the first admitted band"
    assert seen[8] == "4-9 words" and seen[9] == "10-19 words"
    assert seen[18] == "10-19 words" and seen[19] == "20+ words"


def test_a_small_cell_is_reported_but_not_compared(capsys):
    """
    Eight sentences can produce a 0.00 error rate that means nothing. It is shown -- hiding a
    subgroup is its own kind of dishonesty -- and marked so it is not read as the best cell.
    """
    big = (np.zeros(200, dtype=bool), np.zeros(200, dtype=bool))
    tiny = (np.ones(8, dtype=bool), np.zeros(8, dtype=bool))
    rows = report("by nothing", {"large": big, "tiny": tiny})

    printed = capsys.readouterr().out
    assert "tiny" in printed and "(small)" in printed
    assert {r["group"]: r["comparable"] for r in rows} == {"large": True, "tiny": False}
    assert "widest gap" not in printed, "one comparable cell is not a comparison"


def test_a_separated_pair_is_named_as_disjoint(capsys):
    """A gap whose intervals overlap is not a finding, and the line has to distinguish the cases."""
    clean = (np.zeros(400, dtype=bool), np.zeros(400, dtype=bool))
    dirty = (np.ones(400, dtype=bool), np.zeros(400, dtype=bool))
    report("by nothing", {"clean": clean, "dirty": dirty})
    printed = capsys.readouterr().out
    assert "disjoint intervals: clean vs dirty" in printed
    assert "inside the noise" not in printed


def test_a_disjoint_pair_is_found_even_when_it_is_not_the_widest_gap(capsys):
    """
    Wilson width varies with n, so the two extreme point estimates are not necessarily the most
    separated pair. Comparing only those can print "inside the noise" while a genuinely disjoint
    pair sits in the same table -- here the widest gap overlaps and a narrower one does not.
    """
    def cell(errors: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        predicted = np.zeros(n, dtype=bool)
        predicted[:errors] = True
        return predicted, np.zeros(n, dtype=bool)

    report("by nothing", {"wide": cell(11, 40), "tight": cell(499, 1500), "high": cell(19, 35)})
    printed = capsys.readouterr().out
    assert "widest gap: high" in printed and "against wide" in printed
    assert "disjoint intervals" in printed, "the tight/high pair is separated even though the "
    assert "no pair has disjoint" not in printed


def test_two_similar_groups_are_called_noise(capsys):
    a = np.zeros(300, dtype=bool)
    b = np.zeros(300, dtype=bool)
    a[:45] = True
    b[:48] = True
    report("by nothing", {"one": (a, np.zeros(300, dtype=bool)),
                          "two": (b, np.zeros(300, dtype=bool))})
    assert "inside the noise" in capsys.readouterr().out


def test_the_base_rate_is_reported_beside_the_error_rate():
    """
    Without it, a group with more claims looks like a group the detector is biased toward. The
    two numbers only mean something together.
    """
    rows = report("by nothing", {"g": (np.ones(50, dtype=bool), np.ones(50, dtype=bool))})
    assert rows[0]["base_rate"] == pytest.approx(1.0)
    assert rows[0]["admitted_rate"] == pytest.approx(1.0)
    assert rows[0]["error_rate"] == pytest.approx(0.0)


def test_the_minimum_cell_is_large_enough_to_mean_something():
    assert MIN_CELL >= 30


def test_the_interval_stays_inside_zero_and_one():
    low, high = wilson(0, 40)
    assert low >= 0.0 and high <= 1.0


def test_party_is_one_of_the_cuts():
    """
    The cut a reader will ask about first, and the one with an obvious way to be unfair. Leaving
    it out would make the audit look like it had been chosen to avoid the question.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"by party"' in source
    assert '"by sentence length"' in source
    assert '"by speaker"' in source


def test_a_subgroup_of_one_is_still_grouped_rather_than_dropped():
    """
    The cell filter lived in main(), so a test driving report() directly could not see it. Cells
    below ten were silently discarded -- two speakers and six sentences vanished from the
    groundtruth split while the closing line claimed every cell was shown.
    """
    rows = [{"speaker": "A"}] * 40 + [{"speaker": "B"}] * 9 + [{"speaker": "C"}]
    grouped = group_rows(rows, lambda r: r["speaker"])
    assert {g: len(idx) for g, idx in grouped.items()} == {"A": 40, "B": 9, "C": 1}
    assert sum(len(idx) for idx in grouped.values()) == len(rows), "a partition, not a filter"


def test_every_cell_is_printed_so_the_table_is_a_complete_decomposition(capsys):
    """
    Cells were dropped below n=10, which made the table read as a full breakdown while two
    speakers and six sentences were absent from the groundtruth split -- and the closing line
    still claimed every cell was shown. A subgroup too small to compare is still a subgroup that
    was audited, and hiding it is its own kind of dishonesty.
    """
    groups = {
        "big": (np.zeros(200, dtype=bool), np.zeros(200, dtype=bool)),
        "middling": (np.zeros(15, dtype=bool), np.zeros(15, dtype=bool)),
        "single": (np.ones(1, dtype=bool), np.zeros(1, dtype=bool)),
    }
    rows = report("by nothing", groups)
    printed = capsys.readouterr().out

    assert {r["group"] for r in rows} == {"big", "middling", "single"}
    for group in groups:
        assert group in printed, f"{group} was audited but never shown"
    assert printed.count("(small)") == 2, "both sub-threshold cells are marked"


def test_a_single_comparable_cell_says_so_rather_than_printing_nothing(capsys):
    """
    Silence reads as "no gap found". With one comparable cell there is no comparison to make, and
    the output has to distinguish that from a comparison that came back clean.
    """
    report("by nothing", {
        "big": (np.zeros(200, dtype=bool), np.zeros(200, dtype=bool)),
        "tiny": (np.zeros(5, dtype=bool), np.zeros(5, dtype=bool)),
    })
    printed = capsys.readouterr().out
    assert "no comparison: 1 cell(s) reach n >= 30" in printed
    assert "inside the noise" not in printed


def test_the_length_bands_follow_the_filter_s_own_floor():
    """
    The audit's first boundary is the filter's floor. Two copies of the number desync the moment
    anyone moves one, and the audit would then report a band the filter does not use.
    """
    from src.pipeline.detector import MIN_WORDS

    assert length_band(" ".join(["word"] * (MIN_WORDS - 1))) == f"under {MIN_WORDS} words"
    assert length_band(" ".join(["word"] * MIN_WORDS)) != f"under {MIN_WORDS} words"
