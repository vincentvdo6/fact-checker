"""
The check-worthiness dataset, where the failure mode is a leak that looks like skill.

Adjacent sentences in a debate share speaker, topic and often clause structure. Split at the
sentence level and the model memorises a debate then scores it back, which inflates every number
without raising anything -- the leak is indistinguishable from competence in the report. So the
partition is by debate file, and that property is asserted rather than assumed.

The second thing under test is the three-class label space. ClaimBuster separates
factual-but-unimportant from check-worthy factual, and the difference is *importance*, not
verifiability. The Phase 07 rubric labelled the 120 SOTU sentences by verifiability alone, so the
two definitions disagree precisely on that middle band. Binarizing at build time would bury the
disagreement; keeping three classes is what lets the evaluation measure which reading the demo
should act on.
"""

from __future__ import annotations

import pytest

from scripts.build_checkworthy_dataset import FROM_VERDICT, LABELS, rank, summarise, to_row


def record(verdict: str = "1", sentence_id: str = "1", debate: str = "1988-09-25.txt") -> dict:
    return {
        "Sentence_id": sentence_id, "Text": " We consumed 50 percent of it. ",
        "Speaker": "Michael Dukakis", "Speaker_title": "Governor",
        "Speaker_party": "DEMOCRAT", "File_id": debate, "Verdict": verdict,
    }


# --- the label space -----------------------------------------------------------------------------

def test_the_three_native_verdicts_map_to_the_three_classes():
    """-1/0/1 is ClaimBuster's encoding; getting it wrong silently relabels the whole corpus."""
    assert FROM_VERDICT == {
        "-1": "non_factual", "0": "unimportant_factual", "1": "check_worthy",
    }
    assert set(FROM_VERDICT.values()) == set(LABELS)


def test_class_order_is_the_on_disk_contract():
    """Index order is what a trained head learns. Append, never reorder."""
    assert LABELS == ("non_factual", "unimportant_factual", "check_worthy")


@pytest.mark.parametrize(("verdict", "label"), list(FROM_VERDICT.items()))
def test_a_row_carries_its_label_and_its_debate(verdict, label):
    row = to_row(record(verdict=verdict))
    assert row["label"] == label
    assert row["debate"] == "1988-09-25.txt"
    assert row["text"] == "We consumed 50 percent of it.", "surrounding whitespace is stripped"
    assert row["id"] == "cb-1"


# --- the split is by debate ------------------------------------------------------------------------

def test_the_same_debate_always_lands_in_the_same_place():
    """The split is a function of the debate name, so a refetch cannot reshuffle it."""
    assert rank("1988-09-25.txt") == rank("1988-09-25.txt")


def test_different_debates_get_different_ranks():
    names = ["1988-09-25.txt", "1992-10-11.txt", "2016-09-26.txt", "2000-10-03.txt"]
    assert len({rank(n) for n in names}) == len(names)


def test_the_ordering_is_salted_rather_than_alphabetical():
    """
    Debate files are dated names. Sorting them raw would put every early debate in one split and
    every late one in another, so the test set would differ from the train set by era as well as
    by sample -- a confound baked into the partition.
    """
    names = ["1988-09-25.txt", "1992-10-11.txt", "1996-10-06.txt",
             "2000-10-03.txt", "2004-09-30.txt", "2008-09-26.txt",
             "2012-10-03.txt", "2016-09-26.txt"]
    assert sorted(names, key=rank) != sorted(names)


# --- the two binarizations ---------------------------------------------------------------------

def test_factual_counts_both_factual_classes_and_check_worthy_only_one():
    """
    The whole reason three classes survive to disk. Asymmetric counts on purpose -- 2 non-factual,
    3 unimportant, 5 check-worthy -- so a rate computed from the wrong classes cannot coincide.
    """
    rows = ([{"label": "non_factual", "debate": "a.txt"}] * 2
            + [{"label": "unimportant_factual", "debate": "a.txt"}] * 3
            + [{"label": "check_worthy", "debate": "a.txt"}] * 5)
    summary = summarise(rows)
    assert summary["rows"] == 10
    assert summary["factual_rate"] == pytest.approx(0.8), "unimportant + check-worthy, of 10"
    assert summary["check_worthy_rate"] == pytest.approx(0.5), "check-worthy alone, of 10"


def test_the_prior_covers_every_class_even_one_that_never_occurs():
    """A missing key would read as a KeyError in the notebook, or worse as a silent zero shift."""
    summary = summarise([{"label": "check_worthy", "debate": "a.txt"}] * 4)
    assert set(summary["prior"]) == set(LABELS)
    assert summary["prior"]["non_factual"] == 0.0
    assert summary["prior"]["check_worthy"] == pytest.approx(1.0)


def test_a_summary_counts_the_debates_it_spans():
    rows = [{"label": "check_worthy", "debate": d} for d in ("a.txt", "a.txt", "b.txt")]
    assert summarise(rows)["debates"] == 2


# --- the built artifact ---------------------------------------------------------------------------

def test_the_shipped_manifest_has_no_debate_in_two_splits():
    """
    The property the whole scheme exists for, checked against the file that was actually built
    rather than against the code that builds it.
    """
    import json
    from pathlib import Path

    path = Path("data/kaggle/checkworthy-v1/dataset_manifest.json")
    if not path.exists():
        pytest.skip("dataset not built; run scripts.build_checkworthy_dataset")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    debates = {}
    for split in ("train", "calibration", "test"):
        rows = [json.loads(line) for line in
                (path.parent / f"checkworthy_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        debates[split] = {r["debate"] for r in rows}
        assert manifest["splits"][split]["rows"] == len(rows)
    assert not debates["train"] & debates["test"]
    assert not debates["train"] & debates["calibration"]
    assert not debates["calibration"] & debates["test"]
