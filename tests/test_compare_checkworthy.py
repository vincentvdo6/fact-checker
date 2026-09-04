"""
The rules-against-detector comparison, and the discipline that keeps it from being a tuned number.

This script reports a +0.45 F1 swing, which is exactly the size of result that invites a mistake:
sweep the threshold, quote the best cell, call it the headline. That is selection on the test set
and it would void the comparison entirely -- the 120 SOTU labels are the only held-out data this
task has.

So the headline is pinned to 0.5, the model's own decision boundary, fixed before the sweep was
looked at. The sweep is printed as a robustness check and the script must say so; if the result
only held at one threshold, that would be the finding instead of a footnote.

The same discipline is why the table carries a fourth row. The three raw rows measure the model at
a bare 0.5; the sidebar runs the calibrated filter at a threshold frozen on ClaimBuster's debates
behind a length floor taken as a definitional prior. Publishing only the raw rows would quote a
number for a system nobody runs, and printing the configured row without saying where its operating
point came from would leave a reader unable to tell it from one fitted on these 120 labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_checkworthy import main, prf
from src.pipeline.detector import Decision, Scored
from src.pipeline.segment import segment

SCRIPT = Path("scripts/compare_checkworthy.py")


# --- the arithmetic ------------------------------------------------------------------------------

def test_precision_and_recall_are_not_interchangeable():
    """
    3 true positives, 5 false positives, 2 false negatives: precision 3/8, recall 3/5. Different
    values, so a swap changes both rather than cancelling.
    """
    predicted = np.array([True] * 3 + [True] * 5 + [False] * 2)
    actual = np.array([True] * 3 + [False] * 5 + [True] * 2)
    got = prf(predicted, actual)
    assert got["precision"] == pytest.approx(0.375)
    assert got["recall"] == pytest.approx(0.6)
    assert got["f1"] == pytest.approx(2 * 0.375 * 0.6 / (0.375 + 0.6))


def test_predicting_everything_gives_full_recall_and_base_rate_precision():
    actual = np.array([True, True, False, False, False, False])
    got = prf(np.ones(6, dtype=bool), actual)
    assert got["recall"] == pytest.approx(1.0)
    assert got["precision"] == pytest.approx(1 / 3)


def test_predicting_nothing_scores_zero_rather_than_a_perfect_precision():
    """The 0/0 guard must not resolve in the flattering direction."""
    got = prf(np.zeros(4, dtype=bool), np.array([True, True, False, False]))
    assert got["precision"] == 0.0
    assert got["recall"] == 0.0
    assert got["f1"] == pytest.approx(0.0)


def test_accuracy_counts_both_kinds_of_agreement():
    """Two right and two wrong, arranged so a precision-only reading would disagree."""
    got = prf(np.array([True, False, True, False]), np.array([True, False, False, True]))
    assert got["accuracy"] == pytest.approx(0.5)


# --- the operating point is not chosen on the test set ------------------------------------------

def test_the_headline_is_the_pre_specified_boundary_not_the_sweep_maximum():
    """
    An earlier version of this script printed `best: max(results)`, which picks a setting by its
    score on the only held-out labels there are. The headline must name 0.5 explicitly.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "pre-specified 0.5 boundary" in source
    assert 'results["detector, check_worthy >= 0.5"]' in source
    assert "max(results, key=" not in source, "no argmax over settings scored on the test set"


def test_the_sweep_is_labelled_as_robustness_rather_than_selection():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "robustness, not as a choice" in source
    assert "NOT a selection" in source


def test_the_sweep_reports_whether_each_threshold_beats_the_rules():
    """
    The claim being supported is "every threshold beats the heuristic", so each row has to say so
    or not. A sweep that only printed F1 would leave the reader to eyeball it.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "beats the rules?" in source
    assert "'yes' if min(f_one, w_one) > baseline else 'NO'" in source


def test_labels_out_of_step_with_the_segmentation_are_a_hard_error():
    """Same guard as Checkpoint 3: index 51 must still be the sentence that was labelled."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "raise SystemExit(" in source
    assert "set(labelled) - set(sentences)" in source


def test_both_systems_are_scored_on_the_same_sentences():
    """
    The heuristic and the detector must see one list, in one order. Scoring them on separately
    built lists is how an off-by-one becomes a +0.45 result.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "order = sorted(labelled)" in source
    assert "texts = [sentences[i].text for i in order]" in source
    assert "heuristic = np.array([check_worthy(t).worthy for t in texts]" in source
    assert "detector.score_batch(texts)" in source


def test_the_labels_limitation_is_carried_into_the_output():
    """
    The shared-author caveat bounds the heuristic's number. It does not bound the detector's, but
    the reader still needs it to interpret the baseline being beaten.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "limitation carried from the labels" in source


# --- the row the demo actually runs --------------------------------------------------------------

SPEECH = (
    "Unemployment fell to 5 percent in 2015. That was a terrible decision. "
    "The deficit was cut by half since 2010. Thank you."
)
# Index 1 is labelled check-worthy and the rules reject it for having no anchor, so the heuristic
# baseline is neither perfect nor degenerate and a delta against it carries information.
HAND_LABELS = {"0": True, "1": True, "2": True, "3": False}


class RawStub:
    """A raw model that calls every sentence check-worthy, so any row read from it is all True."""

    threads = 1

    def __init__(self, *args, **kwargs) -> None:
        self.seen: list[str] = []

    def score_batch(self, sentences, **kwargs) -> list[Scored]:
        self.seen.extend(sentences)
        return [Scored(probabilities=np.array([0.0, 0.0, 1.0]), label="check_worthy")
                for _ in sentences]


def filter_stub(threshold: float, min_words: int) -> type:
    """
    A configured filter that declines everything, at an operating point unlike the frozen one.

    Both halves are load-bearing. Its verdicts are the exact opposite of `RawStub`'s, so a pipeline
    row taken from the softmax cannot coincide with one taken from the filter; and 0.37 / 6 words
    are values no hard-coded label could happen to contain.
    """

    class Stub:
        seen: list[str] = []

        def __init__(self, binarization: str = "factual", **kwargs) -> None:
            self.binarization = binarization
            self.threshold = threshold
            self.min_words = min_words

        def decide_batch(self, sentences) -> list[Decision]:
            Stub.seen = list(sentences)
            return [Decision(False, "below_factual_threshold", 0.1) for _ in sentences]

    return Stub


def run_main(tmp_path, monkeypatch, *, threshold: float = 0.37, min_words: int = 6):
    """Drive `main` over four sentences with both models stubbed; hand back what it reported."""
    import scripts.compare_checkworthy as module

    transcripts, labels, out = tmp_path / "t", tmp_path / "l", tmp_path / "out"
    transcripts.mkdir()
    labels.mkdir()
    (transcripts / "toy.txt").write_text(SPEECH, encoding="utf-8")
    (labels / "checkworthy-toy.json").write_text(
        json.dumps({"labels": HAND_LABELS, "limitation": "one labeller, one speech"}),
        encoding="utf-8",
    )

    stub = filter_stub(threshold, min_words)
    monkeypatch.setattr(module, "TRANSCRIPTS", transcripts)
    monkeypatch.setattr(module, "LABELS", labels)
    monkeypatch.setattr(module, "CheckworthyDetector", RawStub)
    monkeypatch.setattr(module, "DetectorFilter", stub)
    monkeypatch.setattr(sys, "argv",
                        ["compare_checkworthy", "--transcript", "toy", "--out", str(out)])

    assert main() == 0
    return json.loads((out / "metrics.json").read_text(encoding="utf-8")), stub


def test_the_configured_pipeline_is_scored_beside_the_raw_model(tmp_path, monkeypatch, capsys):
    """
    Three rows at a bare 0.5 describe the model; none of them describes the sidebar, which runs the
    calibrator, the frozen threshold and the floor together. Drop this row and the repository's
    headline comparison is against a system that is never run.
    """
    metrics, _ = run_main(tmp_path, monkeypatch)
    configured = [name for name in metrics["systems"] if name.startswith("pipeline (calibrated,")]
    assert len(configured) == 1, "exactly one configured row, reported not just computed"
    assert configured[0] in capsys.readouterr().out, "and printed in the table, not only written"


@pytest.mark.parametrize(("threshold", "min_words"), [(0.37, 6), (0.62, 9)])
def test_the_pipeline_row_names_the_operating_point_it_actually_ran_at(
    tmp_path, monkeypatch, threshold, min_words
):
    """
    A hard-coded ">=0.35, >=4w" keeps reading correctly after `calibrate_checkworthy` refits the
    frozen threshold, so the table would describe an operating point the numbers did not come from
    -- and the provenance line below it would then be a false statement rather than a stale one.
    """
    metrics, _ = run_main(tmp_path, monkeypatch, threshold=threshold, min_words=min_words)
    assert f"pipeline (calibrated, >={threshold}, >={min_words}w)" in metrics["systems"]


def test_the_pipeline_row_is_the_filter_s_verdicts_not_a_second_read_of_the_softmax(
    tmp_path, monkeypatch
):
    """
    `factual >= 0.5` under a pipeline label silently drops all three things the row exists to
    include -- vector scaling, the frozen threshold, the length floor -- and still prints a
    plausible number. Here the two disagree on every sentence, so the substitution cannot hide.
    """
    metrics, stub = run_main(tmp_path, monkeypatch)
    name = next(n for n in metrics["systems"] if n.startswith("pipeline (calibrated,"))
    assert metrics["systems"][name]["recall"] == pytest.approx(0.0), "the filter declined all four"
    assert metrics["systems"]["detector, factual >= 0.5"]["recall"] == pytest.approx(1.0)
    assert stub.seen == [s.text for s in segment(SPEECH)], "and on the same sentences, in order"


def test_the_configured_number_is_printed_and_quotes_its_own_row(tmp_path, monkeypatch, capsys):
    """
    The headline sentence names the 0.5 boundary; without a second one the configured F1 reaches
    the reader only as a table cell beside three rows it is not comparable to. Quoting a detector
    row here would attribute a raw-model score to the pipeline.
    """
    metrics, _ = run_main(tmp_path, monkeypatch)
    name = next(n for n in metrics["systems"] if n.startswith("pipeline (calibrated,"))
    stated = [ln for ln in capsys.readouterr().out.splitlines()
              if ln.startswith("as the pipeline is configured:")]
    assert len(stated) == 1
    delta = metrics["systems"][name]["f1"] - metrics["systems"]["heuristic (Phase 07 rules)"]["f1"]
    assert f"F1 {metrics['systems'][name]['f1']:.4f}" in stated[0]
    assert f"{delta:+.4f}" in stated[0]


def test_the_operating_point_is_printed_with_where_it_came_from(tmp_path, monkeypatch, capsys):
    """
    A threshold and a floor beside an F1, with no account of their origin, read exactly like two
    numbers chosen on these 120 labels -- the selection this script's headline discipline exists to
    prevent. The account has to sit with the number, so it is read as its provenance.
    """
    run_main(tmp_path, monkeypatch)
    lines = capsys.readouterr().out.splitlines()
    at = [i for i, ln in enumerate(lines) if ln.startswith("as the pipeline is configured:")]
    assert at, "the configured number is stated"
    provenance = " ".join(lines[at[0] + 1:at[0] + 3])
    assert "threshold frozen on the calibration debates" in provenance
    assert "floor taken as a definitional prior" in provenance
    assert "Neither was chosen from these labels." in provenance
