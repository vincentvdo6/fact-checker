"""
The assembled pipeline, and the three ways it can have no verdict.

Nothing here re-measures the model. What it guards is the wiring, because every mistake available
in this module is silent: a gate reading one condition instead of two, a band assigned from raw
logits instead of calibrated probability, or a declined claim quietly carrying its suppressed
prediction into the page as a verdict. All three produce output that looks entirely reasonable
and promises something nobody measured.

The frozen artifacts are the other half. A threshold tuned so the demo shows more verdicts would
void every promise Phases 03 and 04 measured, so the fact that this module only ever *loads*
them is asserted rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.calibration.bands import Band
from src.pipeline.verify import Judgement, Outcome, Verifier
from src.retrieval.features import RETRIEVAL_NAMES
from src.verdict.labels import Verdict

VERIFY = Path("src/pipeline/verify.py")
CHECK = Path("scripts/check_pipeline.py")
MODELS = Path("models/verdict")

requires_artifacts = pytest.mark.skipif(
    not (MODELS / "retrieved" / "sufficiency.json").exists(),
    reason="frozen artifacts absent; install them with scripts/install_artifacts.py",
)


def grounded_features(value: float) -> dict[str, float]:
    """Retrieval features that are constant except in scale -- shape is not what is under test."""
    return dict.fromkeys(RETRIEVAL_NAMES, value)


# --- nothing in the pipeline may fit anything --------------------------------------------------

def test_the_pipeline_only_loads_its_artifacts():
    """
    Phases 03 and 04 measured what the bands and the gate promise. A `fit` anywhere in the
    pipeline would mean the demo is calibrating on whatever it happens to be shown, which is the
    circularity every previous phase was built to avoid.
    """
    source = VERIFY.read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "fit_bands" not in source
    for loader in ("from_dict", "BandPolicy.from_dict", "SufficiencyModel.from_dict"):
        assert loader in source


def test_the_threshold_comes_from_the_artifact_not_a_literal():
    source = VERIFY.read_text(encoding="utf-8")
    assert 'float(gate["threshold"])' in source
    assert "threshold = 0.5" not in source


def test_the_band_is_assigned_from_calibrated_probability():
    """
    Assigning from raw softmax would keep every band populated and every promise false: the
    thresholds were fitted on the calibrator's output, and Phase 03 measured ECE 0.1287 raw
    against 0.0314 calibrated.
    """
    source = VERIFY.read_text(encoding="utf-8")
    assert "self.calibrator.transform(" in source
    assert "probability.max()" in source


# --- the gate is two conditions, not one and not a product ------------------------------------

@requires_artifacts
@pytest.mark.parametrize(
    ("confident", "adequate", "outcome"),
    [
        (True, True, Outcome.ANSWERED),
        (True, False, Outcome.DECLINED_INSUFFICIENT_EVIDENCE),
        (False, True, Outcome.DECLINED_LOW_CONFIDENCE),
        (False, False, Outcome.DECLINED_BOTH),
    ],
)
def test_every_corner_of_the_gate_has_its_own_outcome(confident, adequate, outcome):
    """
    Phase 04 found groundedness and correctness near-orthogonal on FEVER (r = +0.03) and measured
    that multiplying them degrades the ranking, E-AURC 0.1136 -> 0.1492. So the gate is two
    independent conditions, and each corner has to be reachable and distinguishable.
    """
    verifier = Verifier()
    probability = np.array([0.99, 0.005, 0.005] if confident else [0.4, 0.35, 0.25])
    judgement = verifier._decide(
        "a claim", (), probability, verifier.threshold + (0.1 if adequate else -0.1)
    )
    assert judgement.outcome is outcome
    assert judgement.answered == (outcome is Outcome.ANSWERED)


@requires_artifacts
def test_a_declined_claim_carries_no_verdict_but_keeps_its_prediction():
    """
    The verdict is withheld; the prediction is not hidden. A suppressed prediction that cannot be
    inspected makes the abstention unfalsifiable -- there would be no way to ask whether the
    system declined the claims it was wrong about.
    """
    verifier = Verifier()
    judgement = verifier._decide("a claim", (), np.array([0.4, 0.35, 0.25]), 0.0)
    assert judgement.verdict is None
    assert judgement.predicted == Verdict.SUPPORTED
    assert judgement.confidence == pytest.approx(0.4)


@requires_artifacts
def test_a_confident_not_enough_evidence_is_an_answer_not_an_abstention():
    """
    Per src/verdict/labels.py these are different claims: NOT ENOUGH EVIDENCE says something
    about the world, abstention says something about the model. Collapsing them would make the
    two indistinguishable in the one place a reader can see them.
    """
    verifier = Verifier()
    judgement = verifier._decide(
        "a claim", (), np.array([0.005, 0.005, 0.99]), verifier.threshold + 0.1
    )
    assert judgement.outcome is Outcome.ANSWERED
    assert judgement.verdict == Verdict.NOT_ENOUGH_EVIDENCE
    assert judgement.band is Band.STRONG


@requires_artifacts
def test_an_empty_batch_judges_nothing_rather_than_erroring():
    """A transcript can filter every sentence out; that is an empty result, not a failure."""
    assert Verifier().judge_batch([]) == []


@requires_artifacts
def test_judge_scored_returns_one_judgement_per_claim_in_order():
    verifier = Verifier()
    got = verifier.judge_scored(
        ["first", "second", "third"],
        [[5.0, -1.0, -1.0], [0.1, 0.0, -0.1], [-1.0, 5.0, -1.0]],
        [grounded_features(1.0), grounded_features(0.0), grounded_features(1.0)],
    )
    assert [j.claim for j in got] == ["first", "second", "third"]
    assert all(isinstance(j, Judgement) for j in got)


# --- Checkpoint 2 must be able to fail --------------------------------------------------------

def test_checkpoint_two_compares_against_the_published_report():
    """
    The report is the published artifact; the pipeline is the thing under test. Reading the
    expectations from anywhere else -- or recomputing them -- would make the check a tautology.
    """
    source = CHECK.read_text(encoding="utf-8")
    assert 'f"sufficiency-{args.split}" / "metrics.json"' in source
    assert 'report["policies"]["plus sufficiency"]' in source
    assert "return 1" in source


def test_checkpoint_two_checks_the_bands_and_not_only_the_totals():
    """
    Coverage and accuracy can both land exactly right while the band thresholds are mis-wired --
    the totals are sums over the bands, and errors inside them cancel. The bands are what the
    page actually shows a reader, so they are checked one at a time.
    """
    source = CHECK.read_text(encoding="utf-8")
    assert 'for band, promise in report["bands"].items():' in source
    assert 'promise["measured"]' in source


def test_checkpoint_two_tolerance_admits_no_real_disagreement():
    """
    Two arithmetic paths over identical inputs agree to floating-point noise or they disagree for
    a reason. A tolerance loose enough to absorb a mis-wiring is not a checkpoint.
    """
    from scripts.check_pipeline import TOLERANCE

    assert TOLERANCE <= 1e-9
