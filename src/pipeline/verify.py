"""
One claim, end to end: retrieve, score, calibrate, and decline when the evidence does not support
a verdict.

Everything here is assembly. The calibrator, the band thresholds and the sufficiency gate are
frozen artifacts fitted in Phases 03 and 04 on splits this module never reads, and nothing in the
pipeline may re-fit or re-tune any of them. A threshold nudged so the demo shows more verdicts
would void the only promises the project has actually measured.

**One code path, because there must not be two.** The CLI, the rendered page and anything later
all call `judge`. The offline evaluation in `scripts/eval_sufficiency.py` computed coverage
0.7580 and accuracy 0.7553 on the test split; `scripts/check_pipeline.py` runs this module over
the same rows and requires the same numbers. If the assembled pipeline disagrees with the offline
report, the pipeline is wrong -- that check is the only thing that catches a demo which looks
plausible while quietly mis-wiring the calibrator or the gate.

**Three ways to have no verdict, and they are not the same thing.** Per `src/verdict/labels.py`,
NOT ENOUGH EVIDENCE is a claim about the world -- we looked and found nothing that settles it --
and a confident one is an *answer*, carried through as a verdict. Declining is a claim about the
model: either its confidence earned no band, or the retrieval features say it was not given the
evidence it needed, or both. `Outcome` keeps the three apart end to end so the page can too.

**Retrieval is separated from judgement on purpose.** `judge` takes evidence it is handed, so the
checkpoint can feed it the stored retrieval and isolate the wiring from the corpus -- which needs
3.9 GB of SQLite and a 1.5 GB index, and whose determinism was already established in Phase 01.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from src.calibration.bands import Band, BandPolicy
from src.calibration.scaling import from_dict as calibrator_from_dict
from src.calibration.sufficiency import SufficiencyModel, rows_from
from src.retrieval import wiki
from src.retrieval.features import RETRIEVAL_NAMES, retrieval_features
from src.retrieval.search import BM25Index, load_index, retrieve
from src.verdict.encode import pack
from src.verdict.labels import FEVER, Verdict
from src.verdict.runtime import VerdictRuntime

MODELS = Path("models/verdict")
PAGES = 25          # Phase 01's operating point: recall@25 0.811 on test, k=25, injector cap 3
SENTENCES = 25


class Outcome(StrEnum):
    """Why this claim does or does not carry a verdict."""

    ANSWERED = "answered"
    DECLINED_LOW_CONFIDENCE = "declined_low_confidence"
    DECLINED_INSUFFICIENT_EVIDENCE = "declined_insufficient_evidence"
    DECLINED_BOTH = "declined_both"

    @property
    def answered(self) -> bool:
        return self is Outcome.ANSWERED


@dataclass(frozen=True, slots=True)
class Judgement:
    """
    What the system concluded, and everything a reader would need to audit it.

    `verdict` is None whenever the claim was declined. `predicted` always carries what the model
    said, because a suppressed prediction is the interesting thing to inspect and hiding it would
    make the abstention unfalsifiable.
    """

    claim: str
    outcome: Outcome
    verdict: Verdict | None
    predicted: Verdict
    band: Band | None
    confidence: float
    sufficiency: float
    probabilities: tuple[float, ...]
    evidence: tuple[tuple[str, int, str], ...] = field(default=())

    @property
    def answered(self) -> bool:
        return self.outcome.answered


class Verifier:
    """
    The frozen artifacts, loaded once, wired the way the offline evaluation wired them.

    The corpus connection is opened lazily: `judge` needs none, and the checkpoint runs without
    the 3.9 GB database present.
    """

    def __init__(self, variant: str = "retrieved", *, models: str | Path = MODELS) -> None:
        root = Path(models) / variant
        self.variant = variant
        self.runtime = VerdictRuntime(variant, models=models)
        self.labels = FEVER
        self.calibrator = calibrator_from_dict(
            json.loads((root / "calibration.json").read_text(encoding="utf-8"))["calibrator"]
        )
        gate = json.loads((root / "sufficiency.json").read_text(encoding="utf-8"))
        self.sufficiency = SufficiencyModel.from_dict(gate["model"])
        self.bands = BandPolicy.from_dict(gate["bands"])
        self.threshold = float(gate["threshold"])
        self._conn: sqlite3.Connection | None = None
        self._index: BM25Index | None = None

    # --- judgement, corpus not required ---------------------------------------------------

    def judge(self, claim: str, evidence: Sequence[tuple[str, int, str]],
              scores: Sequence[float]) -> Judgement:
        """One claim against evidence already retrieved for it."""
        return self.judge_batch([(claim, evidence, scores)])[0]

    def judge_batch(
        self, items: Sequence[tuple[str, Sequence[tuple[str, int, str]], Sequence[float]]]
    ) -> list[Judgement]:
        if not items:
            return []

        # The packing budget is the encoder's, measured with the model's own tokenizer, so the
        # sufficiency features describe the sentences the model actually read rather than
        # everything retrieval returned.
        packed = [
            list(evidence)[: pack(claim, list(evidence), self.runtime.contract.max_length,
                                 self.runtime.measure)]
            for claim, evidence, _ in items
        ]
        scored = self.runtime.score_batch(
            [(claim, chosen) for (claim, _, _), chosen in zip(items, packed, strict=True)]
        )
        features = [
            retrieval_features(
                [(title, index) for title, index, _ in evidence], list(scores), budget=len(chosen)
            )
            for (_, evidence, scores), chosen in zip(items, packed, strict=True)
        ]
        return self.judge_scored(
            [claim for claim, _, _ in items],
            [s.logits for s in scored],
            features,
            evidence=[tuple(map(tuple, chosen)) for chosen in packed],
        )

    def judge_scored(
        self,
        claims: Sequence[str],
        logits: Sequence[Sequence[float]],
        features: Sequence[dict[str, float]],
        *,
        evidence: Sequence[Sequence[tuple[str, int, str]]] | None = None,
    ) -> list[Judgement]:
        """
        Calibrate, gate and decide, from logits and retrieval features already computed.

        This is the seam Checkpoint 2 drives. Checkpoint 1 proves the ONNX graph reproduces the
        logits the calibration was fitted on; this half proves that everything built on top of
        those logits -- calibrator, sufficiency model, band thresholds, the two-condition gate --
        reproduces the offline report. Composed, the two cover the whole path, and neither has to
        spend an hour re-scoring 2,000 claims to do it.
        """
        probabilities = self.calibrator.transform(np.asarray(logits, dtype=np.float64))
        sufficiency = self.sufficiency.predict(rows_from(list(features), RETRIEVAL_NAMES))
        chosen = evidence if evidence is not None else [()] * len(claims)
        return [
            self._decide(claim, tuple(map(tuple, rows)), probability, float(adequate))
            for claim, rows, probability, adequate in zip(
                claims, chosen, probabilities, sufficiency, strict=True
            )
        ]

    def _decide(self, claim, evidence, probability, adequate: float) -> Judgement:
        """
        The gate, exactly as Phase 04 fitted it: a band assigned **and** sufficiency at or above
        the model's own 0.5 boundary. Two independent conditions rather than a product -- Phase 04
        measured that multiplying them degrades the ranking, E-AURC 0.1136 to 0.1492, because
        groundedness and correctness are near-orthogonal on FEVER (r = +0.03).
        """
        confidence = float(probability.max())
        predicted = self.labels.verdicts[int(probability.argmax())]
        band = self.bands.assign(confidence)
        grounded = adequate >= self.threshold

        if band is not None and grounded:
            outcome = Outcome.ANSWERED
        elif band is None and not grounded:
            outcome = Outcome.DECLINED_BOTH
        elif band is None:
            outcome = Outcome.DECLINED_LOW_CONFIDENCE
        else:
            outcome = Outcome.DECLINED_INSUFFICIENT_EVIDENCE

        return Judgement(
            claim=claim,
            outcome=outcome,
            verdict=predicted if outcome.answered else None,
            predicted=predicted,
            band=band,
            confidence=confidence,
            sufficiency=adequate,
            probabilities=tuple(float(p) for p in probability),
            evidence=evidence,
        )

    # --- retrieval, corpus required -------------------------------------------------------

    @property
    def corpus(self) -> tuple[sqlite3.Connection, BM25Index]:
        if self._conn is None:
            self._conn = wiki.connect()
            self._index = load_index(self._conn)
        return self._conn, self._index

    def retrieve(self, claim: str) -> tuple[list[tuple[str, int, str]], list[float]]:
        """Phase 01's stack at its measured operating point, with the sentence text attached."""
        conn, index = self.corpus
        result = retrieve(conn, index, claim, n=PAGES, k=SENTENCES)

        texts: dict[str, dict[int, str]] = {}
        evidence: list[tuple[str, int, str]] = []
        scores: list[float] = []
        for (title, sentence), score in zip(result.refs, result.scores, strict=True):
            if title not in texts:
                found = wiki.doc_id(conn, title)
                texts[title] = dict(wiki.sentences(conn, found)) if found is not None else {}
            text = texts[title].get(sentence)
            if text is None:
                continue
            evidence.append((title, sentence, text))
            scores.append(float(score))
        return evidence, scores

    def verify(self, claim: str) -> Judgement:
        """Retrieve and judge, the whole path a transcript claim takes."""
        evidence, scores = self.retrieve(claim)
        return self.judge(claim, evidence, scores)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._index = None
