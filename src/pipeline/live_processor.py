"""Reuse the measured detector and verifier while marking live retrieval as unvalidated.

Offline resources load only when used, on the worker thread that owns inference.
Operational failures propagate to the session's failed event and never become abstentions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from src.pipeline.caption_claims import corrective_contrast
from src.pipeline.caption_concepts import concept_packet, select_context_phrases
from src.pipeline.context import SpeechMetadata, build_query
from src.pipeline.detector import DetectorFilter
from src.pipeline.topic_evidence import topic_evidence
from src.pipeline.transcript import ClaimContext
from src.pipeline.verify import Verifier
from src.pipeline.web_research import WebResearch
from src.verdict.chain import check_claim


class LiveProcessor:
    def __init__(self, metadata: SpeechMetadata, *, query_mode: str = "claim", web: WebResearch | None = None,
                 judge: Callable | None = None, specificity: Callable[[str], float | None] | None = None,
                 notes: list[str] | None = None) -> None:
        if query_mode not in ("claim", "context", "topic"):
            raise ValueError("query mode must be claim, context or topic")
        self.metadata = metadata
        self.query_mode = query_mode
        self.detector = DetectorFilter()
        self.verifier: Verifier | None = None
        self.web = web
        # The decomposed chain's pair judge. Optional and off by default: its output is a rule-composed
        # draft over an unmeasured judge, attached beside the research and never a verdict.
        self.judge = judge
        # Corpus specificity for choosing caption phrases to research; a rule, not a model. Optional.
        self.specificity = specificity
        # What this processor could not load, repeated on every result so the panel can say so.
        self.notes = list(notes or [])

    def begin_check(self, metadata: SpeechMetadata | None = None) -> None:
        """Share one web time budget across the claims from a single click."""
        if metadata is not None:
            self.metadata = metadata
        if self.web is not None:
            self.web.begin_check()

    def process(self, context: ClaimContext) -> dict:
        query = build_query(context, self.metadata, mode=self.query_mode)
        decision = self.detector.decide(query.claim)
        result = {
            "claim": query.claim, "start": context.claim.start, "end": context.claim.end,
            "speaker": context.claim.speaker, "source": asdict(self.metadata),
            "query": query.query, "query_mode": query.mode, "context_ids": list(query.context_ids),
            "references": [asdict(reference) for reference in query.references],
            "filter_score": decision.score, "filter_reason": decision.reason,
            "calibration_scope": "FEVER only; live verdict accuracy is unmeasured",
            "corpus": "FEVER Wikipedia, June 2017",
        }
        if not decision.worthy:
            return result | {"status": "skipped"}
        if query.mode == "topic":
            result["retrieval_context"] = list(query.context_text)
            web = getattr(self, "web", None)
            always_search = web is not None and getattr(web, "always_search", False)
            evidence, scores = [], []
            if not always_search:
                evidence, scores = self._get_verifier().retrieve(query.query)
                evidence, scores = topic_evidence(
                    query.claim, query.query, evidence, scores,
                    context=query.context_text,
                )
            if web is not None and (always_search or not evidence or corrective_contrast(query.claim)):
                specificity = getattr(self, "specificity", None)
                if specificity is not None:
                    selection = select_context_phrases(concept_packet(context), specificity)
                    review = web.review(context, self.metadata, concept_selection=selection)
                else:
                    review = web.review(context, self.metadata)
                outcome = result | {
                    "status": "verified", "outcome": "declined_web_review", "verdict": None,
                    "corpus": "Live web research; applicability unverified", "evidence": [], "research": review,
                }
                judge = getattr(self, "judge", None)     # tests build instances without __init__, as for web
                if judge is not None:
                    outcome["decomposed"] = check_claim(query.claim, review, judge)
                if getattr(self, "notes", None):
                    outcome["notes"] = list(self.notes)
                return outcome
            if not evidence:
                return result | {
                    "status": "verified", "outcome": "declined_no_relevant_evidence",
                    "verdict": None, "evidence": [],
                }
            if corrective_contrast(query.claim):
                return result | {
                    "status": "verified", "outcome": "declined_compound_claim",
                    "verdict": None, "evidence": [list(row) for row in evidence],
                }
            judgement = self._get_verifier().judge(query.claim, evidence, scores)
        else:
            judgement = self._get_verifier().verify(query.claim, query=query.query)
        return result | {
            "status": "verified", "outcome": str(judgement.outcome),
            "verdict": str(judgement.verdict) if judgement.verdict is not None else None,
            "predicted": str(judgement.predicted),
            "band": str(judgement.band) if judgement.band is not None else None,
            "confidence": judgement.confidence, "sufficiency": judgement.sufficiency,
            "probabilities": list(judgement.probabilities),
            "evidence": [list(row) for row in judgement.evidence],
        }

    def _get_verifier(self) -> Verifier:
        if self.verifier is None:
            self.verifier = Verifier()
        return self.verifier

    def close(self) -> None:
        if self.verifier is not None:
            self.verifier.close()
