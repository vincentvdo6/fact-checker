"""Order existing contextual quotations without giving a ranker verdict authority.

The scorer is injected so artifact validation and inference stay separate. The input matches the assertion-only ranking
experiment; neither preceding speech nor generated claim interpretations enter it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from src.retrieval.visible_text import visible_text
from src.verdict.chain import Judge

Scores = Callable[[list[dict]], Sequence[float]]


class ContextOrderingUnavailable(ValueError):
    """The ranker cannot run without missing artifacts or truncating an input."""


class ContextOrderedJudge:
    def __init__(self, primary: Judge, scorer: Scores, *, model: str) -> None:
        self.primary, self.scorer, self.model = primary, scorer, model
        self.measurement = getattr(primary, "measurement", None)
        self.direction_measurement = getattr(primary, "direction_measurement", None)

    def __call__(self, assertions: list[dict], units: list[dict]) -> list[dict]:
        return self.primary(assertions, units)

    @property
    def last_input_overflows(self) -> list[dict]:
        return getattr(self.primary, "last_input_overflows", [])

    def review_composed(self, verdict: dict, assertions: list[dict], units: list[dict]) -> dict:
        review = getattr(self.primary, "review_composed", None)
        return review(verdict, assertions, units) if review else verdict

    def order_context(self, verdict: dict, assertions: list[dict], units: list[dict]) -> dict:
        by_assertion = {assertion["id"]: assertion for assertion in assertions}
        by_unit = {unit["id"]: unit for unit in units}
        inputs, keys = [], []
        for assertion in verdict["assertions"]:
            original = by_assertion[assertion["id"]]
            if assertion["text"] != original["text"]:
                raise ValueError("Context ordering cannot rewrite the assertion.")
            for row in assertion["relevant"]:
                unit = by_unit[row["unit_id"]]
                if row["relation"] != "bears_on" or row["text"] != unit["text"] or row["definitions"] != unit["definitions"]:
                    raise ValueError("Context ordering requires an unchanged contextual sentence.")
                keys.append((assertion["id"], row["unit_id"]))
                inputs.append({"hypothesis": original["text"], "visible_sentence": visible_text(unit["text"])[0],
                               "visible_definitions": [visible_text(item["text"])[0] for item in unit["definitions"]]})
        if not inputs:
            return verdict
        if len(set(keys)) != len(keys):
            raise ValueError("Context ordering requires distinct assertion/sentence pairs.")
        try:
            scores = [float(value) for value in self.scorer(inputs)]
        except ContextOrderingUnavailable as error:
            return verdict | {"context_ordering": {"model": self.model, "status": "not_applied", "note": str(error)}}
        if len(scores) != len(keys) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Context ordering requires one finite relevance score per pair in [0, 1].")
        by_pair = dict(zip(keys, scores, strict=True))
        results = []
        for assertion in verdict["assertions"]:
            rows = [row | {"context_relevance_score": by_pair[assertion["id"], row["unit_id"]]}
                    for row in assertion["relevant"]]
            results.append(assertion | {"relevant": sorted(rows, key=lambda row: (
                row["origin"] != "assertion", -row["context_relevance_score"]))})
        audit = {"model": self.model, "status": "applied",
                 "purpose": "context display order only; scores are not truth confidence"}
        if providers := getattr(self.scorer, "providers", None):
            audit["providers"] = list(providers)
        return verdict | {"assertions": results, "context_ordering": audit}
