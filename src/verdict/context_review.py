"""Let a second judge withhold context without changing the verdict judge's counts.

The news-adapted judge rejected unrelated sentences better but did not earn adoption
as a verdict judge. Its only authority here is to remove a bears_on pair when it reads
that pair as unrelated. It cannot add evidence, promote context or change support and
contradiction decisions. Both readings remain in the saved judgment for inspection.
Composition can demote additional counts to context; review those afterwards so
withholding one cannot revive the opposing count on a self-conflicting page.
"""

from __future__ import annotations

from pathlib import Path

from src.verdict.pair_judge import PairJudge

MODEL_DIR = Path("models/pair_judge/v5")


class ContextReviewedJudge:
    def __init__(self, primary: PairJudge, reviewer: PairJudge) -> None:
        self.primary, self.reviewer = primary, reviewer
        self.measurement = primary.measurement
        self.direction_measurement = primary.direction_measurement
        self.last_input_overflows: list[dict] = []

    def __call__(self, assertions: list[dict], units: list[dict]) -> list[dict]:
        self.last_input_overflows = []
        rows = self.primary(assertions, units)
        self.last_input_overflows.extend(row | {"stage": "primary"}
                                         for row in getattr(self.primary, "last_input_overflows", []))
        return self._review(assertions, units, rows)

    def review_composed(self, verdict: dict, assertions: list[dict], units: list[dict]) -> dict:
        """Review newly demoted context without recomputing counts or their safeguards."""
        candidates = [row | {"assertion_id": assertion["id"]} for assertion in verdict["assertions"]
                      for row in assertion["relevant"] if row.get("not_counted")]
        if not candidates:
            return verdict
        reviewed = {(row["assertion_id"], row["unit_id"]): row
                    for row in self._review(assertions, units, candidates)}
        results = []
        for assertion in verdict["assertions"]:
            rows = [reviewed.get((assertion["id"], row["unit_id"]), row) for row in assertion["relevant"]]
            results.append(assertion | {"relevant": [row for row in rows if row["relation"] == "bears_on"],
                                        "withheld_context": [row for row in rows if row["relation"] == "unrelated"]})
        return verdict | {"assertions": results}

    def _review(self, assertions: list[dict], units: list[dict], rows: list[dict]) -> list[dict]:
        by_unit = {unit["id"]: unit for unit in units}
        reviewed = {}
        for assertion in assertions:
            candidates = [row for row in rows if row["assertion_id"] == assertion["id"] and row["relation"] == "bears_on"]
            if not candidates:
                continue
            readings = self.reviewer([assertion], [by_unit[row["unit_id"]] for row in candidates])
            expected = {(row["assertion_id"], row["unit_id"]) for row in candidates}
            actual = {(row["assertion_id"], row["unit_id"]) for row in readings}
            overflows = getattr(self.reviewer, "last_input_overflows", [])
            skipped = {(row["assertion_id"], row["unit_id"]): row for row in overflows}
            if (len(readings) != len(actual) or len(skipped) != len(overflows)
                    or actual & skipped.keys() or actual | skipped.keys() != expected):
                raise ValueError("Context review must return exactly the requested assertion/sentence pairs.")
            self.last_input_overflows.extend(row | {"stage": "context_review"} for row in overflows)
            for reading in readings:
                reviewed[(reading["assertion_id"], reading["unit_id"])] = reading
            for key, overflow in skipped.items():
                reviewed[key] = {"overflow": overflow}
        result = []
        for row in rows:
            reading = reviewed.get((row["assertion_id"], row["unit_id"]))
            if reading is not None:
                if "overflow" in reading:
                    # No second reading is not a finding of irrelevance; retain the primary context.
                    result.append(row | {"context_review": reading["overflow"] | {
                        "model": str(self.reviewer.root), "status": "not_judged",
                        "prior_relation": row["relation"]}})
                    continue
                row = row | {"context_review": {"model": str(self.reviewer.root), "relation": reading["relation"],
                             "confidence": reading["confidence"], "prior_relation": row["relation"]}}
                if reading["relation"] == "unrelated":
                    row = row | {"relation": "unrelated", "span": "", "qualifiers": []}
            result.append(row)
        return result
