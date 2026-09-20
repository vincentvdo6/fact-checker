"""Let a second judge withhold context without changing the verdict judge's counts.

The news-adapted judge rejected unrelated sentences better but did not earn adoption
as a verdict judge. Its only authority here is to remove a bears_on pair when it reads
that pair as unrelated. It cannot add evidence, promote context or change support and
contradiction decisions. Both readings remain in the saved judgment for inspection.
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

    def __call__(self, assertions: list[dict], units: list[dict]) -> list[dict]:
        rows = self.primary(assertions, units)
        by_unit = {unit["id"]: unit for unit in units}
        reviewed = {}
        for assertion in assertions:
            candidates = [row for row in rows if row["assertion_id"] == assertion["id"] and row["relation"] == "bears_on"]
            if not candidates:
                continue
            readings = self.reviewer([assertion], [by_unit[row["unit_id"]] for row in candidates])
            expected = {(row["assertion_id"], row["unit_id"]) for row in candidates}
            actual = {(row["assertion_id"], row["unit_id"]) for row in readings}
            if len(readings) != len(expected) or actual != expected:
                raise ValueError("Context review must return exactly the requested assertion/sentence pairs.")
            for reading in readings:
                reviewed[(reading["assertion_id"], reading["unit_id"])] = reading
        result = []
        for row in rows:
            reading = reviewed.get((row["assertion_id"], row["unit_id"]))
            if reading is not None:
                row = row | {"context_review": {"model": str(self.reviewer.root), "relation": reading["relation"],
                             "confidence": reading["confidence"], "prior_relation": row["relation"]}}
                if reading["relation"] == "unrelated":
                    row = row | {"relation": "unrelated", "span": "", "qualifiers": []}
            result.append(row)
        return result
