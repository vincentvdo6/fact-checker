"""Score a pair judge against hand-labelled (assertion, sentence) relations.

Two numbers matter more than accuracy. The counting error is the share of pairs the
judge would let *count* -- states or states_negation -- whose label says otherwise:
every one of those moves a verdict the wrong way. The recall of counted relations says
how much evidence the judge lets through at all. Both are reported with their
denominators, because the labelled sets here are small and a rate without its n is a
way of not saying so.
"""

from __future__ import annotations

from src.verdict.pair_judgment import RELATIONS

COUNTED = ("states", "states_negation")


def confusion(labels: list[str], predictions: list[str]) -> dict[str, dict[str, int]]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions differ in length")
    table = {actual: {predicted: 0 for predicted in RELATIONS} for actual in RELATIONS}
    for actual, predicted in zip(labels, predictions, strict=True):
        if actual not in RELATIONS or predicted not in RELATIONS:
            raise ValueError(f"unknown relation in ({actual!r}, {predicted!r})")
        table[actual][predicted] += 1
    return table


def report(labels: list[str], predictions: list[str]) -> dict:
    table = confusion(labels, predictions)
    per_relation = {}
    for relation in RELATIONS:
        true_positive = table[relation][relation]
        predicted = sum(table[actual][relation] for actual in RELATIONS)
        actual = sum(table[relation].values())
        precision = true_positive / predicted if predicted else None
        recall = true_positive / actual if actual else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if actual or predicted else None)
        per_relation[relation] = {"precision": precision, "recall": recall, "f1": f1, "support": actual, "predicted": predicted}
    counted_predictions = sum(table[actual][predicted] for actual in RELATIONS for predicted in COUNTED)
    counted_wrong = sum(table[actual][predicted] for actual in RELATIONS for predicted in COUNTED if actual != predicted)
    counted_actual = sum(table[actual][predicted] for actual in COUNTED for predicted in RELATIONS)
    counted_found = sum(table[actual][actual] for actual in COUNTED)
    scored = [row["f1"] for row in per_relation.values() if row["f1"] is not None]
    return {"n": len(labels), "accuracy": sum(table[r][r] for r in RELATIONS) / len(labels) if labels else None,
            "macro_f1": sum(scored) / len(scored) if scored else None, "per_relation": per_relation,
            "counting_error": {"wrong": counted_wrong, "counted": counted_predictions,
                               "rate": counted_wrong / counted_predictions if counted_predictions else None},
            "counted_recall": {"found": counted_found, "actual": counted_actual,
                               "rate": counted_found / counted_actual if counted_actual else None},
            "confusion": table}
