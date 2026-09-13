"""
Measure the rule-based role typer against `labels/roles-*.json`.

Scored against the hand labels (`role`) where they are filled, otherwise against the archived
model annotations (`role_hint`) as a comparison -- the model's roles are not a truth, and the
report says which reference it used. The number that matters is agreement on the gate's decision:
would this sentence be eligible as evidence under the rules, and under the reference? Every
disagreement is printed with the cue that fired, because the point of rules is that a wrong one
can be read and fixed rather than retrained.

    python -m scripts.measure_role_rules --labels labels/roles-frozen-2026-09-10.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from src.verdict.eligibility import CONTEXT_ROLES
from src.verdict.role_rules import ROLES, type_sentence

OUT = Path("runs/role-rules")


def eligible(roles: list[str]) -> bool:
    return "reported_observation" in roles and not any(role in CONTEXT_ROLES for role in roles) and "unknown" not in roles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--show", type=int, default=40, help="disagreements to print")
    args = parser.parse_args()
    payload = json.loads(args.labels.read_text(encoding="utf-8"))
    labelled = [item for item in payload["items"] if item.get("role")]
    reference = "role" if labelled else "role_hint"
    items = labelled or [item for item in payload["items"] if item.get("role_hint")]
    if not items:
        raise SystemExit("no roles or role hints to compare against")
    rows, gate = [], {"agree_eligible": 0, "agree_withheld": 0, "rules_eligible_only": 0, "reference_eligible_only": 0}
    per_role = {role: {"tp": 0, "fp": 0, "fn": 0} for role in ROLES}
    previous_by_paragraph: dict[str, dict | None] = {}
    for item in items:
        # Items are in source order; a new paragraph resets the run.
        key = f"{item['case']}|{item['paragraph']}"
        typed = type_sentence(item["sentence"], previous=previous_by_paragraph.get(key))
        previous_by_paragraph[key] = typed
        expected = list(item[reference])
        for role in ROLES:
            got, want = role in typed["roles"], role in expected
            per_role[role]["tp" if got and want else "fp" if got else "fn" if want else "tp"] += int(got or want)
        rules_ok, ref_ok = eligible(typed["roles"]), eligible(expected)
        key = ("agree_eligible" if rules_ok else "agree_withheld") if rules_ok == ref_ok else (
            "rules_eligible_only" if rules_ok else "reference_eligible_only")
        gate[key] += 1
        rows.append({"id": item["id"], "sentence": item["sentence"], "rules": typed["roles"], "cues": typed["cues"],
                     reference: expected, "gate_agrees": rules_ok == ref_ok})
    total = len(items)
    exact = sum(1 for row in rows if sorted(row["rules"]) == sorted(row[reference]))
    summary = {"labels": str(args.labels), "reference": reference, "n": total, "exact_set_agreement": exact / total,
               "gate_agreement": (gate["agree_eligible"] + gate["agree_withheld"]) / total, "gate": gate,
               "per_role": {role: {"precision": counts["tp"] / max(counts["tp"] + counts["fp"], 1),
                                   "recall": counts["tp"] / max(counts["tp"] + counts["fn"], 1),
                                   "support": counts["tp"] + counts["fn"]} for role, counts in per_role.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.labels.stem}-{reference}-{date.today().isoformat()}.json"
    path.write_text(json.dumps(summary | {"rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_role"}, indent=1))
    for role, counts in summary["per_role"].items():
        print(f"  {role:<21} precision {counts['precision']:.2f} recall {counts['recall']:.2f} support {counts['support']}")
    print(f"\ndisagreements on the gate decision (reference = {reference}):")
    shown = 0
    for row in rows:
        if row["gate_agrees"] or shown >= args.show:
            continue
        shown += 1
        print(f"  rules={row['rules']} cues={row['cues']} | {reference}={row[reference]}\n     {row['sentence'][:140]}")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
