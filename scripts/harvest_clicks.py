"""
Turn recorded extension clicks into label files, so the judge is measured on what viewers checked.

`scripts/youtube_host.py` records a click when FACT_CHECKER_RECORD_CLICKS is set. Each record
holds the caption payload and the full result, including every researched claim's plan. This
reads every claim the detector selected, rebuilds the exact packet the chain read (assertion
excerpts, the passages read around every match, concept-research passages as context), types
the sentences with the same rules the live path uses, and writes the (assertion, eligible
sentence) pairs and every sentence's roles to `labels/pairs-<name>.json` and
`labels/roles-<name>.json` in the shape `scripts/label_pairs.py` serves and
`scripts/eval_pair_judge.py` / `scripts/measure_role_rules.py` score. The judge's own answer is
kept beside each pair as `judged` so labelling can be checked against it afterwards, never
shown while labelling. Nothing is labelled here.

    python -m scripts.harvest_clicks --name clicks-2026-09-13
    python -m scripts.harvest_clicks --clicks runs/clicks --name clicks-2026-09-13 --max-pairs-per-claim 40
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from scripts.harvest_pairs import LABELS, PAIR_RUBRIC, ROLE_RUBRIC, harvest_packet
from src.verdict.chain import research_packet
from src.verdict.pair_judgment import RELATIONS
from src.verdict.reading import ROLES, reading_packet
from src.verdict.role_rules import annotate

CLICKS = Path("runs/clicks")


def claims_from(record: dict, tag: str) -> list[tuple[str, str, dict, dict | None]]:
    """(case name, claim, research plan, decomposed result) for every researched row of a recorded click.

    `tag` is the record's own name: two clicks at one position of one video retrieve different
    pages on different days, so their pairs must not share an id.
    """
    # The host records the result as `result`; `runs/live-click/click.py` keeps the same object as `full`.
    result = record.get("result") or record.get("full") or {}
    video = str(result.get("video_id") or "unknown")
    found = []
    for index, row in enumerate(result.get("rows") or []):
        outcome = row.get("result") or {}
        if "research" not in outcome:       # a skipped or failed row carries no plan
            continue
        found.append((f"{tag}:{video}@{int(result.get('clicked_at') or 0)}#{index}", row["text"], outcome["research"],
                      outcome.get("decomposed")))
    return found


def judged_relations(decomposed: dict | None) -> dict[str, str]:
    """What the judge said per (assertion, unit), keyed the way pair ids are, for later comparison only."""
    if not decomposed:
        return {}
    return {f"{row['assertion_id']}/{row['unit_id']}": row["relation"] for row in decomposed.get("judgments", [])}


def harvest_click(name: str, claim: str, plan: dict, decomposed: dict | None, limit: int) -> tuple[list[dict], list[dict]]:
    packet = research_packet(claim, plan)
    roles = annotate(reading_packet(packet))
    pairs, sentences = harvest_packet(name, packet, roles)
    judged = judged_relations(decomposed)
    for pair in pairs:
        key = "/".join(pair["id"].split("/")[1:])
        pair["judged"] = judged.get(key, "")
    return pairs[:limit], sentences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clicks", type=Path, default=CLICKS, help="directory of click-*.json records")
    parser.add_argument("--name", default=f"clicks-{date.today().isoformat()}")
    parser.add_argument("--max-pairs-per-claim", type=int, default=60)
    args = parser.parse_args()
    records = sorted(args.clicks.glob("click-*.json"))
    if not records:
        raise SystemExit(f"no click records under {args.clicks}; set FACT_CHECKER_RECORD_CLICKS for the host first")
    pairs, sentences, cases = [], [], []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        for case, claim, plan, decomposed in claims_from(record, path.stem):
            found_pairs, found_sentences = harvest_click(case, claim, plan, decomposed, args.max_pairs_per_claim)
            if not found_pairs and not found_sentences:
                continue
            pairs.extend(found_pairs)
            sentences.extend(found_sentences)
            cases.append({"name": case, "claim": claim, "record": path.name, "roles": "closed-class rules",
                          "country": (plan.get("claim_scope") or {}).get("country", "")})
    if not cases:
        raise SystemExit("the recorded clicks contain no researched claims with sources")
    LABELS.mkdir(exist_ok=True)
    header = {"labeller": "", "labelled_on": "", "limitation": "single annotator unless stated; label before looking at `judged`",
              "cases": cases}
    (LABELS / f"pairs-{args.name}.json").write_text(json.dumps(header | {
        "question": "Read literally, what does this one sentence (with its own-paragraph definitions) do to the assertion?",
        "relations": list(RELATIONS), "rubric": PAIR_RUBRIC, "items": pairs}, indent=1, ensure_ascii=False), encoding="utf-8")
    (LABELS / f"roles-{args.name}.json").write_text(json.dumps(header | {
        "question": "What does this sentence do in its source, with no claim in view?",
        "roles": list(ROLES), "rubric": ROLE_RUBRIC, "items": sentences}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"records": len(records), "claims": len(cases), "pairs": len(pairs), "sentences": len(sentences),
                      "files": [f"labels/pairs-{args.name}.json", f"labels/roles-{args.name}.json"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
