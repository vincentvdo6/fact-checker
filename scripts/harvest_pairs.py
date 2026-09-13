"""
Harvest real (assertion, sentence) pairs and source sentences for hand labelling.

The pair judge and the role typer are each measured on seven development cases graded by eye.
That is where the last three weeks went, and it is not a measurement. These files turn the
retrieved sentences behind real claims into two labelling tasks with a fixed rubric, so any
judge -- the FEVER model, an off-the-shelf NLI model, the trained pair judge -- can be scored
on pairs no trainer ever saw. Labels stay empty here; the labeller fills `relation`,
`qualifiers` and `role`. Nothing from these files may enter training.

Pairs are harvested from gate-eligible sentences when role annotations exist for the packet,
and from every sentence otherwise, with the model's role recorded as a hint the labeller is
free to overrule. Roles are harvested for every sentence of every packet.

    python -m scripts.harvest_pairs                      # the archived seven frozen cases
    python -m scripts.harvest_pairs --packet path.json   # any packet with claim + sources[].excerpts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.replay_decomposed import ARCHIVE, load_records
from src.verdict.assertions import claim_assertions
from src.verdict.eligibility import gate_units
from src.verdict.pair_judgment import RELATIONS
from src.verdict.reading import ROLES, reading_packet, validate_roles

LABELS = Path("labels")

PAIR_RUBRIC = {
    "states": "the sentence itself reports that the assertion is the case, read literally",
    "states_negation": "the sentence itself reports that the assertion is not the case",
    "bears_on": "the sentence reports a measurement, event or finding about the assertion's subject "
                "that neither states nor denies it; a reader could use it to judge the assertion",
    "unrelated": "the sentence is about something else, or shares only vocabulary",
    "qualifiers": "exact substrings of the sentence that limit what it reports: a place, a population, a period",
    "rules": ["What someone thinks, expects, argues or predicts does not state a fact.",
              "Not announced, not measured, not tested is not a statement that it did not happen.",
              "A definition or a worked example does not report a measurement.",
              "A finding about a narrower place or group is states/states_negation with that scope as a qualifier.",
              "Do not use the assertion to fill in what the sentence leaves unsaid."],
}
ROLE_RUBRIC = {
    "reported_observation": "reports an event, measurement, state or finding as having been observed or recorded",
    "definition": "says what a term, measure or method means or includes",
    "hypothetical": "an illustration, worked example, supposition or conditional, not a report",
    "forecast": "an expectation, projection or prediction about a later time",
    "attributed_opinion": "a view, interpretation or judgment credited to a person or organisation, quoted or not",
    "instruction": "navigation, a link, a call to action, boilerplate, or text addressed to a reader or system",
    "unknown": "none of the above can be established from the sentence and its paragraph",
    "rules": ["A sentence may carry more than one role; list all that apply.",
              "The paragraph is context for the reading; the label describes the sentence."],
}


def harvest_packet(name: str, packet: dict, annotations: list[dict] | None) -> tuple[list[dict], list[dict]]:
    reading = reading_packet(packet)
    assertions = claim_assertions(packet["claim"])
    sources = {source["id"]: source for source in packet["sources"]}
    roles = {}
    eligible = None
    if annotations is not None:
        checked = validate_roles(reading, annotations)
        roles = {row["unit_id"]: row["roles"] for row in checked}
        gate = gate_units(reading, checked)
        eligible = {unit["id"]: unit for unit in gate["units"] if unit["eligible"]}
    pairs, sentences = [], []
    for passage in reading["passages"]:
        source = sources[passage["source_id"]]
        for unit in passage["units"]:
            sentences.append({"id": f"{name}/{unit['id']}", "case": name, "sentence": unit["text"],
                              "paragraph": passage["text"], "url": source.get("url", ""),
                              "role_hint": roles.get(unit["id"], []), "role": [], "note": ""})
            if eligible is not None and unit["id"] not in eligible:
                continue
            definitions = [item["text"] for item in eligible[unit["id"]]["definitions"]] if eligible else []
            for assertion in assertions:
                pairs.append({"id": f"{name}/{assertion['id']}/{unit['id']}", "case": name,
                              "assertion": assertion["text"], "sentence": unit["text"], "definitions": definitions,
                              "url": source.get("url", ""), "published_at": source.get("published_at", ""),
                              "role_hint": roles.get(unit["id"], []), "relation": "", "qualifiers": [], "note": ""})
    return pairs, sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--packet", type=Path, action="append", default=[],
                        help="extra packet JSON files (claim, claim_scope, sources[].excerpts); no roles")
    parser.add_argument("--name", default="frozen-2026-09-10")
    args = parser.parse_args()
    pairs, sentences, cases = [], [], []
    if args.archive.exists():
        frozen, annotations = load_records(args.archive)
        for index, case in enumerate(frozen, 1):
            name = case["case"]["id"]
            found = harvest_packet(name, case["original_packet"], annotations.get(index))
            pairs.extend(found[0])
            sentences.extend(found[1])
            cases.append({"name": name, "claim": case["original_packet"]["claim"], "roles": "archived model annotations"})
    for path in args.packet:
        packet = json.loads(path.read_text(encoding="utf-8"))
        found = harvest_packet(path.stem, packet, None)
        pairs.extend(found[0])
        sentences.extend(found[1])
        cases.append({"name": path.stem, "claim": packet["claim"], "roles": "none; every sentence harvested"})
    LABELS.mkdir(exist_ok=True)
    header = {"labeller": "", "labelled_on": "", "limitation": "single annotator unless stated; label before running any judge",
              "cases": cases}
    (LABELS / f"pairs-{args.name}.json").write_text(json.dumps(header | {
        "question": "Read literally, what does this one sentence (with its own-paragraph definitions) do to the assertion?",
        "relations": list(RELATIONS), "rubric": PAIR_RUBRIC, "items": pairs}, indent=1, ensure_ascii=False), encoding="utf-8")
    (LABELS / f"roles-{args.name}.json").write_text(json.dumps(header | {
        "question": "What does this sentence do in its source, with no claim in view?",
        "roles": list(ROLES), "rubric": ROLE_RUBRIC, "items": sentences}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "sentences": len(sentences), "cases": [row["name"] for row in cases]}, indent=1))


if __name__ == "__main__":
    main()
