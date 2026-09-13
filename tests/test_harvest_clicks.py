"""Recorded clicks become label files holding exactly the pairs the judge saw, with its answers kept aside."""

from __future__ import annotations

import json

from scripts.harvest_clicks import claims_from, harvest_click


def record():
    plan = {"context": "", "search_cutoff": "2025-07-20", "claim_scope": {"country": "United States", "spoken_at": ""},
            "sources": [{"id": "source-1", "url": "https://www.wpr.org/news/labor", "published_at": "2025-06-25",
                         "temporal_status": "published_by_cutoff", "publication_basis": "meta",
                         "excerpts": ["Wisconsin's labor shortage is a major barrier, a new report finds. In our view that is bad."]}]}
    decomposed = {"judgments": [{"assertion_id": "assertion-1", "unit_id": "source-1:p1:u1", "relation": "bears_on"}]}
    return {"result": {"video_id": "abcdefghijk", "clicked_at": 3173.0, "rows": [
        {"text": "Okay?", "result": {"status": "skipped"}},
        {"text": "We don't have a labor shortage.", "result": {"status": "verified", "research": plan, "decomposed": decomposed}},
        {"text": "No sources here.", "result": {"status": "verified", "research": {**plan, "sources": []}}}]}}


def test_researched_rows_are_harvested_with_rule_roles_and_the_judges_reading_kept_beside_each_pair():
    found = claims_from(record(), "click-1")
    assert [name for name, *_ in found] == ["click-1:abcdefghijk@3173#1", "click-1:abcdefghijk@3173#2"]
    assert claims_from(record(), "click-2")[0][0] != found[0][0], "the same click recorded twice keeps two ids"
    pairs, sentences = harvest_click(*found[0], limit=60)
    assert [item["id"] for item in sentences] == ["click-1:abcdefghijk@3173#1/source-1:p1:u1",
                                                  "click-1:abcdefghijk@3173#1/source-1:p1:u2"]
    assert sentences[1]["role_hint"] == ["reported_observation", "attributed_opinion"], "the live typer's roles travel as hints"
    assert [item["id"] for item in pairs] == ["click-1:abcdefghijk@3173#1/assertion-1/source-1:p1:u1"], "only the eligible sentence is a pair"
    assert pairs[0]["judged"] == "bears_on" and pairs[0]["relation"] == "" and pairs[0]["qualifiers"] == []
    assert harvest_click(*found[1], limit=60) == ([], []), "a claim without sources yields nothing"
    assert claims_from({"full": record()["result"]}, "c")[0][0] == "c:abcdefghijk@3173#1", "click.py diagnostics harvest too"
    assert json.dumps(pairs)  # serialisable as written
