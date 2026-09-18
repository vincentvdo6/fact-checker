"""The live chain reads assertion research as evidence and caption-concept research as context only."""

from __future__ import annotations

from src.verdict.chain import check_claim, research_packet

CLAIM = "We don't have a labor shortage."
PLAN = {"context": "captions", "search_cutoff": "2025-06-01", "claim_scope": {"country": "US", "spoken_at": "2025-06-01"},
        "sources": [
            {"id": "source-1", "url": "https://www.amnesty.org/en/latest/what-is-a-living-wage/", "published_at": "2025-01-22",
             "temporal_status": "published_by_cutoff", "publication_basis": "meta", "excerpts": [],
             "reading_passages": ["Everyone has economic, social and cultural rights."]},
            {"id": "source-2", "url": "https://www.wpr.org/news/labor-shortage", "published_at": "2024-12-13",
             "temporal_status": "published_by_cutoff", "publication_basis": "meta",
             "excerpts": ["Wisconsin's labor shortage grew 4% last year."],
             "reading_passages": ["Wisconsin's labor shortage grew 4% last year.", "The state counted 40,000 openings."]},
            {"id": "source-3", "url": "https://late.example/x", "temporal_status": "later_publication",
             "excerpts": ["Anything."]},
            {"id": "source-4", "url": "https://www.jsonline.com/story/labor", "published_at": "2025-03-01",
             "temporal_status": "published_by_cutoff", "publication_basis": "meta",
             "excerpts": ["Employers reported 40,000 unfilled jobs."],
             "linked_from": [{"parent_source_id": "source-2", "parent_url": "https://www.wpr.org/news/labor-shortage"}]},
            {"id": "source-5", "url": "https://www.bls.gov/news/wisconsin", "published_at": "2025-04-01",
             "temporal_status": "published_by_cutoff", "publication_basis": "meta",
             "excerpts": ["Openings rose 3% over the year."]}]}


def states_everything(assertions, units):
    return [{"assertion_id": assertion["id"], "unit_id": unit["id"], "relation": "states", "span": unit["text"],
             "qualifiers": [], "confidence": 0.9} for assertion in assertions for unit in units]


def test_the_live_judge_receives_a_complete_sentence_with_its_citation_intact():
    sentence = "The [vaccination](https://example.org/coverage(2025).) rate reached 30%."
    plan = {**PLAN, "sources": [{**PLAN["sources"][1], "excerpts": [sentence], "reading_passages": []}]}
    seen = []

    def judge(assertions, units):
        seen.extend(units)
        return [{"assertion_id": a["id"], "unit_id": u["id"], "relation": "bears_on",
                 "span": u["text"], "qualifiers": []} for a in assertions for u in units]

    result = check_claim("Vaccination coverage reached 30%.", plan, judge)
    assert [u["text"] for u in seen] == [sentence]
    assert result["judgments"][0]["span"] == sentence
    assert result["verdict"]["assertions"][0]["relevant"][0]["text"] == sentence
    assert result["verdict"]["assertions"][0]["evidence"] == []


def test_research_packet_keeps_concept_passages_apart_and_drops_later_sources():
    packet = research_packet(CLAIM, PLAN)
    assert [source["id"] for source in packet["sources"]] == ["source-1", "source-2", "source-4", "source-5"]
    assert packet["sources"][2]["linked_from"][0]["parent_source_id"] == "source-2", "lineage travels with the source"
    assert packet["sources"][0]["excerpts"] == [] and packet["sources"][0]["context_excerpts"] == [
        "Everyone has economic, social and cultural rights."]
    assert packet["sources"][1]["excerpts"] == ["Wisconsin's labor shortage grew 4% last year."]
    assert packet["sources"][1]["context_excerpts"] == ["The state counted 40,000 openings."], "no passage is read twice"
    assert packet["claim_scope"] == {"country": "US", "spoken_at": "2025-06-01"}


def test_a_judge_that_counts_every_sentence_moves_the_verdict_only_through_assertion_research():
    result = check_claim(CLAIM, PLAN, states_everything)
    assertion = result["verdict"]["assertions"][0]
    assert [row["unit_id"] for row in assertion["evidence"]] == ["source-2:p1:u1", "source-4:p1:u1", "source-5:p1:u1"]
    assert sorted(row["unit_id"] for row in assertion["relevant"]) == ["source-1:p1:u1", "source-2:p2:u1"]
    assert all(row["origin"] == "context" and row["not_counted"].startswith("states:") for row in assertion["relevant"])
    assert assertion["status"] == "supported" and assertion["sources_for"] == 2 and result["status"].startswith("draft")
    assert "Found by caption-concept research, which does not resolve the claim." in result["text"]
    without = {**PLAN, "sources": [dict(PLAN["sources"][0]), dict(PLAN["sources"][2])]}
    assert check_claim(CLAIM, without, states_everything)["verdict"]["relationship"] == "insufficient"
    # WPR and the page it links to are one source, named as one; only BLS makes a second.
    lineage_only = {**PLAN, "sources": [dict(row) for row in PLAN["sources"][:4]]}
    one = check_claim(CLAIM, lineage_only, states_everything)["verdict"]["assertions"][0]
    assert one["status"] == "supported" and one["sources_for"] == 1 and one["basis"].startswith("one source: ")
    assert assertion["basis"].startswith("2 independent sources: ")


def test_a_regional_sentence_counted_for_a_national_claim_is_qualified_by_its_place():
    declared = {"country": "United States", "spoken_at": "", "country_basis": "supplied by the viewer"}
    regional = {**PLAN, "claim_scope": declared, "sources": [dict(row) for row in PLAN["sources"][:2]]}   # amnesty context, WPR
    result = check_claim(CLAIM, regional, states_everything)
    assertion = result["verdict"]["assertions"][0]
    assert [row["qualifiers"] for row in assertion["evidence"]] == [["Wisconsin"]], "the judge gave none; the gazetteer did"
    assert assertion["status"] == "qualified_support" and assertion["limits"] == ["every counted sentence states a narrower scope"]
    assert "Scope stated in the sentence: Wisconsin." in result["text"]
    national = check_claim(CLAIM, {**PLAN, "claim_scope": declared}, states_everything)["verdict"]["assertions"][0]
    assert national["status"] == "supported", "two national dated sources still establish"
    assert [row["qualifiers"] for row in national["evidence"]] == [["Wisconsin"], [], []]
    unset = check_claim(CLAIM, {**PLAN, "claim_scope": {"country": "", "spoken_at": ""}, "sources": regional["sources"]}, states_everything)
    assert all(row["qualifiers"] == [] for row in unset["verdict"]["assertions"][0]["evidence"]), "no declaration, no narrowing"


def test_a_quantified_group_in_a_counted_sentence_qualifies_it_without_a_country():
    listing = "Several states in America are facing a worker shortage crisis, with too many open jobs and nobody to fill them."
    plan = {**PLAN, "claim_scope": {"country": "", "spoken_at": ""},
            "sources": [{**PLAN["sources"][1], "excerpts": [listing], "reading_passages": [listing]}]}
    assertion = check_claim(CLAIM, plan, states_everything)["verdict"]["assertions"][0]
    assert assertion["evidence"][0]["qualifiers"] == ["Several states"]
    assert assertion["status"] == "qualified_support"


def test_a_sentence_naming_many_places_is_qualified_not_a_crash():
    declared = {"country": "United States", "spoken_at": "", "country_basis": "supplied by the viewer"}
    listing = ("Labor shortages were reported by employers in Texas, Florida, Arizona, Colorado, Nevada and Georgia "
               "during the survey period, the association said in its annual review.")
    plan = {**PLAN, "claim_scope": declared, "sources": [{**PLAN["sources"][1], "excerpts": [listing], "reading_passages": [listing]}]}
    assertion = check_claim(CLAIM, plan, states_everything)["verdict"]["assertions"][0]
    assert assertion["evidence"][0]["qualifiers"] == ["Texas", "Florida", "Arizona", "Colorado"], "capped at the judgment limit"
    assert assertion["status"] == "qualified_support"


def test_assertion_passages_are_read_as_assertion_origin_and_can_count():
    plan = {**PLAN, "claim_scope": {"country": "United States", "spoken_at": "", "country_basis": "supplied by the viewer"},
            "sources": [dict(row) for row in PLAN["sources"][:2]]}
    plan["sources"][1] = {**plan["sources"][1],
                          "assertion_passages": ["Wisconsin's labor shortage grew 4% last year.",
                                                 "Employers nationwide reported 7 million unfilled jobs in June."]}
    packet = research_packet(CLAIM, plan)
    assert packet["sources"][1]["excerpts"] == ["Wisconsin's labor shortage grew 4% last year.",
                                                "Employers nationwide reported 7 million unfilled jobs in June."]
    assert packet["sources"][1]["context_excerpts"] == ["The state counted 40,000 openings."]
    result = check_claim(CLAIM, plan, states_everything)
    counted = {row["unit_id"]: row for row in result["verdict"]["assertions"][0]["evidence"]}
    assert set(counted) == {"source-2:p1:u1", "source-2:p2:u1"}, "both paragraphs of the admitted page count"
    assert counted["source-2:p2:u1"]["origin"] == "assertion" and counted["source-2:p2:u1"]["qualifiers"] == []


def test_a_measured_judges_counting_record_reaches_the_page_only_when_it_counts():
    class Measured:
        measurement = {"claims": 35, "measured_on": "2026-09-12", "counted": 18, "wrong": 10}

        def __call__(self, assertions, units):
            return states_everything(assertions, units)

    declared = {"country": "United States", "spoken_at": "", "country_basis": "supplied by the viewer"}
    result = check_claim(CLAIM, {**PLAN, "claim_scope": declared}, Measured())
    assert result["verdict"]["judge_note"].startswith("When this judge counted a sentence on 35 unseen claims (2026-09-12), it was wrong 10 of 18")
    assert "Reliability: When this judge counted" in result["text"]
    silent = check_claim(CLAIM, {**PLAN, "claim_scope": declared}, states_everything)
    assert silent["verdict"]["judge_note"] == "", "a judge nobody measured says nothing"
