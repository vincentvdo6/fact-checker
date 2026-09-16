"""The verdict is a fixed function of gate-eligible judgments; nothing withheld can move it."""

from __future__ import annotations

import pytest

from src.verdict.composition import compose

ASSERTIONS = [{"id": "assertion-1", "text": "We don't have a labor shortage.", "negated": True, "contrast": True},
              {"id": "assertion-2", "text": "We have a good job shortage.", "negated": False, "contrast": True}]
SOURCES = [{"id": "s1", "url": "https://a.example/x", "published_at": "2025-05-01",
            "temporal_status": "published_by_cutoff", "publication_basis": "meta"},
           {"id": "s2", "url": "https://b.example/y", "published_at": "2025-05-05",
            "temporal_status": "published_by_cutoff", "publication_basis": "meta"},
           {"id": "s5", "url": "https://d.example/undated", "published_at": "", "temporal_status": "date_unconfirmed",
            "publication_basis": "No explicit publication date found."},
           {"id": "s3", "url": "https://www.a.example/z", "published_at": "2025-05-02",       # same publisher as s1
            "temporal_status": "published_by_cutoff", "publication_basis": "meta"},
           {"id": "s4", "url": "https://c.example/w", "published_at": "2025-05-03",           # reached from s2
            "temporal_status": "published_by_cutoff", "publication_basis": "meta",
            "linked_from": [{"parent_source_id": "s2", "parent_url": "https://b.example/y"}]}]


def unit(identity, source, text, eligible=True, withheld=None):
    row = {"id": identity, "passage_id": identity.rsplit(":", 1)[0], "source_id": source, "text": text,
           "roles": ["reported_observation"] if eligible else ["attributed_opinion"], "eligible": eligible}
    if eligible:
        row["definitions"] = [{"id": identity + "d", "text": "Defined as the labor force share."}]
    else:
        row["withheld"] = withheld or "attributed opinion"
    return row


def gate(*units):
    return {"units": list(units), "eligible_ids": [row["id"] for row in units if row["eligible"]],
            "withheld": {"attributed opinion": sum(1 for row in units if not row["eligible"])}, "rule": "test"}


def judgment(assertion, identity, relation, qualifiers=()):
    return {"assertion_id": assertion, "unit_id": identity, "relation": relation,
            "span": "x" if relation != "unrelated" else "", "qualifiers": list(qualifiers), "status": "unverified"}


GATE = gate(unit("s1:p1:u1", "s1", "Rate was 24.3%."), unit("s2:p1:u1", "s2", "Wisconsin's labor shortage grew."),
            unit("s1:p2:u1", "s1", "In our view there is a shortage.", eligible=False),
            unit("s3:p1:u1", "s3", "The publisher's second page agrees."), unit("s4:p1:u1", "s4", "The linked page agrees."),
            unit("s5:p1:u1", "s5", "The undated page agrees."))
FOR = [judgment("assertion-1", "s1:p1:u1", "states"), judgment("assertion-1", "s2:p1:u1", "states")]


def test_judgments_on_withheld_unknown_or_repeated_pairs_are_rejected():
    with pytest.raises(ValueError):
        compose(ASSERTIONS, GATE, [judgment("assertion-1", "s1:p2:u1", "states_negation")], SOURCES)
    with pytest.raises(ValueError):
        compose(ASSERTIONS, GATE, [judgment("assertion-9", "s1:p1:u1", "states")], SOURCES)
    with pytest.raises(ValueError):
        compose(ASSERTIONS, GATE, [judgment("assertion-1", "s1:p1:u1", "states")] * 2, SOURCES)
    with pytest.raises(ValueError):
        compose(ASSERTIONS, GATE, [], SOURCES + [{"id": "s3", "url": "", "temporal_status": "later_publication"}])
    with pytest.raises(ValueError):
        compose([], GATE, [], SOURCES)


def test_only_states_relations_count_and_bears_on_is_shown_not_counted():
    rows = [judgment("assertion-2", "s1:p1:u1", "bears_on"), judgment("assertion-2", "s2:p1:u1", "unrelated"),
            judgment("assertion-1", "s1:p1:u1", "unrelated"), judgment("assertion-1", "s2:p1:u1", "unrelated")]
    verdict = compose(ASSERTIONS, GATE, rows, SOURCES, {"country": "", "spoken_at": ""})
    second = verdict["assertions"][1]
    assert second["status"] == "insufficient" and second["evidence"] == [] and len(second["relevant"]) == 1
    assert second["relevant"][0]["text"] == "Rate was 24.3%." and second["relevant"][0]["url"] == "https://a.example/x"
    assert second["relevant"][0]["definitions"][0]["text"] == "Defined as the labor force share."
    assert second["judged"] == 2 and second["eligible"] == 5
    assert verdict["relationship"] == "insufficient" and verdict["summary"] == "unresolved"
    assert verdict["scope_limits"] == ["The claim's country has not been established.",
                                       "The claim's speech date has not been established."]
    assert verdict["withheld"] == {"attributed opinion": 1} and verdict["eligible"] == 5


def test_qualifiers_weaken_and_disagreement_contests():
    against = [judgment("assertion-1", "s1:p1:u1", "states_negation"), judgment("assertion-1", "s2:p1:u1", "states_negation")]
    plain = compose(ASSERTIONS[:1], GATE, against, SOURCES)
    assert plain["assertions"][0]["status"] == "contradicted" and plain["relationship"] == "contradicted"
    assert plain["summary"] == "established" and plain["assertions"][0]["sources_against"] == 2
    scoped = compose(ASSERTIONS[:1], GATE, [against[0], judgment("assertion-1", "s2:p1:u1", "states_negation", ["Wisconsin"])],
                     SOURCES)
    assert scoped["assertions"][0]["status"] == "contradicted", "a qualified count weakens nothing an unqualified one establishes"
    assert scoped["assertions"][0]["basis"] == "one source: a.example (published 2025-05-01)"
    narrow = compose(ASSERTIONS[:1], GATE, [FOR[1], judgment("assertion-1", "s1:p1:u1", "states", ["in April"])], SOURCES)
    assert narrow["assertions"][0]["status"] == "supported" and narrow["relationship"] == "supported"
    assert narrow["assertions"][0]["basis"] == "one source: b.example (published 2025-05-05)"
    alone = compose(ASSERTIONS[:1], GATE, [judgment("assertion-1", "s1:p1:u1", "states", ["in April"])], SOURCES)
    assert alone["assertions"][0]["status"] == "qualified_support" and alone["assertions"][0]["basis"] == ""
    assert alone["assertions"][0]["limits"] == ["every counted sentence states a narrower scope"]
    assert plain["assertions"][0]["limits"] == []
    assert plain["assertions"][0]["basis"] == ("2 independent sources: a.example (published 2025-05-01), "
                                               "b.example (published 2025-05-05)")
    both = compose(ASSERTIONS[:1], GATE, [judgment("assertion-1", "s2:p1:u1", "states_negation", ["Wisconsin"]),
                                          judgment("assertion-1", "s1:p1:u1", "states")], SOURCES)
    assert both["assertions"][0]["status"] == "contested" and both["assertions"][0]["direction"] == "mixed"
    assert both["relationship"] == "insufficient" and both["summary"] == "contested"
    supported = compose(ASSERTIONS[:1], GATE, FOR + [judgment("assertion-1", "s3:p1:u1", "states", ["Wisconsin"])], SOURCES)
    assert supported["assertions"][0]["status"] == "supported" and supported["assertions"][0]["sources_for"] == 2


def test_one_dated_source_establishes_and_is_named_and_lineage_or_publisher_is_one_source():
    single = compose(ASSERTIONS[:1], GATE, FOR[:1], SOURCES)
    assert single["assertions"][0]["status"] == "supported" and single["assertions"][0]["sources_for"] == 1
    assert single["assertions"][0]["basis"] == "one source: a.example (published 2025-05-01)"
    assert single["relationship"] == "supported" and single["summary"] == "established"
    denied = compose(ASSERTIONS[:1], GATE, [judgment("assertion-1", "s2:p1:u1", "states_negation")], SOURCES)
    assert denied["assertions"][0]["status"] == "contradicted" and denied["summary"] == "established"
    assert denied["assertions"][0]["basis"] == "one source: b.example (published 2025-05-05)"
    publisher = compose(ASSERTIONS[:1], GATE, [FOR[0], judgment("assertion-1", "s3:p1:u1", "states")], SOURCES)
    assert publisher["assertions"][0]["sources_for"] == 1, "two pages on a.example are one source"
    assert publisher["assertions"][0]["basis"] == "one source: a.example (published 2025-05-01)", "and are named as one"
    linked = compose(ASSERTIONS[:1], GATE, [FOR[1], judgment("assertion-1", "s4:p1:u1", "states")], SOURCES)
    assert {row["independent_source"] for row in linked["assertions"][0]["evidence"]} == {"b.example"}, "a page reached from s2 is s2's lineage"
    assert linked["assertions"][0]["basis"] == "one source: b.example (published 2025-05-05)"
    two = compose(ASSERTIONS[:1], GATE, [FOR[0], judgment("assertion-1", "s4:p1:u1", "states")], SOURCES)
    assert two["assertions"][0]["status"] == "supported" and two["assertions"][0]["sources_for"] == 2
    assert two["assertions"][0]["basis"] == "2 independent sources: a.example (published 2025-05-01), b.example (published 2025-05-03)",         "the linked page counts under its lineage root's name with its own date"
    assert any("one independent source with a confirmed publication date, which the page names" in rule for rule in two["rules"])
    assert all(item["basis"] == "" for item in compose(ASSERTIONS[:1], GATE, [], SOURCES)["assertions"])
    unknown = compose(ASSERTIONS[:1], GATE, FOR[:1], SOURCES, {"country": "", "spoken_at": ""})["assertions"][0]
    assert unknown["status"] == "qualified_support" and unknown["basis"] == "", "a qualified assertion rests on no basis"


def test_an_undated_source_is_shown_and_counted_but_cannot_corroborate():
    undated = judgment("assertion-1", "s5:p1:u1", "states")
    alone = compose(ASSERTIONS[:1], GATE, [undated], SOURCES)
    assert alone["assertions"][0]["status"] == "qualified_support" and len(alone["assertions"][0]["evidence"]) == 1
    assert alone["assertions"][0]["limits"] == ["no counted sentence comes from a source with a confirmed publication date"]
    assert alone["assertions"][0]["basis"] == ""
    with_dated = compose(ASSERTIONS[:1], GATE, [FOR[0], undated], SOURCES)
    assert with_dated["assertions"][0]["status"] == "supported" and with_dated["assertions"][0]["limits"] == []
    assert with_dated["assertions"][0]["basis"] == "one source: a.example (published 2025-05-01)", "the undated page is not a basis"
    assert with_dated["assertions"][0]["sources_for"] == 2, "the display count is every counted source"
    assert compose(ASSERTIONS[:1], GATE, FOR, SOURCES)["assertions"][0]["status"] == "supported"


def test_contrast_is_established_only_when_every_half_is():
    second = [judgment("assertion-2", "s1:p1:u1", "states"), judgment("assertion-2", "s2:p1:u1", "states")]
    half = compose(ASSERTIONS, GATE, second, SOURCES)
    assert [row["status"] for row in half["assertions"]] == ["insufficient", "supported"]
    assert half["relationship"] == "qualified" and half["summary"] == "partially_established"
    whole = compose(ASSERTIONS, GATE, second + FOR, SOURCES)
    assert whole["relationship"] == "supported" and whole["summary"] == "established"
    split = compose(ASSERTIONS, GATE, second + [judgment("assertion-1", "s1:p1:u1", "states_negation"),
                                                judgment("assertion-1", "s2:p1:u1", "states_negation")], SOURCES)
    assert split["relationship"] == "qualified" and split["summary"] == "contested", "one half for, one half against"
    against = compose(ASSERTIONS, GATE, [judgment("assertion-1", "s2:p1:u1", "states_negation")], SOURCES)
    assert [row["status"] for row in against["assertions"]] == ["contradicted", "insufficient"]
    assert against["relationship"] == "qualified" and against["summary"] == "partially_contradicted", \
        "counts against one half and nothing for any half is partially contradicted, never partially established"
    undated = judgment("assertion-1", "s5:p1:u1", "states_negation")
    held = compose(ASSERTIONS, GATE, [undated], SOURCES)
    assert [row["status"] for row in held["assertions"]] == ["qualified_contradiction", "insufficient"]
    assert held["summary"] == "qualified_contradiction", "a qualified count on one half establishes nothing and says so"
    forward = compose(ASSERTIONS, GATE, [judgment("assertion-2", "s5:p1:u1", "states")], SOURCES)
    assert forward["summary"] == "qualified_support"
    mixed = compose(ASSERTIONS, GATE, second + [judgment("assertion-1", "s1:p1:u1", "states_negation")], SOURCES)
    assert [row["status"] for row in mixed["assertions"]] == ["contradicted", "supported"]
    assert mixed["summary"] == "contested"
    partly = compose(ASSERTIONS, GATE, second + [judgment("assertion-1", "s5:p1:u1", "states")], SOURCES)
    assert [row["status"] for row in partly["assertions"]] == ["qualified_support", "supported"]
    assert partly["summary"] == "partially_established", "partial only once a half is established"
    disputed = compose(ASSERTIONS, GATE, [judgment("assertion-1", "s1:p1:u1", "states"), judgment("assertion-1", "s2:p1:u1", "states_negation"),
                                          judgment("assertion-2", "s5:p1:u1", "states")], SOURCES)
    assert [row["status"] for row in disputed["assertions"]] == ["contested", "qualified_support"]
    assert disputed["relationship"] == "qualified" and disputed["summary"] == "contested", "a disputed half is not buried"
    denied = compose(ASSERTIONS, GATE, [judgment(a, u, "states_negation") for a in ("assertion-1", "assertion-2")
                                        for u in ("s1:p1:u1", "s2:p1:u1")], SOURCES)
    assert denied["relationship"] == "contradicted"


def test_unknown_claim_country_can_qualify_but_never_establish():
    rows = FOR + [judgment("assertion-2", "s1:p1:u1", "states_negation"), judgment("assertion-2", "s2:p1:u1", "states_negation")]
    unknown = compose(ASSERTIONS, GATE, rows, SOURCES, {"country": "", "spoken_at": ""})
    assert [row["status"] for row in unknown["assertions"]] == ["qualified_support", "qualified_contradiction"]
    assert unknown["relationship"] == "qualified"
    assert [row["limits"] for row in unknown["assertions"]] == [["the claim's country is not established"]] * 2
    known = compose(ASSERTIONS, GATE, rows, SOURCES, {"country": "US", "spoken_at": ""})
    assert [row["status"] for row in known["assertions"]] == ["supported", "contradicted"]
    unmodelled = compose(ASSERTIONS, GATE, rows, SOURCES, {"referent": "the reactor"})
    assert [row["status"] for row in unmodelled["assertions"]] == ["supported", "contradicted"]


def test_relevant_rows_are_ordered_by_origin_then_confidence_which_never_changes_a_status():
    rows = [judgment("assertion-2", "s1:p1:u1", "bears_on") | {"confidence": 0.4},
            judgment("assertion-2", "s2:p1:u1", "bears_on") | {"confidence": 0.9}]
    verdict = compose(ASSERTIONS, GATE, rows, SOURCES)
    assert [row["unit_id"] for row in verdict["assertions"][1]["relevant"]] == ["s2:p1:u1", "s1:p1:u1"]
    assert verdict["assertions"][1]["relevant"][0]["confidence"] == 0.9
    assert verdict["assertions"][1]["status"] == "insufficient"
    context = unit("s2:p2:u1", "s2", "The mission of LISEP is research.") | {"origin": "context"}
    rows.append(judgment("assertion-2", "s2:p2:u1", "bears_on") | {"confidence": 0.99})
    verdict = compose(ASSERTIONS, gate(*GATE["units"], context), rows, SOURCES)
    assert [row["unit_id"] for row in verdict["assertions"][1]["relevant"]] == ["s2:p1:u1", "s1:p1:u1", "s2:p2:u1"]


def test_a_count_on_a_caption_concept_sentence_is_shown_never_counted():
    context = unit("s2:p2:u1", "s2", "Everyone has economic, social and cultural rights.") | {"origin": "context"}
    both = gate(*GATE["units"], context)
    rows = [judgment("assertion-1", "s2:p2:u1", "states"), judgment("assertion-1", "s1:p1:u1", "unrelated")]
    verdict = compose(ASSERTIONS[:1], both, rows, SOURCES)
    first = verdict["assertions"][0]
    assert first["status"] == "insufficient" and first["evidence"] == [] and first["sources_for"] == 0
    assert first["relevant"][0]["relation"] == "bears_on" and first["relevant"][0]["origin"] == "context"
    assert first["relevant"][0]["not_counted"] == "states: found by caption-concept research; shown, never counted"
    assert "A sentence found by caption-concept research is shown, never counted." in verdict["rules"]
    counted = compose(ASSERTIONS[:1], both, FOR, SOURCES)
    assert counted["assertions"][0]["status"] == "supported" and counted["assertions"][0]["evidence"][0]["origin"] == "assertion"


@pytest.mark.parametrize("claim,sentence", [
    ("That's around 25% of the population.", "Formerly incarcerated individuals are 24% less likely to return to prison."),
    ("Around 25% of households lack electricity.", "Electricity prices rose 24% during the same year."),
    ("The median salary is $50,000.", "The vehicle costs $50,000 before taxes."),
    ("Reservoirs supply 20 million residents.", "The reservoir stores 20 million cubic metres."),
    ("Around 25% of adults are unemployed.", "The unemployment rate for teenagers was 24.3%."),
])
def test_numeric_similarity_neither_confirms_a_measure_nor_promotes_a_bearing_sentence(claim, sentence):
    basis = [{"id": "assertion-1", "text": claim, "negated": False, "contrast": False}]
    figure = unit("s1:p3:u1", "s1", sentence)
    other = unit("s2:p3:u1", "s2", "The survey describes the claim's measure and its population.")
    both = gate(*GATE["units"], figure, other)
    rows = [judgment("assertion-1", "s2:p3:u1", "bears_on") | {"confidence": 0.95},
            judgment("assertion-1", "s1:p3:u1", "bears_on") | {"confidence": 0.60},
            judgment("assertion-1", "s1:p1:u1", "unrelated")]
    verdict = compose(basis, both, rows, SOURCES)
    result = verdict["assertions"][0]
    assert result["status"] == "insufficient" and result["evidence"] == [], "a figure match never counts"
    assert [row["unit_id"] for row in result["relevant"]] == ["s2:p3:u1", "s1:p3:u1"], "matching digits do not boost relevance"
    assert result["relevant"][1]["text"] == sentence
    assert all("figures" not in row for row in result["relevant"])
    assert not result.get("figures_confirmed")
    assert not any("confirms the figure" in rule for rule in verdict["rules"])


def test_a_declared_country_lifts_the_limit_and_is_noted_with_its_basis():
    declared = compose(ASSERTIONS[:1], GATE, FOR, SOURCES, {"country": "United States", "spoken_at": "",
                                                            "country_basis": "supplied by the viewer"})
    assert declared["assertions"][0]["status"] == "supported"
    assert declared["scope_limits"] == ["The claim's speech date has not been established."]
    assert declared["scope_notes"] == ["Claim country: United States (supplied by the viewer)."]
    unset = compose(ASSERTIONS[:1], GATE, FOR, SOURCES, {"country": "", "spoken_at": ""})
    assert unset["assertions"][0]["status"] == "qualified_support" and unset["scope_notes"] == []


def test_a_count_dated_outside_the_claims_period_is_shown_never_counted():
    dated = [{"id": "assertion-1", "text": "The Federal Reserve cut interest rates in September 2025.", "negated": False, "contrast": False}]
    old = unit("s2:p3:u1", "s2", "After inflation peaked in June 2022, the Fed implemented a series of rate hikes.")
    same = unit("s1:p3:u1", "s1", "The Fed cut its benchmark rate on Sept. 17, 2025.")
    both = gate(*GATE["units"], old, same)
    verdict = compose(dated, both, [judgment("assertion-1", "s2:p3:u1", "states_negation"), judgment("assertion-1", "s1:p3:u1", "states")], SOURCES)
    first = verdict["assertions"][0]
    assert [row["unit_id"] for row in first["evidence"]] == ["s1:p3:u1"] and first["status"] == "supported"
    demoted = first["relevant"][0]
    assert demoted["relation"] == "bears_on" and demoted["period"] == ["June 2022"]
    assert demoted["not_counted"] == "states_negation: states a different period (June 2022)"
    assert any("dated wholly outside" in rule for rule in verdict["rules"])


def test_a_declared_speech_date_is_noted_and_lifts_its_limit():
    scope = {"country": "", "spoken_at": "2025-07-15", "spoken_at_basis": "supplied by the viewer"}
    verdict = compose(ASSERTIONS[:1], GATE, FOR, SOURCES, scope)
    assert verdict["scope_limits"] == ["The claim's country has not been established."]
    assert verdict["scope_notes"] == ["Speech date: 2025-07-15 (supplied by the viewer)."]


def test_the_judges_measured_counting_record_travels_only_with_a_page_that_counts():
    record = {"claims": 36, "measured_on": "2026-09-12", "counted": 18, "wrong": 10}
    counted = compose(ASSERTIONS[:1], GATE, FOR[:1], SOURCES, judge_measurement=record)
    assert counted["judge_note"] == ("When this judge counted a sentence on 36 unseen claims (2026-09-12), it was wrong "
                                     "10 of 18 times; a counted sentence is a lead to read, not a finding.")
    assert compose(ASSERTIONS[:1], GATE, [], SOURCES, judge_measurement=record)["judge_note"] == "", "nothing counted, nothing to weigh"
    assert compose(ASSERTIONS[:1], GATE, FOR[:1], SOURCES)["judge_note"] == "", "unmeasured judges say nothing"
    direction = {"claims": 153, "measured_on": "2026-09-14", "directional": 47, "right": 47, "wrong": 0}
    composed = compose(ASSERTIONS[:1], GATE, FOR[:1], SOURCES, judge_measurement=record, direction_measurement=direction)
    assert composed["judge_note"] == ("When this reading pointed a direction on 153 unseen claims (2026-09-14), it pointed the "
                                      "right way 47 of 47 times; the sentences it rests on are quoted below."), \
        "the direction record is what a viewer experiences and takes precedence over the sentence record"
    assert compose(ASSERTIONS[:1], GATE, [], SOURCES, direction_measurement=direction)["judge_note"] == ""


def test_one_page_read_both_ways_is_a_misreading_and_counts_for_nothing():
    conflicted = GATE["units"] + [unit("s2:p2:u1", "s2", "The second page sentence.")]
    both_ways = gate(*conflicted)
    rows = [judgment("assertion-1", "s2:p1:u1", "states_negation"), judgment("assertion-1", "s2:p2:u1", "states"),
            judgment("assertion-1", "s1:p1:u1", "states_negation"), judgment("assertion-1", "s4:p1:u1", "states_negation")]
    verdict = compose(ASSERTIONS[:1], both_ways, rows, SOURCES)
    first = verdict["assertions"][0]
    assert first["status"] == "contradicted", "s1 and s4 (independent, dated) still establish; the b.example page is set aside"
    assert [row["source_id"] for row in first["evidence"]] == ["s1", "s4"]
    shown = {row["unit_id"]: row.get("not_counted", "") for row in first["relevant"]}
    assert shown == {"s2:p1:u1": "states_negation: this page is read both as stating and as denying the assertion; a misreading, not a dispute",
                     "s2:p2:u1": "states: this page is read both as stating and as denying the assertion; a misreading, not a dispute"}
    dispute = compose(ASSERTIONS[:1], both_ways, [judgment("assertion-1", "s2:p1:u1", "states_negation"),
                                                  judgment("assertion-1", "s1:p1:u1", "states")], SOURCES)
    assert dispute["assertions"][0]["status"] == "contested", "two pages disagreeing is a dispute"
    assert any("misreading, not a dispute" in rule for rule in verdict["rules"])
