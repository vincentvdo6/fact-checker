"""Rendered text is built from quoted sentences; a figure from nowhere is rejected."""

from __future__ import annotations

import pytest

from src.verdict.span_render import figures, render, ungrounded_figures

CLAIM = "We don't have a labor shortage. We have a good job shortage."


def evidence(relation, text, qualifiers=(), url="https://www.example.org/report", published="2025-05-15"):
    return {"unit_id": f"s1:p1:{text}", "passage_id": "s1:p1", "source_id": "s1", "relation": relation, "span": text[:10],
            "text": text, "qualifiers": list(qualifiers),
            "definitions": [{"id": "s1:p1:u2", "text": "Defined as the share of the labor force below $25,000."}],
            "url": url, "published_at": published, "publication_basis": "meta"}


def verdict(**overrides):
    base = {"status": "draft", "relationship": "qualified", "summary": "partially_established",
            "assertions": [
                {"id": "assertion-1", "text": "We don't have a labor shortage.", "negated": True,
                 "status": "qualified_contradiction", "direction": "against",
                 "limits": ["every counted sentence states a narrower scope"],
                 "evidence": [evidence("states_negation", "Wisconsin's labor shortage is a major barrier.",
                                       ["Wisconsin"], published="")],
                 "relevant": [], "sources_for": 0, "sources_against": 1, "judged": 3, "eligible": 3},
                {"id": "assertion-2", "text": "We have a good job shortage.", "negated": False,
                 "status": "insufficient", "direction": "none", "evidence": [],
                 "relevant": [evidence("bears_on", "The true rate was 24.3% in April.")],
                 "sources_for": 0, "sources_against": 0, "judged": 3, "eligible": 3}],
            "withheld": {"attributed opinion": 10, "instruction or navigation text": 5}, "eligible": 3,
            "scope_limits": ["The claim's country has not been established."], "rules": []}
    base.update(overrides)
    return base


def test_render_quotes_sentences_definitions_scope_and_source_record_only():
    text = render(verdict(), CLAIM)
    assert "Partially established (qualified)" in text
    assert "\"We don't have a labor shortage.\" is contradicted but not established: every counted sentence states a narrower scope." in text
    assert "\"We have a good job shortage.\" is not established by the sources." in text
    assert "example.org (publication date unconfirmed) [read as denying it]: \"Wisconsin's labor shortage is a major barrier.\"" in text
    assert "Scope stated in the sentence: Wisconsin." in text
    assert "Relevant, not counted:" in text and "example.org (published 2025-05-15): \"The true rate was 24.3% in April.\"" in text
    assert "Definition in the same paragraph: \"Defined as the share of the labor force below $25,000.\"" in text
    assert "Read as context, never as evidence: 10 attributed opinion, 5 instruction or navigation text." in text
    assert "Limit: The claim's country has not been established." in text


def test_withheld_sentences_and_invented_figures_never_reach_the_page():
    empty = verdict(assertions=[{"id": "assertion-1", "text": "The bridge opened in 1932.", "negated": False,
                                 "status": "insufficient", "direction": "none", "evidence": [], "relevant": [],
                                 "sources_for": 0, "sources_against": 0, "judged": 0, "eligible": 0}],
                    withheld={"instruction or navigation text": 4}, scope_limits=[], relationship="insufficient",
                    summary="unresolved")
    text = render(empty, "The bridge opened in 1932.")
    assert "Output supported" not in text and "No eligible sentence addresses it (0 of 0 judged)." in text
    bad = verdict()
    bad["assertions"][1]["relevant"][0]["text"] = "The rate rose."
    bad["assertions"][1]["relevant"][0]["definitions"] = []
    bad["assertions"][1]["text"] = "We have a 24.3% good job shortage."
    with pytest.raises(ValueError, match="24.3%"):
        render(bad, CLAIM)


def test_figure_detection_targets_statistics_not_counts():
    assert figures("24.3%, $25,000, 2017, 5.7 million and 39 workers per 100 openings") == [
        "24.3%", "$25,000", "2017", "5.7"]
    assert figures("13 sentences, 3 of 44 judged") == []
    assert ungrounded_figures("Rate 24.3% of 163.5 million", ["The rate was 24.3%"]) == ["163.5"]


def test_relevant_sentences_are_capped_on_the_page_but_kept_in_the_data():
    many = verdict()
    many["assertions"][1]["relevant"] = [evidence("bears_on", f"Sentence number {index} about jobs.") for index in range(9)]
    text = render(many, CLAIM)
    assert text.count("Sentence number") == 6 and "... and 3 more relevant sentences not shown." in text
    assert len(many["assertions"][1]["relevant"]) == 9


def test_context_research_rows_say_so_on_the_page():
    row = evidence("bears_on", "Everyone has economic, social and cultural rights.") | {"origin": "context"}
    text = render(verdict(assertions=[verdict()["assertions"][1] | {"relevant": [row]}]), CLAIM)
    assert "Found by caption-concept research, which does not resolve the claim." in text
    assert "caption-concept" not in render(verdict(), CLAIM)


def test_link_labels_are_shown_without_destinations_and_figures_stay_grounded():
    from src.verdict.span_render import display_text

    raw = "The rate was [24.3%](https://c212.net/c/link/?t=0&l=en&o=4428867-1) in April, see https://x.y/z?n=99999 now."
    assert display_text(raw) == "The rate was 24.3% in April, see  now."
    text = render(verdict(assertions=[verdict()["assertions"][1] | {"relevant": [evidence("bears_on", raw)]}]), CLAIM)
    assert "The rate was 24.3% in April, see  now." in text and "c212.net" not in text and "4428867" not in text
    joined = "Openings reached [1](https://a.b/c),000 by May."      # the figure exists only once the markup is gone
    text = render(verdict(assertions=[verdict()["assertions"][1] | {"relevant": [evidence("bears_on", joined)]}]), CLAIM)
    assert "Openings reached 1,000 by May." in text
    unquoted = verdict()["assertions"][1] | {"text": "Rate 77.7% claim.", "relevant": [evidence("bears_on", raw)]}
    with pytest.raises(ValueError, match="77.7%"):
        render(verdict(assertions=[unquoted]), CLAIM)


def test_a_sentence_quoted_under_one_half_is_counted_not_requoted_under_the_next():
    def rows(start, stop):
        return [evidence("bears_on", f"Sentence number {index} about jobs.") | {"unit_id": f"s1:p1:u{index}"}
                for index in range(start, stop)]

    both = verdict()
    both["assertions"][0]["relevant"] = rows(1, 4)
    both["assertions"][1]["relevant"] = [both["assertions"][0]["evidence"][0], *rows(1, 12)]   # one counted, three shown, eight fresh
    text = render(both, CLAIM)
    first, second = text.split("\"We have a good job shortage.\"")
    assert first.count("Sentence number") == 3
    assert second.count("Sentence number") == 6 and "4 relevant sentences already shown above." in second
    assert "... and 2 more relevant sentences not shown." in second
    assert "Sentence number 4 about jobs." in second and "Sentence number 1 about jobs." not in second
    assert second.count("Wisconsin's labor shortage") == 0 and first.count("Wisconsin's labor shortage") == 1
    assert len(both["assertions"][1]["relevant"]) == 12
    single = render(verdict(assertions=[verdict()["assertions"][1] | {"relevant": rows(0, 2)}] * 2), CLAIM)
    assert "2 relevant sentences already shown above." in single and single.count("Sentence number") == 2
    assert single.count("Relevant, not counted:") == 1, "no empty heading under the half whose sentences were all shown above"


def test_legacy_arithmetic_annotations_cannot_be_rendered_as_confirmation():
    row = evidence("bears_on", "Formerly incarcerated individuals are 24% less likely to return to prison.") | {
        "figures": [{"claimed": "around 25%", "relation": "near", "read_as": "24%"}]}
    basis = verdict()["assertions"][1] | {"text": "That's around 25% of the population.", "relevant": [row],
                                         "figures_confirmed": [{"claimed": "around 25%", "read_as": "24%", "relation": "near",
                                                                "unit_id": row["unit_id"], "source_id": "s1", "url": "https://www.example.org/report",
                                                                "independent_source": "example.org", "origin": "assertion"}]}
    spoken = "If you count all of those people together, that's around 25% of the population."
    text = render(verdict(assertions=[basis]), spoken)
    assert "\"That's around 25% of the population.\" is not established by the sources." in text
    assert row["text"] in text and "example.org (published 2025-05-15)" in text
    assert "Its figure checks" not in text and "Gives the claim's figure" not in text


def test_scope_notes_are_rendered_beside_the_limits():
    text = render(verdict(scope_limits=[], scope_notes=["Claim country: United States (supplied by the viewer)."]), CLAIM)
    assert "Scope: Claim country: United States (supplied by the viewer)." in text and "Limit:" not in text
    assert "Scope:" not in render(verdict(), CLAIM)


def test_a_partial_summary_carries_its_direction():
    assert "Partially contradicted (qualified)" in render(verdict(summary="partially_contradicted"), CLAIM)


def test_a_qualified_summary_says_not_established_and_names_the_part_on_a_contrast():
    both = verdict(summary="qualified_contradiction")
    assert "Verdict: One part contradicted, not established (qualified)." in render(both, CLAIM)
    single = verdict(summary="qualified_support", assertions=both["assertions"][:1])
    assert "Verdict: Supported, not established (qualified)." in render(single, CLAIM)


def test_a_sentence_dated_outside_the_claims_period_says_so_on_the_page():
    row = evidence("bears_on", "After inflation peaked in June 2022, the Fed raised rates.") | {"period": ["June 2022"]}
    text = render(verdict(assertions=[verdict()["assertions"][1] | {"relevant": [row]}]), CLAIM)
    assert "Dated to June 2022, outside the claim's stated period; shown, not counted." in text


def test_the_reliability_line_is_rendered_and_its_figures_are_not_stray():
    note = "When this judge counted a sentence on 36 unseen claims (2026-09-12), it was wrong 10 of 18 times; a counted sentence is a lead to read, not a finding."
    text = render(verdict(judge_note=note), CLAIM)
    assert f"Reliability: {note}" in text
    assert "Reliability:" not in render(verdict(judge_note=""), CLAIM)
    direction = "When this reading pointed a direction on 153 unseen claims (2026-09-14), it pointed the right way 47 of 47 times; the sentences it rests on are quoted below."
    assert f"Reliability: {direction}" in render(verdict(judge_note=direction), CLAIM)


def test_an_established_assertion_names_its_basis_from_the_source_record():
    established = verdict()["assertions"][0] | {
        "status": "contradicted", "limits": [], "basis": "one source: rbc.com (published 2025-07-09)",
        "evidence": [evidence("states_negation", "At the core of this labor shortage is an aging US population.",
                              url="https://www.rbc.com/en/x", published="2025-07-09")]}
    text = render(verdict(assertions=[established], relationship="qualified", summary="partially_contradicted"), CLAIM)
    assert "\"We don't have a labor shortage.\" is contradicted (one source: rbc.com (published 2025-07-09))." in text
    invented = established | {"basis": "one source: rbc.com (published 2025-07-10)"}
    with pytest.raises(ValueError):
        render(verdict(assertions=[invented], relationship="qualified", summary="partially_contradicted"), CLAIM)


def test_a_page_read_both_ways_is_shown_with_the_reason_it_does_not_count():
    row = evidence("bears_on", "The labor market showed remarkable resilience.") | {
        "not_counted": "states: this page is read both as stating and as denying the assertion; a misreading, not a dispute"}
    text = render(verdict(assertions=[verdict()["assertions"][1] | {"relevant": [row]}]), CLAIM)
    assert "Shown, not counted: states: this page is read both as stating and as denying the assertion; a misreading, not a dispute." in text
    assert "[read as" not in text.split("The labor market")[0].split("Relevant, not counted:")[-1], "a bearing row carries no direction"
