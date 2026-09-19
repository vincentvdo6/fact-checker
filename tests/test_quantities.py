"""A hedge licenses a figure by arithmetic stated in advance; years, kinds and unhedged figures never move."""

from __future__ import annotations

import pytest

from src.verdict.quantities import APPROXIMATION, hedged_form, hedged_quantities


def test_hedges_are_read_with_their_relation_and_figures_with_their_kind():
    found = hedged_quantities("Around 25% of workers earn about $25,000, nearly 5.7 million people, more than 1,200 firms.")
    assert [(item["hedge"], item["relation"], item["kind"], item["value"]) for item in found] == [
        ("Around", "near", "percent", 25.0), ("about", "near", "currency", 25000.0),
        ("nearly", "below", "count", 5.7e6), ("more than", "at_least", "count", 1200.0)]
    assert hedged_quantities("It opened in about 2005 and cost 7 dollars.") == [], "a year and a single digit are not quantities"
    assert hedged_quantities("The rate was 24.3% in April.") == [], "an unhedged figure is a claim of that figure"


def test_substitution_needs_a_fitting_figure_of_the_same_kind_and_keeps_the_hedge():
    assert APPROXIMATION == 0.05
    claim = "That's around 25% of the population."
    assert hedged_form(claim, "The rate increased from 24% to 24.3%, while the official rate stayed at 4.2%.") == (
        "That's around 24.3% of the population.", [{"hedge": "around", "relation": "near", "claimed": "25%", "read_as": "24.3%"}])
    assert hedged_form(claim, "The rate for Black workers rose to 26.7%.") is None, "6.8% away is outside the hedge"
    assert hedged_form(claim, "Some 25 million people were affected.") is None, "a count is not a percentage"
    assert hedged_form(claim, "Some 25 people were affected.") is None, "the same number of a different kind does not fit"
    assert hedged_form("They earn about $25,000 a year.", "About 24,900 workers were surveyed.") is None
    assert hedged_form(claim, "Wisconsin's labor shortage is a major barrier.") is None
    assert hedged_form("Exactly 25% of the population.", "The rate was 24.3%.") is None
    assert hedged_form("Nearly 25% of the population.", "The rate was 25.5%.") is None, "nearly means below"
    assert hedged_form("Just over 25% of the population.", "The rate was 24.8%.") is None, "just over means above"
    assert hedged_form("Unemployment is above 25 percent.", "It stood at 26.5% in March.")[0] == "Unemployment is above 26.5%."
    assert hedged_form("Unemployment is above 25 percent.", "It stood at 19.8% in March.") is None
    assert hedged_form("Fewer than 1,000 people attended.", "Police counted 640 people.")[0] == "Fewer than 640 people attended."
    assert hedged_form("The deficit was about $1.8 trillion.", "It came to $1,833 billion.")[0] == "The deficit was about $1,833 billion."


def test_every_hedged_figure_is_read_against_the_nearest_fit_and_unmatched_ones_stay():
    claim = "People who make around $25,000 a year: that's around 25% of the population."
    form, figures = hedged_form(claim, "TRU increased from 24% to 24.3% while the official rate was 4.2%.")
    assert form == "People who make around $25,000 a year: that's around 24.3% of the population."
    assert [item["read_as"] for item in figures] == ["24.3%"], "24.3% is nearer 25% than 24% is; $25,000 has no fit and stays"
    both = hedged_form(claim, "Those earning under $25,000 a year were 24.3% of the labor force.")
    assert both[0] == "People who make around $25,000 a year: that's around 24.3% of the population."
    assert [item["claimed"] for item in both[1]] == ["$25,000", "25%"]


@pytest.mark.parametrize("sentence", [
    "Formerly incarcerated people were 24% [less likely](https://example.org/report) to return to prison.",
    "The treatment group was 24 percent less likely to develop symptoms.",
    "Prices were 24% higher than last year.",
    "The study found a 24% reduction in missed appointments.",
    "The response rate increased by 24%.",
    "The response rate increased by roughly 24%.",
    "Costs could drop by 24%.",
    "The response rate fell 24%.",
    "The group had a 24% relative risk reduction.",
    "The response rate rose by 24 percentage points.",
])
def test_a_population_share_cannot_borrow_a_relative_change_or_percentage_points(sentence):
    assert hedged_form("Around 25% of the population responded.", sentence) is None


def test_percentage_kinds_stay_distinct_in_both_directions():
    assert hedged_form("Costs were around 25% higher.", "Costs were 24% of revenue.") is None
    assert hedged_form("Coverage rose by around 25 percentage points.", "Coverage was 24%.") is None
    assert hedged_form("Coverage rose by around 25 percentage points.", "Coverage rose by 24%.") is None
    assert hedged_form("Coverage rose by around 25%.", "Coverage rose by 24 percentage points.") is None
    assert hedged_form("Costs were around 25% higher.", "Costs were 24% higher than before.")[0] == "Costs were around 24% higher."
    assert hedged_form("Coverage rose by around 25 percentage points.", "Coverage rose by 24 percentage points.")[0] == (
        "Coverage rose by around 24 percentage points.")


def test_only_visible_source_numbers_can_supply_a_hedge_substitution():
    claim = "Around 25% of households responded."
    assert hedged_form(claim, 'Read the [survey](https://example.org/24.3% "24.3% responded").') is None
    assert hedged_form(claim, "The survey is at https://example.org/24.3% today.") is None
    assert hedged_form(claim, "The rate was [24.3%](https://example.org/report). ")[0] == "Around 24.3% of households responded."
    assert hedged_form(claim, "The rate was 24% lower, reaching 24.3% of households.")[0] == "Around 24.3% of households responded."
