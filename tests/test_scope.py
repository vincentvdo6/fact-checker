"""A subdivision of the declared country named in a sentence narrows it; nothing is read without a declaration."""

from __future__ import annotations

from src.verdict.scope import group_qualifiers, place_qualifiers


def test_subdivisions_of_the_declared_country_qualify_unless_the_claim_names_them():
    denial = "We don't have a labor shortage."
    wpr = "Wisconsin’s labor shortage is a major barrier to growing the state’s economy, a new report finds."
    assert place_qualifiers(wpr, denial, "United States") == ["Wisconsin"]
    assert place_qualifiers(wpr, denial, "") == [], "no country declared, nothing narrows"
    assert place_qualifiers(wpr, denial, "Canada") == [], "Wisconsin is not a Canadian subdivision"
    assert place_qualifiers(wpr, "Wisconsin has a labor shortage.", "United States") == [], "the claim's own scope"
    assert place_qualifiers("Shootings in New York fell.", "Shootings in New York City fell to a record low.", "United States") == []
    assert place_qualifiers("Openings rose in West Virginia and Texas.", denial, "US") == ["West Virginia", "Texas"]
    assert place_qualifiers("Openings rose in Virginia.", denial, "U.S.") == ["Virginia"]
    assert place_qualifiers("WASHINGTON, May 15, 2025 /PRNewswire/ -- the rate rose.", denial, "United States") == []
    assert place_qualifiers("Unemployment in Scotland fell.", "Unemployment fell.", "UK") == ["Scotland"]
    assert place_qualifiers("Unemployment in Bavaria fell.", "Unemployment fell.", "Germany") == [], "no list for Germany yet"


def test_periods_are_read_as_month_ranges_and_relative_periods_are_not():
    from src.verdict.scope import periods

    assert [text for *_, text in periods("After inflation peaked in June 2022, the Fed raised rates.")] == ["June 2022"]
    assert periods("Openings have exceeded job seekers since 2021.")[0][1] > 10**8, "an open range reaches the present"
    start, end, text = periods("From 2020 to 2024, food prices rose almost 24%.")[0]
    assert text == "2020 to 2024" and end - start == 59
    assert periods("The rate was 24.3% in April.") == [] and periods("Restart on May 14.") == []
    assert periods("last year's figure") == [], "relative periods are not read"


def test_a_sentence_dated_outside_the_claims_period_is_a_different_period_and_overlap_is_not():
    from src.verdict.scope import different_period

    claim = "The Federal Reserve cut interest rates in September 2025."
    assert different_period(claim, "After inflation peaked in June 2022, the Fed implemented a series of rate hikes.") == ["June 2022"]
    assert different_period(claim, "The Fed cut its benchmark rate on Sept. 17, 2025, the first cut of the year.") == []
    assert different_period(claim, "The Fed has cut rates twice since 2024.") == [], "an open range covers the claim's month"
    assert different_period(claim, "The Fed cut rates in 2025.") == [], "a year contains its months"
    assert different_period(claim, "The Fed cut rates.") == [], "an undated sentence is not a different period"
    assert different_period("We don't have a labor shortage.", "In June 2022 the Fed raised rates.") == [], "no claim period, no comparison"
    assert different_period("Rates rose between January 2023 and March 2023.", "Rates fell from April 2023 through 2024.") == ["April 2023 through 2024"]


def test_a_quantified_group_is_a_qualifier_unless_the_claim_names_it():
    claim = "We don't have a labor shortage."
    assert group_qualifiers("Several states in America are facing a worker shortage crisis.", claim) == ["Several states"]
    assert group_qualifiers("Many industries in several states are hiring.", claim) == ["Many industries", "several states"]
    assert group_qualifiers("A number of large firms cut jobs.", claim) == ["A number of large firms"]
    assert group_qualifiers("Every state and all industries reported a shortage.", claim) == [], "all and every narrow nothing"
    assert group_qualifiers("In many industries the shortage grew.", "Unemployment in many industries rose.") == [],         "a group the claim itself names is the claim's own scope"
    assert group_qualifiers("The state hired several people.", claim) == [], "only the listed group nouns"
    assert group_qualifiers("Most states reported gains.", claim) == ["Most states"]
    assert group_qualifiers("China and India are the world's most populous countries.", claim) == [], "a superlative, not a quantity"
