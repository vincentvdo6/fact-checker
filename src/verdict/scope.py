"""Read a sentence's place scope against the claim's declared country, by a closed list, not a model.

The pair judge says whether a sentence states an assertion; it does not say for where. A
sentence about Wisconsin's labor shortage that states "we have a labor shortage" counts, under
the composition rules, as an unqualified national statement unless something names the
narrower scope. With the viewer's country declared, a first-level subdivision of that country
named in the sentence -- a state, province, territory or nation -- is narrower by construction,
and its exact wording becomes a qualifier, which can only weaken a count. Nothing is inferred
when no country is declared, and a place the claim itself names is the claim's own scope, not
a narrowing. Ambiguous names (Washington the state against the capital's dateline) are left
out rather than guessed.
"""

from __future__ import annotations

import re

SUBDIVISIONS: dict[str, tuple[str, ...]] = {
    "United States": (
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
        "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
        "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
        "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
        "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
        "Vermont", "Virginia", "West Virginia", "Wisconsin", "Wyoming", "Puerto Rico"),
    "Canada": ("Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador", "Nova Scotia",
               "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Northwest Territories", "Nunavut", "Yukon"),
    "Australia": ("New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia", "Tasmania",
                  "Northern Territory", "Australian Capital Territory"),
    "United Kingdom": ("England", "Scotland", "Wales", "Northern Ireland"),
}
ALIASES = {"United States of America": "United States", "USA": "United States", "US": "United States",
           "U.S.": "United States", "UK": "United Kingdom", "Britain": "United Kingdom", "Great Britain": "United Kingdom"}


def _names(country: str) -> tuple[str, ...]:
    return SUBDIVISIONS.get(ALIASES.get(country.strip(), country.strip()), ())


# A quantified group -- "several states", "many industries", "some regions" -- reports a finding about a
# part of the claim's scope, whatever the country; the exact phrase becomes a qualifier. "All" and
# "every" narrow nothing and are not listed.
_QUANTIFIERS = r"(?:several|some|many|certain|numerous|various|a few|a number of|a handful of)"
_GROUPS = (r"(?:states|provinces|territories|regions|cities|counties|towns|areas|industries|sectors|markets|"
           r"occupations|fields|employers|companies|firms|businesses|communities|countries|hospitals|schools|districts)")
# "most states" quantifies; "most populous countries" is a superlative, so "most" takes no adjective between.
_GROUP = re.compile(rf"\b(?:{_QUANTIFIERS}\s+(?:\w+\s+)?|most\s+){_GROUPS}\b", re.IGNORECASE)


def group_qualifiers(sentence: str, claim: str) -> list[str]:
    """Exact substrings of the sentence naming a quantified part of the scope that the claim does not name."""
    found = []
    for match in _GROUP.finditer(sentence):
        phrase = match.group(0)
        if phrase.lower() not in claim.lower() and phrase not in found:
            found.append(phrase)
    return found


def place_qualifiers(sentence: str, claim: str, country: str) -> list[str]:
    """Exact substrings of the sentence naming a subdivision of the declared country that the claim does not name."""
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for name in sorted(_names(country or ""), key=len, reverse=True):     # "West Virginia" before "Virginia"
        if re.search(rf"\b{re.escape(name)}\b", claim):
            continue
        for match in re.finditer(rf"\b{re.escape(name)}\b", sentence):
            if any(start <= match.start() < end for start, end in taken):
                continue
            taken.append(match.span())
            found.append((match.start(), match.group(0)))
            break
    return [name for _, name in sorted(found)]      # in sentence order


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december")
_MONTH = r"(?:Jan(?:uary|\.)?|Feb(?:ruary|\.)?|Mar(?:ch|\.)?|Apr(?:il|\.)?|May|Jun(?:e|\.)?|Jul(?:y|\.)?|Aug(?:ust|\.)?|" \
         r"Sep(?:t(?:ember|\.)?|\.)?|Oct(?:ober|\.)?|Nov(?:ember|\.)?|Dec(?:ember|\.)?)"
_YEAR = r"(?:19|20)\d{2}"
_POINT = rf"(?:{_MONTH}\s+(?:\d{{1,2}},?\s+)?{_YEAR}|{_YEAR})"
_PERIOD = re.compile(
    rf"\b(?:(?P<open>since|from|starting in|beginning in)\s+(?P<a>{_POINT})(?!\s*(?:to|through|until|-|–)\s*{_POINT})"
    rf"|(?:between\s+)?(?P<b>{_POINT})\s*(?:to|through|until|and|-|–)\s*(?P<c>{_POINT})"
    rf"|(?P<d>{_POINT}))\b")


def _month_index(text: str) -> tuple[int, int] | None:
    """(start, end) months since year 0 for a point: a year spans twelve months, a month one."""
    year = int(re.search(_YEAR, text).group(0))
    month = re.match(_MONTH, text)
    if month is None:
        return year * 12, year * 12 + 11
    number = next(index for index, name in enumerate(_MONTHS) if name.startswith(month.group(0).rstrip(".").lower()[:3]))
    return year * 12 + number, year * 12 + number


def periods(text: str) -> list[tuple[int, int, str]]:
    """Dated periods a sentence states, as month ranges with the exact wording; relative periods are not read."""
    found = []
    for match in _PERIOD.finditer(text):
        if match.group("open"):
            start, _ = _month_index(match.group("a"))
            found.append((start, 10**9, match.group(0)))
        elif match.group("b"):
            start, _ = _month_index(match.group("b"))
            _, end = _month_index(match.group("c"))
            found.append((start, end, match.group(0)))
        else:
            start, end = _month_index(match.group("d"))
            found.append((start, end, match.group(0)))
    return found


def different_period(claim: str, sentence: str) -> list[str]:
    """The sentence's stated periods when the claim states a period and none of them overlap it; else nothing."""
    claimed, stated = periods(claim), periods(sentence)
    if not claimed or not stated:
        return []
    if any(start <= c_end and c_start <= end for start, end, _ in stated for c_start, c_end, _ in claimed):
        return []
    return [text for _, _, text in stated]
