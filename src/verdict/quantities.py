"""Read a hedged quantity in a claim against the figures a sentence actually gives.

"Around 25%" is not a claim of 25.000%: the hedge licenses any figure close to it, and
"more than 25%" any figure above it. Whether a figure is close is arithmetic, and it is
decided here, in code, by relations stated in advance. Whether the sentence's figure
measures the same thing as the claim's is not arithmetic, and it is left to the judge --
but the judge trained on FEVER reads a differing number as a contradiction before it reads
anything else, so it is asked about the sentence's own figure in the claim's place, with
the hedge kept. This is the negation probe's shape: a mechanical form the model can read,
with the transformation carried by code and recorded on the judgment.

A year is never a quantity here. Percentages, recognized relative-change forms, percentage
points, currencies and counts are separate kinds. Recognizing a kind does not establish
that two figures measure the same population or outcome. A missing match leaves the
claim as written; the judge can still misread an unrelated sentence.
"""

from __future__ import annotations

import re

from src.retrieval.visible_text import visible_text

APPROXIMATION = 0.05        # relative tolerance for "around", "about", "nearly", "just over" ...

_HEDGES = {
    "near": ("around", "about", "roughly", "approximately", "some", "circa", "close to", "in the region of",
             "on the order of"),
    "below": ("nearly", "almost", "just under", "just short of", "a little under", "slightly under",
              "slightly below", "approaching"),
    "above": ("just over", "a little over", "slightly over", "slightly more than", "slightly above"),
    "at_least": ("more than", "over", "above", "at least", "upwards of", "in excess of", "exceeding", "exceeds",
                 "greater than", "higher than", "no less than", "north of"),
    "at_most": ("less than", "under", "below", "at most", "fewer than", "no more than", "lower than",
                "south of"),
}
_HEDGE_WORDS = sorted((word, kind) for kind, words in _HEDGES.items() for word in words)
_HEDGE = "|".join(re.escape(word) for word, _ in sorted(_HEDGE_WORDS, key=lambda item: -len(item[0])))
_NUMBER = r"(?P<currency>[$€£])?(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_UNIT = r"(?:\s?(?P<percent>%|percent|per cent|percentage points?)|\s(?P<scale>thousand|million|billion|trillion))?"
_QUANTITY = re.compile(rf"(?<![\w.]){_NUMBER}{_UNIT}(?!\w|\.\d)", re.I)
_HEDGED = re.compile(rf"\b(?P<hedge>{_HEDGE})\s+(?={_NUMBER})", re.I)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_RELATIVE_AFTER = re.compile(r"^\s+(?:(?:more|less)\s+likely|higher|lower|larger|smaller|"
                             r"(?:relative\s+(?:risk\s+)?)?(?:increase|decrease|reduction|growth))\b", re.I)
_RELATIVE_BEFORE = re.compile(rf"\b(?:increased?|decreased?|reduced?|rose|fell|grew|drop(?:ped)?)\s+(?:by\s+)?"
                              rf"(?:(?:{_HEDGE})\s+)?$", re.I)


def _figures(text: str) -> list[dict]:
    found = []
    for match in _QUANTITY.finditer(text):
        number = match["number"].replace(",", "")
        value = float(number)
        if match["percent"]:
            if match["percent"].lower().startswith("percentage point"):
                kind = "percentage_points"
            elif _RELATIVE_AFTER.match(text[match.end():]) or _RELATIVE_BEFORE.search(text[:match.start()]):
                kind = "relative_percent"
            else:
                kind = "percent"
        elif match["currency"]:
            kind = "currency"
        elif match["scale"] or "," in match["number"] or "." in match["number"]:
            kind = "count"
        elif 1500 <= value <= 2099 or len(number) < 2:
            continue                # a year or a single digit is not a quantity here
        else:
            kind = "count"
        if match["scale"]:
            value *= _SCALE[match["scale"].lower()]
        found.append({"text": match.group(0), "value": value, "kind": kind, "start": match.start(), "end": match.end()})
    return found


def _fits(relation: str, claimed: float, given: float) -> bool:
    if claimed <= 0 or given <= 0:
        return False
    close = abs(given - claimed) / max(given, claimed) <= APPROXIMATION
    return {"near": close, "below": close and given <= claimed, "above": close and given >= claimed,
            "at_least": given >= claimed, "at_most": given <= claimed}[relation]


def hedged_quantities(text: str) -> list[dict]:
    """Every hedged figure in a claim: the hedge, its relation, and the figure's exact span."""
    figures = {figure["start"]: figure for figure in _figures(text)}
    found = []
    for match in _HEDGED.finditer(text):
        figure = figures.get(match.end())
        if figure is not None:
            relation = next(kind for word, kind in _HEDGE_WORDS if word == match["hedge"].lower())
            found.append({"hedge": match["hedge"], "relation": relation, **figure})
    return found


def hedged_form(claim: str, sentence: str) -> tuple[str, list[dict]] | None:
    """The claim with each hedged figure replaced by the sentence's nearest fitting figure of its kind.

    None when nothing fits: the claim then reads as written. The record lists every
    substitution so a judgment made on the form can be traced to the figure it was read with.
    """
    # Hidden destinations and link titles cannot supply a quantity or hide its comparator.
    given = _figures(visible_text(sentence)[0])
    substitutions = []
    for hedged in hedged_quantities(claim):
        fitting = [figure for figure in given if figure["kind"] == hedged["kind"]
                   and _fits(hedged["relation"], hedged["value"], figure["value"])]
        if not fitting:
            continue
        nearest = min(fitting, key=lambda figure: abs(figure["value"] - hedged["value"]))
        substitutions.append({"hedge": hedged["hedge"], "relation": hedged["relation"], "claimed": hedged["text"],
                              "read_as": nearest["text"], "start": hedged["start"], "end": hedged["end"]})
    if not substitutions:
        return None
    form = claim
    for item in sorted(substitutions, key=lambda item: -item["start"]):
        form = form[:item["start"]] + item["read_as"] + form[item["end"]:]
    return form, [{key: item[key] for key in ("hedge", "relation", "claimed", "read_as")} for item in substitutions]
