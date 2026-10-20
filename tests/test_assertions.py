"""Assertions are exact spans of the claim; a corrective contrast is two of them."""

from __future__ import annotations

import pytest

from src.verdict.assertions import claim_assertions, compound_parts, negate_form, object_phrase, positive_form


def test_contrast_splits_into_two_exact_spans_with_polarity():
    claim = "We don't have a labor shortage. We have a good job shortage."
    rows = claim_assertions(claim)
    assert [row["text"] for row in rows] == ["We don't have a labor shortage.", "We have a good job shortage."]
    assert [row["id"] for row in rows] == ["assertion-1", "assertion-2"]
    assert [row["negated"] for row in rows] == [True, False]
    assert all(row["contrast"] for row in rows)
    assert all(row["text"] in claim for row in rows)


def test_plain_claim_stays_whole_and_unrewritten():
    rows = claim_assertions("The central bank raised its interest rate.")
    assert rows == [{"id": "assertion-1", "text": "The central bank raised its interest rate.",
                     "negated": False, "contrast": False}]
    denial = claim_assertions("The reactor was not shut down in May.")
    assert denial[0]["negated"] and not denial[0]["contrast"]


def test_empty_claim_is_rejected():
    with pytest.raises(ValueError):
        claim_assertions("   ")


def test_positive_form_is_a_mechanical_inversion_of_the_contrast_grammar_only():
    assert positive_form("We don't have a labor shortage.") == "We have a labor shortage."
    assert positive_form("We do not have a labor shortage.") == "We have a labor shortage."
    assert positive_form("There is no housing shortage") == "There is housing shortage"
    assert positive_form("It isn't a shortage of workers.") == "It is a shortage of workers."
    assert positive_form("We have a good job shortage.") is None
    assert positive_form("The reactor was not shut down in May.") is None
    assert positive_form("We don't have a labor shortage. We have a good job shortage.") is None


def test_object_phrase_is_an_exact_span_of_the_clause():
    assert object_phrase("We don't have a labor shortage.") == "a labor shortage"
    assert object_phrase("We have a good job shortage.") == "a good job shortage"
    assert object_phrase("There is no housing shortage") == "housing shortage"
    assert object_phrase("The central bank raised its interest rate.") is None


def test_negate_form_inserts_one_negation_after_the_first_copula_or_auxiliary():
    assert negate_form("Tilda Swinton is a British actress.") == "Tilda Swinton is not a British actress."
    assert negate_form("We have a good job shortage.") == "We have not a good job shortage."
    assert negate_form("The reactor was shut down in May") == "The reactor was not shut down in May"
    assert negate_form("Taxes will rise next year.") == "Taxes will not rise next year."
    assert negate_form("We don't have a labor shortage.") is None
    assert negate_form("There is no port.") is None
    assert negate_form("Kyoto hosts a festival.") is None


def test_a_compound_of_two_clauses_joined_by_but_is_two_exact_assertions():
    claim = ("As of July, the unemployment rate stood at 4.1% but according to data from the Ludwig Institute, "
             "the true rate of unemployment was actually at 24.9%.")
    rows = claim_assertions(claim)
    assert [row["text"] for row in rows] == ["As of July, the unemployment rate stood at 4.1%",
                                             "according to data from the Ludwig Institute, the true rate of "
                                             "unemployment was actually at 24.9%."]
    assert all(row["contrast"] and row["text"] in claim for row in rows)
    for whole in ("Inflation is falling but still too high.", "The council voted to close the library.",
                  "He should have had him out, but Putin's probably asking for billions.",
                  "Not really, but the council voted to close the library."):
        assert compound_parts(whole) == () and len(claim_assertions(whole)) == 1, whole
