"""Literal text references preserve source spans; parsing never establishes evidence."""

from __future__ import annotations

import pytest

from src.retrieval.visible_text import inline_links, visible_text


@pytest.mark.parametrize("label,destination", [
    ("Water survey: 24.3% of households", "https://example.org/water/(samples)/2024"),
    ("Battery capacity (not output)", "https://example.org/(energy(storage))/report"),
    ("Housing definition", r"https://example.org/definition\(housing\)?edition=2&literal=%28"),
    ("River measurements", r"https://example.org/escaped\)not-the-end"),
])
def test_complete_references_keep_exact_labels_destinations_and_unicode_source_spans(label, destination):
    markup = f"[{label}]({destination})"
    text = "Résumé — “quoted context”\n" + markup + " followed by the original qualification."
    original = text
    links = inline_links(text)
    assert links == [{"label": label, "destination": destination,
                      "start": text.index("["), "end": text.index("[") + len(markup)}]
    assert text == original
    assert text[links[0]["start"]:links[0]["end"]] == markup


def test_adjacent_references_keep_order_and_do_not_consume_outer_parentheses():
    text = "([survey](https://example.org/a))[definition](../definitions#section)"
    links = inline_links(text)
    assert [(row["label"], row["destination"]) for row in links] == [
        ("survey", "https://example.org/a"), ("definition", "../definitions#section")]
    assert text[links[0]["end"]] == ")"
    assert links[1]["end"] == len(text)


@pytest.mark.parametrize("text", [
    "[survey](https://example.org/unclosed",
    "[survey](https://example.org/(unclosed)",
    "[survey](https://example.org/escaped\\)",
    "[survey](https://example.org/trailing\\",
    "[good](https://example.org/good) then [broken](https://example.org/(unfinished)",
])
def test_malformed_destinations_fail_closed_without_yielding_a_partial_url(text):
    assert inline_links(text) == []
    assert visible_text(text) == ("", 0)


@pytest.mark.parametrize("text", [
    "![water chart](https://example.org/chart.png)",
    "[![battery chart](https://example.org/chart.png)](https://example.org/article)",
    r"\[escaped label](https://example.org/escaped)",
    "[[nested label](https://example.org/unsupported)](https://example.org/outer)",
    "](https://example.org/orphan)",
    "[reference][footnote]",
])
def test_images_escaped_openers_and_unsupported_labels_are_not_text_references(text):
    assert inline_links(text) == []


def test_an_image_does_not_hide_a_later_text_reference_or_expose_its_own_destination():
    text = "![diagram](https://example.org/chart(a).png) and [survey](https://example.org/report)."
    assert inline_links(text) == [{"label": "survey", "destination": "https://example.org/report",
                                  "start": text.index("[survey]"), "end": len(text) - 1}]
    assert visible_text(text) == ("!diagram and survey.", len("diagram") + len("survey"))


@pytest.mark.parametrize("prefix", ["", "!"])
@pytest.mark.parametrize("outer_end", ["](https://example.org/outer)", "]"])
def test_unsupported_enclosing_labels_cannot_expose_later_nested_references(prefix, outer_end):
    text = prefix + "[outer [first](https://example.org/first) and [second](https://example.org/second)" + outer_end
    assert inline_links(text) == []


@pytest.mark.parametrize("prefix", ["", "!"])
def test_skipping_an_entire_nested_construct_preserves_the_next_unrelated_reference(prefix):
    nested = prefix + "[outer [first](https://example.org/first) and [second](https://example.org/second)](https://example.org/outer)"
    text = nested + " [actual](https://example.org/actual)"
    assert inline_links(text) == [{"label": "actual", "destination": "https://example.org/actual",
                                  "start": len(nested) + 1, "end": len(text)}]


def test_escaped_image_marker_is_literal_punctuation_before_a_text_reference():
    text = r"\![survey](https://example.org/report)"
    assert inline_links(text)[0]["start"] == 2
    assert inline_links(text)[0]["label"] == "survey"
    assert inline_links(r"\\![survey](https://example.org/report)") == []


def test_escaped_closing_bracket_is_not_a_label_terminator_and_preserves_later_references():
    malformed = r"[a\](https://example.org/not-a-link)"
    assert inline_links(malformed) == []
    text = malformed + " and [real](https://example.org/real)"
    assert inline_links(text) == [{"label": "real", "destination": "https://example.org/real",
                                  "start": len(malformed) + 5, "end": len(text)}]
    text = r"[a\\](https://example.org/real)"
    assert inline_links(text) == [{"label": r"a\\", "destination": "https://example.org/real",
                                  "start": 0, "end": len(text)}]


def test_destination_syntax_does_not_normalize_or_authorize_transport():
    destination = "https://User:Pass@EXAMPLE.org:443/a%2Fb?term=a&term=b#part"
    assert inline_links(f"[literal]({destination})")[0]["destination"] == destination


@pytest.mark.parametrize("text,expected", [
    ("No references here. https://example.org/hidden", ("No references here. ", 0)),
    (r"\[escaped](https://example.org/x)", ("\\escaped", len("escaped"))),
    ("![diagram](https://example.org/chart)", ("!diagram", len("diagram"))),
    ("[[nested](https://example.org/x)](https://example.org/y)", ("[nested](", len("[nested"))),
])
def test_shared_scanner_preserves_existing_visible_text_behavior(text, expected):
    assert visible_text(text) == expected
