"""The train/serve contract: packing, grouping, and the variants' evidence selection."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.verdict.encode import (
    FEVER_TO_LABEL,
    LABELS,
    MAX_SENTENCE_WORDS,
    build_input,
    pack,
    readable_title,
    render,
    select_evidence,
    shuffle_pages,
)
from src.verdict.labels import FEVER, Verdict


def words(first: str, second: str) -> int:
    """Stand-in for a tokenizer; monotone in length, which is all pack() requires."""
    return len(first.split()) + len(second.split())


def ev(title: str, index: int, text: str):
    return [title, index, text]


def sentences(n: int, each: int = 5, title: str = "Page_A"):
    return [ev(title, i, " ".join(f"w{i}_{j}" for j in range(each))) for i in range(n)]


def test_label_literal_matches_the_label_space():
    """encode.py cannot import src on Kaggle, so the literal is checked here instead."""
    assert LABELS == tuple(v.value for v in FEVER.verdicts)
    assert {FEVER_TO_LABEL[k] for k in FEVER.native} == set(LABELS)
    for native, mapped in FEVER_TO_LABEL.items():
        assert FEVER.to_verdict(native) == Verdict(mapped)


def test_the_shipped_copy_is_byte_identical():
    """The notebook imports the copy; a drift here is a silent train/serve skew."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "verdict" / "encode.py").read_bytes() == (
        root / "notebooks" / "encode_spec.py"
    ).read_bytes()


def test_encode_does_not_import_src():
    """There is no src package on Kaggle; an import here breaks every notebook."""
    source = (Path(__file__).resolve().parents[1] / "src" / "verdict" / "encode.py").read_text()
    assert "import src" not in source and "from src" not in source


def test_titles_are_rendered_as_prose():
    assert readable_title("Soul_Food_-LRB-film-RRB-") == "Soul Food (film)"


def test_consecutive_sentences_share_one_title():
    text = render(sentences(3))
    assert text.count("Page A:") == 1


def test_a_new_page_starts_a_new_title():
    evidence = sentences(2, title="Page_A") + sentences(2, title="Page_B")
    text = render(evidence)
    assert text.count("Page A:") == 1 and text.count("Page B:") == 1


def test_a_returning_page_repeats_its_title():
    """Grouping is by run, not by set -- interleaved pages must stay attributable."""
    evidence = [ev("A", 0, "one"), ev("B", 0, "two"), ev("A", 1, "three")]
    assert render(evidence).count("A:") == 2


def test_overlong_sentences_are_clipped():
    long = " ".join(f"w{i}" for i in range(MAX_SENTENCE_WORDS + 40))
    rendered = render([ev("A", 0, long)])
    assert len(rendered.split()) == MAX_SENTENCE_WORDS + 1   # + the title token


def test_pack_is_monotone_in_budget():
    evidence = sentences(20)
    counts = [pack("c", evidence, budget, words) for budget in range(0, 200, 10)]
    assert counts == sorted(counts)


def test_pack_returns_a_prefix_that_fits_and_one_more_that_does_not():
    evidence = sentences(20)
    n = pack("c", evidence, 40, words)
    assert 0 < n < len(evidence)
    assert words(*build_input("c", evidence[:n])) <= 40
    assert words(*build_input("c", evidence[: n + 1])) > 40


def test_pack_handles_empty_and_impossible_budgets():
    assert pack("c", [], 100, words) == 0
    assert pack("c", sentences(5), 0, words) == 0


def test_claim_only_sees_no_evidence():
    row = {"claim": "a claim", "evidence": sentences(10), "gold": [ev("A", 0, "g")]}
    assert select_evidence(row, "claim_only", 500, words) == []


def test_retrieved_takes_a_packed_prefix():
    row = {"claim": "a claim", "evidence": sentences(20)}
    chosen = select_evidence(row, "retrieved", 40, words)
    assert chosen == row["evidence"][: len(chosen)]
    assert 0 < len(chosen) < 20


def test_gold_is_always_present_when_it_fits():
    gold = [ev("Gold_Page", 7, "the gold sentence")]
    row = {"claim": "a claim", "evidence": sentences(20), "gold": gold}
    chosen = select_evidence(row, "gold", 60, words)
    assert ("Gold_Page", 7) in {(c[0], c[1]) for c in chosen}


def test_gold_rows_are_budget_matched_against_retrieved():
    """
    The failure this exists to prevent: gold-only for verifiable and fifteen retrieved sentences
    for NOT ENOUGH INFO teaches the model to count sentences, and the oracle then scores near
    perfectly while measuring nothing.
    """
    budget = 60
    verifiable = {"claim": "a claim", "evidence": sentences(20), "gold": [ev("Gold_Page", 7, "the gold sentence")]}
    nei = {"claim": "a claim", "evidence": sentences(20), "gold": []}

    n_verifiable = len(select_evidence(verifiable, "gold", budget, words))
    n_nei = len(select_evidence(nei, "gold", budget, words))
    assert abs(n_verifiable - n_nei) <= 1


def test_gold_is_not_duplicated_by_the_retrieved_fill():
    gold = [ev("Page_A", 3, "w3_0 w3_1 w3_2 w3_3 w3_4")]
    row = {"claim": "a claim", "evidence": sentences(10), "gold": gold}
    chosen = select_evidence(row, "gold", 200, words)
    refs = [(c[0], c[1]) for c in chosen]
    assert len(refs) == len(set(refs))


def test_a_claim_without_gold_keeps_the_retrieved_condition():
    row = {"claim": "a claim", "evidence": sentences(20), "gold": []}
    assert select_evidence(row, "gold", 60, words) == select_evidence(row, "retrieved", 60, words)


def test_the_shuffle_reorders_without_changing_the_set():
    row = {"claim": "a claim", "evidence": sentences(20), "gold": [ev("Gold_Page", 7, "gold")]}
    plain = select_evidence(row, "gold", 80, words)
    shuffled = select_evidence(row, "gold", 80, words, rng=random.Random(0))
    assert {(c[0], c[1]) for c in plain} == {(c[0], c[1]) for c in shuffled}


def test_an_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        select_evidence({"claim": "a claim", "evidence": []}, "oracle", 100, words)


def test_packing_counts_the_claim_against_the_budget():
    """
    The claim shares the sequence with the evidence. Packing against an empty first segment
    overruns by the claim's length on every example.
    """
    evidence = sentences(20)
    short = pack("c", evidence, 60, words)
    long_claim = pack(" ".join(f"q{i}" for i in range(20)), evidence, 60, words)
    assert long_claim < short


def test_a_packed_prefix_never_exceeds_the_budget():
    evidence = sentences(20)
    for budget in range(5, 120, 7):
        n = pack("a claim here", evidence, budget, words)
        assert words(*build_input("a claim here", evidence[:n])) <= budget


def test_shuffling_pages_does_not_change_the_token_count():
    """
    Shuffling rows breaks render()'s consecutive-title runs, adding one title per break. Measured
    at +12 tokens on a three-page fixture, which can push a packed set over its budget where the
    tokenizer truncates in silence.
    """
    rows = [ev(f"Page_{p}", i, f"s{p}{i} w w w w") for p in "ABC" for i in range(4)]
    shuffled = shuffle_pages(rows, random.Random(0))
    assert sorted(map(tuple, shuffled)) == sorted(map(tuple, rows))
    assert words(*build_input("c", shuffled)) == words(*build_input("c", rows))


def test_shuffling_pages_keeps_each_page_contiguous():
    rows = [ev(f"Page_{p}", i, "w w") for p in "ABC" for i in range(3)]
    shuffled = shuffle_pages(rows, random.Random(1))
    titles = [r[0] for r in shuffled]
    assert len(set(titles)) == 3
    # One run per page, so no title is ever reintroduced after another has started.
    runs = [t for i, t in enumerate(titles) if i == 0 or titles[i - 1] != t]
    assert len(runs) == len(set(runs))


def test_gold_rows_stay_within_budget_after_shuffling():
    """The regression this pair of fixes exists for."""
    rows = [ev(f"Page_{p}", i, f"s{p}{i} w w w w") for p in "ABC" for i in range(4)]
    row = {"claim": "a claim", "evidence": rows, "gold": [ev("Page_A", 0, "sA0 w w w w")]}
    for budget in range(20, 90, 6):
        chosen = select_evidence(row, "gold", budget, words, rng=random.Random(0))
        assert words(*build_input(row["claim"], chosen)) <= budget or len(chosen) == 1
