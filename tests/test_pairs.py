"""Pair construction: gold single sentences count, joint groups do not, and no group crosses a split."""

from __future__ import annotations

import random

import pytest

from src.data.fever import Claim, claim_key
from src.data.mnli import NliPair
from src.verdict.pairs import Pair, fever_pairs, mnli_pairs, negated, negation_augment, validate

PAGES = {
    "Tilda_Swinton": {0: "Tilda Swinton is a British actress.", 1: "She was born in 1960.", 2: "She won an Oscar."},
    "Edinburgh": {0: "Edinburgh is the capital of Scotland.", 1: "It hosts a festival."},
    "Kyoto": {0: "Kyoto is a city in Japan.", 1: "It was the imperial capital."},
    "Mars": {0: "Mars is the fourth planet.", 1: "It has two moons."},
}


class Lookup:
    def text(self, title, index):
        return PAGES.get(title, {}).get(index)


def claim(identity, label, text, groups, retrieved):
    parsed = tuple(frozenset(group) for group in groups)
    pages = tuple(sorted({title for group in parsed for title, _ in group}))
    return Claim(id=identity, label=label, text=text, key=claim_key(text), groups=parsed, pages=pages), retrieved


CLAIMS = [
    claim(1, "SUPPORTS", "Tilda Swinton is an actress.", [[("Tilda_Swinton", 0)]],
          [["Tilda_Swinton", 0], ["Tilda_Swinton", 1], ["Tilda_Swinton", 2]]),
    claim(2, "REFUTES", "Tilda Swinton was not born in 1960.", [[("Tilda_Swinton", 1)], [("Tilda_Swinton", 0), ("Tilda_Swinton", 1)]],
          [["Tilda_Swinton", 1], ["Tilda_Swinton", 2], ["Edinburgh", 1]]),
    claim(3, "SUPPORTS", "Edinburgh hosts a festival in its capital.", [[("Edinburgh", 0), ("Edinburgh", 1)]],
          [["Edinburgh", 0], ["Edinburgh", 1]]),
    claim(4, "NOT ENOUGH INFO", "Kyoto has a large port.", [], [["Kyoto", 0], ["Kyoto", 1]]),
    claim(5, "SUPPORTS", "Mars is the fourth planet.", [[("Mars", 0)]], [["Mars", 0], ["Mars", 1]]),
    claim(6, "SUPPORTS", "Tilda Swinton is an actress.", [[("Tilda_Swinton", 0)]], [["Tilda_Swinton", 0]]),
]


def build(seed=0, **kwargs):
    claims = [row for row, _ in CLAIMS]
    retrieved = {row.id: refs for row, refs in CLAIMS}
    return fever_pairs(claims, retrieved, Lookup(), rng=random.Random(seed), **kwargs)


def test_gold_single_sentences_become_states_and_joint_groups_do_not():
    pairs = build()
    by = {(pair.provenance["claim_id"], pair.relation): pair for pair in pairs}
    assert by[(1, "states")].premise == "Tilda Swinton is an actress." or by[(1, "states")].premise == PAGES["Tilda_Swinton"][0]
    assert by[(2, "states_negation")].premise == "She was born in 1960." and by[(2, "states_negation")].negated_hypothesis
    assert not any(pair.provenance["claim_id"] == 3 and pair.relation in ("states", "states_negation") for pair in pairs)
    assert by[(4, "bears_on")].premise in PAGES["Kyoto"].values()
    assert not by[(1, "states")].negated_hypothesis


def test_bears_on_never_uses_a_gold_sentence_even_from_a_joint_group():
    for seed in range(8):
        for pair in build(seed=seed, bears_per_claim=3):
            if pair.relation != "bears_on":
                continue
            claim_id = pair.provenance["claim_id"]
            gold = {(title, index) for row, _ in CLAIMS if row.id == claim_id for group in row.groups for title, index in group}
            assert (pair.provenance["title"], pair.provenance["index"]) not in gold


def test_unrelated_comes_from_pages_the_claim_never_touched():
    for seed in range(8):
        pairs = build(seed=seed, unrelated_per_claim=2)
        for pair in pairs:
            if pair.relation != "unrelated":
                continue
            claim_id = pair.provenance["claim_id"]
            own = {row.id: {title for title, _ in refs} | set(row.pages) for row, refs in CLAIMS}[claim_id]
            assert pair.provenance["title"] not in own
            assert pair.provenance["donor_claim_id"] != claim_id


def test_repeated_claim_text_does_not_repeat_a_pair_and_shares_a_group():
    pairs = build()
    keys = [(pair.premise, pair.hypothesis) for pair in pairs]
    assert len(keys) == len(set(keys))
    groups = {pair.provenance["claim_id"]: pair.group for pair in pairs}
    assert groups[1] == groups[6]


def test_missing_retrieval_is_an_error_not_a_skip():
    with pytest.raises(ValueError):
        fever_pairs([CLAIMS[0][0]], {}, Lookup(), rng=random.Random(0))


def rows():
    return [NliPair("p1e", "P1", "fiction", "entailment", "The cat sat on the mat.", "A cat was sitting."),
            NliPair("p1n", "P1", "fiction", "neutral", "The cat sat on the mat.", "The cat was tired."),
            NliPair("p1c", "P1", "fiction", "contradiction", "The cat sat on the mat.", "No cat was there."),
            NliPair("p2e", "P2", "government", "entailment", "Taxes rose in 2019.", "Taxes increased."),
            NliPair("p3n", "P3", "travel", "neutral", "The hotel is by the sea.", "The hotel is expensive.")]


def test_mnli_relations_map_and_cross_genre_swaps_are_unrelated():
    pairs = mnli_pairs(rows(), rng=random.Random(1), unrelated_every=1)
    native = {pair.provenance["pair_id"]: pair for pair in pairs if "donor_pair_id" not in pair.provenance}
    assert native["p1e"].relation == "states" and native["p1c"].relation == "states_negation"
    assert native["p1n"].relation == "bears_on" and native["p1c"].negated_hypothesis
    assert {pair.group for pair in pairs if pair.provenance["pair_id"].startswith("p1")} == {"mnli:P1"}
    swapped = [pair for pair in pairs if "donor_pair_id" in pair.provenance]
    assert swapped and all(pair.relation == "unrelated" for pair in swapped)
    assert all(pair.provenance["donor_genre"] != pair.provenance["genre"] for pair in swapped)
    assert all(pair.premise != next(r.premise for r in rows() if r.pair_id == pair.provenance["pair_id"]) for pair in swapped)
    assert len(mnli_pairs(rows(), rng=random.Random(1), unrelated_every=0)) == 5
    copied = rows() + [NliPair("p4e", "P4", "fiction", "entailment", "Same words.", "Same words.")]
    assert len(mnli_pairs(copied, rng=random.Random(1), unrelated_every=0)) == 5


def test_validate_rejects_split_crossing_groups_and_duplicates():
    a = Pair("x1", "fever", "states", "s", "h", "g1", False)
    b = Pair("x2", "fever", "bears_on", "s2", "h", "g1", False)
    with pytest.raises(ValueError, match="g1"):
        validate({"train": [a], "test": [b]})
    with pytest.raises(ValueError, match="duplicate"):
        validate({"train": [a, Pair("x3", "fever", "states", "s", "h", "g1", False)]})
    with pytest.raises(ValueError, match="unknown relation"):
        validate({"train": [Pair("x4", "fever", "proves", "s", "h", "g1", False)]})
    with pytest.raises(ValueError, match="identical"):
        validate({"train": [Pair("x5", "fever", "states", "same", "same", "g1", False)]})
    counts = validate({"train": [a, b], "test": [Pair("y", "mnli", "unrelated", "p", "q", "g9", True)]})
    assert counts == {"train": {"fever:bears_on": 1, "fever:states": 1}, "test": {"mnli:unrelated": 1}}


def test_negation_cue_is_a_closed_class():
    assert negated("Taxes did not rise.") and negated("There is no port.") and negated("It can't fly.")
    assert not negated("Notably, taxes rose.") and not negated("The knot held.")


def test_negation_augment_flips_states_keeps_other_labels_and_never_touches_denials():
    base = [Pair("a", "fever", "states", "Tilda is an actress.", "Tilda Swinton is an actress.", "g1", False),
            Pair("b", "fever", "bears_on", "She won an Oscar.", "Tilda Swinton is an actress.", "g1", False),
            Pair("c", "fever", "unrelated", "Mars is red.", "Tilda Swinton is an actress.", "g1", False),
            Pair("d", "fever", "states_negation", "She was born in 1960.", "Tilda was not born in 1960.", "g2", True),
            Pair("e", "mnli", "states", "It rained.", "Rain fell.", "g3", False),
            Pair("f0", "fever", "states_negation", "She was born in 1960.", "Tilda Swinton was born in 1970.", "g2", False),
            Pair("g0", "mnli", "bears_on", "Both plans cost money.", "Neither plan is cheap.", "g4", True)]
    added = negation_augment(base, rng=random.Random(0), rate=1.0)
    by_id = {pair.id: pair for pair in added}
    assert by_id["a:neg"].relation == "states_negation" and by_id["a:neg"].hypothesis == "Tilda Swinton is not an actress."
    assert by_id["b:neg"].relation == "bears_on" and by_id["c:neg"].relation == "unrelated"
    assert all(pair.negated_hypothesis and pair.group in ("g1",) for pair in added)
    assert "d:neg" not in by_id, "a refuted denial is never un-negated"
    assert "f0:neg" not in by_id, "a refuted claim is never negated either: its gold need not support the negation"
    assert "g0:neg" not in by_id, "a hypothesis the pair regex reads as negated is left alone"
    assert "e:neg" not in by_id, "no copula to hang a negation on"
    assert negation_augment(base, rng=random.Random(0), rate=0.0) == []
    clash = base + [Pair("f", "fever", "states_negation", "Tilda is an actress.", "Tilda Swinton is not an actress.", "g1", True)]
    assert "a:neg" not in {pair.id for pair in negation_augment(clash, rng=random.Random(0), rate=1.0)}
    held = {("Tilda is an actress.", "Tilda Swinton is not an actress.")}
    assert "a:neg" not in {pair.id for pair in negation_augment(base, rng=random.Random(0), rate=1.0, exclude=held)}
    with pytest.raises(ValueError):
        negation_augment(base, rng=random.Random(0), rate=1.5)
