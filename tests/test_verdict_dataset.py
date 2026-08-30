"""The gate that has to catch a bad dataset before it reaches a GPU."""

from __future__ import annotations

import gzip
import json
import sqlite3

import pytest

from src.data.fever import NOT_ENOUGH_INFO, Claim, claim_key
from src.retrieval import wiki
from src.retrieval.text import title_key, title_norm
from src.verdict.dataset import (
    Row,
    SentenceStore,
    assert_splits_disjoint,
    build_rows,
    read_rows,
    sha256,
    smallest_gold,
    validate,
    write_rows,
)

DECOMPOSED = "Besiktas\u0327_J.K."   # s + combining cedilla, as FEVER releases it
COMPOSED = title_key(DECOMPOSED)            # as the store holds it

PAGES = [
    ("Page_A", "0\tfirst sentence\n2\tthird sentence"),   # index 1 dropped: gaps are real
    ("Page_B", "0\tbee one\n1\tbee two"),
    (COMPOSED, "0\tthe composed page"),
]


@pytest.fixture
def store(tmp_path):
    conn = sqlite3.connect(tmp_path / "wiki.sqlite3")
    wiki.create_tables(conn)
    conn.executemany(
        "INSERT INTO pages (doc_id, title, norm, sentences) VALUES (?, ?, ?, ?)",
        [(i, title_key(t), title_norm(t), s) for i, (t, s) in enumerate(PAGES)],
    )
    wiki.create_indexes(conn)
    conn.commit()
    return SentenceStore(conn)


def claim(id_, label, text="a claim", groups=()):
    return Claim(id=id_, label=label, text=text, key=claim_key(text), groups=tuple(groups), pages=())


def test_sentences_resolve_by_index_not_position(store):
    """
    The store drops empty sentences, so Page_A holds indices 0 and 2. Resolving by list position
    would return "third sentence" for index 1 -- a different, entirely plausible sentence, with
    nothing raising and the model training on mismatched evidence.
    """
    assert store.text("Page_A", 0) == "first sentence"
    assert store.text("Page_A", 2) == "third sentence"
    assert store.text("Page_A", 1) is None


def test_a_decomposed_title_resolves_against_the_composed_store(store):
    """170 of 14,533 gold titles are NFD; an unnormalised lookup silently loses every one."""
    assert DECOMPOSED != COMPOSED
    assert store.text(DECOMPOSED, 0) == "the composed page"


def test_resolve_composes_titles_so_dedup_can_work(store):
    """encode's gold/fill dedup compares titles, so both sides must arrive in one form."""
    rows = store.resolve([(DECOMPOSED, 0), (COMPOSED, 0)])
    assert {r[0] for r in rows} == {COMPOSED}


def test_unresolvable_refs_are_dropped_not_faked(store):
    assert store.resolve([("Page_A", 1), ("Nonexistent", 0), ("Page_B", 0)]) == [
        ("Page_B", 0, "bee one")
    ]


def test_smallest_gold_prefers_the_cheapest_group(store):
    c = claim(1, "SUPPORTS", groups=[
        frozenset({("Page_A", 0), ("Page_A", 2)}),
        frozenset({("Page_B", 1)}),
    ])
    gold, resolved = smallest_gold(c, store)
    assert resolved and [g[:2] for g in gold] == [("Page_B", 1)]


def test_smallest_gold_falls_through_an_unresolvable_group(store):
    c = claim(1, "SUPPORTS", groups=[
        frozenset({("Page_A", 1)}),                       # index 1 does not exist
        frozenset({("Page_B", 0), ("Page_B", 1)}),
    ])
    gold, resolved = smallest_gold(c, store)
    assert resolved and len(gold) == 2


def test_a_claim_whose_gold_never_resolves_is_flagged(store):
    """Silently keeping it would train the oracle on retrieved evidence and call it gold."""
    c = claim(1, "SUPPORTS", groups=[frozenset({("Page_A", 1)})])
    gold, resolved = smallest_gold(c, store)
    assert gold == [] and resolved is False


def test_build_rows_maps_labels_and_resolves_text(store):
    claims = [claim(1, "SUPPORTS", groups=[frozenset({("Page_B", 0)})])]
    rows = list(build_rows(claims, {1: [["Page_A", 0], ["Page_B", 1]]}, store))
    assert rows[0].label == "supported"
    assert rows[0].evidence == (("Page_A", 0, "first sentence"), ("Page_B", 1, "bee two"))
    assert rows[0].gold == (("Page_B", 0, "bee one"),)


def test_nei_carries_no_gold_and_is_not_flagged(store):
    rows = list(build_rows([claim(1, NOT_ENOUGH_INFO)], {1: [["Page_A", 0]]}, store))
    assert rows[0].label == "not_enough_evidence"
    assert rows[0].gold == () and rows[0].gold_resolved is True


def test_a_claim_without_retrieval_raises(store):
    with pytest.raises(ValueError, match="no retrieved evidence"):
        list(build_rows([claim(1, "SUPPORTS")], {}, store))


def row(id_=1, label="supported", evidence=(("Page_A", 0, "x"),), gold=(), resolved=True):
    return Row(id=id_, label=label, claim="a claim", evidence=evidence, gold=gold, gold_resolved=resolved)


def test_validate_accepts_a_well_formed_split():
    claims = {1: claim(1, "SUPPORTS", groups=[frozenset({("Page_A", 0)})])}
    validate([row(gold=(("Page_A", 0, "x"),))], claims, split="train")


def test_validate_rejects_a_duplicated_claim():
    # Otherwise valid rows, so the duplicate is what fails rather than the gold check upstream.
    claims = {1: claim(1, "SUPPORTS", groups=[frozenset({("Page_A", 0)})])}
    good = row(gold=(("Page_A", 0, "x"),))
    with pytest.raises(ValueError, match="appears twice"):
        validate([good, good], claims, split="train")


def test_validate_rejects_a_claim_not_in_the_split():
    with pytest.raises(ValueError, match="not in the split"):
        validate([row(id_=9)], {1: claim(1, "SUPPORTS")}, split="train")


def test_validate_rejects_a_mismatched_label():
    with pytest.raises(ValueError, match="does not match"):
        validate([row(label="contradicted")], {1: claim(1, "SUPPORTS")}, split="train")


def test_validate_rejects_an_empty_evidence_list():
    """A row with no evidence trains the model on nothing and reports as an ordinary miss."""
    with pytest.raises(ValueError, match="has no evidence"):
        validate([row(evidence=())], {1: claim(1, "SUPPORTS")}, split="train")


def test_validate_rejects_gold_from_another_claim():
    """An oracle reserving someone else's evidence is an oracle for a different question."""
    claims = {1: claim(1, "SUPPORTS", groups=[frozenset({("Page_A", 0)})])}
    with pytest.raises(ValueError, match="outside its own groups"):
        validate([row(gold=(("Page_B", 1, "x"),))], claims, split="train")


def test_validate_accepts_decomposed_gold_against_composed_groups():
    """Groups hold FEVER's raw NFD title; the row holds the composed one. Both must pass."""
    claims = {1: claim(1, "SUPPORTS", groups=[frozenset({(DECOMPOSED, 0)})])}
    validate([row(gold=((COMPOSED, 0, "x"),))], claims, split="train")


def test_validate_rejects_a_verifiable_claim_with_no_gold():
    with pytest.raises(ValueError, match="carries no gold"):
        validate([row(gold=(), resolved=True)], {1: claim(1, "SUPPORTS")}, split="train")


def test_an_unresolved_verifiable_claim_is_allowed_through_flagged():
    validate([row(gold=(), resolved=False)], {1: claim(1, "SUPPORTS")}, split="train")


def test_splits_must_not_share_a_claim_id():
    claims = {"train": {1: claim(1, "SUPPORTS")}, "test": {1: claim(1, "SUPPORTS")}}
    with pytest.raises(ValueError, match="is in both"):
        assert_splits_disjoint({"train": [row()], "test": [row()]}, claims)


def test_splits_must_not_share_a_claim_key():
    """
    Different ids, same text. drop_leaked already removes these upstream, but this is the last
    point before the data leaves for a GPU and a leak found later invalidates every number.
    """
    claims = {
        "train": {1: claim(1, "SUPPORTS", text="same wording")},
        "test": {2: claim(2, "SUPPORTS", text="same wording")},
    }
    with pytest.raises(ValueError, match="spans"):
        assert_splits_disjoint({"train": [row(id_=1)], "test": [row(id_=2)]}, claims)


def test_disjoint_splits_pass():
    claims = {
        "train": {1: claim(1, "SUPPORTS", text="one")},
        "test": {2: claim(2, "SUPPORTS", text="two")},
    }
    assert_splits_disjoint({"train": [row(id_=1)], "test": [row(id_=2)]}, claims)


def test_rows_round_trip_through_the_written_file(tmp_path):
    rows = [row(id_=1, gold=(("Page_A", 0, "x"),)), row(id_=2, resolved=False)]
    path = tmp_path / "rows.jsonl.gz"
    digest = write_rows(rows, path)
    assert list(read_rows(path)) == rows
    assert digest == sha256(path)


def test_the_written_file_is_gzipped_jsonl(tmp_path):
    """The notebook reads it directly, so the on-disk shape is part of the contract."""
    path = tmp_path / "rows.jsonl.gz"
    write_rows([row()], path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline())["id"] == 1


def test_the_digest_covers_the_whole_file(tmp_path):
    """A capped hash would collide across files sharing a long prefix."""
    shared = "x" * (1 << 20)
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_text(shared + "one")
    second.write_text(shared + "two")
    assert sha256(first) != sha256(second)
