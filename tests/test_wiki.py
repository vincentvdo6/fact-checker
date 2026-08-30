"""Wiki corpus parsing and the page store: index fidelity, dedup, and title normalization."""

from __future__ import annotations

import json
import unicodedata
from contextlib import closing
from pathlib import Path

import pytest

from scripts.build_wiki import Stats, build
from src.retrieval import wiki
from src.retrieval.text import tokenize

requires_store = pytest.mark.skipif(
    not wiki.DB_PATH.exists(),
    reason="wiki store absent; run python -m scripts.build_wiki",
)


def page(page_id: str, lines: str = "") -> dict[str, str]:
    # `text` is written but never read: the parser must take sentences from `lines` alone.
    return {"id": page_id, "text": "ignored", "lines": lines}


def store(tmp_path: Path, records: list[dict[str, str]]) -> tuple[Path, Stats]:
    shard = tmp_path / "wiki-001.jsonl"
    with open(shard, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    db = tmp_path / "wiki.sqlite3"
    return db, build([shard], db)


def test_index_comes_from_the_record_not_the_position():
    """The trap this pins: enumerate() would call the second sentence 1 instead of 2."""
    assert wiki.parse_lines("0\tA\n2\tC") == [(0, "A"), (2, "C")]


def test_non_integer_index_is_skipped_without_shifting_the_rest():
    lines = "0\tA\nWarwick Farm\tWarwick Farm Raceway\n1\tB"
    rows, malformed = wiki.scan_lines(lines)
    assert rows == [(0, "A"), (1, "B")]
    assert malformed == 1


def test_blank_sentences_are_dropped_and_neighbours_keep_their_indices():
    assert wiki.parse_lines("0\tA\n1\t\n2\t   \n3\tD") == [(0, "A"), (3, "D")]


def test_hyperlink_columns_are_stripped():
    lines = "0\tThe 1986 NBA Finals .\tNBA\tNational Basketball Association\t1981\t1981 NBA Finals"
    assert wiki.parse_lines(lines) == [(0, "The 1986 NBA Finals .")]


def test_empty_lines_parses_to_nothing():
    assert wiki.scan_lines("") == ([], 0)


def test_a_record_with_no_tab_is_not_a_sentence():
    assert wiki.parse_lines("0\tA\n1") == [(0, "A")]


def test_format_sentences_round_trips_through_parse_lines():
    rows = [(0, "First ."), (3, "Fourth , with a comma .")]
    assert wiki.parse_lines(wiki.format_sentences(rows)) == rows


def test_repeated_empty_ids_and_empty_lines_survive_the_unique_index(tmp_path):
    """The dump opens with an empty-id record; a second one would break UNIQUE NOT NULL."""
    records = [
        page(""),
        page("Real_Page", "0\tA sentence ."),
        page(""),
        page("Blank_Page"),
    ]
    db, stats = store(tmp_path, records)

    assert stats.empty_id == 2
    assert stats.pages == 2
    assert stats.blank_pages == 1
    with closing(wiki.connect(db)) as conn:
        assert wiki.page_count(conn) == 2
        assert wiki.sentences(conn, wiki.doc_id(conn, "Blank_Page")) == []


def test_duplicate_titles_keep_the_first_occurrence(tmp_path):
    db, stats = store(tmp_path, [page("Dup", "0\tfirst ."), page("Dup", "0\tsecond .")])

    assert stats.duplicate_titles == 1
    with closing(wiki.connect(db)) as conn:
        assert wiki.sentences(conn, 0) == [(0, "first .")]


def test_doc_ids_are_dense_from_zero_even_with_skipped_records(tmp_path):
    """doc_id is the BM25 column index, so a gap would misalign an already-built matrix."""
    db, _ = store(tmp_path, [page(""), page("A", "0\ta ."), page("A", "0\tdup ."), page("B", "0\tb .")])

    with closing(wiki.connect(db)) as conn:
        assert [row[0] for row in wiki.iter_pages(conn)] == [0, 1]
        assert [row[1] for row in wiki.iter_pages(conn)] == ["A", "B"]


def test_an_nfd_gold_title_resolves_to_the_nfc_dump_title(tmp_path):
    """
    The finding this guards: dump titles are NFC, gold evidence titles are NFD, and 170 of
    14,533 differ. Without title_key on the stored side the lookup misses and nothing raises.
    """
    composed = unicodedata.normalize("NFC", "Beyoncé_Knowles")
    decomposed = unicodedata.normalize("NFD", "Beyoncé_Knowles")
    assert composed != decomposed

    db, _ = store(tmp_path, [page(composed, "0\tShe is a singer .")])
    with closing(wiki.connect(db)) as conn:
        assert wiki.doc_id(conn, decomposed) == 0
        assert wiki.doc_id(conn, composed) == 0


def test_escapes_stay_in_the_title_and_leave_the_norm(tmp_path):
    """
    norm has to live in the same token space the injector searches with, which is why it is
    built from tokenize. Keeping the unescaped brackets instead put 4,201 of the 14,533 gold
    titles beyond reach: a claim tokenizes to "soul food film" and never to "soul food (film)".
    """
    db, _ = store(tmp_path, [page("Soul_Food_-LRB-film-RRB-", "0\tA 1997 film .")])

    with closing(wiki.connect(db)) as conn:
        assert wiki.titles(conn, [0]) == {0: "Soul_Food_-LRB-film-RRB-"}
        stored_norm = conn.execute("SELECT norm FROM pages WHERE doc_id = 0").fetchone()[0]
        assert stored_norm == "soul food film"
        assert stored_norm == " ".join(tokenize("Soul_Food_-LRB-film-RRB-"))
        assert "-LRB-" not in stored_norm and "(" not in stored_norm
        assert wiki.find_by_norm(conn, "soul food film") == [0]
        assert wiki.find_by_norm(conn, "Soul_Food_-LRB-film-RRB-") == [0]


def test_find_by_norm_returns_every_page_sharing_a_loose_form(tmp_path):
    db, _ = store(tmp_path, [page("Mercury", "0\tone ."), page("mercury", "0\ttwo ."), page("Venus", "0\tthree .")])

    with closing(wiki.connect(db)) as conn:
        assert wiki.find_by_norm(conn, "mercury") == [0, 1]
        assert wiki.find_by_norm(conn, "pluto") == []


def test_sentences_round_trip_through_the_store(tmp_path):
    lines = "0\tFirst .\tanchor\ttarget\n1\t\n2\tThird , with punctuation .\n4\tFifth ."
    db, _ = store(tmp_path, [page("Round_Trip", lines)])

    with closing(wiki.connect(db)) as conn:
        did = wiki.doc_id(conn, "Round_Trip")
        assert wiki.sentences(conn, did) == [(0, "First ."), (2, "Third , with punctuation ."), (4, "Fifth .")]


def test_unknown_title_returns_none_rather_than_raising(tmp_path):
    db, _ = store(tmp_path, [page("Known", "0\tA .")])

    with closing(wiki.connect(db)) as conn:
        assert wiki.doc_id(conn, "Missing_Page") is None
        assert wiki.sentences(conn, 99) == []
        assert wiki.titles(conn, [0, 99]) == {0: "Known"}


def test_iter_pages_yields_doc_id_order_for_every_page(tmp_path):
    records = [page(f"Page_{i:03d}", f"0\tSentence {i} .") for i in range(50)]
    db, _ = store(tmp_path, records)

    with closing(wiki.connect(db)) as conn:
        rows = list(wiki.iter_pages(conn))
        assert [row[0] for row in rows] == list(range(50))
        assert wiki.page_count(conn) == 50
        assert rows[7][2] == "0\tSentence 7 ."


def test_building_twice_replaces_rather_than_appends(tmp_path):
    db, _ = store(tmp_path, [page("A", "0\ta ."), page("B", "0\tb .")])
    shard = tmp_path / "wiki-001.jsonl"
    stats = build([shard], db)

    assert stats.pages == 2
    with closing(wiki.connect(db)) as conn:
        assert wiki.page_count(conn) == 2


@pytest.mark.slow
@requires_store
def test_real_store_holds_the_whole_corpus():
    with closing(wiki.connect()) as conn:
        assert 5_000_000 < wiki.page_count(conn) < 6_000_000


@pytest.mark.slow
@requires_store
def test_a_known_page_resolves_and_carries_its_sentences():
    with closing(wiki.connect()) as conn:
        did = wiki.doc_id(conn, "Barack_Obama")
        assert did is not None
        rows = wiki.sentences(conn, did)
        assert rows[0][0] == 0
        assert "Obama" in rows[0][1]
        assert len(rows) > 5
