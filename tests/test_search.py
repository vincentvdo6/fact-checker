"""Two-stage retrieval over a toy corpus: injector merge, cutoffs, dedup, determinism."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from src.retrieval import wiki
from src.retrieval.bm25 import BM25Index, build_index
from src.retrieval.search import (
    DEFAULT_MAX_TITLES,
    SqliteVocab,
    retrieve,
    retrieve_pages,
    retrieve_sentences,
    title_candidates,
)
from src.retrieval.text import title_key, title_norm, tokenize

# Titles chosen so one-token matches are plentiful: "House" and "Shooter" are exactly the
# generic pages an uncapped injector floods the page budget with.
PAGES = [
    ("Soul_Food_-LRB-film-RRB-", "0\tSoul Food is a 1997 American comedy drama film .\n1\tIt was released by Fox ."),
    ("House", "0\tA house is a building that people live in ."),
    ("Shooter", "0\tShooter is a 2007 American action film ."),
    ("Savages", "0\tSavages is a 2012 American thriller film ."),
    ("Fox_Broadcasting_Company", "0\tFox is an American television network .\n1\tIt launched in 1986 ."),
    ("Beyonce\u0301_Knowles", "0\tBeyonce is an American singer ."),
    ("Film", "0\tA film is a work of visual art ."),
    ("Kenneth_Edmonds", "0\tKenneth Edmonds is an American record producer ."),
]


@pytest.fixture
def corpus(tmp_path):
    """An eight-page store plus a matching index, wired the way the real pipeline is."""
    conn = sqlite3.connect(tmp_path / "wiki.sqlite3")
    wiki.create_tables(conn)
    rows = [
        (doc_id, title_key(title), title_norm(title), sentences)
        for doc_id, (title, sentences) in enumerate(PAGES)
    ]
    conn.executemany("INSERT INTO pages (doc_id, title, norm, sentences) VALUES (?, ?, ?, ?)", rows)
    wiki.create_indexes(conn)

    def source():
        for doc_id, title, _, sentences in rows:
            tokens = tokenize(title)
            for _, sentence in wiki.parse_lines(sentences):
                tokens.extend(tokenize(sentence))
            yield doc_id, tokens

    lengths = [len(tokenize(s)) for _, _, _, blob in rows for _, s in wiki.parse_lines(blob)]
    index, _ = build_index(source, sentence_avgdl=lambda: sum(lengths) / len(lengths))
    conn.execute("CREATE TABLE terms (term TEXT PRIMARY KEY, term_id INTEGER NOT NULL, df INTEGER NOT NULL)")
    conn.executemany(
        "INSERT INTO terms (term, term_id, df) VALUES (?, ?, ?)",
        [(term, term_id, index.df(term)) for term, term_id in index.vocab.items()],
    )
    conn.commit()
    return conn, BM25Index(
        indices=index.indices,
        tf=index.tf,
        indptr=index.indptr,
        doclen=index.doclen,
        vocab=SqliteVocab(conn),
        n_docs=index.n_docs,
        avgdl=index.avgdl,
        sentence_avgdl=index.sentence_avgdl,
    )


def titles_of(conn, doc_ids):
    resolved = wiki.titles(conn, doc_ids)
    return [resolved[doc_id] for doc_id in doc_ids]


def test_sqlite_vocab_matches_the_built_vocabulary(corpus):
    """SqliteVocab is the production vocabulary, so it has to agree with the stored terms."""
    conn, index = corpus
    stored = dict(conn.execute("SELECT term, term_id FROM terms").fetchall())
    assert stored
    assert len(index.vocab) == len(stored)
    assert {term: index.vocab[term] for term in stored} == stored
    assert index.vocab.get("nonexistent") is None


def test_title_candidates_prefers_the_longest_match(corpus):
    """A disambiguated title only matches because norm lives in tokenize's space."""
    conn, _ = corpus
    found = titles_of(conn, title_candidates(conn, "Soul Food film was about a house"))
    assert found[0] == "Soul_Food_-LRB-film-RRB-"
    assert "House" in found


def test_injector_cap_limits_how_many_pages_it_takes(corpus):
    conn, index = corpus
    claim = "Soul Food is a film about a house with savages and a shooter"
    assert len(title_candidates(conn, claim)) > 2

    capped = retrieve_pages(conn, index, claim, n=6, max_titles=2)
    injected = title_candidates(conn, claim)[:2]
    assert capped[: len(injected)] == injected
    assert len(capped) <= 6


def test_the_cap_bounds_how_many_injected_pages_survive(corpus):
    """
    The assertion the cap exists for: with more candidates than the budget, max_titles decides
    how many slots BM25 keeps. Without the slice both calls return the same three pages.
    """
    conn, index = corpus
    claim = "Soul Food film house shooter savages fox"
    assert len(title_candidates(conn, claim)) > 2

    narrow = retrieve_pages(conn, index, claim, n=3, max_titles=1)
    wide = retrieve_pages(conn, index, claim, n=3, max_titles=99)
    assert narrow != wide
    assert narrow[:1] == title_candidates(conn, claim)[:1]
    assert wide == title_candidates(conn, claim)[:3]


def test_the_shipped_default_caps_injection(corpus):
    """
    scripts/eval_retrieval.py never passes max_titles, so DEFAULT_MAX_TITLES is the value behind
    every reported number. Without this the constant can be set to anything and stay green.
    """
    conn, index = corpus
    # Non-title terms so BM25 prefers a page the injector never proposes.
    claim = "house shooter savages film american record producer television network"
    candidates = title_candidates(conn, claim)
    assert len(candidates) > DEFAULT_MAX_TITLES

    n = len(candidates)
    shipped = retrieve_pages(conn, index, claim, n=n)
    uncapped = retrieve_pages(conn, index, claim, n=n, max_titles=n)

    assert uncapped == candidates[:n]
    assert shipped[:DEFAULT_MAX_TITLES] == candidates[:DEFAULT_MAX_TITLES]
    assert shipped != uncapped
    assert any(page not in candidates for page in shipped)


def test_uncapped_injection_can_fill_the_whole_budget(corpus):
    """The bug the cap exists for: a long candidate list evicting BM25 entirely."""
    conn, index = corpus
    claim = "house shooter savages"
    wide = retrieve_pages(conn, index, claim, n=3, max_titles=99)
    assert wide == title_candidates(conn, claim)[:3]


def test_disabling_injection_leaves_pure_bm25(corpus):
    conn, index = corpus
    claim = "Soul Food film"
    plain = retrieve_pages(conn, index, claim, n=5, inject_titles=False)
    assert plain == [doc_id for doc_id, _ in index.search(claim, k=5)][:5]
    assert retrieve_pages(conn, index, claim, n=5, max_titles=0) == plain


def test_injected_pages_come_before_bm25(corpus):
    conn, index = corpus
    claim = "Fox released Soul Food"
    pages = retrieve_pages(conn, index, claim, n=6, max_titles=3)
    injected = title_candidates(conn, claim)[:3]
    assert pages[: len(injected)] == injected


def test_sentences_are_deduplicated_and_capped(corpus):
    conn, index = corpus
    pages = [0, 4]
    refs, scores = retrieve_sentences(conn, index, "Soul Food Fox", pages, k=3)
    assert len(refs) == len(set(refs)) == len(scores) <= 3


def test_sentence_scoring_uses_the_sentence_length_norm(corpus):
    """
    Sentences run an order of magnitude shorter than pages (17.9 against 86.2 on the corpus), so
    scoring them with the page norm reorders everything. If search read index.avgdl instead of
    sentence_avgdl, both rankings below would be identical.
    """
    conn, index = corpus
    pages = [0, 1, 2, 3, 4]
    short = replace(index, sentence_avgdl=2.0)
    long_norm = replace(index, sentence_avgdl=200.0)
    assert retrieve_sentences(conn, short, "American film", pages, k=6) != retrieve_sentences(
        conn, long_norm, "American film", pages, k=6
    )


def test_an_index_without_a_sentence_norm_is_rejected(corpus):
    """
    Substituting the page norm silently swaps 17.9 for 86.2 and reorders every sentence, so a
    smoke-built or older artifact has to fail loudly rather than score with the wrong length.
    """
    conn, index = corpus
    unmeasured = replace(index, sentence_avgdl=0.0)
    with pytest.raises(ValueError, match="no sentence_avgdl"):
        retrieve_sentences(conn, unmeasured, "American film", [0, 1], k=5)


def test_sentence_ranking_is_deterministic(corpus):
    conn, index = corpus
    pages = [0, 1, 2, 3, 4]
    first = retrieve_sentences(conn, index, "American film", pages, k=10)
    second = retrieve_sentences(conn, index, "American film", pages, k=10)
    assert first == second


def test_ties_break_by_page_rank_then_sentence_index(corpus):
    """Pages 2 and 3 carry near-identical sentences, so order must follow the page order given."""
    conn, index = corpus
    forward, _ = retrieve_sentences(conn, index, "zzz", [2, 3], k=4)
    reverse, _ = retrieve_sentences(conn, index, "zzz", [3, 2], k=4)
    assert [title for title, _ in forward][0] == "Shooter"
    assert [title for title, _ in reverse][0] == "Savages"


def test_retrieve_is_deterministic_and_consistent(corpus):
    conn, index = corpus
    first = retrieve(conn, index, "Soul Food is a 1997 film", n=4, k=5)
    second = retrieve(conn, index, "Soul Food is a 1997 film", n=4, k=5)
    assert first == second
    assert len(first.refs) == len(first.scores)
    assert len(first.pages) <= 4
    assert set(title for title, _ in first.refs) <= set(first.pages)


def test_retrieved_titles_are_composed(corpus):
    """
    Refs carry store titles, which are composed. Written as escapes because the two forms are
    indistinguishable on screen, and an assertion nobody can read is one nobody can trust.
    """
    conn, index = corpus
    decomposed = "Beyonce\u0301_Knowles"
    composed = "Beyonc\u00e9_Knowles"
    assert decomposed != composed

    result = retrieve(conn, index, "Beyonce is an American singer", n=3, k=5)
    assert composed in result.pages
    assert decomposed not in result.pages


def test_empty_page_list_yields_nothing(corpus):
    conn, index = corpus
    assert retrieve_sentences(conn, index, "anything", [], k=5) == ([], [])
