"""BM25: the ranking function pinned to hand-computed values, CSC invariants, and tf clipping."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from src.retrieval.bm25 import K1, MAX_TF, B, BM25Index, BuildStats, build_index
from src.retrieval.text import tokenize

# Five documents, chosen so "cat" lands in a short, a medium and a long one and so that the
# terms pinned below sit at df 1, 2 and 3. Lengths: 6, 6, 3, 7, 11 -- avgdl 6.6 over N = 5.
TOY = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cat and dog",
    "the cat cat cat chased the dog",
    "a long document about nothing much at all in particular here",
]


def index_of(documents: Sequence[str], **kwargs) -> tuple[BM25Index, BuildStats]:
    """Build an in-memory index over raw texts, ids 0..n-1."""

    def source() -> Iterator[tuple[int, list[str]]]:
        return ((doc_id, tokenize(text)) for doc_id, text in enumerate(documents))

    return build_index(source, **kwargs)


def reference_scores(query: str, documents: Sequence[str]) -> list[float]:
    """
    Textbook BM25 written out scalar by scalar, with no clipping and no index machinery.

    Deliberately slow and dumb: it exists to disagree with src/retrieval/bm25.py if the CSC
    bookkeeping there ever drifts, so it must not share any code with it.
    """
    corpus = [tokenize(text) for text in documents]
    n_docs = len(corpus)
    avgdl = sum(len(tokens) for tokens in corpus) / n_docs

    scores = []
    for tokens in corpus:
        total = 0.0
        for term in tokenize(query):
            df = sum(1 for other in corpus if term in other)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            tf = tokens.count(term)
            if tf:
                norm = K1 * (1.0 - B + B * len(tokens) / avgdl)
                total += idf * tf * (K1 + 1.0) / (tf + norm)
        scores.append(total)
    return scores


def test_scores_match_hand_computed_values():
    """
    The single most valuable test here: it pins the idf variant, k1, b and the length norm.

    Worked by hand for document 0 and the term "cat", N = 5, df = 3, tf = 1, len = 6, avgdl = 6.6:

        idf  = ln(1 + (5 - 3 + 0.5) / (3 + 0.5))        = ln(1.7142857142857142)  = 0.5389965007
        norm = 1.2 * (1 - 0.75 + 0.75 * 6 / 6.6)                                  = 1.1181818182
        score = 0.5389965007 * 1 * 2.2 / (1 + 1.1181818182)                       = 0.5598161081

    Swapping in the Robertson/Sparck Jones idf, ln((N - df + 0.5) / (df + 0.5)), turns that
    0.5390 into -0.3567 and every number below changes sign somewhere. Nothing else notices.
    """
    index, _ = index_of(TOY)

    assert index.n_docs == 5
    assert index.avgdl == pytest.approx(6.6, abs=1e-12)
    assert index.df("cat") == 3
    assert index.idf("cat") == pytest.approx(0.5389965007326871, abs=1e-12)

    expected = [0.5598161080571258, 0.0, 0.6938146445601612, 0.8361355972904507, 0.0]
    assert index.score("cat") == pytest.approx(expected, abs=1e-9)

    # Two terms, so the per-term sum is exercised as well as the per-document norm.
    expected = [0.5598161080571258, 0.5598161080571258, 1.3876292891203224, 1.3620918601021859, 0.0]
    assert index.score("cat dog") == pytest.approx(expected, abs=1e-9)


def test_scores_agree_with_a_scalar_reference():
    index, _ = index_of(TOY)
    for query in ("cat", "cat dog", "the sat mat", "chased"):
        assert index.score(query) == pytest.approx(reference_scores(query, TOY), abs=1e-12)


def test_out_of_vocabulary_term_scores_zero():
    index, _ = index_of(TOY)
    assert index.df("photosynthesis") == 0
    assert index.idf("photosynthesis") == 0.0
    assert not index.score("photosynthesis").any()
    assert index.search("photosynthesis") == []
    # A known term beside an unknown one is unaffected by the unknown one.
    assert index.score("cat photosynthesis") == pytest.approx(index.score("cat"), abs=1e-12)


def test_term_in_every_document_keeps_a_small_positive_idf():
    """The Lucene variant floors at df = N; the classic one would go negative past df/N of a half."""
    documents = [f"wikipedia article number {number}" for number in range(5)]
    index, _ = index_of(documents)

    assert index.df("wikipedia") == index.n_docs == 5
    # ln(1 + 0.5 / 5.5)
    assert index.idf("wikipedia") == pytest.approx(0.08701137698962981, abs=1e-12)
    assert 0.0 < index.idf("wikipedia") < index.idf("0")
    assert (index.score("wikipedia") > 0.0).all()


def test_index_round_trips_through_save_and_load(tmp_path):
    index, _ = index_of(TOY)
    before = index.search("cat dog", k=5)

    index.save(tmp_path)
    restored = BM25Index.load(tmp_path, index.vocab)

    assert restored.n_docs == index.n_docs
    assert restored.avgdl == pytest.approx(index.avgdl, abs=1e-12)
    assert restored.search("cat dog", k=5) == before


def test_streamed_build_round_trips_through_the_memory_maps(tmp_path):
    """The directory build writes indices.npy and tf.npy as memory maps rather than np.save."""
    index, stats = index_of(TOY, directory=tmp_path, sentence_avgdl=lambda: 4.25)
    expected = index.search("cat dog", k=5)

    restored = BM25Index.load(tmp_path, index.vocab)
    assert restored.search("cat dog", k=5) == expected
    assert restored.sentence_avgdl == pytest.approx(4.25, abs=1e-12)
    assert restored.n_postings == stats.postings
    assert restored.indices.dtype == np.int32
    assert restored.tf.dtype == np.uint8


def test_indptr_is_non_decreasing_and_ends_at_the_posting_count():
    index, stats = index_of(TOY)
    assert index.indptr[0] == 0
    assert (np.diff(index.indptr) >= 0).all()
    assert int(index.indptr[-1]) == stats.postings == index.indices.size == index.tf.size


def test_every_column_is_exactly_its_terms_df():
    index, _ = index_of(TOY)
    for term, term_id in index.vocab.items():
        start, stop = int(index.indptr[term_id]), int(index.indptr[term_id + 1])
        column = index.indices[start:stop]
        assert stop - start == index.df(term)
        # A CSC column is a set of documents, and score() adds into it in one shot.
        assert len(set(column.tolist())) == column.size
        expected = {doc_id for doc_id, text in enumerate(TOY) if term in tokenize(text)}
        assert set(column.tolist()) == expected


def test_tf_below_the_ceiling_is_not_clipped():
    _, stats = index_of(["spam " * MAX_TF + "eggs", "eggs"])
    assert stats.tf_clipped == 0


def test_clip_counter_fires_one_past_the_ceiling():
    index, stats = index_of(["spam " * (MAX_TF + 1) + "eggs", "eggs"])
    assert stats.tf_clipped == 1
    assert index.tf.max() == MAX_TF


def test_clipping_does_not_change_ranking_on_a_toy_corpus():
    documents = ["spam " * 300 + "eggs", "spam spam spam eggs", "eggs and ham"]
    index, stats = index_of(documents)
    assert stats.tf_clipped == 1

    clipped = [doc_id for doc_id, _ in index.search("spam", k=3)]
    unclipped = reference_scores("spam", documents)
    assert clipped == sorted(range(3), key=lambda doc: -unclipped[doc])[:2]


def test_longer_documents_score_lower_for_the_same_term_count():
    short = "quantum entanglement"
    long = "quantum " + " ".join(f"filler{number}" for number in range(40))
    index, _ = index_of([short, long])

    scores = index.score("quantum")
    assert index.tf[index.indptr[index.vocab["quantum"]] : index.indptr[index.vocab["quantum"] + 1]].tolist() == [1, 1]
    assert scores[0] > scores[1]


def test_top_k_larger_than_the_corpus_does_not_crash():
    index, _ = index_of(TOY)
    hits = index.search("cat dog", k=1000)
    assert len(hits) == 4       # document 4 shares no term and is never returned
    assert index.search("cat dog", k=4) == hits


def test_zero_or_negative_k_is_rejected():
    index, _ = index_of(TOY)
    with pytest.raises(ValueError, match="k must be positive"):
        index.search("cat", k=0)


def test_ties_break_deterministically():
    documents = ["cat sat", "cat sat", "cat sat", "dog"]
    index, _ = index_of(documents)

    hits = index.search("cat", k=3)
    assert hits == index.search("cat", k=3)
    assert [doc_id for doc_id, _ in hits] == [0, 1, 2]   # identical scores, ascending document id
    assert len({score for _, score in hits}) == 1


def test_sentence_scorer_uses_corpus_idf_and_the_supplied_avgdl():
    """
    Sentence selection is why this exists: idf over a handful of sentences means nothing, and
    the page-length norm would call every sentence short.

    Hand-worked over the corpus below -- N = 4, df("president") = 4, df("nixon") = 1 -- with a
    supplied avgdl of 4.0:

        "the president and the president again president"  tf 3, len 7
            idf  = ln(1 + 0.5 / 4.5)                                = 0.1053605157
            norm = 1.2 * (1 - 0.75 + 0.75 * 7 / 4)                  = 1.875
            score = 0.1053605157 * 3 * 2.2 / (3 + 1.875)            = 0.1426419289

        "nixon"                                            tf 1, len 1
            idf  = ln(1 + 3.5 / 1.5)                                = 1.2039728043
            norm = 1.2 * (1 - 0.75 + 0.75 * 1 / 4)                  = 0.525
            score = 1.2039728043 * 1 * 2.2 / (1 + 0.525)            = 1.7368787997

    Computed over the two sentences alone both terms would sit at df 1 of 2 and share an idf,
    and the three occurrences of "president" would win. Corpus idf is what makes "nixon" win.
    """
    corpus = [
        "president nixon resigned in august",
        "the president gave a speech",
        "president of the united states",
        "a president is elected",
    ]
    index, _ = index_of(corpus)
    assert index.df("president") == 4
    assert index.df("nixon") == 1

    sentences = ["the president and the president again president", "nixon"]
    scores = index.score_texts("president nixon", sentences, avgdl=4.0)
    assert scores == pytest.approx([0.1426419288905957, 1.736878799683318], abs=1e-9)
    assert scores[1] > scores[0]


def test_sentence_scorer_reproduces_document_scores_at_the_page_avgdl():
    """Same formula on both sides: feed it the pages and the page avgdl and it must agree."""
    index, _ = index_of(TOY)
    assert index.score_texts("cat dog", TOY, avgdl=index.avgdl) == pytest.approx(index.score("cat dog"), abs=1e-12)


def test_sentence_scorer_honours_the_avgdl_it_is_given():
    index, _ = index_of(TOY)
    sentences = ["the cat sat on the mat"]
    # A larger avgdl makes the same text relatively shorter, so it is penalized less.
    assert index.score_texts("cat", sentences, avgdl=30.0) > index.score_texts("cat", sentences, avgdl=3.0)
    with pytest.raises(ValueError, match="avgdl must be positive"):
        index.score_texts("cat", sentences, avgdl=0.0)


def test_sentence_scorer_ignores_terms_the_corpus_has_never_seen():
    index, _ = index_of(TOY)
    assert index.score_texts("photosynthesis", ["photosynthesis photosynthesis"], avgdl=4.0) == pytest.approx([0.0])


def test_document_ids_may_have_gaps_but_must_increase():
    """iter_pages hands over the corpus id, which is a row id and not necessarily 0-based."""

    def gapped() -> Iterator[tuple[int, list[str]]]:
        return iter([(1, ["cat"]), (4, ["cat", "dog"])])

    index, stats = build_index(gapped)
    assert stats.docs == 2
    assert index.doclen.tolist() == [0, 1, 0, 0, 2]
    assert index.n_docs == 2                      # the padding is addressing, not documents
    assert index.avgdl == pytest.approx(1.5)
    assert [doc_id for doc_id, _ in index.search("cat")] == [1, 4]

    def backwards() -> Iterator[tuple[int, list[str]]]:
        return iter([(4, ["cat"]), (1, ["dog"])])

    with pytest.raises(ValueError, match="must strictly increase"):
        build_index(backwards)


def test_a_source_that_does_not_replay_is_rejected():
    """Both passes must see the same corpus, and a source that drifts is caught, not indexed."""
    calls = []

    def drifting() -> Iterator[tuple[int, list[str]]]:
        calls.append(None)
        documents = ["cat dog"] if len(calls) == 1 else ["cat dog", "cat"]
        return ((doc_id, tokenize(text)) for doc_id, text in enumerate(documents))

    with pytest.raises(ValueError, match="does not replay"):
        build_index(drifting)


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="empty corpus"):
        build_index(lambda: iter([]))
    with pytest.raises(ValueError, match="no tokens at all"):
        index_of(["", "   "])
