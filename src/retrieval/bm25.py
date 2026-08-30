"""
BM25 over a memory-mapped inverted index.

The corpus is 5.4M pages and 292.5M postings, so the index is built in two streaming
passes and is never held in RAM whole. Pass one tokenizes every document to learn the
vocabulary, each term's document frequency, and each document's length, discarding the tokens
as it goes. Because df is the exact length of every column, the CSC column pointer is fully
determined before a single posting is written -- so pass two writes each posting straight into
its final slot and no sort is ever required. Buying away a 292.5M-element sort, and the several
GB that sort would need, is the entire reason for tokenizing twice.

Postings hold raw term frequencies in uint8 rather than precomputed impacts. Impacts would be
four times larger and would bake k1 and b into the artifact, and the retrieval ceiling has to
be measured against more than one setting. tf above 255 is clipped and the clips are counted:
the shipped corpus clips 330 of 292.5M, on the ~191 pages long enough to use one term that
often, so a trickle is normal and orders of magnitude more means the tokenizer.

Scoring uses the Lucene idf variant

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

which is always positive, so a term present in every document still contributes a little
instead of subtracting. This is the exact line where a reimplementation silently diverges from
published BM25 numbers -- the classic Robertson/Sparck Jones form goes negative once df passes
half the corpus -- so it is pinned by test against hand-computed values.

The vocabulary is deliberately not part of the artifact. 3.4M terms is several hundred MB of
Python strings, and a query needs a handful of them; it lives in the corpus SQLite beside df,
and is handed to BM25Index.load by the caller.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from src.retrieval.text import tokenize

K1 = 1.2
B = 0.75

# tf is one byte per posting. See the module docstring on why clipping here is harmless.
MAX_TF = 255

DOC_DTYPE = np.int32   # 5.4M page ids; int32 halves the 1.08 GB posting array against int64
TF_DTYPE = np.uint8
PTR_DTYPE = np.int64   # 292.5M postings overflows int32
LEN_DTYPE = np.int32

INDICES_FILE = "indices.npy"
TF_FILE = "tf.npy"
INDPTR_FILE = "indptr.npy"
DOCLEN_FILE = "doclen.npy"
META_FILE = "meta.json"


def _idf(df: int, n_docs: int) -> float:
    """
    Lucene's BM25 idf. Never negative.

    df of zero returns 0.0 rather than the maximal value the formula would give: a term the
    corpus has never seen carries no evidence, and both scorers depend on that so an unknown
    query term contributes nothing instead of dominating.
    """
    if df <= 0:
        return 0.0
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def _terms(query: str | Sequence[str]) -> list[str]:
    """Query terms, tokenizing only if the query arrives as raw text."""
    return tokenize(query) if isinstance(query, str) else list(query)


@dataclass(frozen=True, slots=True)
class BuildStats:
    """What a build reports. A --limit-docs run is extrapolated from these before a full build."""

    docs: int
    terms: int
    postings: int
    tokens: int
    tf_clipped: int

    @property
    def postings_per_doc(self) -> float:
        return self.postings / self.docs

    @property
    def tokens_per_doc(self) -> float:
        return self.tokens / self.docs

    def to_dict(self) -> dict[str, int]:
        return {
            "docs": self.docs,
            "terms": self.terms,
            "postings": self.postings,
            "tokens": self.tokens,
            "tf_clipped": self.tf_clipped,
        }


@dataclass(frozen=True, slots=True)
class BM25Index:
    """
    A CSC inverted index and the corpus statistics BM25 needs.

    Column term_id occupies indices[indptr[term_id]:indptr[term_id + 1]], holding the ids of the
    documents containing the term, with the matching term frequencies at the same offsets in tf.
    Document ids are unique within a column, which is what lets score() accumulate with a plain
    fancy-index += instead of an unbuffered add.

    doclen is indexed by document id directly, so its length is the highest id plus one and may
    exceed n_docs if the corpus ids have gaps. n_docs -- documents actually seen -- is the N in idf.

    Arrays are normally memory maps (np.load(mmap_mode="r")), so a loaded index costs a few tens of
    MB resident plus whatever the OS chooses to keep in page cache.
    """

    indices: np.ndarray
    tf: np.ndarray
    indptr: np.ndarray
    doclen: np.ndarray
    vocab: Mapping[str, int]
    n_docs: int
    avgdl: float
    sentence_avgdl: float = 0.0   # persisted default for score_texts; 0.0 means never measured
    k1: float = K1
    b: float = B

    def __post_init__(self) -> None:
        if self.indptr.size != len(self.vocab) + 1:
            raise ValueError(f"vocabulary has {len(self.vocab)} terms but indptr holds {self.indptr.size - 1}")
        if int(self.indptr[-1]) != self.indices.size or self.indices.size != self.tf.size:
            raise ValueError(f"indptr ends at {int(self.indptr[-1])} but indices holds {self.indices.size}")
        if self.n_docs <= 0 or self.avgdl <= 0.0:
            raise ValueError(f"n_docs={self.n_docs} and avgdl={self.avgdl} must both be positive")

    @property
    def n_terms(self) -> int:
        return self.indptr.size - 1

    @property
    def n_postings(self) -> int:
        return int(self.indptr[-1])

    def df(self, term: str) -> int:
        """Document frequency, or 0 for a term outside the vocabulary."""
        term_id = self.vocab.get(term)
        if term_id is None:
            return 0
        return int(self.indptr[term_id + 1] - self.indptr[term_id])

    def idf(self, term: str) -> float:
        return _idf(self.df(term), self.n_docs)

    def score(self, query: str | Sequence[str]) -> np.ndarray:
        """
        BM25 score of every document against the query, indexed by document id.

        Query terms are summed as they occur, so a term repeated in the query contributes twice --
        the same as a Lucene boolean query with one clause per occurrence. float64 throughout:
        float32 would halve the 43 MB accumulator, but ranking ties would then turn on rounding.
        """
        scores = np.zeros(self.doclen.size, dtype=np.float64)
        for term in _terms(query):
            term_id = self.vocab.get(term)
            if term_id is None:
                continue
            start = int(self.indptr[term_id])
            stop = int(self.indptr[term_id + 1])
            if start == stop:
                continue
            idf = _idf(stop - start, self.n_docs)
            docs = self.indices[start:stop]
            freq = self.tf[start:stop].astype(np.float64)
            norm = self.k1 * (1.0 - self.b + self.b * (self.doclen[docs] / self.avgdl))
            scores[docs] += idf * freq * (self.k1 + 1.0) / (freq + norm)
        return scores

    def search(self, query: str | Sequence[str], k: int = 10) -> list[tuple[int, float]]:
        """
        Top-k (document id, score), best first, ties broken by ascending document id.

        Documents scoring zero are never returned, so fewer than k results means fewer than k
        documents matched -- including when k is larger than the corpus.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        scores = self.score(query)
        hits = np.flatnonzero(scores)
        if hits.size == 0:
            return []
        values = scores[hits]
        if k < hits.size:
            # argpartition is O(n) against a full sort of 5.4M scores for a handful of results.
            top = np.argpartition(-values, k - 1)[:k]
            hits, values = hits[top], values[top]
        order = np.lexsort((hits, -values))   # document id ascending within a score tie
        return [(int(hits[row]), float(values[row])) for row in order]

    def score_texts(self, query: str | Sequence[str], texts: Sequence[str], *, avgdl: float) -> np.ndarray:
        """
        BM25 of a handful of texts against the query, using corpus idf but a caller-supplied avgdl.

        This is how a page's sentences get ranked. idf has to come from the corpus -- computed over
        five candidate sentences it would be statistically meaningless -- while the length norm must
        not, because sentences run an order of magnitude shorter than pages and the page avgdl would
        make every one of them look short. Pass the index's own sentence_avgdl, measured over the
        same corpus during the build.
        """
        if avgdl <= 0.0:
            raise ValueError(f"avgdl must be positive, got {avgdl}")
        weighted = [(term, self.idf(term)) for term in _terms(query)]
        weighted = [(term, weight) for term, weight in weighted if weight > 0.0]

        scores = np.zeros(len(texts), dtype=np.float64)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            norm = self.k1 * (1.0 - self.b + self.b * sum(counts.values()) / avgdl)
            scores[row] = sum(
                weight * freq * (self.k1 + 1.0) / (freq + norm)
                for term, weight in weighted
                if (freq := counts.get(term, 0))
            )
        return scores

    def meta(self) -> dict[str, float | int]:
        return {"n_docs": self.n_docs, "avgdl": self.avgdl, "sentence_avgdl": self.sentence_avgdl}

    def save(self, directory: str | Path) -> None:
        """
        Write the four arrays and meta.json. The vocabulary is the caller's to persist.

        Not for an index build_index already streamed into this directory -- its posting arrays are
        memory-mapped from those very files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / INDICES_FILE, self.indices)
        np.save(directory / TF_FILE, self.tf)
        np.save(directory / INDPTR_FILE, self.indptr)
        np.save(directory / DOCLEN_FILE, self.doclen)
        _write_meta(directory, self.meta())

    @classmethod
    def load(
        cls,
        directory: str | Path,
        vocab: Mapping[str, int],
        *,
        k1: float = K1,
        b: float = B,
    ) -> BM25Index:
        """Map the arrays read-only. vocab comes from the corpus SQLite terms table."""
        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))

        def mapped(name: str) -> np.ndarray:
            return np.load(directory / name, mmap_mode="r")

        return cls(
            indices=mapped(INDICES_FILE),
            tf=mapped(TF_FILE),
            indptr=mapped(INDPTR_FILE),
            doclen=mapped(DOCLEN_FILE),
            vocab=vocab,
            n_docs=int(meta["n_docs"]),
            avgdl=float(meta["avgdl"]),
            sentence_avgdl=float(meta["sentence_avgdl"]),
            k1=k1,
            b=b,
        )


def _write_meta(directory: Path, payload: Mapping[str, object]) -> None:
    (directory / META_FILE).write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def build_index(
    source: Callable[[], Iterable[tuple[int, Sequence[str]]]],
    *,
    directory: str | Path | None = None,
    sentence_avgdl: Callable[[], float] | None = None,
    k1: float = K1,
    b: float = B,
) -> tuple[BM25Index, BuildStats]:
    """
    Build the index in two streaming passes. See the module docstring for why there are two.

    source is a factory, not an iterator: it is called once per pass and must yield the same
    documents both times, as (doc_id, tokens) with strictly increasing non-negative ids. A source
    that does not replay is caught at the end of pass two rather than producing a corrupt index.

    With directory given, the posting arrays are written straight into indices.npy and tf.npy as
    memory maps, so 1.35 GB of postings never has to be resident and the files load with
    np.load(mmap_mode="r") afterwards. Without it everything stays in RAM, which is what the unit
    tests use.

    sentence_avgdl is a callable because a sentence-length average is only complete once the corpus
    has been streamed -- which is a thing this function does and its caller has not yet done.
    """
    vocab: dict[str, int] = {}
    df: list[int] = []
    doclen: list[int] = []
    n_docs = 0
    n_tokens = 0
    previous = -1

    for doc_id, tokens in source():
        if doc_id <= previous:
            raise ValueError(f"document ids must strictly increase; {doc_id} follows {previous}")
        previous = doc_id
        # Gaps are padded so a document id stays a direct index into doclen and into the score
        # vector, which is what keeps scoring a single fancy-index add.
        doclen.extend([0] * (doc_id - len(doclen)))
        doclen.append(len(tokens))
        n_docs += 1
        n_tokens += len(tokens)
        # dict.fromkeys, not set: df counts documents not occurrences either way, but set
        # iteration order follows string hashing, which is randomized per process. That made
        # term ids -- and so the whole index -- differ between builds of the same corpus.
        for term in dict.fromkeys(tokens):
            term_id = vocab.get(term)
            if term_id is None:
                vocab[term] = len(df)
                df.append(1)
            else:
                df[term_id] += 1

    if n_docs == 0:
        raise ValueError("cannot build an index over an empty corpus")
    if n_tokens == 0:
        raise ValueError(f"{n_docs} documents produced no tokens at all")
    if len(doclen) > np.iinfo(DOC_DTYPE).max:
        raise ValueError(f"document id {len(doclen) - 1} does not fit {DOC_DTYPE.__name__}")

    df_array = np.asarray(df, dtype=PTR_DTYPE)
    indptr = np.empty(df_array.size + 1, dtype=PTR_DTYPE)
    indptr[0] = 0
    np.cumsum(df_array, out=indptr[1:])
    n_postings = int(indptr[-1])

    if directory is None:
        indices = np.empty(n_postings, dtype=DOC_DTYPE)
        freqs = np.empty(n_postings, dtype=TF_DTYPE)
    else:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # open_memmap writes a real .npy header, so these load with np.load(mmap_mode="r") later.
        indices = open_memmap(directory / INDICES_FILE, mode="w+", dtype=DOC_DTYPE, shape=(n_postings,))
        freqs = open_memmap(directory / TF_FILE, mode="w+", dtype=TF_DTYPE, shape=(n_postings,))

    cursor = indptr[:-1].copy()   # next free slot in each column
    clipped = 0
    for doc_id, tokens in source():
        for term, count in Counter(tokens).items():
            term_id = vocab[term]
            position = cursor[term_id]
            indices[position] = doc_id
            if count > MAX_TF:
                clipped += 1
                count = MAX_TF   # numpy 2 raises on an out-of-range store rather than wrapping
            freqs[position] = count
            cursor[term_id] = position + 1

    if not np.array_equal(cursor, indptr[1:]):
        raise ValueError("the second pass did not reproduce the first; the source does not replay")

    index = BM25Index(
        indices=indices,
        tf=freqs,
        indptr=indptr,
        doclen=np.asarray(doclen, dtype=LEN_DTYPE),
        vocab=vocab,
        n_docs=n_docs,
        avgdl=n_tokens / n_docs,
        sentence_avgdl=0.0 if sentence_avgdl is None else sentence_avgdl(),
        k1=k1,
        b=b,
    )
    stats = BuildStats(
        docs=n_docs,
        terms=len(vocab),
        postings=n_postings,
        tokens=n_tokens,
        tf_clipped=clipped,
    )

    if directory is not None:
        indices.flush()
        freqs.flush()
        np.save(directory / INDPTR_FILE, index.indptr)
        np.save(directory / DOCLEN_FILE, index.doclen)
        _write_meta(directory, index.meta() | stats.to_dict())

    return index, stats
