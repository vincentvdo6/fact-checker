"""
Text normalization for the retrieval layer.

FEVER is internally inconsistent about Unicode. Page titles in the wiki dump are NFC, but
the gold evidence titles in train.jsonl and shared_task_dev.jsonl are NFD: of 14,533 distinct
gold titles, 257 are non-ASCII and 170 of those differ between the forms. Claim *text* in the
same files is NFC throughout.

Left alone that costs roughly 1.2% of gold titles, and it costs them twice over -- the title
lookup misses, and a \\w+ tokenizer splits the two forms differently ("beyonce" followed by a
combining accent, versus "beyonce" with a precomposed one), so term matching misses as well.
Neither failure raises anything. It reads as ordinary retrieval difficulty.

So every string crossing into retrieval is normalized here, on both sides of every comparison.
Claim records keep their titles as released -- src/data/fever.py promises records as released
and that promise holds -- which makes this module the one place the two forms are reconciled.

Titles also carry the dump's bracket escapes (Soul_Food_-LRB-film-RRB-). They are decoded for
display keys and dropped entirely from tokens, where they would otherwise be high-frequency
terms appearing on about one page in eight (673,980 of 5,416,536).
"""

from __future__ import annotations

import re
import unicodedata

# The dump escapes characters that are awkward in a page id. Only these appear in titles.
ESCAPES = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-COLON-": ":",
}

# Not escape residue: unescape() runs before the word match, so "-LRB-" is already "(" and the
# word match drops it. What this actually removes is the English words, and "Colon" tokenizes
# to nothing at all. Measured cost on FEVER is 2 of 145,449 train claims and 0 of 19,998 dev,
# so no reported number moves -- but the filter buys nothing and should go when the index is
# next rebuilt. Removing it now would leave the shipped index without postings for these terms.
ESCAPE_TOKENS = frozenset({"lrb", "rrb", "lsb", "rsb", "colon"})

_ESCAPE_RE = re.compile("|".join(re.escape(key) for key in ESCAPES))
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def nfc(text: str) -> str:
    """Normalize to composed form. Applied on both sides of every title comparison."""
    return unicodedata.normalize("NFC", text)


def unescape(text: str) -> str:
    return _ESCAPE_RE.sub(lambda match: ESCAPES[match.group()], text)


def title_key(title: str) -> str:
    """
    Exact join key between a gold evidence title and a dump page id.

    NFC only -- no lowercasing, no unescaping. The dump's ids are the ground truth for what
    a page is called, and case distinguishes real pages.
    """
    return nfc(title)


def title_norm(title: str) -> str:
    """
    Loose form for matching a title against claim text, in the token space tokenize produces.

    Built from tokenize rather than beside it, because the two must agree exactly: the injector
    looks a title up by joining claim tokens, so any character one side keeps and the other drops
    makes the title unmatchable. Unescaping brackets and keeping them cost 4,201 of the 14,533
    gold titles -- the disambiguated ones, "Soul_Food_-LRB-film-RRB-" and its kind, which are
    precisely the pages a title match is most useful for.
    """
    return " ".join(tokenize(title))


def tokenize(text: str) -> list[str]:
    """Terms for BM25. Underscores split, escapes dropped, case folded."""
    lowered = unescape(nfc(text)).replace("_", " ").lower()
    return [token for token in _WORD_RE.findall(lowered) if token not in ESCAPE_TOKENS]
