"""
Retrieval scoring against FEVER gold evidence.

Gold evidence is a list of groups and a group only verifies a claim in full:
conjunctive within a group, disjunctive between them. Everything here is scored over
verifiable claims alone -- NOT ENOUGH INFO carries no gold evidence, so recall is
undefined for it rather than zero -- and empty `groups` raises, because a verifiable
claim without evidence is an impossible state and scoring it would silently record a
miss instead of a bug.

`recall_at_k` is the metric the project reports. The rest exist to be read against it:

  recall_any_at_k   credits a claim as soon as any single gold sentence lands, which
                    overstates -- it is the number reported by accident. The gap is
                    bounded by the 9.03% of verifiable dev claims needing two or more sentences.
  mrr_at_k          group-aware: a group completes at the rank of its LAST arriving
                    member, so members at ranks 1 and 3 give 1/3, not 1/1. Ranking one
                    member of a pair first is worth nothing on its own, and the
                    first-member reading of MRR is quietly higher for the same run.
  ndcg_any_at_k     binary gain over the union of all groups. nDCG has no way to
                    express a conjunction, so it grades ranking quality only and can
                    sit high while `recall_at_k` is 0. It never decides anything.
  doc_recall_at_n   the same strict form over page titles, scoring stage 1 alone. It
                    is a hard ceiling on `recall_at_k`, since a sentence on a page
                    that was never retrieved cannot be. 7.9% of verifiable dev claims need two
                    or more pages.

`Ceiling.overall` is the bound on end-to-end accuracy and it is inflated: an oracle
answers NEI correctly with no evidence at all, so the NEI share is accuracy given
away. It sits in the same structure as `nei_share` so it cannot be quoted alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.data.fever import NOT_ENOUGH_INFO, Claim
from src.retrieval.text import title_key

Ref = tuple[str, int]                    # (page_title, sentence_index)
Groups = Sequence[frozenset[Ref]]

DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20)     # sentence cutoffs
DEFAULT_NS: tuple[int, ...] = (1, 5, 10, 20)     # page cutoffs, stage 1


def _refs(refs: Sequence[Ref]) -> list[Ref]:
    """
    Both sides of every comparison, with titles composed.

    Gold titles arrive as FEVER released them, which is NFD for 170 of the 14,533 distinct
    ones. Retrieved titles come from the store, which composed them at build time. Comparing
    the two forms directly makes those claims unscoreable -- not lower-scoring, but incapable
    of a hit at any k -- and the result reads as retrieval difficulty rather than a bug.
    """
    return [(title_key(title), index) for title, index in refs]


def _groups(groups: Groups) -> list[frozenset[Ref]]:
    return [frozenset(_refs(group)) for group in groups]


def _checked(groups: Groups, cutoff: int) -> None:
    if not groups:
        raise ValueError("no gold evidence: a verifiable claim must carry at least one group")
    # An empty group is a subset of everything, so one would make recall unconditionally true.
    # fever._groups drops null-page groups whole and cannot produce one; this is the second lock.
    if any(not group for group in groups):
        raise ValueError("an empty evidence group would be satisfied by retrieving nothing")
    # A negative cutoff would slice a suffix off the ranking and score whatever remained.
    if cutoff < 0:
        raise ValueError(f"cutoff must be non-negative, got {cutoff}")


def recall_at_k(retrieved: Sequence[Ref], groups: Groups, k: int) -> bool:
    """Strict recall: one whole group inside the top k. The primary metric."""
    retrieved, groups = _refs(retrieved), _groups(groups)
    _checked(groups, k)
    top = set(retrieved[:k])
    return any(group <= top for group in groups)


def recall_any_at_k(retrieved: Sequence[Ref], groups: Groups, k: int) -> bool:
    """Loose recall: any one gold sentence inside the top k. Overstates; see module docstring."""
    retrieved, groups = _refs(retrieved), _groups(groups)
    _checked(groups, k)
    top = set(retrieved[:k])
    return any(top & group for group in groups)


def mrr_at_k(retrieved: Sequence[Ref], groups: Groups, k: int) -> float:
    """Reciprocal rank of the first group to complete, a group completing at its last member."""
    retrieved, groups = _refs(retrieved), _groups(groups)
    _checked(groups, k)
    rank: dict[Ref, int] = {}
    for position, ref in enumerate(retrieved[:k], start=1):
        # First occurrence wins: the caller deduplicates, but a stray repeat must not
        # be able to push a group's completion rank later than it really was.
        rank.setdefault(ref, position)
    completed = [max(rank[ref] for ref in group) for group in groups if group <= rank.keys()]
    return 1.0 / min(completed) if completed else 0.0


def ndcg_any_at_k(retrieved: Sequence[Ref], groups: Groups, k: int) -> float:
    """Binary-gain nDCG over the union of all groups. Ranking context only; it decides nothing."""
    retrieved, groups = _refs(retrieved), _groups(groups)
    _checked(groups, k)
    union = set().union(*groups)
    dcg = sum(1.0 / math.log2(i + 2) for i, ref in enumerate(retrieved[:k]) if ref in union)
    # Ideal ranking holds as many gold sentences as the cutoff has room for.
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(union))))
    return dcg / ideal if ideal else 0.0


def doc_recall_at_n(pages_retrieved: Sequence[str], groups: Groups, n: int) -> bool:
    """Stage-1 recall: one whole group's pages inside the top n. A hard ceiling on recall_at_k."""
    pages_retrieved = [title_key(title) for title in pages_retrieved]
    groups = _groups(groups)
    _checked(groups, n)
    top = set(pages_retrieved[:n])
    return any({page for page, _ in group} <= top for group in groups)


@dataclass(frozen=True, slots=True)
class Ceiling:
    """
    Best reachable accuracy given what retrieval returned. `overall` is not reportable alone.

    It bounds *evidence-derived* accuracy, which is not the same as accuracy. A model reading
    FEVER's claim-only artifacts is not bounded by it and can sail past: the evidence-free
    baseline scores 0.5865 on test against a ceiling of 0.0, because it reads no evidence at all.
    That gap is the artifact, and measuring it is the reason the baseline is a separate control
    rather than a footnote.
    """

    verifiable: float    # identical to strict recall; the name later phases cite
    overall: float       # 1 - verifiable_share * (1 - verifiable)
    nei_share: float     # the share `overall` gets for free: an oracle answers NEI evidence-free

    def to_dict(self) -> dict[str, float]:
        return {"verifiable": self.verifiable, "overall": self.overall, "nei_share": self.nei_share}


@dataclass(frozen=True, slots=True)
class CutoffMetrics:
    k: int
    recall: float
    recall_any: float
    mrr: float
    ndcg_any: float
    ceiling: Ceiling

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "recall": self.recall,
            "recall_any": self.recall_any,
            "mrr": self.mrr,
            "ndcg_any": self.ndcg_any,
            "ceiling": self.ceiling.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DocCutoffMetrics:
    n: int
    doc_recall: float

    def to_dict(self) -> dict[str, object]:
        return {"n": self.n, "doc_recall": self.doc_recall}


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    claims: int          # handed in, verifiable and NEI together
    verifiable: int
    nei: int
    cutoffs: tuple[CutoffMetrics, ...]
    doc_cutoffs: tuple[DocCutoffMetrics, ...] = ()

    @property
    def verifiable_share(self) -> float:
        return self.verifiable / self.claims

    @property
    def nei_share(self) -> float:
        return self.nei / self.claims

    def at(self, k: int) -> CutoffMetrics:
        for cutoff in self.cutoffs:
            if cutoff.k == k:
                return cutoff
        raise KeyError(f"no metrics at k={k}")

    def doc_at(self, n: int) -> DocCutoffMetrics:
        for cutoff in self.doc_cutoffs:
            if cutoff.n == n:
                return cutoff
        raise KeyError(f"no page metrics at n={n}")

    def to_dict(self) -> dict[str, object]:
        return {
            "claims": self.claims,
            "verifiable": self.verifiable,
            "nei": self.nei,
            "verifiable_share": self.verifiable_share,
            "nei_share": self.nei_share,
            "cutoffs": [cutoff.to_dict() for cutoff in self.cutoffs],
            "doc_cutoffs": [cutoff.to_dict() for cutoff in self.doc_cutoffs],
        }


def evaluate_retrieval(
    claims: Sequence[Claim],
    retrieved: Sequence[Sequence[Ref]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    pages_retrieved: Sequence[Sequence[str]] | None = None,
    ns: Sequence[int] = DEFAULT_NS,
) -> RetrievalReport:
    """
    Score a run: every metric at every cutoff, plus the ceilings later phases cite.

    NEI claims are counted but never scored. They are still handed in because the NEI
    share is what separates the two ceilings, and a report built from verifiable claims
    alone cannot state either one.
    """
    if len(claims) != len(retrieved):
        raise ValueError(f"got {len(claims)} claims and {len(retrieved)} retrieved lists")
    if pages_retrieved is not None and len(pages_retrieved) != len(claims):
        raise ValueError(f"got {len(claims)} claims and {len(pages_retrieved)} retrieved page lists")
    if not claims:
        raise ValueError("cannot score an empty run")

    rows: list[tuple[Groups, Sequence[Ref], Sequence[str]]] = []
    nei = 0
    for index, claim in enumerate(claims):
        if claim.label == NOT_ENOUGH_INFO:
            nei += 1
            continue
        if not claim.groups:
            raise ValueError(f"claim {claim.id} is labelled {claim.label} but carries no evidence")
        pages = () if pages_retrieved is None else pages_retrieved[index]
        rows.append((claim.groups, retrieved[index], pages))

    if not rows:
        raise ValueError("no verifiable claims to score")

    total = len(claims)
    verifiable_share = len(rows) / total
    nei_share = nei / total

    cutoffs = []
    for k in sorted(set(ks)):
        recall = sum(recall_at_k(refs, groups, k) for groups, refs, _ in rows) / len(rows)
        cutoffs.append(
            CutoffMetrics(
                k=k,
                recall=recall,
                recall_any=sum(recall_any_at_k(refs, groups, k) for groups, refs, _ in rows) / len(rows),
                mrr=sum(mrr_at_k(refs, groups, k) for groups, refs, _ in rows) / len(rows),
                ndcg_any=sum(ndcg_any_at_k(refs, groups, k) for groups, refs, _ in rows) / len(rows),
                ceiling=Ceiling(
                    verifiable=recall,
                    overall=1.0 - verifiable_share * (1.0 - recall),
                    nei_share=nei_share,
                ),
            )
        )

    doc_cutoffs = []
    if pages_retrieved is not None:
        for n in sorted(set(ns)):
            hits = sum(doc_recall_at_n(pages, groups, n) for groups, _, pages in rows)
            doc_cutoffs.append(DocCutoffMetrics(n=n, doc_recall=hits / len(rows)))

    return RetrievalReport(
        claims=total,
        verifiable=len(rows),
        nei=nei,
        cutoffs=tuple(cutoffs),
        doc_cutoffs=tuple(doc_cutoffs),
    )
