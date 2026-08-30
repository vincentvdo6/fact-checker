"""Retrieval metrics: group completion, the loose/strict gap, ranking, and the two ceilings."""

from __future__ import annotations

import json
import random

import pytest

from src.data.fever import NOT_ENOUGH_INFO, SUPPORTS, Claim, claim_key
from src.eval.retrieval import (
    CutoffMetrics,
    doc_recall_at_n,
    evaluate_retrieval,
    mrr_at_k,
    ndcg_any_at_k,
    recall_any_at_k,
    recall_at_k,
)

ADA_0 = ("Ada Lovelace", 0)
ADA_4 = ("Ada Lovelace", 4)
BABBAGE_1 = ("Charles Babbage", 1)
ENGINE_2 = ("Analytical Engine", 2)
NOISE = [("Punched card", i) for i in range(10)]

PAIR = (frozenset({ADA_0, BABBAGE_1}),)          # one group, two sentences
SINGLE = (frozenset({ADA_0}),)


def make(groups: tuple[frozenset[tuple[str, int]], ...], label: str = SUPPORTS, id_: int = 0) -> Claim:
    text = f"claim {id_}"
    return Claim(id=id_, label=label, text=text, key=claim_key(text), groups=groups, pages=())


def pages_of(retrieved: list[tuple[str, int]]) -> list[str]:
    """Page titles in retrieval order, deduplicated -- what stage 1 would have returned."""
    return list(dict.fromkeys(page for page, _ in retrieved))


POOL = [(f"Page {letter}", sentence) for letter in "ABCD" for sentence in range(3)]


def sample(rng: random.Random) -> tuple[list[tuple[str, int]], tuple[frozenset[tuple[str, int]], ...]]:
    groups = tuple({frozenset(rng.sample(POOL, rng.randint(1, 3))) for _ in range(rng.randint(1, 3))})
    retrieved = rng.sample(POOL, rng.randint(0, len(POOL)))
    return retrieved, groups


def sweep(seed: int = 17, draws: int = 300):
    """Random rankings against random gold groups, at every cutoff around the pool size."""
    rng = random.Random(seed)
    for _ in range(draws):
        retrieved, groups = sample(rng)
        for k in range(len(POOL) + 3):
            yield retrieved, groups, k


# --- strict, group-complete recall ------------------------------------------------------------


def test_half_a_group_scores_nothing_and_the_last_member_scores_it():
    assert recall_at_k([ADA_0, NOISE[0]], PAIR, 5) is False
    assert recall_at_k([ADA_0, NOISE[0], BABBAGE_1], PAIR, 5) is True


def test_a_later_group_completing_is_enough():
    groups = (frozenset({ADA_0, BABBAGE_1}), frozenset({ENGINE_2}))
    assert recall_at_k([ADA_0, NOISE[0]], groups, 5) is False
    assert recall_at_k([ADA_0, NOISE[0], ENGINE_2], groups, 5) is True


def test_recall_ignores_order_within_the_cutoff():
    forward = [ADA_0, NOISE[0], BABBAGE_1]
    assert recall_at_k(forward, PAIR, 3) is recall_at_k(list(reversed(forward)), PAIR, 3) is True


def test_recall_respects_the_cutoff():
    retrieved = [ADA_0, NOISE[0], BABBAGE_1]
    assert recall_at_k(retrieved, PAIR, 2) is False
    assert recall_at_k(retrieved, PAIR, 3) is True


def test_extra_gold_sentences_from_the_wrong_group_do_not_complete_one():
    """Two disjoint groups half-retrieved each: the union is satisfying, no group is."""
    groups = (frozenset({ADA_0, BABBAGE_1}), frozenset({ADA_4, ENGINE_2}))
    assert recall_at_k([ADA_0, ENGINE_2], groups, 5) is False
    assert recall_any_at_k([ADA_0, ENGINE_2], groups, 5) is True


# --- the loose metric and the gap it hides ----------------------------------------------------


def test_recall_any_credits_a_single_member_that_strict_recall_refuses():
    assert recall_any_at_k([ADA_0], PAIR, 5) is True
    assert recall_at_k([ADA_0], PAIR, 5) is False


def test_recall_any_never_falls_below_strict_recall():
    for retrieved, groups, k in sweep():
        assert recall_any_at_k(retrieved, groups, k) >= recall_at_k(retrieved, groups, k)


def test_the_two_recalls_agree_on_single_sentence_groups():
    for retrieved, groups, k in sweep(seed=5, draws=100):
        singles = tuple(frozenset({ref}) for group in groups for ref in group)
        assert recall_any_at_k(retrieved, singles, k) == recall_at_k(retrieved, singles, k)


# --- group-aware MRR --------------------------------------------------------------------------


def test_mrr_completes_a_group_at_its_last_member():
    """Members at ranks 1 and 3. Reading the rank off the first member gives 1/1 instead."""
    assert mrr_at_k([ADA_0, NOISE[0], BABBAGE_1], PAIR, 5) == pytest.approx(1 / 3)


def test_mrr_is_zero_when_no_group_completes_within_the_cutoff():
    assert mrr_at_k([ADA_0, NOISE[0], BABBAGE_1], PAIR, 2) == 0.0


def test_mrr_takes_the_earliest_completing_group():
    groups = (frozenset({ADA_0, BABBAGE_1}), frozenset({ENGINE_2}))
    # The pair completes at rank 4, the singleton at rank 2.
    assert mrr_at_k([ADA_0, ENGINE_2, NOISE[0], BABBAGE_1], groups, 5) == pytest.approx(1 / 2)


def test_mrr_is_sensitive_to_order_where_recall_is_not():
    late = [ADA_0, NOISE[0], BABBAGE_1]
    early = [ADA_0, BABBAGE_1, NOISE[0]]
    assert recall_at_k(late, PAIR, 3) == recall_at_k(early, PAIR, 3)
    assert mrr_at_k(late, PAIR, 3) == pytest.approx(1 / 3)
    assert mrr_at_k(early, PAIR, 3) == pytest.approx(1 / 2)


def test_mrr_is_positive_exactly_when_a_group_is_complete():
    for retrieved, groups, k in sweep():
        assert (mrr_at_k(retrieved, groups, k) > 0.0) == recall_at_k(retrieved, groups, k)


def test_mrr_never_exceeds_the_reciprocal_of_the_smallest_group():
    """A group of size m cannot complete before rank m, so its best reciprocal rank is 1/m."""
    for retrieved, groups, k in sweep(seed=3, draws=100):
        smallest = min(len(group) for group in groups)
        assert mrr_at_k(retrieved, groups, k) <= 1 / smallest + 1e-12


# --- nDCG -------------------------------------------------------------------------------------


def test_ndcg_matches_a_hand_computed_value():
    groups = (frozenset({ADA_0}), frozenset({BABBAGE_1, ENGINE_2}))
    retrieved = [NOISE[0], ADA_0, BABBAGE_1]
    # gains at ranks 2 and 3 over an ideal of three gold sentences:
    # (1/log2(3) + 1/log2(4)) / (1 + 1/log2(3) + 1/log2(4))
    assert ndcg_any_at_k(retrieved, groups, 3) == pytest.approx(0.5307212739772434)


def test_ndcg_ideal_is_capped_at_the_cutoff():
    """Three gold sentences, room for two: a perfect top-2 is 1.0, not 1.63/2.13."""
    groups = (frozenset({ADA_0, BABBAGE_1, ENGINE_2}),)
    assert ndcg_any_at_k([ADA_0, BABBAGE_1], groups, 2) == pytest.approx(1.0)


def test_ndcg_falls_when_gold_is_pushed_down_the_ranking():
    top = ndcg_any_at_k([ADA_0, NOISE[0], NOISE[1]], SINGLE, 3)
    bottom = ndcg_any_at_k([NOISE[0], NOISE[1], ADA_0], SINGLE, 3)
    assert top == pytest.approx(1.0)
    assert bottom == pytest.approx(0.5)  # 1/log2(4) over an ideal of 1.0


def test_ndcg_can_be_high_while_strict_recall_is_zero():
    """Why the metric is named _any: it grades the ranking, not the verification."""
    groups = (frozenset({ADA_0, BABBAGE_1}), frozenset({ADA_4, ENGINE_2}))
    retrieved = [ADA_0, ADA_4]
    assert ndcg_any_at_k(retrieved, groups, 2) == pytest.approx(1.0)
    assert recall_at_k(retrieved, groups, 2) is False


def test_ndcg_stays_within_the_unit_interval_and_tracks_the_loose_recall():
    for retrieved, groups, k in sweep():
        score = ndcg_any_at_k(retrieved, groups, k)
        assert 0.0 <= score <= 1.0
        assert (score > 0.0) == recall_any_at_k(retrieved, groups, k)


# --- page-level recall ------------------------------------------------------------------------


def test_doc_recall_ignores_sentence_indices_but_still_needs_a_whole_group():
    groups = (frozenset({ADA_0, BABBAGE_1}),)
    assert doc_recall_at_n(["Ada Lovelace"], groups, 5) is False
    assert doc_recall_at_n(["Ada Lovelace", "Charles Babbage"], groups, 5) is True


def test_doc_recall_can_pass_where_sentence_recall_fails():
    """The gap between the stages: right page, wrong sentence."""
    assert doc_recall_at_n(["Ada Lovelace"], SINGLE, 1) is True
    assert recall_at_k([ADA_4], SINGLE, 1) is False


def test_doc_recall_bounds_sentence_recall():
    for retrieved, groups, k in sweep():
        pages = pages_of(retrieved[:k])
        if recall_at_k(retrieved, groups, k):
            assert doc_recall_at_n(pages, groups, len(pages))


# --- degenerate inputs ------------------------------------------------------------------------


@pytest.mark.parametrize("metric", [recall_at_k, recall_any_at_k, mrr_at_k, ndcg_any_at_k])
def test_sentence_metrics_reject_a_claim_with_no_evidence(metric):
    with pytest.raises(ValueError, match="no gold evidence"):
        metric([ADA_0], (), 5)


def test_doc_recall_rejects_a_claim_with_no_evidence():
    with pytest.raises(ValueError, match="no gold evidence"):
        doc_recall_at_n(["Ada Lovelace"], (), 5)


def test_cutoff_beyond_the_retrieved_list_is_fine():
    retrieved = [ADA_0, BABBAGE_1]
    assert recall_at_k(retrieved, PAIR, 500) is True
    assert mrr_at_k(retrieved, PAIR, 500) == pytest.approx(1 / 2)
    assert ndcg_any_at_k(retrieved, PAIR, 500) == pytest.approx(1.0)
    assert doc_recall_at_n(pages_of(retrieved), PAIR, 500) is True


def test_zero_cutoff_scores_nothing():
    retrieved = [ADA_0, BABBAGE_1]
    assert recall_at_k(retrieved, PAIR, 0) is False
    assert recall_any_at_k(retrieved, PAIR, 0) is False
    assert mrr_at_k(retrieved, PAIR, 0) == 0.0
    assert ndcg_any_at_k(retrieved, PAIR, 0) == 0.0
    assert doc_recall_at_n(pages_of(retrieved), PAIR, 0) is False


def test_empty_ranking_scores_nothing():
    assert recall_at_k([], PAIR, 5) is False
    assert mrr_at_k([], PAIR, 5) == 0.0
    assert ndcg_any_at_k([], PAIR, 5) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, recall_any_at_k, mrr_at_k, ndcg_any_at_k])
def test_negative_cutoff_rejected(metric):
    with pytest.raises(ValueError, match="non-negative"):
        metric([ADA_0, BABBAGE_1], PAIR, -1)


def test_a_repeated_ranking_entry_does_not_delay_completion():
    assert mrr_at_k([ADA_0, ADA_0, BABBAGE_1], PAIR, 5) == pytest.approx(1 / 3)


# --- the aggregate ----------------------------------------------------------------------------


def run(hits: int, misses: int, neis: int, *, with_pages: bool = False):
    """A run whose strict recall at any k >= 1 is hits / (hits + misses)."""
    claims: list[Claim] = []
    retrieved: list[list[tuple[str, int]]] = []
    pages: list[list[str]] = []
    for i in range(hits):
        claims.append(make(SINGLE, id_=i))
        retrieved.append([ADA_0])
        pages.append(["Ada Lovelace"])
    for i in range(misses):
        claims.append(make(SINGLE, id_=100 + i))
        retrieved.append([NOISE[0]])
        pages.append(["Punched card"])
    for i in range(neis):
        claims.append(make((), label=NOT_ENOUGH_INFO, id_=200 + i))
        retrieved.append([NOISE[1]])
        pages.append(["Punched card"])
    return evaluate_retrieval(
        claims,
        retrieved,
        ks=(1, 5),
        pages_retrieved=pages if with_pages else None,
        ns=(1,),
    )


def test_report_counts_split_verifiable_from_nei():
    report = run(hits=2, misses=2, neis=2)
    assert (report.claims, report.verifiable, report.nei) == (6, 4, 2)
    assert report.verifiable_share == pytest.approx(2 / 3)
    assert report.nei_share == pytest.approx(1 / 3)


def test_recall_is_averaged_over_verifiable_claims_only():
    """Adding NEI rows must not dilute recall the way a denominator of all claims would."""
    without_nei = run(hits=2, misses=2, neis=0).at(5)
    with_nei = run(hits=2, misses=2, neis=6).at(5)
    assert without_nei.recall == pytest.approx(0.5)
    assert with_nei.recall == pytest.approx(0.5)
    assert with_nei.mrr == pytest.approx(without_nei.mrr)


def test_ceiling_verifiable_is_strict_recall():
    for cutoff in run(hits=3, misses=1, neis=2).cutoffs:
        assert cutoff.ceiling.verifiable == cutoff.recall


def test_ceiling_overall_exceeds_the_verifiable_ceiling_by_the_nei_term():
    ceiling = run(hits=2, misses=2, neis=2).at(5).ceiling
    assert ceiling.verifiable == pytest.approx(0.5)
    assert ceiling.nei_share == pytest.approx(1 / 3)
    assert ceiling.overall == pytest.approx(1 - (2 / 3) * (1 - 0.5))
    assert ceiling.overall > ceiling.verifiable
    # The whole inflation: free accuracy on NEI, which needs no evidence at all.
    assert ceiling.overall - ceiling.verifiable == pytest.approx(ceiling.nei_share * (1 - ceiling.verifiable))


def test_the_two_ceilings_coincide_without_nei_claims():
    ceiling = run(hits=1, misses=1, neis=0).at(5).ceiling
    assert ceiling.nei_share == 0.0
    assert ceiling.overall == pytest.approx(ceiling.verifiable)


def test_nei_share_travels_inside_the_ceiling_structure():
    payload = run(hits=2, misses=2, neis=2).at(5).ceiling.to_dict()
    assert set(payload) == {"verifiable", "overall", "nei_share"}


def test_cutoffs_cover_every_requested_k_once_and_in_order():
    report = run(hits=1, misses=1, neis=1)
    assert [cutoff.k for cutoff in report.cutoffs] == [1, 5]
    with pytest.raises(KeyError):
        report.at(10)


def test_recall_does_not_fall_as_the_cutoff_grows():
    claims = [make(PAIR, id_=0)]
    retrieved = [[ADA_0, NOISE[0], BABBAGE_1]]
    report = evaluate_retrieval(claims, retrieved, ks=(1, 2, 3))
    assert [report.at(k).recall for k in (1, 2, 3)] == [0.0, 0.0, 1.0]
    assert [report.at(k).recall_any for k in (1, 2, 3)] == [1.0, 1.0, 1.0]


def test_doc_cutoffs_appear_only_when_pages_are_supplied():
    assert run(hits=1, misses=1, neis=0).doc_cutoffs == ()
    assert run(hits=1, misses=1, neis=0, with_pages=True).doc_at(1).doc_recall == pytest.approx(0.5)


def test_a_verifiable_claim_without_evidence_raises():
    with pytest.raises(ValueError, match="carries no evidence"):
        evaluate_retrieval([make((), label=SUPPORTS, id_=7)], [[ADA_0]])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="2 claims and 1 retrieved lists"):
        evaluate_retrieval([make(SINGLE, id_=0), make(SINGLE, id_=1)], [[ADA_0]])


def test_mismatched_page_lengths_rejected():
    with pytest.raises(ValueError, match="retrieved page lists"):
        evaluate_retrieval([make(SINGLE, id_=0)], [[ADA_0]], pages_retrieved=[])


def test_empty_run_rejected():
    with pytest.raises(ValueError, match="empty run"):
        evaluate_retrieval([], [])


def test_a_run_of_only_nei_claims_is_rejected():
    with pytest.raises(ValueError, match="no verifiable claims"):
        evaluate_retrieval([make((), label=NOT_ENOUGH_INFO, id_=0)], [[ADA_0]])


def test_report_is_json_serializable():
    report = run(hits=2, misses=1, neis=1, with_pages=True)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["claims"] == 4
    assert payload["cutoffs"][0]["ceiling"]["nei_share"] == pytest.approx(0.25)
    assert payload["doc_cutoffs"] == [{"n": 1, "doc_recall": pytest.approx(2 / 3)}]


def test_report_metrics_match_the_single_claim_functions():
    """The aggregate is a mean of the functions above, not a second implementation."""
    rng = random.Random(29)
    claims, retrieved = [], []
    for i in range(25):
        refs, groups = sample(rng)
        claims.append(make(groups, id_=i))
        retrieved.append(refs)
    report = evaluate_retrieval(claims, retrieved, ks=(3,))
    metrics: CutoffMetrics = report.at(3)
    expected = [
        sum(fn(r, c.groups, 3) for c, r in zip(claims, retrieved, strict=True)) / len(claims)
        for fn in (recall_at_k, recall_any_at_k, mrr_at_k, ndcg_any_at_k)
    ]
    assert [metrics.recall, metrics.recall_any, metrics.mrr, metrics.ndcg_any] == pytest.approx(expected)


def test_nfd_gold_titles_score_against_nfc_retrieved():
    """
    Gold titles arrive as FEVER released them (NFD for 170 of 14,533); retrieved titles come
    from the store composed. Without normalizing both sides these claims cannot score a hit at
    any k, which reads as retrieval difficulty rather than a bug.
    """
    nfd = "Beyonce\u0301_Knowles"
    nfc = "Beyonc\u00e9_Knowles"
    assert nfd != nfc

    groups = [frozenset({(nfd, 3)})]
    retrieved = [(nfc, 3)]

    assert recall_at_k(retrieved, groups, 5)
    assert recall_any_at_k(retrieved, groups, 5)
    assert mrr_at_k(retrieved, groups, 5) == 1.0
    assert ndcg_any_at_k(retrieved, groups, 5) == 1.0
    assert doc_recall_at_n([nfc], groups, 5)


def test_normalization_does_not_merge_distinct_titles():
    """Composing must not make two different pages equal; 14,533 gold titles stay 14,533."""
    groups = [frozenset({("Paris", 0)})]
    assert not recall_at_k([("Paris_Hilton", 0)], groups, 5)
    assert not doc_recall_at_n(["Paris_Hilton"], groups, 5)


def test_an_empty_group_is_rejected_rather_than_satisfied():
    """The empty set is a subset of everything, so one would score a hit against nothing."""
    for call in (
        lambda: recall_at_k([], [frozenset()], 5),
        lambda: recall_any_at_k([], [frozenset()], 5),
        lambda: mrr_at_k([], [frozenset()], 5),
        lambda: ndcg_any_at_k([], [frozenset()], 5),
        lambda: doc_recall_at_n([], [frozenset()], 5),
    ):
        with pytest.raises(ValueError, match="empty evidence group"):
            call()
