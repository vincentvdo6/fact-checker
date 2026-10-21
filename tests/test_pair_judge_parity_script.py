from __future__ import annotations

from scripts.check_pair_judge_parity import comparable


class LengthJudge:
    """Token length is the premise's word count; the limit is the judge's own."""

    max_length = 3

    def _token_lengths(self, pairs: list[tuple[str, str]]) -> list[int]:
        return [len(premise.split()) for premise, _ in pairs]


def test_only_pairs_the_judge_scores_whole_are_compared_and_the_rest_are_counted():
    pairs = [{"id": "fits", "premise": "a b c", "hypothesis": "h"},
             {"id": "long", "premise": "a b c d", "hypothesis": "h"},
             {"id": "short", "premise": "a", "hypothesis": "h"}]
    kept, over_limit = comparable(LengthJudge(), pairs)
    assert [pair["id"] for pair in kept] == ["fits", "short"], "a pair exactly at the limit is scored, in order"
    assert over_limit == 1
