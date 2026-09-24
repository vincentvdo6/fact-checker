"""Complete pair-judge inputs must fit the contract before any model prediction."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.verdict.pair_judge import PairJudge


class WordTokenizer:
    def __init__(self):
        self.calls = []
        self.vocabulary = {}

    def token_id(self, word):
        if word not in self.vocabulary:
            self.vocabulary[word] = len(self.vocabulary) + 1
        return self.vocabulary[word]

    def __call__(self, premises, hypotheses, *, padding, truncation, return_tensors=None):
        self.calls.append((list(premises), list(hypotheses), padding, truncation))
        assert truncation is False
        rows = [[101, *(self.token_id(word) for word in premise.split()), 102,
                 *(self.token_id(word) for word in hypothesis.split()), 102]
                for premise, hypothesis in zip(premises, hypotheses, strict=True)]
        if not padding:
            return {"input_ids": rows}
        longest = max(map(len, rows))
        return {"input_ids": np.array([row + [0] * (longest - len(row)) for row in rows]),
                "attention_mask": np.array([[1] * len(row) + [0] * (longest - len(row)) for row in rows])}


class Session:
    def __init__(self):
        self.calls = []

    def run(self, _, inputs):
        self.calls.append(inputs)
        return [np.array([[float(row[1]) / 10, 0.0, 0.0, 0.0] for row in inputs["input_ids"]])]


def judge(tmp_path, max_length=8):
    (tmp_path / "contract.json").write_text(json.dumps({
        "labels": ["states", "states_negation", "bears_on", "unrelated"],
        "max_length": max_length, "template_id": "premise_hypothesis_v1"}), encoding="utf-8")
    result = PairJudge(model_dir=tmp_path)
    result._tokenizer = WordTokenizer()
    result._session = Session()
    return result


def assertion(identity, text="one two"):
    return {"id": identity, "text": text, "negated": False}


def unit(identity, text, definitions=()):
    return {"id": identity, "text": text, "definitions": [{"text": definition} for definition in definitions]}


def test_exact_boundary_is_scored_and_full_definition_overflow_is_omitted(tmp_path):
    pair_judge = judge(tmp_path)
    rows = pair_judge([assertion("a")], [unit("fits", "one two three"),
                                         unit("too_long", "one two", ["three four five six"])])
    assert [(row["assertion_id"], row["unit_id"], row["relation"]) for row in rows] == [
        ("a", "fits", "states")]
    assert pair_judge.last_input_overflows == [{"assertion_id": "a", "unit_id": "too_long",
                                                "reason": "input_over_limit", "input_tokens": 11,
                                                "max_input_tokens": 8}]
    assert len(pair_judge._session.calls) == 1
    assert all(call[3] is False for call in pair_judge._tokenizer.calls)


def test_long_assertion_does_not_displace_short_pairs_in_batch(tmp_path):
    pair_judge = judge(tmp_path)
    rows = pair_judge([assertion("short"), assertion("long", "one two three four five six")],
                      [unit("u1", "one"), unit("u2", "two")])
    assert [(row["assertion_id"], row["unit_id"]) for row in rows] == [("short", "u1"), ("short", "u2")]
    assert [entry["unit_id"] for entry in pair_judge.last_input_overflows] == ["u1", "u2"]
    assert len(pair_judge._session.calls) == 1
    assert len(pair_judge._session.calls[0]["input_ids"]) == 2
    assert all(row["relation"] == "states" and row["probabilities"] is not None for row in rows)


def test_valid_pair_order_survives_overflow_between_model_batches(tmp_path):
    pair_judge = judge(tmp_path)
    units = [unit(f"u{index}", f"word{index}") for index in range(17)]
    units.insert(8, unit("too_long", "one two three four five six"))
    rows = pair_judge([assertion("a")], units)
    assert [row["unit_id"] for row in rows] == [f"u{index}" for index in range(17)]
    assert [row["confidence"] for row in rows] == sorted(row["confidence"] for row in rows)
    assert len({row["confidence"] for row in rows}) == 17
    assert [entry["unit_id"] for entry in pair_judge.last_input_overflows] == ["too_long"]
    assert [len(call["input_ids"]) for call in pair_judge._session.calls] == [16, 1]


def test_all_oversize_and_empty_calls_never_load_session(tmp_path):
    pair_judge = judge(tmp_path)
    pair_judge._session = None
    assert pair_judge([assertion("long", "one two three four five six")], [unit("u", "one")]) == []
    assert len(pair_judge.last_input_overflows) == 1
    assert pair_judge._session is None
    assert pair_judge([], []) == []
    assert pair_judge.last_input_overflows == []
    assert pair_judge._session is None


def test_direct_logits_refuses_oversize_without_loading_session(tmp_path):
    pair_judge = judge(tmp_path)
    pair_judge._session = None
    with pytest.raises(ValueError, match="no input was truncated"):
        pair_judge.logits([("one", "two"), ("one two three four five six", "one")])
    assert pair_judge._session is None
