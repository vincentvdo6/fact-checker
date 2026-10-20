"""The NLI model may only add a support the pair judge read just below its band."""

from __future__ import annotations

from src.verdict.nli_confirm import SUPPORT_LOWER, SUPPORT_THRESHOLD, SupportConfirmedJudge


class Primary:
    """A pair judge stand-in: each unit's id names its (read_as, relation, confidence, raw_label)."""

    probe_negation = True
    measurement = {"counted": 7}
    direction_measurement = {"directional": 3}

    def __init__(self, readings):
        self.readings = readings
        self.last_input_overflows = [{"assertion_id": "a", "unit_id": "huge", "reason": "input_over_limit"}]

    def __call__(self, assertions, units):
        rows = []
        for assertion in assertions:
            for unit in units:
                read_as, relation, confidence, raw = self.readings[unit["id"]]
                rows.append({"assertion_id": assertion["id"], "unit_id": unit["id"], "relation": relation,
                             "read_as": read_as, "raw_label": raw, "confidence": confidence,
                             "judged_text": f"judged {assertion['text']}",
                             "span": unit["text"] if relation != "unrelated" else "", "qualifiers": [],
                             "status": "unverified"})
        return rows


class Nli:
    def __init__(self, entail=None, contradict=None):
        self.entail, self.contradict, self.calls = entail or {}, contradict or {}, []

    def __call__(self, pairs):
        self.calls.extend(pairs)
        return [None if premise.startswith("LONG") else
                {"entailment": self.entail.get(premise, 0.0), "neutral": 0.0,
                 "contradiction": self.contradict.get(premise, 0.0)} for premise, _ in pairs]


ASSERTION = [{"id": "a", "text": "Employers cut 92,000 jobs in February.", "negated": False}]


def units(*names):
    return [{"id": name, "text": name} for name in names]


def run(readings, nli, assertions=ASSERTION):
    judge = SupportConfirmedJudge(Primary(readings), nli, model_dir="missing")
    return {row["unit_id"]: row["relation"] for row in judge(assertions, units(*readings))}, judge


def test_a_near_miss_support_counts_only_when_the_nli_model_confirms_it():
    rows, _ = run({"near": ("states", "bears_on", 0.83, "states"), "doubted": ("states", "bears_on", 0.83, "states")},
                  Nli({"near": SUPPORT_THRESHOLD, "doubted": SUPPORT_THRESHOLD - 0.01}))
    assert rows == {"near": "states", "doubted": "bears_on"}


def test_only_near_misses_are_asked_and_nothing_counted_or_contradicted_changes():
    readings = {"banded": ("states", "states", 0.95, "states"),
                "denies": ("states_negation", "states_negation", 0.95, "states_negation"),
                "near denial": ("states_negation", "bears_on", 0.8, "states_negation"),
                "context": ("bears_on", "bears_on", 0.95, "bears_on"),
                "weak": ("states", "bears_on", SUPPORT_LOWER - 0.01, "states"),
                "noise": ("unrelated", "unrelated", 0.99, "unrelated")}
    nli = Nli({name: 1.0 for name in readings}, {name: 1.0 for name in readings})
    rows, _ = run(readings, nli)
    assert rows == {name: relation for name, (_, relation, _, _) in readings.items()}, "only near-miss supports may change"
    assert nli.calls == [], "the NLI model is asked about nothing outside the near-miss supports"


def test_a_support_read_through_a_positive_form_is_confirmed_by_contradiction():
    denial = [{"id": "a", "text": "We don't have a labor shortage.", "negated": True}]
    # The judge saw "We have a labor shortage" and read the page as denying it, which states the original.
    rows, _ = run({"page": ("states", "bears_on", 0.8, "states_negation")}, Nli(contradict={"page": 0.99}), denial)
    assert rows == {"page": "states"}
    rows, _ = run({"page": ("states", "bears_on", 0.8, "states_negation")}, Nli(entail={"page": 0.99}), denial)
    assert rows == {"page": "bears_on"}, "entailment of the positive form would deny the claim"


def test_the_nli_model_reads_the_text_the_judge_saw_with_its_definitions():
    nli = Nli({"near Defined.": 1.0})
    judge = SupportConfirmedJudge(Primary({"near": ("states", "bears_on", 0.8, "states")}), nli, model_dir="missing")
    judge(ASSERTION, [{"id": "near", "text": "near", "definitions": [{"text": "Defined."}]}])
    assert nli.calls == [("near Defined.", "judged Employers cut 92,000 jobs in February.")]


def test_a_pair_too_long_for_the_nli_model_stays_context_and_is_reported_beside_the_judges_own():
    rows, judge = run({"LONG page": ("states", "bears_on", 0.8, "states")}, Nli())
    assert rows == {"LONG page": "bears_on"}
    assert judge.last_input_overflows == [
        {"assertion_id": "a", "unit_id": "huge", "reason": "input_over_limit"},
        {"assertion_id": "a", "unit_id": "LONG page", "reason": "input_over_limit", "stage": "nli_confirm"}]


def test_the_card_quotes_the_record_of_this_system_not_the_pair_judge_alone(tmp_path):
    (tmp_path / "direction_measurement.json").write_text('{"directional": 9}', encoding="utf-8")
    judge = SupportConfirmedJudge(Primary({}), Nli(), model_dir=tmp_path)
    assert judge.direction_measurement == {"directional": 9} and judge.measurement == {"counted": 7}
    assert SupportConfirmedJudge(Primary({}), Nli(), model_dir=tmp_path / "none").direction_measurement is None


def test_the_runtime_refuses_pairs_longer_than_the_model_reads_instead_of_cutting_them(tmp_path, monkeypatch):
    import numpy as np

    from src.verdict import nli_confirm

    (tmp_path / "config.json").write_text('{"id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}')
    monkeypatch.delenv("VERDICT_THREADS", raising=False)
    runtime = nli_confirm.NliRuntime(tmp_path)
    assert runtime.threads == 4, "never every core"
    fed = []

    def tokenizer(premises, hypotheses, **kwargs):
        ids = [[0] * (len(p.split()) + len(h.split())) for p, h in zip(premises, hypotheses, strict=True)]
        if kwargs.get("return_tensors") == "np":
            width = max(map(len, ids))
            return {"input_ids": np.array([row + [0] * (width - len(row)) for row in ids])}
        assert kwargs.get("truncation") is False
        return {"input_ids": ids}

    class Session:
        def run(self, _, feed):
            fed.append(feed["input_ids"].shape[0])
            return [np.tile(np.array([[3.0, 0.0, 0.0]]), (feed["input_ids"].shape[0], 1))]

    runtime._tokenizer, runtime._session, runtime._inputs = tokenizer, Session(), {"input_ids"}
    long = " ".join(["word"] * nli_confirm.MAX_LENGTH)
    out = runtime([("short premise", "short claim"), (long, "claim")])
    assert out[1] is None and fed == [1], "the long pair never reaches the model"
    assert max(out[0], key=out[0].get) == "entailment", "labels follow the model's own id2label order"
