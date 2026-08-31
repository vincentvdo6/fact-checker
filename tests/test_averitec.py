"""
Loading AVeriTeC: the label mapping Phase 00 guessed at, and the evidence shape FEVER does not have.

Two things here are load-bearing. The native label strings were written from release notes before
any data existed, and a wrong one would mislabel a whole verdict silently. And a Boolean answer's
reasoning lives in a separate field, so dropping it would leave the model a bare "Yes" to reason
from -- which looks like evidence and is not.
"""

from __future__ import annotations

import pytest

from src.data.averitec import (
    DATA,
    MIXED,
    VERDICTS,
    AveritecClaim,
    _answer_text,
    claim_key,
    label_counts,
    load_all,
    load_claims,
    native_labels_seen,
)
from src.verdict.labels import AVERITEC, Verdict

requires_averitec = pytest.mark.skipif(
    not (DATA / "train.json").exists() or not (DATA / "dev.json").exists(),
    reason="AVeriTeC data absent; run python -m scripts.fetch_averitec",
)


def record(label="Supported", claim="a claim", questions=None):
    return {"claim": claim, "label": label, "questions": questions or [], "speaker": "S"}


def write(tmp_path, records):
    import json

    path = tmp_path / "part.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


# --- the label space, guessed in Phase 00 -----------------------------------------------------

def test_the_label_space_has_four_verdicts_including_mixed():
    """FEVER has no notion of a claim true in a way that misleads. This is the first data with one."""
    assert len(AVERITEC.verdicts) == 4
    assert Verdict.MIXED in AVERITEC.verdicts
    assert MIXED in VERDICTS


def test_an_unmapped_native_label_raises_on_the_first_record(tmp_path):
    """
    A renamed verdict upstream must fail loudly. Defaulting it to NEI would quietly relabel a
    whole class and every metric downstream would still compute.
    """
    with pytest.raises(ValueError, match="unmapped label"):
        load_claims(write(tmp_path, [record(label="Mostly True")]))


@requires_averitec
@pytest.mark.slow
def test_every_native_label_in_the_release_is_mapped():
    """
    AVERITEC.native was written from release notes in Phase 00 and marked unconfirmed. This is the
    check that confirms it against the actual files.
    """
    for name in ("train.json", "dev.json"):
        seen = native_labels_seen(DATA / name)
        unmapped = seen - set(AVERITEC.native)
        assert not unmapped, f"{name} carries unmapped labels: {sorted(unmapped)}"


# --- evidence shape ---------------------------------------------------------------------------

def test_a_boolean_answer_keeps_its_explanation():
    """A bare "No" is not evidence. The annotator's reasoning is in a separate field."""
    text = _answer_text({"answer": "No", "boolean_explanation": "He worked in law, not energy."})
    assert text.startswith("No")
    assert "law, not energy" in text


def test_an_explanation_already_contained_is_not_repeated():
    text = _answer_text({"answer": "No, he worked in law", "boolean_explanation": "no, he worked in law"})
    assert text.count("worked in law") == 1


def test_an_answer_without_an_explanation_is_unchanged():
    assert _answer_text({"answer": "1998", "answer_type": "Extractive"}) == "1998"


def test_questions_and_answers_are_paired(tmp_path):
    path = write(tmp_path, [record(questions=[
        {"question": "Did he?", "answers": [{"answer": "No", "boolean_explanation": "because X"}]},
        {"question": "When?", "answers": [{"answer": "1998"}]},
    ])])
    claim = load_claims(path)[0]
    assert len(claim.questions) == 2
    assert claim.questions[0][0] == "Did he?"
    assert "because X" in claim.questions[0][1]
    assert claim.questions[1] == ("When?", "1998")


def test_evidence_text_joins_every_pair():
    claim = AveritecClaim(
        id=0, label="supported", text="c", key="c",
        questions=(("Q1", "A1"), ("Q2", "A2")), speaker=None, claim_date=None,
    )
    assert claim.evidence_text == "Q1 A1\nQ2 A2"


def test_a_claim_with_no_questions_loads_with_empty_evidence(tmp_path):
    """282 train claims are NOT ENOUGH EVIDENCE; some carry nothing at all."""
    claim = load_claims(write(tmp_path, [record(label="Not Enough Evidence")]))[0]
    assert claim.questions == ()
    assert claim.evidence_text == ""


def test_multiple_answers_to_one_question_are_joined(tmp_path):
    path = write(tmp_path, [record(questions=[
        {"question": "Q", "answers": [{"answer": "first"}, {"answer": "second"}]},
    ])])
    assert load_claims(path)[0].questions[0][1] == "first second"


# --- identity and ids -------------------------------------------------------------------------

def test_claim_key_normalises_whitespace_and_case():
    assert claim_key("  The   Sky  Is Blue ") == claim_key("the sky is blue")


def test_ids_are_offset_so_train_and_dev_do_not_collide(tmp_path):
    first = load_claims(write(tmp_path, [record(claim="a"), record(claim="b")]))
    second = load_claims(write(tmp_path, [record(claim="c")]), start_id=len(first))
    assert [c.id for c in first] == [0, 1]
    assert [c.id for c in second] == [2]


@requires_averitec
@pytest.mark.slow
def test_the_release_loads_at_the_expected_size():
    claims = load_all()
    assert len(claims) == 3568, f"expected 3,068 train + 500 dev, got {len(claims):,}"
    assert len({c.id for c in claims}) == len(claims), "ids collide across train and dev"


@requires_averitec
@pytest.mark.slow
def test_every_verdict_including_mixed_is_present():
    counts = label_counts(load_all())
    assert all(counts[v] > 0 for v in VERDICTS), counts
    assert counts[MIXED] > 0, "MIXED is the verdict FEVER cannot express; it must appear here"


@requires_averitec
@pytest.mark.slow
def test_the_majority_baseline_is_recorded():
    """
    Read every AVeriTeC accuracy against this, the way FEVER's 0.3410 is quoted. AVeriTeC is far
    more skewed -- Refuted dominates -- so a model can look strong by never saying anything else.
    """
    counts = label_counts(load_all())
    total = sum(counts.values())
    majority = max(counts.values()) / total
    assert 0.55 < majority < 0.60, f"majority baseline moved: {majority:.4f} of {total:,}"
