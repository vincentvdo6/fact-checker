"""
The claim-only probe: it must find leakage that is there and none that is not.

Both directions matter. A probe that cannot detect a planted signal would report every dataset
clean and retire the Phase 02 finding by accident. A probe that scores above the floor on random
text would manufacture an artifact in any dataset it was pointed at.
"""

from __future__ import annotations

import random

import pytest

from src.eval.artifact import NaiveBayes, measure_artifact, tokenize


def planted(n: int, seed: int = 0):
    """Claims whose label is fully determined by one marker word."""
    rng = random.Random(seed)
    filler = ["the", "a", "claim", "about", "something", "in", "year"]
    texts, labels = [], []
    for _ in range(n):
        label = rng.choice(["supported", "contradicted"])
        marker = "alpha" if label == "supported" else "beta"
        texts.append(" ".join(rng.choices(filler, k=6) + [marker]))
        labels.append(label)
    return texts, labels


def noise(n: int, seed: int = 1):
    """Claims whose label is independent of the text."""
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(60)]
    texts = [" ".join(rng.choices(words, k=8)) for _ in range(n)]
    labels = [rng.choice(["supported", "contradicted", "not_enough_evidence"]) for _ in range(n)]
    return texts, labels


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("The Sky, in 1998: it's BLUE!") == ["the", "sky", "in", "1998", "it's", "blue"]


def test_the_probe_finds_a_planted_marker():
    """If this fails the probe cannot see leakage at all, and a clean report means nothing."""
    train_texts, train_labels = planted(600)
    test_texts, test_labels = planted(200, seed=9)
    report = measure_artifact(train_texts, train_labels, test_texts, test_labels, dataset="planted")
    assert report.accuracy > 0.95
    assert report.lift > 0.4


def test_the_probe_finds_nothing_in_noise():
    """
    The opposite failure. A probe that beats the floor on random text would invent an artifact in
    whatever dataset it was pointed at, including one that genuinely has none.
    """
    train_texts, train_labels = noise(900)
    test_texts, test_labels = noise(400, seed=7)
    report = measure_artifact(train_texts, train_labels, test_texts, test_labels, dataset="noise")
    assert report.lift < 0.06, f"found {report.lift:+.4f} of leakage in random text"


def test_unknown_tokens_do_not_let_length_decide():
    """
    Scoring unseen tokens would add a constant per word, so a long claim would drift toward
    whichever class had the sparsest vocabulary -- an artifact of the probe, not the data.
    """
    model = NaiveBayes().fit(*planted(400))
    short = model.predict("zzz alpha")
    long = model.predict("zzz " * 200 + "alpha")
    assert short == long == "supported"


def test_the_majority_floor_comes_from_the_split_it_is_compared_against():
    """
    Taking it from train would flatter or punish the lift whenever the splits differ in balance,
    and AVeriTeC's do.
    """
    # Deliberately asymmetric floors: train 0.90, test 0.60. A fixture where both come out equal
    # cannot tell the two sources apart, which is how a vacuous version of this passed until
    # mutation testing swapped the source and nothing failed.
    train_texts = ["x"] * 90 + ["y"] * 10
    train_labels = ["supported"] * 90 + ["contradicted"] * 10
    test_texts = ["x"] * 60 + ["y"] * 40
    test_labels = ["supported"] * 60 + ["contradicted"] * 40

    report = measure_artifact(train_texts, train_labels, test_texts, test_labels, dataset="d")
    assert report.majority == pytest.approx(0.60)


def test_lift_is_accuracy_above_the_floor():
    report = measure_artifact(*planted(200), *planted(100, seed=3), dataset="d")
    assert report.lift == pytest.approx(report.accuracy - report.majority)


def test_the_report_serialises():
    import json

    payload = measure_artifact(*planted(200), *planted(100, seed=4), dataset="d").to_dict()
    assert json.loads(json.dumps(payload))["dataset"] == "d"


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="2 texts and 1 labels"):
        NaiveBayes().fit(["a", "b"], ["supported"])


def test_fitting_on_nothing_is_refused():
    with pytest.raises(ValueError, match="empty"):
        NaiveBayes().fit([], [])
