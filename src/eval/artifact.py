"""
How much of a dataset's label is readable from the claim wording alone.

This is the measurement the whole of Phase 06 turns on. Phases 02 to 05 all traced back to one
property of FEVER: its claims were written *from* Wikipedia sentences, so the phrasing carries the
verdict. An evidence-free DeBERTa scored 0.5865 against a 0.3410 majority, and that leak is why
groundedness and correctness diverged, why honest abstention cost coverage, and why relabelling
ungrounded rows backfired. Whether those are findings about fact-checking or findings about FEVER
depends entirely on whether the leak exists elsewhere.

Answering that does not need a GPU. A multinomial naive Bayes over unigrams is a weak model on
purpose: it has no world knowledge, no reasoning, and no evidence -- it can only exploit the
correlation between wording and label. Whatever it scores above the majority baseline is leakage,
and comparing that *lift* across datasets is the point. The absolute numbers are not comparable
between a three-class and a four-class problem; the lift over each dataset's own floor is.

Naive Bayes rather than a fitted linear model because it has no optimiser, no learning rate and no
seed: the number it produces is a deterministic property of the text and labels, which is what a
diagnostic should be. Add-one smoothing, and unseen tokens are ignored rather than assigned a
prior, so a long claim full of unknown words is not pushed toward whichever class is largest.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass(frozen=True, slots=True)
class ArtifactReport:
    dataset: str
    classes: int
    train_claims: int
    test_claims: int
    majority: float
    accuracy: float

    @property
    def lift(self) -> float:
        """Accuracy above the dataset's own majority floor. The comparable quantity."""
        return self.accuracy - self.majority

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "dataset": self.dataset, "classes": self.classes,
            "train_claims": self.train_claims, "test_claims": self.test_claims,
            "majority": self.majority, "accuracy": self.accuracy, "lift": self.lift,
        }


class NaiveBayes:
    """Multinomial naive Bayes over unigrams, add-one smoothed."""

    def __init__(self) -> None:
        self.log_prior: dict[str, float] = {}
        self.log_likelihood: dict[str, dict[str, float]] = {}
        self.vocabulary: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> NaiveBayes:
        if len(texts) != len(labels):
            raise ValueError(f"got {len(texts)} texts and {len(labels)} labels")
        if not texts:
            raise ValueError("cannot fit on an empty sample")

        counts: dict[str, Counter] = defaultdict(Counter)
        totals: Counter = Counter()
        for text, label in zip(texts, labels, strict=True):
            tokens = tokenize(text)
            counts[label].update(tokens)
            totals[label] += 1
            self.vocabulary.update(tokens)

        size = len(self.vocabulary)
        for label, label_counts in counts.items():
            self.log_prior[label] = math.log(totals[label] / len(labels))
            denominator = sum(label_counts.values()) + size
            self.log_likelihood[label] = {
                token: math.log((label_counts[token] + 1) / denominator) for token in self.vocabulary
            }
            # One entry for tokens the class never saw, so scoring does not need a branch.
            self.log_likelihood[label]["<unseen>"] = math.log(1 / denominator)
        return self

    def predict(self, text: str) -> str:
        # Unknown tokens are skipped rather than smoothed: they carry no evidence about the label,
        # and scoring them would let claim *length* decide the answer.
        tokens = [t for t in tokenize(text) if t in self.vocabulary]
        best, best_score = None, -math.inf
        for label, prior in self.log_prior.items():
            likelihood = self.log_likelihood[label]
            score = prior + sum(likelihood.get(t, likelihood["<unseen>"]) for t in tokens)
            if score > best_score:
                best, best_score = label, score
        return best


def measure_artifact(
    train_texts: list[str], train_labels: list[str],
    test_texts: list[str], test_labels: list[str],
    *, dataset: str,
) -> ArtifactReport:
    """
    Fit on train claim text only, score on test, and report the lift over the majority floor.

    The majority baseline is computed on *test*, because that is what the accuracy is measured
    against. Taking it from train would flatter or punish the lift whenever the splits differ in
    balance, which they do.
    """
    model = NaiveBayes().fit(train_texts, train_labels)
    correct = sum(model.predict(t) == y for t, y in zip(test_texts, test_labels, strict=True))
    counts = Counter(test_labels)
    return ArtifactReport(
        dataset=dataset,
        classes=len(set(train_labels) | set(test_labels)),
        train_claims=len(train_texts),
        test_claims=len(test_texts),
        majority=max(counts.values()) / len(test_labels),
        accuracy=correct / len(test_labels),
    )
