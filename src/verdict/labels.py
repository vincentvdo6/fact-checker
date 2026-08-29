"""
The project's verdict space, and how each dataset maps onto it.

Four verdicts, matching AVeriTeC so the Phase 04 reproduction needs no translation.
FEVER carries only three of them -- it has no notion of a claim with genuine evidence
on both sides -- so it declares a three-verdict space and MIXED never appears when
training on it. A four-way head trained on FEVER would leave one class with no
examples and a softmax that can still emit it.

NOT_ENOUGH_EVIDENCE is a prediction about the world: we looked and found nothing that
settles the claim. It is not abstention, which is a statement about the model's own
confidence (see src/calibration/bands.py). Both reach a reader as "no verdict", but
they are measured separately, and a calibrated system can be very sure evidence is
absent.

Verdict order inside a LabelSpace is the model's class-index order. It is part of the
on-disk contract for any trained head, so append rather than reorder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"


@dataclass(frozen=True, slots=True)
class LabelSpace:
    name: str
    verdicts: tuple[Verdict, ...]
    native: dict[str, Verdict]

    def to_verdict(self, label: str) -> Verdict:
        try:
            verdict = self.native[label]
        except KeyError:
            raise ValueError(f"{self.name}: unmapped label {label!r}") from None
        if verdict not in self.verdicts:
            raise ValueError(f"{self.name}: {label!r} maps to {verdict}, which is outside this space")
        return verdict

    def index(self, verdict: Verdict) -> int:
        """Class index for a model head. Stable across runs; see module docstring."""
        return self.verdicts.index(verdict)

    @property
    def size(self) -> int:
        return len(self.verdicts)


FEVER = LabelSpace(
    name="fever",
    verdicts=(Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.NOT_ENOUGH_EVIDENCE),
    native={
        "SUPPORTS": Verdict.SUPPORTED,
        "REFUTES": Verdict.CONTRADICTED,
        "NOT ENOUGH INFO": Verdict.NOT_ENOUGH_EVIDENCE,
    },
)

# Native strings are taken from the AVeriTeC release notes and are unconfirmed until
# the data lands in Phase 04. to_verdict raises on anything unmapped, so a mismatch
# fails loudly on the first record rather than silently mislabelling a split.
AVERITEC = LabelSpace(
    name="averitec",
    verdicts=(Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.MIXED, Verdict.NOT_ENOUGH_EVIDENCE),
    native={
        "Supported": Verdict.SUPPORTED,
        "Refuted": Verdict.CONTRADICTED,
        "Conflicting Evidence/Cherrypicking": Verdict.MIXED,
        "Not Enough Evidence": Verdict.NOT_ENOUGH_EVIDENCE,
    },
)

LABEL_SPACES: dict[str, LabelSpace] = {space.name: space for space in (FEVER, AVERITEC)}
