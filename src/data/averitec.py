"""
Load AVeriTeC claims into the project's verdict space.

This is the external-validity check. Everything Phases 02 to 05 found traces to one property of
FEVER: claims were written *from* Wikipedia sentences, so their wording leaks the verdict and an
evidence-free model scores 0.5865 against a 0.3410 majority. AVeriTeC claims come from
fact-checking organisations and were written by people making a point, not by annotators looking
at a passage -- so if the artifact is FEVER's, it should be much weaker here.

Two structural differences from `src/data/fever.py`, and neither is cosmetic:

**Evidence is question-answer pairs, not sentence references.** An AVeriTeC annotator asked a
question, found a web page, and wrote down the answer. There is no fixed corpus, so there is
nothing for `recall_at_k` to be computed against and the Phase 01 retrieval stack does not
transfer at all. What transfers is the verdict model and every measurement built on it.

**The label space has four verdicts, not three.** `Conflicting Evidence/Cherrypicking` maps to
MIXED, which FEVER has no notion of -- a claim can be true in a way that misleads. `LabelSpace`
has carried that since Phase 00 and this is the first data to exercise it.

The test split ships unlabelled (a 14-byte shared-task file), so the project's three-way split is
carved from train and dev here, the same way `split_dev` carves FEVER's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.verdict.labels import AVERITEC, Verdict

DATA = Path("data/averitec")


@dataclass(frozen=True, slots=True)
class AveritecClaim:
    """
    One real-world claim and the evidence an annotator assembled for it.

    `key` groups near-duplicate claim text the way FEVER's does, so a claim cannot land in two
    splits. `questions` holds (question, answer) pairs flattened from the release: the answer
    types vary -- Boolean, Extractive, Abstractive -- and a Boolean answer carries its reasoning
    in a separate field, so both are joined rather than dropping the half that explains why.
    """

    id: int
    label: str                          # the project verdict, not AVeriTeC's native string
    text: str
    key: str
    questions: tuple[tuple[str, str], ...]
    speaker: str | None
    claim_date: str | None

    @property
    def evidence_text(self) -> str:
        """The annotator's evidence as one block, question and answer per line."""
        return "\n".join(f"{q} {a}" for q, a in self.questions)


def claim_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _answer_text(answer: dict) -> str:
    """
    One answer, including a Boolean's explanation.

    A bare "Yes" or "No" carries no evidence on its own -- the annotator's reasoning lives in
    `boolean_explanation`, and dropping it would leave the model a yes/no token to reason from.
    """
    text = str(answer.get("answer") or "").strip()
    explanation = str(answer.get("boolean_explanation") or "").strip()
    if explanation and explanation.lower() not in text.lower():
        return f"{text}. {explanation}" if text else explanation
    return text


def load_claims(path: str | Path, *, start_id: int = 0) -> list[AveritecClaim]:
    """
    Read one AVeriTeC release file.

    Ids are positional and offset by `start_id`, because the release carries none of its own and
    the splits have to be addressable. That makes the id a function of file order, so a changed
    upstream file would silently renumber everything -- which is why `scripts/fetch_averitec.py`
    asserts the claim count rather than only the byte size.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[AveritecClaim] = []
    for offset, record in enumerate(records):
        text = str(record["claim"]).strip()
        questions = tuple(
            (str(q.get("question") or "").strip(), " ".join(
                filter(None, (_answer_text(a) for a in q.get("answers") or []))
            ))
            for q in record.get("questions") or []
        )
        out.append(
            AveritecClaim(
                id=start_id + offset,
                label=AVERITEC.to_verdict(record["label"]).value,
                text=text,
                key=claim_key(text),
                questions=tuple((q, a) for q, a in questions if q or a),
                speaker=record.get("speaker") or None,
                claim_date=record.get("claim_date") or None,
            )
        )
    return out


def load_all(directory: str | Path = DATA) -> list[AveritecClaim]:
    """
    Train and dev together, ids disjoint.

    They are combined before splitting rather than used as-is: AVeriTeC's dev is only 500 claims,
    and the project needs three splits, so the boundary the release drew is not the boundary this
    project needs. Splitting is a separate concern -- see `src/data/splits.py`.
    """
    directory = Path(directory)
    train = load_claims(directory / "train.json")
    dev = load_claims(directory / "dev.json", start_id=len(train))
    return train + dev


def label_counts(claims: list[AveritecClaim]) -> dict[str, int]:
    counts = {verdict.value: 0 for verdict in AVERITEC.verdicts}
    for claim in claims:
        counts[claim.label] += 1
    return counts


def native_labels_seen(path: str | Path) -> set[str]:
    """
    Every native label string in a release file.

    `AVERITEC.native` was written from the release notes in Phase 00 and marked unconfirmed until
    data arrived. This is what confirms it: `to_verdict` raises on anything unmapped, so a new or
    renamed verdict fails on the first record rather than silently mislabelling a split.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(record["label"]) for record in records}


VERDICTS: tuple[str, ...] = tuple(v.value for v in AVERITEC.verdicts)
MIXED = Verdict.MIXED.value
