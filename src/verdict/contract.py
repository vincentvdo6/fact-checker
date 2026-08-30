"""
What a trained artifact must agree with before it is allowed anywhere near a number.

A tabular model can pin a list of feature names. A transformer cannot: its inputs are text, so
the things that must not drift are the tokenizer that turns text into ids, the template that
assembled the text, the sequence budget that truncated it, and the label order that turns three
logits into three verdicts. Any one of those can change silently and produce plausible wrong
numbers rather than a crash -- a permuted label order is the worst, because every metric still
computes, just against the wrong classes.

The contract travels inside the artifact zip as contract.json. It is written by the notebook,
which has no access to this module, so the field names are themselves the interface: from_dict
raises on a missing key rather than defaulting one, because a default here would paper over
exactly the drift this exists to catch.

Nothing in this file loads a model. Phase 02 runs no inference locally -- the notebooks export
predictions and every local number is computed from those -- so validation is a comparison of
declarations, not of behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

CONTRACT_VERSION = 1
CONTRACT_FILE = "contract.json"


@dataclass(frozen=True, slots=True)
class EncoderContract:
    base_model: str
    base_revision: str          # pinned commit; tokenizer files do change under a moving tag
    tokenizer_sha256: str       # over the concatenated tokenizer files -- the real identity check
    labels: tuple[str, ...]     # class-index order; append, never reorder
    max_length: int
    template_id: str
    variant: str                # retrieved | claim_only | gold
    seed: int
    torch_version: str
    transformers_version: str
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            **{f.name: getattr(self, f.name) for f in fields(self) if f.name != "labels"},
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> EncoderContract:
        expected = {f.name for f in fields(cls)}
        missing = expected - set(payload)
        if missing:
            raise ValueError(f"contract is missing {sorted(missing)}")
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"contract carries unknown fields {sorted(unknown)}")
        return cls(**{**payload, "labels": tuple(payload["labels"])})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EncoderContract:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def assert_compatible(self, other: EncoderContract) -> None:
        """
        Raise unless `other` was produced under the same input contract as this one.

        `variant` and `seed` are deliberately excluded: three variants ship from the same
        contract by design, and two seeds of one variant are still comparable artifacts. What
        must match is everything that changes what the model was shown.
        """
        for field in ("contract_version", "labels", "template_id", "max_length", "base_model", "tokenizer_sha256"):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                raise ValueError(f"contract mismatch on {field}: expected {mine!r}, artifact has {theirs!r}")
