"""
How many sentences the encoder will actually pack for each row.

`n_evidence_used` is exported by the training notebook, but only for the splits it scored. Every
question about what the model *read* -- was its gold inside the prefix, is this row grounded --
needs that number on splits the notebook never touched, so it has to be recomputable here.

It is recomputed through the shipped `select_evidence`, advancing one Random across the split in
file order exactly as the notebook's `encode()` does. That equivalence is not assumed: on the
calibration split it reproduces the notebook's own exported counts 2,000 of 2,000, and
`tests/test_train_serve_skew.py` holds it there.

The tokenizer is the base checkpoint's, not a fine-tuned one -- fine-tuning does not change the
vocabulary -- so an installed artifact's copy and the upstream model give identical counts. The
directory is a parameter because the dataset builder runs before any artifact is installed.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from functools import lru_cache

from src.verdict.encode import select_evidence


@lru_cache(maxsize=4)
def _tokenizer(source: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source)


def budgets_for(
    rows: Sequence[dict],
    *,
    variant: str,
    max_length: int,
    seed: int,
    tokenizer_source: str,
) -> list[int]:
    """
    One packing budget per row, in the order given.

    Order matters and is not an implementation detail: the notebook advances a single Random over
    the split, so a row's evidence shuffle depends on how many rows preceded it. Reordering the
    input silently produces budgets for a run nobody performed.
    """
    tokenizer = _tokenizer(tokenizer_source)

    def measure(first: str, second: str) -> int:
        return len(tokenizer(first, second)["input_ids"])

    rng = random.Random(seed)
    return [len(select_evidence(row, variant, max_length, measure, rng)) for row in rows]
