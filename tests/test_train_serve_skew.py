"""
The exported token lengths must be reproducible from the repository's own encoder.

This is the guard against train/serve skew, and it is the reason `encode_spec.py` is shipped
byte-identical rather than reimplemented. `tests/test_encode.py` proves the two files match;
this proves the file that trained the model is the file this repository runs -- that the same
claim and the same evidence produce the same token ids on Kaggle and here.

A drift here is silent in the worst way. Nothing raises, every metric computes, and the model is
simply scored on text it was never trained on. Comparing lengths catches it because a changed
template, a changed budget, a changed sentence clip or a changed title rendering all move the
count, and matching on 200 rows by accident is not a thing that happens.

Skips unless a full (non-smoke) artifact is installed, since only then is there anything to check.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import pytest

from src.verdict.contract import CONTRACT_FILE, EncoderContract
from src.verdict.encode import build_input, select_evidence

MODELS = Path("models/verdict")
DATA = Path("data/kaggle/fever-verdict-v1")
CHECK_ROWS = 200


def installed() -> list[Path]:
    return sorted(
        p for p in (MODELS.glob("*") if MODELS.exists() else [])
        if (p / CONTRACT_FILE).exists() and (p / "predictions_test.jsonl").exists()
    )


def rows_for(split: str) -> list[dict]:
    path = DATA / f"verdict_{split}.jsonl.gz"
    opener = gzip.open if path.exists() else open
    if not path.exists():
        path = DATA / f"verdict_{split}.jsonl"   # Kaggle decompresses on ingest
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


@pytest.mark.slow
@pytest.mark.parametrize("variant_dir", installed() or [pytest.param(None, marks=pytest.mark.skip(
    reason="no installed artifact; run scripts.install_artifacts first"))])
def test_exported_token_lengths_match_this_repository(variant_dir: Path):
    from transformers import AutoTokenizer

    contract = EncoderContract.from_dict(json.loads((variant_dir / CONTRACT_FILE).read_text()))
    tokenizer = AutoTokenizer.from_pretrained(str(variant_dir / "model_v1"))

    with open(variant_dir / "predictions_test.jsonl", encoding="utf-8") as handle:
        exported = {row["id"]: row for row in map(json.loads, handle)}
    if "token_len" not in next(iter(exported.values())):
        pytest.skip("artifact predates token_len export; retrain to check skew")

    def measure(first: str, second: str) -> int:
        return len(tokenizer(first, second)["input_ids"])

    # The notebook advances one Random across the whole split in file order, so every row must be
    # encoded to keep the stream in step -- assertions are only collected for the first N.
    rng = random.Random(contract.seed)
    checked = 0
    for row in rows_for("test"):
        evidence = select_evidence(row, contract.variant, contract.max_length, measure, rng)
        first, second = build_input(row["claim"], evidence)
        if checked >= CHECK_ROWS:
            break
        if row["id"] not in exported:
            continue
        length = len(tokenizer(first, second, truncation=True, max_length=contract.max_length)["input_ids"])
        assert length == exported[row["id"]]["token_len"], (
            f"{variant_dir.name} claim {row['id']}: this repository encodes {length} tokens, "
            f"the artifact was scored on {exported[row['id']]['token_len']}. "
            "encode_spec.py and encode.py have diverged, or the contract does not describe the run."
        )
        assert len(evidence) == exported[row["id"]]["n_evidence_used"]
        checked += 1

    assert checked > 0, "no exported claim matched the local test split"
