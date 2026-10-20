"""The shipped tokenizer's serialized input recipe survives library upgrades."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from src.pipeline.detector import CheckworthyDetector
from src.verdict.budgets import _tokenizer as budget_tokenizer
from src.verdict.pair_judge import PairJudge
from src.verdict.runtime import VerdictRuntime
from src.verdict.tokenizer import load_frozen_tokenizer

VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
         "café": 4, "rose": 5, "blue": 6, "sky": 7, "extra": 8}


def frozen_fixture(tmp_path: Path, *, with_specials: bool) -> Path:
    directory = tmp_path / "model_v1"
    directory.mkdir()
    backend = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    if with_specials:
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
        )
    else:
        backend.post_processor = processors.TemplateProcessing(single="$A", pair="$A $B", special_tokens=[])
    backend.save(str(directory / "tokenizer.json"))
    # This metadata declares existing tokens but must never define the input recipe.
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "DebertaV2Tokenizer",
        "extra_special_tokens": ["[PAD]", "[CLS]", "[SEP]"],
        "unk_token": "[UNK]", "pad_token": "[PAD]", "cls_token": "[CLS]",
        "sep_token": "[SEP]", "bos_token": "[CLS]", "eos_token": "[SEP]",
    }), encoding="utf-8")
    return directory


@pytest.mark.parametrize("with_specials, expected_single, expected_pair", [
    (False, [4, 5], [4, 5, 6, 7]),
    (True, [2, 4, 5, 3], [2, 4, 5, 3, 6, 7, 3]),
])
def test_serialized_single_and_pair_recipes(tmp_path, with_specials, expected_single, expected_pair):
    directory = frozen_fixture(tmp_path, with_specials=with_specials)
    tokenizer = load_frozen_tokenizer(directory)
    direct = Tokenizer.from_file(str(directory / "tokenizer.json"))
    assert tokenizer("café rose")["input_ids"] == expected_single
    assert tokenizer("café rose", "blue sky")["input_ids"] == expected_pair
    assert direct.encode("café rose", "blue sky").ids == expected_pair
    assert tokenizer("café rose", "blue sky", return_token_type_ids=True)["token_type_ids"] == direct.encode(
        "café rose", "blue sky"
    ).type_ids
    assert tokenizer.pad_token_id == 0


@pytest.mark.parametrize("with_specials", [False, True])
def test_padding_masks_and_limits(tmp_path, with_specials):
    tokenizer = load_frozen_tokenizer(frozen_fixture(tmp_path, with_specials=with_specials))
    batch = tokenizer(["café", "café rose"], ["blue", "blue sky"],
                      padding=True, return_tensors="np")
    assert isinstance(batch["input_ids"], np.ndarray)
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"].sum(axis=1).tolist() == [5 if with_specials else 2,
                                                              7 if with_specials else 4]
    assert batch["input_ids"][0, -1] == 0
    assert batch["attention_mask"][0, -1] == 0
    clipped = tokenizer("café rose extra", "blue sky", truncation=True,
                        max_length=6 if with_specials else 3)
    assert len(clipped["input_ids"]) == (6 if with_specials else 3)


def test_missing_serialized_artifact_fails_locally(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text('{"pad_token":"[PAD]"}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        load_frozen_tokenizer(tmp_path)


def test_configured_special_token_must_be_in_serialized_vocabulary(tmp_path):
    directory = frozen_fixture(tmp_path, with_specials=False)
    config_file = directory / "tokenizer_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["mask_token"] = "[MISSING]"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="mask_token"):
        load_frozen_tokenizer(directory)


def test_serialized_left_padding_direction(tmp_path):
    directory = frozen_fixture(tmp_path, with_specials=False)
    backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
    backend.enable_padding(direction="left", pad_id=0, pad_token="[PAD]")
    backend.save(str(directory / "tokenizer.json"))
    tokenizer = load_frozen_tokenizer(directory)
    batch = tokenizer(["café", "café rose"], padding=True)
    assert tokenizer.padding_side == "left"
    assert batch["input_ids"][0] == [0, 4]
    assert batch["attention_mask"][0] == [0, 1]


@pytest.mark.parametrize("runtime_class", [PairJudge, CheckworthyDetector, VerdictRuntime])
def test_runtime_properties_lazy_load_the_frozen_recipe(tmp_path, runtime_class):
    frozen_fixture(tmp_path, with_specials=False)
    instance = runtime_class.__new__(runtime_class)
    instance.root = tmp_path
    instance._tokenizer = None
    assert instance._tokenizer is None
    assert instance.tokenizer("café", "blue")["input_ids"] == [4, 6]
    assert instance.tokenizer is instance._tokenizer
    if runtime_class is PairJudge:
        assert instance._token_lengths([("café", "blue")]) == [2]
    elif runtime_class is VerdictRuntime:
        assert instance.measure("café", "blue") == 2


def test_local_budget_source_uses_the_frozen_recipe(tmp_path):
    directory = frozen_fixture(tmp_path, with_specials=False)
    tokenizer = budget_tokenizer(str(directory))
    assert tokenizer("café", "blue")["input_ids"] == [4, 6]
