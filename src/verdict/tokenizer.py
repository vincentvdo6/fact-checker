"""Load a local fast tokenizer without replacing its serialized input recipe.

The tokenizer.json post-processor specifies which special tokens, if any, are
inserted for single and paired text. AutoTokenizer may rebuild that processor
from tokenizer_config.json under a newer Transformers release, changing the
token IDs sent to an otherwise unchanged ONNX graph. Load the serialized fast
backend directly and use the config only to name existing special tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


_NAMED_TOKENS = (
    "unk_token", "pad_token", "bos_token", "eos_token", "cls_token", "sep_token", "mask_token"
)


def load_frozen_tokenizer(directory: str | Path) -> PreTrainedTokenizerFast:
    """Return the offline tokenizer whose post-processor is exactly in tokenizer.json."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    directory = Path(directory)
    tokenizer_file = directory / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"frozen tokenizer artifact is missing: {tokenizer_file}")
    config_file = directory / "tokenizer_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
    names = {name: config[name] for name in _NAMED_TOKENS if isinstance(config.get(name), str)}
    backend = Tokenizer.from_file(str(tokenizer_file))
    missing = {name: value for name, value in names.items() if backend.token_to_id(value) is None}
    if missing:
        raise ValueError(f"special tokens absent from frozen tokenizer vocabulary: {missing}")
    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **names)
