"""
The dataset title has to carry its version, for the same reason the slug does.

The slug is versioned so a breaking rebuild cannot silently replace a dataset a checkpoint was
keyed against. A title hard-coded to v1 defeats half of that: two datasets with different contents
appear under the same name, and the one attached to a notebook is identified only by a slug nobody
reads in the Kaggle UI.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripts.upload_verdict_dataset import TITLE


def title_for(slug: str) -> str:
    """Mirrors the derivation in upload_verdict_dataset.main."""
    version = slug.rsplit("-", 1)[-1]
    return f"{TITLE} {version}" if re.fullmatch(r"v[0-9]+", version) else TITLE


def test_the_title_carries_the_version_from_the_slug():
    assert title_for("fever-verdict-v1") == "FEVER verdict training data v1"
    assert title_for("fever-verdict-v2") == "FEVER verdict training data v2"


def test_a_slug_without_a_version_keeps_the_bare_title():
    assert title_for("fever-verdict") == "FEVER verdict training data"
    assert title_for("something-else") == "FEVER verdict training data"
    # "verdict" starts with a v; only a full v<digits> match is a version.
    assert title_for("fever-verdict") != "FEVER verdict training data verdict"


def test_the_constant_no_longer_hard_codes_a_version():
    """It did, so every dataset after the first went out titled v1."""
    assert not TITLE.endswith("v1")


def test_the_script_derives_the_title_rather_than_using_the_constant_directly():
    source = Path("scripts/upload_verdict_dataset.py").read_text(encoding="utf-8")
    assert r're.fullmatch(r"v\d+", version)' in source
    assert '"title": title' in source
